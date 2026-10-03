# -*- coding: utf-8 -*-
import torch

from Dynamics import AcceleratedMovementModel


class SageHusaEKF:
    """Sage-Husa adaptive EKF for the local 6D state / 4D measurement model.

    State: [x, y, z, vx, vy, vz]
    Measurement: [range, vx, vy, vz]

    The implementation follows the local EKF/AEKF style and adapts the
    process/measurement noise statistics online. By default it uses the same
    running-average schedule as the provided MATLAB snippet, and it can also
    use the classic forgetting-factor schedule.
    """

    def __init__(
        self,
        initial_state,
        initial_covariance,
        process_noise,
        measurement_noise,
        dt,
        b=0.95,
        schedule="running_average",
        warmup_steps=10,
        diagonal_only=True,
        adapt_q_mean=False,
        adapt_r_mean=False,
        adapt_q=True,
        adapt_r=True,
        q_mean_init=None,
        r_mean_init=None,
        q_bounds=(1e-8, 1e6),
        r_bounds=(1e-8, 1e6),
        device=None,
    ):
        self.device = torch.device(device or "cpu")
        self.b = float(b)
        self.schedule = schedule
        self.warmup_steps = int(warmup_steps)
        self.diagonal_only = bool(diagonal_only)
        self.adapt_q_mean = bool(adapt_q_mean)
        self.adapt_r_mean = bool(adapt_r_mean)
        self.adapt_q = bool(adapt_q)
        self.adapt_r = bool(adapt_r)
        self.q_bounds = q_bounds
        self.r_bounds = r_bounds

        dynamics = AcceleratedMovementModel(float(dt), device=self.device)
        self.state_dim, self.obs_dim, self.acc_dim, self.F, _, self.B_mat = dynamics.get_dynamics()

        self.state = self._to_batch(initial_state, (self.state_dim,), self.device)
        self.covariance = self._to_batch(initial_covariance, (self.state_dim, self.state_dim), self.device)
        self.covariance_pre = self.covariance.clone()
        self.Q = self._to_batch(process_noise, (self.state_dim, self.state_dim), self.device)
        self.R = self._to_batch(measurement_noise, (self.obs_dim, self.obs_dim), self.device)
        self.Q = self._project_noise_covariance(self.Q, self.q_bounds)
        self.R = self._project_noise_covariance(self.R, self.r_bounds)

        batch_size = self.state.shape[0]
        self.state_pre = self.state.clone()
        self.innovation = torch.zeros(batch_size, self.obs_dim, device=self.device)
        self.innovation_diff = torch.zeros(batch_size, self.obs_dim, device=self.device)
        self.predict_error = torch.zeros(batch_size, self.state_dim, device=self.device)
        self.H = torch.zeros(batch_size, self.obs_dim, self.state_dim, device=self.device)

        self.q_mean = self._init_mean(q_mean_init, batch_size, self.state_dim)
        self.r_mean = self._init_mean(r_mean_init, batch_size, self.obs_dim)
        self.step_count = 0

    @staticmethod
    def _to_batch(x, single_shape, device):
        if isinstance(x, torch.Tensor):
            t = x.detach().clone().float().to(device)
        else:
            t = torch.as_tensor(x, dtype=torch.float32, device=device)

        if t.dim() == len(single_shape) and tuple(t.shape) == tuple(single_shape):
            t = t.unsqueeze(0)
        return t

    def _init_mean(self, x, batch_size, dim):
        if x is None:
            return torch.zeros(batch_size, dim, device=self.device)
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[0] == 1 and batch_size > 1:
            x = x.expand(batch_size, dim).clone()
        return x

    @staticmethod
    def _symmetrize(mat):
        return 0.5 * (mat + mat.transpose(-1, -2))

    def _stabilize_cov(self, cov, bounds):
        cov = self._symmetrize(cov)
        low, high = bounds
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.clamp(min=low, max=high)
        cov = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
        return self._symmetrize(cov)

    def _project_noise_covariance(self, cov, bounds):
        """Stabilize Q/R and optionally restrict only these matrices to diagonal form."""
        cov = self._stabilize_cov(cov, bounds)
        if self.diagonal_only:
            cov = torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
            cov = self._stabilize_cov(cov, bounds)
        return cov

    def _measurement(self, x):
        px, py, pz = x[:, 0], x[:, 1], x[:, 2]
        radius = torch.sqrt(px.square() + py.square() + pz.square() + 1e-8)
        return torch.stack([radius, x[:, 3], x[:, 4], x[:, 5]], dim=-1)

    def _measurement_jacobian(self, x):
        batch_size = x.shape[0]
        px, py, pz = x[:, 0], x[:, 1], x[:, 2]
        radius = torch.sqrt(px.square() + py.square() + pz.square() + 1e-8)
        row0 = torch.stack([px / radius, py / radius, pz / radius], dim=1)
        row0 = torch.cat([row0, torch.zeros(batch_size, 3, device=self.device)], dim=1)
        row1 = torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.float32, device=self.device).expand(batch_size, 6)
        row2 = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32, device=self.device).expand(batch_size, 6)
        row3 = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32, device=self.device).expand(batch_size, 6)
        return torch.stack([row0, row1, row2, row3], dim=1)

    def _gain_weight(self):
        k = self.step_count
        if self.schedule == "classic":
            return (1.0 - self.b) / max(1e-8, 1.0 - self.b ** k)
        return 1.0 / float(k)

    def predict(self, u=None):
        batch_size = self.state.shape[0]
        self.P_prev = self.covariance.clone()
        self.x_nominal_prev = self.state.clone()

        if u is None:
            control_term = torch.zeros(batch_size, self.state_dim, device=self.device)
        else:
            u = torch.as_tensor(u, dtype=torch.float32, device=self.device)
            if u.dim() == 1:
                u = u.unsqueeze(0)
            control_term = u @ self.B_mat.T

        self.control_term = control_term
        x_nominal = self.state @ self.F.T + control_term
        self.state_pre = x_nominal + self.q_mean
        self.x_nominal_pred = x_nominal

        F_batch = self.F.unsqueeze(0).expand(batch_size, self.state_dim, self.state_dim)
        self.covariance_pre = F_batch @ self.P_prev @ F_batch.transpose(1, 2) + self.Q
        # P^- must retain state cross-covariances; only Q/R may be diagonalized.
        self.covariance_pre = self._stabilize_cov(self.covariance_pre, self.q_bounds)

    def update(self, z):
        batch_size = self.state.shape[0]
        self.step_count += 1
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        if z.dim() == 1:
            z = z.unsqueeze(0)

        H = self._measurement_jacobian(self.state_pre)
        H_t = H.transpose(1, 2)
        y_pre = self._measurement(self.state_pre)
        innovation = z - y_pre - self.r_mean

        S = H @ self.covariance_pre @ H_t + self.R
        # Preserve innovation cross-covariances in S.
        S = self._stabilize_cov(S, self.r_bounds)
        K = self.covariance_pre @ H_t @ torch.linalg.pinv(S)

        self.state = self.state_pre + (K @ innovation.unsqueeze(-1)).squeeze(-1)

        identity = torch.eye(self.state_dim, dtype=torch.float32, device=self.device).unsqueeze(0)
        identity = identity.expand(batch_size, self.state_dim, self.state_dim)
        joseph_left = identity - K @ H
        self.covariance = joseph_left @ self.covariance_pre @ joseph_left.transpose(1, 2) + K @ self.R @ K.transpose(1, 2)
        # P^+ also remains a full covariance matrix.
        self.covariance = self._stabilize_cov(self.covariance, self.q_bounds)

        residual = z - self._measurement(self.state) - self.r_mean
        prev_innovation = self.innovation
        self.innovation = innovation
        self.innovation_diff = innovation - prev_innovation
        self.predict_error = self.state - self.state_pre
        self.H = H

        x_res_cov = K @ S @ K.transpose(1, 2)
        x_res_cov = self._stabilize_cov(x_res_cov, self.q_bounds)

        if self.step_count > self.warmup_steps:
            self._adapt_statistics(z, K, H, innovation)

        return S, residual, innovation, self.predict_error, x_res_cov

    def _adapt_statistics(self, z, K, H, innovation):
        d = self._gain_weight()
        d_tensor = torch.tensor(d, dtype=torch.float32, device=self.device)

        y_pre = self._measurement(self.state_pre)
        z_centered = z - y_pre
        if self.adapt_r_mean:
            self.r_mean = (1.0 - d_tensor) * self.r_mean + d_tensor * z_centered

        q_sample = self.state - self.x_nominal_pred
        if self.adapt_q_mean:
            self.q_mean = (1.0 - d_tensor) * self.q_mean + d_tensor * q_sample

        centered_innovation = innovation.unsqueeze(-1) @ innovation.unsqueeze(-2)
        r_hat = centered_innovation - H @ self.covariance_pre @ H.transpose(1, 2)

        F_batch = self.F.unsqueeze(0).expand_as(self.P_prev)
        q_hat = K @ centered_innovation @ K.transpose(1, 2) + self.covariance - F_batch @ self.P_prev @ F_batch.transpose(1, 2)

        if self.adapt_r:
            self.R = self._project_noise_covariance(
                (1.0 - d_tensor) * self.R + d_tensor * r_hat,
                self.r_bounds,
            )
        if self.adapt_q:
            self.Q = self._project_noise_covariance(
                (1.0 - d_tensor) * self.Q + d_tensor * q_hat,
                self.q_bounds,
            )

    def step(self, z, u=None):
        self.predict(u)
        return self.update(z)

    def get_state(self):
        return self.state

    def get_covariance(self):
        return self.covariance

    def get_QandR(self):
        return self.Q, self.R

    def get_noise_means(self):
        return self.q_mean, self.r_mean

    def get_innovation(self):
        return self.innovation

    def get_innovation_diff(self):
        return self.innovation_diff

    def get_predict_error(self):
        return self.predict_error

    def get_H(self):
        return self.H
