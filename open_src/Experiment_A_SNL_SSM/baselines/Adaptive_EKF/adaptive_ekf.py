# -*- coding: utf-8 -*-
import torch

from Dynamics import AcceleratedMovementModel


class WindowAdaptiveEKF:
    """Adaptive EKF with residual/innovation covariance noise estimation.

    The EKF equations follow the local AEKF.py implementation. After each
    correction step, R is estimated from the posterior residual covariance and
    Q is estimated from the innovation covariance, using a sliding window.
    """

    def __init__(
        self,
        initial_state,
        initial_covariance,
        process_noise,
        measurement_noise,
        dt,
        window_size=64,
        forgetting_factor=0.99,
        min_window=5,
        q_bounds=(1e-8, 1e6),
        r_bounds=(1e-8, 1e6),
        adapt_noise=True,
        device=None,
    ):
        self.device = torch.device(device or "cpu")
        self.window_size = int(window_size)
        self.forgetting_factor = float(forgetting_factor)
        self.min_window = int(min_window)
        self.q_bounds = q_bounds
        self.r_bounds = r_bounds
        self.adapt_noise = bool(adapt_noise)

        dynamics = AcceleratedMovementModel(float(dt), device=self.device)
        self.state_dim, self.obs_dim, self.acc_dim, self.F, _, self.B_mat = dynamics.get_dynamics()

        self.state = self._to_batch(initial_state, (self.state_dim,), self.device)
        self.covariance = self._to_batch(initial_covariance, (self.state_dim, self.state_dim), self.device)
        self.covariance_pre = self.covariance.clone()
        self.Q = self._to_batch(process_noise, (self.state_dim, self.state_dim), self.device)
        self.R = self._to_batch(measurement_noise, (self.obs_dim, self.obs_dim), self.device)

        batch_size = self.state.shape[0]
        self.state_pre = self.state.clone()
        self.innovation = torch.zeros(batch_size, self.obs_dim, device=self.device)
        self.innovation_diff = torch.zeros(batch_size, self.obs_dim, device=self.device)
        self.predict_error = torch.zeros(batch_size, self.state_dim, device=self.device)
        self.H = torch.zeros(batch_size, self.obs_dim, self.state_dim, device=self.device)

        self.residual_window = torch.zeros(batch_size, self.window_size, self.obs_dim, device=self.device)
        self.innovation_window = torch.zeros(batch_size, self.window_size, self.obs_dim, device=self.device)
        self.window_count = 0

    @staticmethod
    def _to_batch(x, single_shape, device):
        if isinstance(x, torch.Tensor):
            t = x.detach().clone().float().to(device)
        else:
            t = torch.as_tensor(x, dtype=torch.float32, device=device)

        if t.dim() == len(single_shape) and tuple(t.shape) == tuple(single_shape):
            t = t.unsqueeze(0)
        return t

    def _expand_to_batch(self, x, single_shape, batch_size):
        x = self._to_batch(x, single_shape, self.device)
        if x.shape[0] == 1 and batch_size > 1:
            x = x.expand(batch_size, *x.shape[1:]).clone()
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

    def _window_cov(self, window, count):
        valid_len = min(count, self.window_size)
        samples = window[:, :valid_len, :]
        centered = samples - samples.mean(dim=1, keepdim=True)
        denom = max(valid_len - 1, 1)
        return centered.transpose(1, 2) @ centered / denom

    def _push_window(self, residual, innovation):
        if self.window_count < self.window_size:
            idx = self.window_count
            self.residual_window[:, idx, :] = residual
            self.innovation_window[:, idx, :] = innovation
        else:
            self.residual_window = torch.roll(self.residual_window, shifts=-1, dims=1)
            self.innovation_window = torch.roll(self.innovation_window, shifts=-1, dims=1)
            self.residual_window[:, -1, :] = residual
            self.innovation_window[:, -1, :] = innovation
        self.window_count += 1

    def predict(self, u=None, adaptive_Q=None):
        batch_size = self.state.shape[0]
        if adaptive_Q is not None:
            self.Q = self._expand_to_batch(adaptive_Q, (self.state_dim, self.state_dim), batch_size)

        if u is None:
            self.state_pre = self.state @ self.F.T
        else:
            u = torch.as_tensor(u, dtype=torch.float32, device=self.device)
            if u.dim() == 1:
                u = u.unsqueeze(0)
            self.state_pre = self.state @ self.F.T + u @ self.B_mat.T

        F_batch = self.F.unsqueeze(0).expand(batch_size, self.state_dim, self.state_dim)
        self.covariance_pre = F_batch @ self.covariance @ F_batch.transpose(1, 2) + self.Q
        self.covariance_pre = self._stabilize_cov(self.covariance_pre, self.q_bounds)

    def update(self, z, adaptive_R=None):
        batch_size = self.state.shape[0]
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if adaptive_R is not None:
            self.R = self._expand_to_batch(adaptive_R, (self.obs_dim, self.obs_dim), batch_size)

        H = self._measurement_jacobian(self.state_pre)
        H_t = H.transpose(1, 2)
        y_pre = self._measurement(self.state_pre)
        innovation = z - y_pre

        S = H @ self.covariance_pre @ H_t + self.R
        S = self._stabilize_cov(S, self.r_bounds)
        K = self.covariance_pre @ H_t @ torch.linalg.pinv(S)

        self.state = self.state_pre + (K @ innovation.unsqueeze(-1)).squeeze(-1)

        identity = torch.eye(self.state_dim, dtype=torch.float32, device=self.device).unsqueeze(0)
        identity = identity.expand(batch_size, self.state_dim, self.state_dim)
        KH = K @ H
        joseph_left = identity - KH
        self.covariance = joseph_left @ self.covariance_pre @ joseph_left.transpose(1, 2) + K @ self.R @ K.transpose(1, 2)
        self.covariance = self._stabilize_cov(self.covariance, self.q_bounds)

        residual = z - self._measurement(self.state)
        previous_innovation = self.innovation
        self.innovation = innovation
        self.innovation_diff = innovation - previous_innovation
        self.predict_error = self.state - self.state_pre
        self.H = H

        if self.adapt_noise:
            self._push_window(residual.detach(), innovation.detach())
            if min(self.window_count, self.window_size) >= self.min_window:
                self._adapt_noise(K.detach(), H.detach())

        x_res_cov = K @ S @ K.transpose(1, 2)
        x_res_cov = self._stabilize_cov(x_res_cov, self.q_bounds)
        return S, residual, innovation, self.predict_error, x_res_cov

    def _adapt_noise(self, K, H):
        residual_cov = self._window_cov(self.residual_window, self.window_count)
        innovation_cov = self._window_cov(self.innovation_window, self.window_count)

        r_hat = residual_cov + H @ self.covariance @ H.transpose(1, 2)
        q_hat = K @ innovation_cov @ K.transpose(1, 2)

        beta = self.forgetting_factor
        self.R = self._stabilize_cov(beta * self.R + (1.0 - beta) * r_hat, self.r_bounds)
        self.Q = self._stabilize_cov(beta * self.Q + (1.0 - beta) * q_hat, self.q_bounds)

    def step(self, z, u=None):
        self.predict(u)
        return self.update(z)

    def get_state(self):
        return self.state

    def get_covariance(self):
        return self.covariance

    def get_QandR(self):
        return self.Q, self.R

    def get_innovation(self):
        return self.innovation

    def get_innovation_diff(self):
        return self.innovation_diff

    def get_predict_error(self):
        return self.predict_error

    def get_H(self):
        return self.H
