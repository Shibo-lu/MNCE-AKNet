# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

from Dynamics import AcceleratedMovementModel


def _acc_at(acc, time_index):
    """Return [batch, control_dim] from constant or time-indexed controls."""
    return acc[:, time_index] if acc.dim() == 3 else acc


def measurement_from_state(x):
    px, py, pz = x[:, 0], x[:, 1], x[:, 2]
    radius = torch.sqrt(px.square() + py.square() + pz.square() + 1e-8)
    return torch.stack([radius, x[:, 3], x[:, 4], x[:, 5]], dim=-1)


def jacobian_measurement(x, device):
    batch_size = x.shape[0]
    px, py, pz = x[:, 0], x[:, 1], x[:, 2]
    radius = torch.sqrt(px.square() + py.square() + pz.square() + 1e-8)
    row0 = torch.stack([px / radius, py / radius, pz / radius], dim=1)
    row0 = torch.cat([row0, torch.zeros(batch_size, 3, device=device)], dim=1)
    row1 = torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.float32, device=device).expand(batch_size, 6)
    row2 = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32, device=device).expand(batch_size, 6)
    row3 = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32, device=device).expand(batch_size, 6)
    return torch.stack([row0, row1, row2, row3], dim=1)


class ProjectARKFNetFilter:
    def __init__(self, dt, process_noise, measurement_noise, initial_covariance, slide_window=8, device="cpu"):
        self.device = torch.device(device)
        self.slide_window = slide_window

        dynamics = AcceleratedMovementModel(float(dt), device=self.device)
        self.x_dim, self.y_dim, _, self.F, _, self.B_mat = dynamics.get_dynamics()

        self.Q = process_noise.to(self.device)
        self.R = measurement_noise.to(self.device)
        self.P0 = initial_covariance.to(self.device)

        self.state_post = None
        self.cov_post = None
        self.obs_past = None
        self.y_predict_past = None
        self.residual_seq = None
        self.diff_obs_seq = None
        self.diff_pre_y_seq = None
        self.first = True

    def reset(self, batch_size, initial_state):
        self.state_post = initial_state.to(self.device)
        self.cov_post = self.P0.unsqueeze(0).expand(batch_size, self.x_dim, self.x_dim).clone()
        self.obs_past = torch.zeros(batch_size, self.y_dim, 1, device=self.device)
        self.y_predict_past = torch.zeros(batch_size, self.y_dim, 1, device=self.device)
        self.residual_seq = torch.zeros(batch_size, self.y_dim, self.slide_window, device=self.device)
        self.diff_obs_seq = torch.zeros(batch_size, self.y_dim, self.slide_window, device=self.device)
        self.diff_pre_y_seq = torch.zeros(batch_size, self.y_dim, self.slide_window, device=self.device)
        self.first = True

    @staticmethod
    def _stable_covariance(cov):
        cov = 0.5 * (cov + cov.transpose(1, 2))
        eye = torch.eye(cov.size(-1), device=cov.device).unsqueeze(0)
        return cov + 1e-6 * eye

    def step_arkfnet(self, observation, acc, task_net):
        batch_size = observation.shape[0]
        F_batch = self.F.unsqueeze(0).expand(batch_size, self.x_dim, self.x_dim)

        x_predict = self.state_post @ self.F.T + acc @ self.B_mat.T
        H_jacob = jacobian_measurement(x_predict, self.device)
        cov_pred = F_batch @ self.cov_post @ F_batch.transpose(1, 2) + self.Q.unsqueeze(0)
        cov_pred = self._stable_covariance(cov_pred)

        y_predict = measurement_from_state(x_predict).unsqueeze(-1)
        residual = observation - y_predict
        diff_obs = observation - self.obs_past
        diff_pre_y = y_predict - self.y_predict_past

        if self.first:
            self.residual_seq = residual.repeat(1, 1, self.slide_window)
            self.diff_obs_seq = diff_obs.repeat(1, 1, self.slide_window)
            self.diff_pre_y_seq = diff_pre_y.repeat(1, 1, self.slide_window)
        else:
            self.residual_seq = torch.cat((self.residual_seq[:, :, 1:], residual), dim=2)
            self.diff_obs_seq = torch.cat((self.diff_obs_seq[:, :, 1:], diff_obs), dim=2)
            self.diff_pre_y_seq = torch.cat((self.diff_pre_y_seq[:, :, 1:], diff_pre_y), dim=2)

        residual_rectify, gain_scale = task_net(
            self.residual_seq, self.diff_obs_seq, self.diff_pre_y_seq, is_first=self.first
        )

        s_mat = H_jacob @ cov_pred @ H_jacob.transpose(1, 2) + self.R.unsqueeze(0)
        s_mat = self._stable_covariance(s_mat)
        s_inv_base = torch.linalg.pinv(s_mat)
        sk_inv = gain_scale @ s_inv_base
        K_gain = cov_pred @ H_jacob.transpose(1, 2) @ sk_inv
        x_post = x_predict.unsqueeze(-1) + K_gain @ residual_rectify
        x_post = x_post.squeeze(-1)

        identity = torch.eye(self.x_dim, device=self.device).unsqueeze(0).expand(batch_size, self.x_dim, self.x_dim)
        joseph_left = identity - K_gain @ H_jacob
        self.cov_post = joseph_left @ cov_pred @ joseph_left.transpose(1, 2) + K_gain @ self.R.unsqueeze(0) @ K_gain.transpose(1, 2)
        self.cov_post = self._stable_covariance(self.cov_post)

        self.state_post = x_post.detach()
        self.obs_past = observation.detach()
        self.y_predict_past = y_predict.detach()
        self.first = False
        return x_post

    def step_ekf(self, observation, acc):
        batch_size = observation.shape[0]
        F_batch = self.F.unsqueeze(0).expand(batch_size, self.x_dim, self.x_dim)
        x_predict = self.state_post @ self.F.T + acc @ self.B_mat.T
        y_predict = measurement_from_state(x_predict).unsqueeze(-1)
        residual = observation - y_predict
        H_jacob = jacobian_measurement(x_predict, self.device)

        cov_pred = F_batch @ self.cov_post @ F_batch.transpose(1, 2) + self.Q.unsqueeze(0)
        cov_pred = self._stable_covariance(cov_pred)
        s_mat = H_jacob @ cov_pred @ H_jacob.transpose(1, 2) + self.R.unsqueeze(0)
        k_gain = cov_pred @ H_jacob.transpose(1, 2) @ torch.linalg.pinv(s_mat)

        x_post = x_predict.unsqueeze(-1) + k_gain @ residual
        x_post = x_post.squeeze(-1)

        identity = torch.eye(self.x_dim, device=self.device).unsqueeze(0).expand(batch_size, self.x_dim, self.x_dim)
        self.cov_post = (identity - k_gain @ H_jacob) @ cov_pred
        self.cov_post = self._stable_covariance(self.cov_post)
        self.state_post = x_post.detach()
        return x_post

    def train_loss(self, x_seg, z_seg, acc, task_net, use_true_x0=True, fixed_initial_state=None):
        batch_size, seq_len, _ = x_seg.shape
        if use_true_x0:
            initial_state = x_seg[:, 0, :]
        else:
            initial_state = fixed_initial_state.unsqueeze(0).expand(batch_size, -1).clone()
        self.reset(batch_size, initial_state)
        task_net.reset(batch_size, x_seg.device)

        outputs = []
        for t in range(seq_len):
            y_t = z_seg[:, t, :].unsqueeze(-1)
            x_post = self.step_arkfnet(y_t, _acc_at(acc, t), task_net)
            outputs.append(x_post)
        x_est = torch.stack(outputs, dim=1)
        return F.mse_loss(x_est, x_seg), x_est

    @torch.no_grad()
    def eval_dataset(self, x_seq, z_seq, acc, task_net, batch_size, use_true_x0=True, fixed_initial_state=None):
        total_mse = 0.0
        total_pos_mse = 0.0
        total_vel_mse = 0.0
        num_trajectories = 0
        n_traj = x_seq.shape[0]

        for start in range(0, n_traj, batch_size):
            end = min(start + batch_size, n_traj)
            current_batch_size = end - start
            xb = x_seq[start:end]
            zb = z_seq[start:end]
            ab = acc[start:end]
            if use_true_x0:
                initial_state = xb[:, 0, :]
            else:
                initial_state = fixed_initial_state.unsqueeze(0).expand(current_batch_size, -1).clone()
            self.reset(current_batch_size, initial_state)
            task_net.reset(current_batch_size, x_seq.device)

            outputs = []
            for t in range(xb.shape[1]):
                y_t = zb[:, t, :].unsqueeze(-1)
                outputs.append(self.step_arkfnet(y_t, _acc_at(ab, t), task_net))
            x_est = torch.stack(outputs, dim=1)

            total_mse += F.mse_loss(x_est, xb).item() * current_batch_size
            total_pos_mse += F.mse_loss(x_est[:, :, :3], xb[:, :, :3]).item() * current_batch_size
            total_vel_mse += F.mse_loss(x_est[:, :, 3:], xb[:, :, 3:]).item() * current_batch_size
            num_trajectories += current_batch_size

        denom = max(num_trajectories, 1)
        return {
            "mse": total_mse / denom,
            "pos_mse": total_pos_mse / denom,
            "vel_mse": total_vel_mse / denom,
        }
