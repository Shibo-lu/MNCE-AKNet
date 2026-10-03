# -*- coding: utf-8 -*-
import torch
from torch import nn


class LearnerARKFNet(nn.Module):
    """Project-adapted ARKFNet learner.

    Branch 1 learns a rectified innovation vector.
    Branch 2 learns a positive-definite matrix that plays the role of an
    inverse innovation covariance / gain-shaping matrix.
    """

    def __init__(self, x_dim, y_dim, hidden_scale=1.0, seq_window=8, bidirectional=True):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.seq_window = seq_window
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        h1 = int((x_dim + y_dim) * 32 * hidden_scale)
        h2 = int((x_dim + y_dim) * 40 * hidden_scale)
        h3 = int((x_dim + y_dim) * 16 * hidden_scale)
        h4 = int((x_dim + y_dim) * 16 * hidden_scale)
        gru_hidden = int(2 * (x_dim * x_dim + y_dim * y_dim))

        self.l1 = nn.Sequential(nn.Linear(y_dim * 3, h1), nn.ReLU())
        self.gru1 = nn.GRU(h1, gru_hidden, batch_first=False, bidirectional=bidirectional)
        self.l2 = nn.Sequential(nn.Linear(gru_hidden * self.num_directions, h3), nn.ReLU(), nn.Linear(h3, y_dim))

        self.l3 = nn.Sequential(nn.Linear(y_dim * 3, h2), nn.ReLU())
        self.gru2 = nn.GRU(h2, gru_hidden, batch_first=False)
        self.l4 = nn.Sequential(nn.Linear(gru_hidden, h4), nn.ReLU(), nn.Linear(h4, y_dim * y_dim))

        self.h1_state = None
        self.h2_state = None

    def reset(self, batch_size, device):
        num_layers_1 = self.gru1.num_layers * self.num_directions
        num_layers_2 = self.gru2.num_layers
        hidden_1 = self.gru1.hidden_size
        hidden_2 = self.gru2.hidden_size
        self.h1_state = torch.zeros(num_layers_1, batch_size, hidden_1, device=device)
        self.h2_state = torch.zeros(num_layers_2, batch_size, hidden_2, device=device)

    def _gain_scale(self, raw_mat):
        raw_mat = torch.nan_to_num(raw_mat, nan=0.0, posinf=20.0, neginf=-20.0)
        symm = 0.5 * (raw_mat + raw_mat.transpose(-1, -2))
        diag = torch.diagonal(symm, dim1=-2, dim2=-1)
        # Start near identity and let the network scale each measurement channel gently.
        scale = 0.5 + torch.sigmoid(diag)
        return torch.diag_embed(scale)

    def forward(self, residual_seq, diff_obs_seq, diff_pre_y_seq, is_first=False):
        batch_size = residual_seq.shape[0]
        device = residual_seq.device
        if is_first or self.h1_state is None or self.h1_state.shape[1] != batch_size:
            self.reset(batch_size, device)

        input1 = torch.cat((residual_seq, diff_obs_seq, diff_pre_y_seq), dim=1).permute(2, 0, 1)
        input1 = input1[-self.seq_window :, :, :]

        l1_out = self.l1(input1)
        gru1_out, self.h1_state = self.gru1(l1_out, self.h1_state)
        residual_delta = 0.1 * torch.tanh(self.l2(gru1_out[-1])).reshape(batch_size, self.y_dim, 1)

        current_residual = residual_seq[:, :, -1:]
        current_diff_obs = diff_obs_seq[:, :, -1:]
        residual_rectify = current_residual + residual_delta
        input2 = torch.cat((residual_rectify, current_residual, current_diff_obs), dim=1).permute(2, 0, 1)
        l3_out = self.l3(input2)
        gru2_out, self.h2_state = self.gru2(l3_out, self.h2_state)
        raw_precision = self.l4(gru2_out[-1]).reshape(batch_size, self.y_dim, self.y_dim)
        gain_scale = self._gain_scale(raw_precision)
        return residual_rectify, gain_scale
