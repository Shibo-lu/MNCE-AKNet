# -*- coding: utf-8 -*-
"""Kalman-gain learner matching the paper's FC-GRU-FC backbone."""

import torch
from torch import nn


class Learner(nn.Module):
    """Predict a Kalman gain from the features used by MAML-KalmanNet."""

    def __init__(self, state_dim=6, measurement_dim=4, nonlinear=True):
        super().__init__()
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.nonlinear = nonlinear

        input_dim = 2 * state_dim + 2 * measurement_dim
        if nonlinear:
            input_dim += state_dim**2 + state_dim * measurement_dim
        first_hidden = (state_dim + measurement_dim) * 10 * 4
        self.gru_hidden_dim = 4 * (state_dim**2 + measurement_dim**2)

        # These layer widths and activations are unchanged from the source.
        self.l1 = nn.Sequential(nn.Linear(input_dim, first_hidden), nn.ReLU())
        self.GRU = nn.GRU(
            input_size=first_hidden,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
        )
        self.l2 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, state_dim * measurement_dim * 4),
            nn.ReLU(),
            nn.Linear(state_dim * measurement_dim * 4, state_dim * measurement_dim),
        )

        # The source code uses two fixed random initial hidden states. Keeping
        # one template for support/adaptation and one for query/evaluation
        # preserves that behavior while allowing a dynamic batch size.
        # The source allocates a distinct fixed random hidden vector for every
        # batch member. A non-persistent bank preserves this behavior while
        # permitting the current scripts to use different batch sizes.
        self.register_buffer(
            "hn_train_init",
            torch.randn(1, 1024, self.gru_hidden_dim),
            persistent=False,
        )
        self.register_buffer(
            "hn_query_init",
            torch.randn(1, 1024, self.gru_hidden_dim),
            persistent=False,
        )
        self.hn = None

    def initialize_hidden(self, batch_size, query=False, device=None):
        template = self.hn_query_init if query else self.hn_train_init
        if device is None:
            device = next(self.parameters()).device
        if batch_size > template.size(1):
            raise ValueError("Batch size exceeds the source-style hidden bank.")
        self.hn = template[:, :batch_size].to(device).detach().clone()

    def forward(
        self,
        state_innovation,
        residual,
        state_difference,
        measurement_difference,
        transition_jacobian,
        measurement_jacobian,
    ):
        batch_size = residual.size(0)
        features = [
            state_innovation,
            residual,
            state_difference,
            measurement_difference,
        ]
        if self.nonlinear:
            features.extend(
                [
                    transition_jacobian.reshape(batch_size, -1),
                    measurement_jacobian.reshape(batch_size, -1),
                ]
            )
        network_input = torch.cat(features, dim=1)
        first_output = self.l1(network_input).unsqueeze(0)
        if self.hn is None or self.hn.size(1) != batch_size:
            self.initialize_hidden(batch_size, device=network_input.device)
        gru_output, hidden = self.GRU(first_output, self.hn)
        # This detach is intentional and matches the paper source code.
        self.hn = hidden.detach().clone()
        gain = self.l2(gru_output.squeeze(0))
        return gain.reshape(
            batch_size, self.state_dim, self.measurement_dim
        )
