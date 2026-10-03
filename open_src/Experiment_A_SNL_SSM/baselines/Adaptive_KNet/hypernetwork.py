# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperNetwork(nn.Module):
    """Generate context-modulation shift/gain vectors from a noise descriptor."""

    def __init__(self, sow_dim, output_size, hidden_size=256):
        super().__init__()
        self.sow_dim = sow_dim
        self.output_size = output_size
        self.position_embeddings = (0.0, 1.0)  # shift, gain

        self.fc1 = nn.Linear(sow_dim + 1, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def init_hidden(self, device):
        weight = next(self.parameters())
        self.hgru = weight.new_zeros(1, 1, self.gru.hidden_size, device=device)

    def forward(self, sow):
        if sow.dim() == 0:
            sow = sow.reshape(1)
        sow = torch.log10(sow.float().clamp_min(1e-12))

        outputs = []
        for pe in self.position_embeddings:
            pe_tensor = sow.new_tensor([pe])
            x = torch.cat([pe_tensor, sow], dim=0).unsqueeze(0)
            x = F.relu(self.fc1(x)).unsqueeze(0)
            x, self.hgru = self.gru(x, self.hgru)
            outputs.append(self.fc2(x.squeeze(0)).squeeze(0))
        return outputs[0], outputs[1]
