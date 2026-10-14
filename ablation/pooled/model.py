from __future__ import annotations

import torch
from torch import nn


class PooledDNN(nn.Module):
    """Unified predictor for pooled cohorts without MMoE or task-specific layers."""

    def __init__(self, input_dim: int, dropout: float = 0.4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)
