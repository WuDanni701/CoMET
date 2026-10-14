"""Only-MTL model: shared MMoE representations and cohort heads only."""

from __future__ import annotations

import torch
from torch import nn

from model import sharedEncoder


class SharedOnlyHead(nn.Module):
    def __init__(self, shared_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.shared_projection = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
        )
        self.tower = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.tower(self.shared_projection(features)))


class OnlyMTL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expert_hidden_units: list[int],
        num_experts: int,
        gate_hidden_units: list[int],
        num_tasks: int,
        fusion_dim: int,
        l2_reg: float,
        dropout_rate: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.shared_encoder = sharedEncoder(
            input_dim=input_dim, expert_dnn_hidden_units=expert_hidden_units,
            num_experts=num_experts, gate_dnn_hidden_units=gate_hidden_units,
            num_tasks=num_tasks, l2_reg=l2_reg, dnn_activation="relu",
            dropout_rate=dropout_rate, use_bn=True, init_std=0.001, device=device,
        )
        self.task_heads = nn.ModuleList(
            [SharedOnlyHead(expert_hidden_units[-1], fusion_dim) for _ in range(num_tasks)]
        )
        self.to(device)

    def forward(self, shared_input, specific_input, task_ids, task_specific_slices):
        task_features, regularization_loss = self.shared_encoder(shared_input)
        logits = [None] * self.num_tasks
        for task_id in range(self.num_tasks):
            indices = (task_ids == task_id).nonzero(as_tuple=True)[0]
            if len(indices):
                logits[task_id] = self.task_heads[task_id](task_features[task_id][indices])
        return logits, regularization_loss
