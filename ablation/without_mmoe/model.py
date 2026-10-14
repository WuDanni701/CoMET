import torch
import torch.nn as nn
from model import TaskSpecificHead

class SharedDNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_units=(64,),
        dropout_rate=0.3,
    ):
        super().__init__()

        if len(hidden_units) == 0:
            raise ValueError("hidden_units must contain at least one hidden dimension.")

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_units:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            in_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_units[-1]

    def forward(self, x):
        return self.network(x)

class NoMMoESharedEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        shared_dnn_hidden_units,
        num_tasks,
        dropout_rate=0.3,
    ):
        super().__init__()

        self.num_tasks = num_tasks

        self.shared_dnn = SharedDNN(
            input_dim=input_dim,
            hidden_units=shared_dnn_hidden_units,
            dropout_rate=dropout_rate,
        )

    def forward(self, dnn_input):
        shared_out = self.shared_dnn(dnn_input)

        shared_outs = [
            shared_out for _ in range(self.num_tasks)
        ]

        total_loss = shared_out.new_zeros(())

        return shared_outs, total_loss

class ICIMTL_NoMMoE(nn.Module):

    def __init__(
        self,
        shared_input_dim,
        shared_dnn_hidden_units,
        num_tasks,
        task_feature_dims,
        fusion_dim,
        dropout_rate=0.4,
        seed=42,
        device="cpu",
    ):
        super().__init__()

        torch.manual_seed(seed)

        self.num_tasks = num_tasks
        self.device = device

        self.shared_encoder = NoMMoESharedEncoder(
            input_dim=shared_input_dim,
            shared_dnn_hidden_units=shared_dnn_hidden_units,
            num_tasks=num_tasks,
            dropout_rate=dropout_rate,
        )

        self.task_heads = nn.ModuleList()

        for i in range(num_tasks):
            task_head = TaskSpecificHead(
                shared_dim=shared_dnn_hidden_units[-1],
                cohort_feature_dim=task_feature_dims[i],
                fusion_dim=fusion_dim,
                device=device,
            )

            if i == 0:
                task_head.gate_layer.bias.data.fill_(-1.5)
            elif i == 1:
                task_head.gate_layer.bias.data.fill_(0.2)
            elif i == 2:
                task_head.gate_layer.bias.data.fill_(0.3)
            elif i == 3:
                task_head.gate_layer.bias.data.fill_(0.2)

            self.task_heads.append(task_head)

        self.to(device)

    def forward(
        self,
        shared_input,
        spc_inputs,
        task_ids,
        task_specific_slices,
    ):
        shared_outs, total_loss = self.shared_encoder(shared_input)

        logits_list = [None] * self.num_tasks

        for t_id in range(self.num_tasks):
            mask = task_ids == t_id

            if not mask.any():
                continue

            idx = mask.nonzero(as_tuple=True)[0]
            start, end = task_specific_slices[t_id]

            task_specific = spc_inputs[idx, start:end]
            task_shared = shared_outs[t_id][idx]

            logits = self.task_heads[t_id](
                task_shared,
                task_specific,
            )

            logits_list[t_id] = logits

        return logits_list, total_loss
