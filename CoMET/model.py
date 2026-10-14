import torch
import torch.nn as nn
import torch.nn.functional as F
from DNN import DNN

class sharedEncoder(nn.Module):
    def __init__(self,input_dim,expert_dnn_hidden_units,num_experts,gate_dnn_hidden_units,
                 num_tasks,l2_reg,dnn_activation,dropout_rate,use_bn,init_std,device, tau=0.6):
        super(sharedEncoder, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.device = device
        self.gate_dnn_hidden_units = gate_dnn_hidden_units
        self.regularization_weight = []
        self.tau = tau
        self.l2_reg = l2_reg
        self.expert_dnn= nn.ModuleList([DNN(self.input_dim, expert_dnn_hidden_units, activation=dnn_activation,
                                             l2_reg=l2_reg, dropout_rate=dropout_rate, use_bn=use_bn,
                                             init_std=init_std, device=device) for _ in range(self.num_experts)])

        if len(gate_dnn_hidden_units) > 0:
            self.gate_dnn = nn.ModuleList([DNN(self.input_dim, gate_dnn_hidden_units, activation=dnn_activation,
                                               l2_reg=l2_reg, dropout_rate=dropout_rate, use_bn=use_bn,
                                               init_std=init_std, device=device) for _ in range(self.num_tasks)])
            gate_input_dim = gate_dnn_hidden_units[-1]
        else:
            self.gate_dnn = None
            gate_input_dim = self.input_dim
            #self.add_regularization_weight(gate_params,l2=l2_reg)
        self.gate_dnn_final_layer = nn.ModuleList(
            [nn.Linear(gate_input_dim,
                       self.num_experts, bias=False) for _ in range(self.num_tasks)])
        self._init_regularization_weights()
        self.to(device)

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        weight_list = list(weight_list)
        if len(weight_list) == 0:
            return
        self.regularization_weight.append((weight_list, l1, l2))
    def _init_regularization_weights(self):
        if self.l2_reg <= 0:
            return
        for expert in self.expert_dnn:
            self.add_regularization_weight(
                (p for p in expert.parameters() if p.requires_grad),
                l1=0.0,
                l2=self.l2_reg
            )
        if self.gate_dnn is not None:
            for gate_net in self.gate_dnn:
                self.add_regularization_weight(
                    (p for p in gate_net.parameters() if p.requires_grad),
                    l1=0.0,
                    l2=self.l2_reg
                )
        for gate_layer in self.gate_dnn_final_layer:
            self.add_regularization_weight(
                (p for p in gate_layer.parameters() if p.requires_grad),
                l1=0.0,
                l2=self.l2_reg
            )

    def get_regularization_loss(self):
        reg_loss = 0.0
        for weight_list, l1, l2 in self.regularization_weight:
            for weight in weight_list:
                if l2 > 0.0:
                    reg_loss = reg_loss + l2 * torch.norm(weight, p=2)
                if l1 > 0.0:
                    reg_loss = reg_loss + l1 * torch.norm(weight, p=1)
        return reg_loss

    def forward(self, dnn_input):
        expert_outs = [self.expert_dnn[i](dnn_input) for i in range(self.num_experts)]
        expert_outs = torch.stack(expert_outs, 1)
        #gate outputs
        mmoe_outs = []
        for i in range(self.num_tasks):
            if len(self.gate_dnn_hidden_units) > 0:
                gate_dnn_out = self.gate_dnn[i](dnn_input)
                gate_dnn_out = self.gate_dnn_final_layer[i](gate_dnn_out)
            else:
                gate_dnn_out = self.gate_dnn_final_layer[i](dnn_input)
            gate_logits = gate_dnn_out / self.tau
            gate_weight = F.softmax(gate_logits, dim=1)
            gate_mul_expert = torch.matmul(gate_weight.unsqueeze(1), expert_outs)
            mmoe_outs.append(gate_mul_expert.squeeze(1))
        total_loss = self.get_regularization_loss()
        return mmoe_outs,total_loss

class TaskSpecificHead(nn.Module):
    def __init__(self,
                 shared_dim,
                 cohort_feature_dim,
                 fusion_dim,
                 device):
        super(TaskSpecificHead, self).__init__()
        self.specific_encoder = nn.Sequential(
            nn.Linear(cohort_feature_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.shared_encoder = nn.Sequential(
            nn.Linear(shared_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.gate_layer = nn.Linear(fusion_dim * 2, 1)
        self.tower = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim//2, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 1)
        )
        self.to(device)
    def forward(self, shared_feat, specific_feat):
        sp = self.specific_encoder(specific_feat)
        sh = self.shared_encoder(shared_feat)
        gate_input = torch.cat([sh, sp], dim=1)
        gate_logit = self.gate_layer(gate_input)
        alpha = torch.sigmoid(gate_logit)
        fused = alpha * sh + (1.0 - alpha) * sp
        h = self.tower(fused)
        logit = self.classifier(h)
        return logit

class ICIMTL(nn.Module):
    def __init__(self,shared_input_dim,expert_dnn_hidden_units,num_experts,gate_dnn_hidden_units,
                 num_tasks,task_feature_dims,fusion_dim,tower_dnn_hidden_units,l2_reg,dnn_activation,dropout_rate, use_bn, init_std, seed, device):
        super(ICIMTL, self).__init__()
        self.num_tasks = num_tasks
        self.device = device
        #shared layer
        self.shared_encoder = sharedEncoder(
            input_dim=shared_input_dim,
            expert_dnn_hidden_units=expert_dnn_hidden_units,
            num_experts=num_experts,
            gate_dnn_hidden_units=gate_dnn_hidden_units,
            num_tasks=num_tasks,
            l2_reg=l2_reg,
            dnn_activation=dnn_activation,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
            init_std=init_std,
            device=device
        )
        #task_specifi
        self.task_heads = nn.ModuleList()
        for i in range(num_tasks):
            shared_dim = expert_dnn_hidden_units[-1]
            task_head = TaskSpecificHead(
                shared_dim=shared_dim,
                cohort_feature_dim=task_feature_dims[i],
                fusion_dim=fusion_dim,
                device=device
            )
            if i == 0:
                task_head.gate_layer.bias.data.fill_(-1.5)
            if i == 1:
                task_head.gate_layer.bias.data.fill_(0.2)
            if i == 2:
                task_head.gate_layer.bias.data.fill_(0.3)
            if i == 3:
                task_head.gate_layer.bias.data.fill_(0.2)
            self.task_heads.append(task_head)
        self.to(device)

    def forward(self, shared_input,spc_inputs,task_ids,task_specific_slices):
        mmoe_outs, total_loss = self.shared_encoder(shared_input)
        logits_list = [None] * self.num_tasks
        for t_id in range(self.num_tasks):
            bool_mask = (task_ids == t_id)
            if not bool_mask.any():
                continue
            idx_mask = bool_mask.nonzero(as_tuple=True)[0]
            start, end = task_specific_slices[t_id]
            task_specific = spc_inputs[idx_mask, start:end]
            task_shared = mmoe_outs[t_id][idx_mask]
            logits = self.task_heads[t_id](task_shared, task_specific)
            logits_list[t_id] = logits
        return logits_list, total_loss
