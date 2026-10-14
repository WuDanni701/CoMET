import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, roc_curve

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, monitor="loss", mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best = -np.inf if mode == "max" else np.inf
        self.counter = 0
        self.best_epoch = 0
        self.best_value = None

    def step(self, metrics: dict, epoch: int):
        value = metrics[self.monitor]

        if self.mode == "max":
            improved = value > (self.best + self.min_delta)
        else:
            improved = value < (self.best - self.min_delta)

        if improved:
            self.best = value
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
            return True

        self.counter += 1
        return False

    def should_stop(self):
        return self.counter >= self.patience


class ICITrainer:
    def __init__(
        self,
        model,
        device,
        learning_rate=1e-4,
        task_weights=(0.48, 0.54, 0.56, 0.55),
    ):
        self.model = model
        self.device = device
        self.task_weights = torch.tensor(task_weights, device=device, dtype=torch.float32)
        expert_params = []
        shared_gate_params = []
        task_gate_params = []
        task_encoder_params = []
        task_classifier_params = []
        for name, p in self.model.shared_encoder.named_parameters():
            if "expert_dnn" in name:
                expert_params.append(p)
            else:
                shared_gate_params.append(p)
        if hasattr(self.model, "task_heads"):
            for head in self.model.task_heads:
                for name, p in head.named_parameters():
                    if "gate_layer" in name:
                        task_gate_params.append(p)
                    elif "classifier" in name:
                        task_classifier_params.append(p)
                    else:
                        task_encoder_params.append(p)
        if hasattr(self.model, "prediction_head"):
          task_classifier_params.extend(list(self.model.prediction_head.parameters()))
        param_groups = []
        if expert_params:
            param_groups.append({
                "params": expert_params,
                "lr": learning_rate,
                "weight_decay": 1e-4,
            })

        if shared_gate_params:
            param_groups.append({
                "params": shared_gate_params,
                "lr": learning_rate,
                "weight_decay": 1e-4,
            })

        if task_gate_params:
            param_groups.append({
                "params": task_gate_params,
                "lr": learning_rate,
                "weight_decay": 2e-4,
            })

        if task_encoder_params:
            param_groups.append({
                "params": task_encoder_params,
                "lr": learning_rate,
                "weight_decay": 1e-4,
            })

        if task_classifier_params:
            param_groups.append({
                "params": task_classifier_params,
                "lr": learning_rate,
                "weight_decay": 0.0,
            })

        self.optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=15,
            T_mult=1,
            eta_min=1e-5,
        )
        self.num_tasks = self.model.num_tasks
        self.pos_weights = {
            t: torch.tensor([1.0], device=self.device)
            for t in range(self.num_tasks)
        }
        self.criterion_dict = {
            t: torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights[t])
            for t in range(self.num_tasks)
        }

    def update_pos_weights_from_loader(self, loader):
        pos_counts = torch.zeros(self.num_tasks, device=self.device)
        neg_counts = torch.zeros(self.num_tasks, device=self.device)
        with torch.no_grad():
            for batch in loader:
                labels = batch["label"].to(self.device).long()
                task_ids = batch["task_id"].to(self.device)
                for tid in range(self.num_tasks):
                    mask = task_ids == tid
                    if mask.any():
                        y = labels[mask]
                        pos_counts[tid] += (y == 1).sum()
                        neg_counts[tid] += (y == 0).sum()
        for tid in range(self.num_tasks):
            if pos_counts[tid] > 0:
                w = neg_counts[tid] / pos_counts[tid]
                if tid == 0:
                    w = w.clamp(min=1.0, max=8.0)
                else:
                    w = w.clamp(min=1.0, max=4.0)
                self.pos_weights[tid] = w.view(1)
            else:
                self.pos_weights[tid] = torch.tensor([1.0], device=self.device)
        self.criterion_dict = {
            t: torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights[t])
            for t in range(self.num_tasks)
        }
    def _compute_metrics(self, labels, probs):
        bin_pred = (probs > 0.5).astype(int)

        out = {
            "accuracy": accuracy_score(labels, bin_pred),
            "f1": f1_score(labels, bin_pred, zero_division=0),
            "preds": probs,
            "labels": labels,
        }

        if len(np.unique(labels)) > 1:
            out["auc"] = roc_auc_score(labels, probs)
            out["auprc"] = average_precision_score(labels, probs)
            fpr, tpr, _ = roc_curve(labels, probs)
            out["fpr"] = fpr
            out["tpr"] = tpr
        else:
            out["auc"] = 0.0
            out["auprc"] = 0.0
            out["fpr"] = np.array([])
            out["tpr"] = np.array([])

        return out

    def train_epoch(self, loader):
        self.model.train()

        total_loss = 0.0
        all_preds = {i: [] for i in range(self.num_tasks)}
        all_labels = {i: [] for i in range(self.num_tasks)}

        for batch in loader:
            self.optimizer.zero_grad()

            rna = batch["rna"].to(self.device)
            shared_feat = batch["shared"].to(self.device)
            specific_feat = batch["specific"].to(self.device)
            labels = batch["label"].to(self.device).long()
            task_ids = batch["task_id"].to(self.device)

            logits_list, reg_loss = self.model(
                torch.cat([rna, shared_feat], dim=1),
                specific_feat,
                task_ids,
                loader.dataset.task_slices,
            )

            loss_dict = {}

            for tid in range(self.num_tasks):
                mask = task_ids == tid

                if logits_list[tid] is not None and mask.any():
                    task_logits = logits_list[tid].view(-1)
                    task_labels = labels[mask].float()

                    task_loss = self.criterion_dict[tid](task_logits, task_labels)
                    loss_dict[tid] = task_loss

                    probs = torch.sigmoid(task_logits).detach().cpu().numpy()
                    all_preds[tid].extend(probs)
                    all_labels[tid].extend(task_labels.cpu().numpy())

            if loss_dict:
                weighted_loss = sum(
                    self.task_weights[tid] * loss
                    for tid, loss in loss_dict.items()
                )

                total = weighted_loss + 0.02 * reg_loss
                total.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += total.item()

        metrics = {}

        for tid in range(self.num_tasks):
            if all_labels[tid]:
                metrics[f"task_{tid}"] = self._compute_metrics(
                    np.array(all_labels[tid]),
                    np.array(all_preds[tid]),
                )

        return total_loss / max(len(loader), 1), metrics
    def evaluate(self, loader, return_loss=False):
        self.model.eval()
        total_loss = 0.0
        n_batch = 0
        all_preds = {i: [] for i in range(self.num_tasks)}
        all_labels = {i: [] for i in range(self.num_tasks)}

        with torch.no_grad():
            for batch in loader:
                rna = batch["rna"].to(self.device)
                shared_feat = batch["shared"].to(self.device)
                specific_feat = batch["specific"].to(self.device)
                labels = batch["label"].to(self.device).long()
                task_ids = batch["task_id"].to(self.device)

                logits_list, reg_loss = self.model(
                    torch.cat([rna, shared_feat], dim=1),
                    specific_feat,
                    task_ids,
                    loader.dataset.task_slices,
                )

                loss_dict = {}
                for tid in range(self.num_tasks):
                    mask = task_ids == tid
                    if logits_list[tid] is not None and mask.any():
                        task_logits = logits_list[tid].view(-1)
                        task_labels = labels[mask].float()
                        task_loss = self.criterion_dict[tid](task_logits, task_labels)
                        loss_dict[tid] = task_loss
                        probs = torch.sigmoid(task_logits).cpu().numpy()
                        all_preds[tid].extend(probs)
                        all_labels[tid].extend(task_labels.cpu().numpy())
                if loss_dict and return_loss:
                    weighted_loss = sum(
                        self.task_weights[tid] * loss
                        for tid, loss in loss_dict.items()
                    )

                    total = weighted_loss + 0.02 * reg_loss
                    total_loss += total.item()
                    n_batch += 1
        metrics = {}
        for tid in range(self.num_tasks):
            if all_labels[tid]:
                metrics[f"task_{tid}"] = self._compute_metrics(
                    np.array(all_labels[tid]),
                    np.array(all_preds[tid]),
                )
        if return_loss:
            mean_loss = total_loss / max(n_batch, 1)
            return mean_loss, metrics
        return metrics