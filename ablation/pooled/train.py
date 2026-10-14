from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader

try:
    from .model import PooledDNN
    from .train import (
        COHORTS,
        SHARED_COLUMNS,
        REPOSITORY_ROOT,
        PooledDataset,
        binary_metrics,
        load_and_encode_data,
        load_pretraining,
        preprocess_shared_fold,
        resolve_device,
        set_seed,
    )
except ImportError:
    from model import PooledDNN
    from train import (
        COHORTS,
        SHARED_COLUMNS,
        REPOSITORY_ROOT,
        PooledDataset,
        binary_metrics,
        load_and_encode_data,
        load_pretraining,
        preprocess_shared_fold,
        resolve_device,
        set_seed,
    )

import sys

ICIMTL_DIR = REPOSITORY_ROOT / "ICIMTL"
if str(ICIMTL_DIR) not in sys.path:
    sys.path.insert(0, str(ICIMTL_DIR))

from autoencoder import Autoencoder
from config import (
    SEED,
    batch_size,
    gene_dropout,
    hidden_dims_list,
    latent_dim,
    learning_rate_train,
    logvar_max,
    logvar_min,
    noise_std,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pooled DNN learning")
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "icidata")
    parser.add_argument("--pretraining-dir", type=Path, default=REPOSITORY_ROOT / "TCGA_pretraining")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / "ablation_pooled_dnn")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--learning-rate", type=float, default=learning_rate_train)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


class Trainer:
    def __init__(self, model: PooledDNN, device: torch.device, learning_rate: float) -> None:
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion: nn.Module = nn.BCEWithLogitsLoss()

    def set_class_weight(self, loader: DataLoader) -> None:
        labels = torch.cat([batch["label"] for batch in loader])
        positives = int((labels == 1).sum())
        negatives = int((labels == 0).sum())
        weight = min(max(negatives / positives, 1.0), 6.0)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([weight], device=self.device)
        )

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            inputs = torch.cat([batch["rna"], batch["shared"]], dim=1).to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(inputs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += float(loss.item())
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        probabilities, labels, cohort_ids = [], [], []
        for batch in loader:
            inputs = torch.cat([batch["rna"], batch["shared"]], dim=1).to(self.device)
            probabilities.extend(torch.sigmoid(self.model(inputs)).cpu().numpy())
            labels.extend(batch["label"].numpy())
            cohort_ids.extend(batch["cohort_id"].numpy())
        labels = np.asarray(labels, dtype=int)
        probabilities = np.asarray(probabilities)
        return binary_metrics(labels, probabilities), probabilities, np.asarray(cohort_ids)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_genes, cohort_scalers, encoder_state, _ = load_pretraining(args.pretraining_dir)
    autoencoder = Autoencoder(
        input_dim=len(selected_genes),
        hidden_dims=hidden_dims_list,
        latent_dim=latent_dim,
        dropout=0.4,
        noise_std=noise_std,
        gene_dropout=gene_dropout,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    ).to(device)
    autoencoder.load_state_dict(encoder_state, strict=False)

    encoded_rna, shared, labels, cohort_ids, sample_ids = load_and_encode_data(
        args.data_dir, selected_genes, cohort_scalers, autoencoder, device, args.batch_size
    )
    strata = np.asarray([f"{cohort}:{label}" for cohort, label in zip(cohort_ids, labels)])
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    fold_rows, prediction_rows = [], []

    for fold, (train_idx, validation_idx) in enumerate(splitter.split(labels, strata), start=1):
        set_seed(args.seed + fold)
        train_shared, validation_shared = preprocess_shared_fold(shared, train_idx, validation_idx)
        train_data = PooledDataset(encoded_rna[train_idx], train_shared, labels[train_idx], cohort_ids[train_idx])
        validation_data = PooledDataset(
            encoded_rna[validation_idx], validation_shared, labels[validation_idx], cohort_ids[validation_idx]
        )
        generator = torch.Generator().manual_seed(args.seed + fold)
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, generator=generator
        )
        validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)

        model = PooledDNN(encoded_rna.shape[1] + len(SHARED_COLUMNS)).to(device)
        trainer = Trainer(model, device, args.learning_rate)
        trainer.set_class_weight(train_loader)
        best_auc, stale_epochs = -np.inf, 0
        best_path = args.output_dir / f"best_fold_{fold}.pth"

        for epoch in range(1, args.epochs + 1):
            train_loss = trainer.train_epoch(train_loader)
            fold_metrics, _, _ = trainer.evaluate(validation_loader)
            if fold_metrics["auc"] > best_auc + 1e-4:
                best_auc, stale_epochs = fold_metrics["auc"], 0
                torch.save(model.state_dict(), best_path)
            else:
                stale_epochs += 1
            if epoch == 1 or epoch % 10 == 0:
                print(f"Fold {fold} epoch {epoch}: loss={train_loss:.4f}, AUC={fold_metrics['auc']:.4f}")
            if stale_epochs >= args.patience:
                break

        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        fold_metrics, probabilities, validation_cohorts = trainer.evaluate(validation_loader)
        fold_rows.append({"fold": fold, **fold_metrics})
        for local_index, global_index in enumerate(validation_idx):
            prediction_rows.append({
                "fold": fold,
                "sample_id": sample_ids[global_index],
                "cohort": COHORTS[int(validation_cohorts[local_index])],
                "label": int(labels[global_index]),
                "probability": float(probabilities[local_index]),
            })
        print(f"Fold {fold} result: {fold_metrics}")

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "out_of_fold_predictions.csv", index=False)
    print("\nCross-validation summary (mean ± std)")
    for metric in ("auc", "auprc", "accuracy", "f1"):
        print(f"{metric}: {fold_frame[metric].mean():.4f} ± {fold_frame[metric].std(ddof=0):.4f}")


if __name__ == "__main__":
    run(parse_args())
