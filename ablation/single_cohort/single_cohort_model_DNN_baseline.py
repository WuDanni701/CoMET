from __future__ import annotations
import argparse
import sys
from pathlib import Path
import json
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

_ABLATION_DIR = Path(__file__).resolve().parent
if not (_ABLATION_DIR / 'clinical_preprocessing.py').exists():
    _ABLATION_DIR = _ABLATION_DIR.parent
if str(_ABLATION_DIR) not in sys.path:
    sys.path.insert(0, str(_ABLATION_DIR))
from clinical_preprocessing import DATA_ROOT, preprocess_single_cohort_fold
from autoencoder import Autoencoder
from config import (
    batch_size,
    gene_dropout,
    hidden_dims_list,
    latent_dim,
    learning_rate_train,
    logvar_max,
    logvar_min,
    noise_std,
    output_dir_tcga,
)
from ICIdataset import ICIDataset, ICIProcessor

SEED = 269
COHORTS = {
    "DavidA": {
        "specific": [
            "Age", "FS_Counts", "NEO_Weak", "ITH", "MSKCC_FAVORABLE",
            "MSKCC_INTERMEDIATE", "MSKCC_POOR", "homozygous_ANY_0",
            "homozygous_ANY_1", "Deletion_9p21.3_MUT",
            "Deletion_9p21.3_WT", "Deletion_11q23.1_MUT",
"Deletion_11q23.1_WT",
            "Amplification_12q24.32_MUT", "Amplification_12q24.32_WT",
            "Amplification_12q24.32_WUT", "Amplification_6q21_MUT",
            "Amplification_6q21_WT",
        ]
    },
    "DavidLiu": {
        "specific": [
            "CNA_prop", "heterogeneity", "Purity", "MHC-II", "ECOG_0",
            "ECOG_1", "ECOG_2", "ECOG_3", "Primary_Type_binary_0",
            "Primary_Type_binary_1", "TMB_clonal", "TMB_subclonal",
        ]
    },
    "Ravi": {
        "specific": [
            "Purity", "Smoking_Pack_Years", "Neoantigens", "Smoking_Status_0",
            "Smoking_Status_1", "Smoking_Status_2",
            "Histology_Harmonized_Adeno", "Histology_Harmonized_LC-NE",
            "Histology_Harmonized_Other", "Histology_Harmonized_Squamous",
            "Prior_Platinum_0", "Prior_Platinum_1", "TMB_clonal",
            "TMB_subclonal",
        ]
    },
    "IMvigor210": {
        "specific": [
            "Neoantigen_burden", "ECOG_0", "ECOG_1", "ECOG_2",
            "Smoking_Status_CURRENT", "Smoking_Status_NEVER",
            "Smoking_Status_PREVIOUS", "IC.Level_IC0", "IC.Level_IC1",
            "IC.Level_IC2+", "TC.Level_TC0", "TC.Level_TC1", "TC.Level_TC2+", "Lund_MS1a",
            "Lund_MS1b", "Lund_MS2a1", "Lund_MS2a2", "Lund_MS2b1",
            "Lund_MS2b2.1", "Lund_MS2b2.2", "Neo_missing_0", "Neo_missing_1",
        ]
    },
}
SHARED_COLUMNS = ["Sex_F", "Sex_M", "TMB"]

DNN_HIDDEN_UNITS = [64,32]
SINGLE_FUSION_DIM = 16

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SingleCohortDNN(nn.Module):

    def __init__(self, shared_input_dim: int, specific_dim: int, device: torch.device):
        super().__init__()

        fusion_input_dim = shared_input_dim + specific_dim
        hidden_units = list(DNN_HIDDEN_UNITS)

        layers = []
        in_dim = fusion_input_dim

        for hidden_dim in hidden_units:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                ]
            )
            in_dim = hidden_dim

        layers.extend(
            [
                nn.Linear(in_dim, SINGLE_FUSION_DIM),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(SINGLE_FUSION_DIM, 1),
            ]
        )

        self.network = nn.Sequential(*layers)

    def forward(self, shared_input: torch.Tensor, specific_input: torch.Tensor):
        x = torch.cat([shared_input, specific_input], dim=1)
        logits = self.network(x)

        reg_loss = torch.tensor(
            0.0,
            device=x.device,
        )

        return logits.view(-1), reg_loss

def load_tcga_artifacts():
    artifact_dir = Path(output_dir_tcga)
    with open(artifact_dir / "selected_genes.json", encoding="utf-8") as handle:
        selected_genes = json.load(handle)
    encoder_state = torch.load(
        artifact_dir / "encoder.pth", map_location="cpu", weights_only=True
    )
    load(artifact_dir / "std_scaler.joblib")
    return selected_genes, encoder_state, artifact_dir / "shared_encoder.pth"

def load_one_cohort(cohort: str, autoencoder: Autoencoder, selected_genes, device):
    clinical_path = DATA_ROOT / "clinical" / f"{cohort}_clinical.csv"
    rna_path = DATA_ROOT / "rna" / f"{cohort}_rna.csv"
    processor = ICIProcessor(
        rna_paths=[str(rna_path)],
        cohort_paths=[str(clinical_path)],
        shared_cols=SHARED_COLUMNS,
        specific_cols=[COHORTS[cohort]["specific"]],
    )
    rna, shared, specific, response, task_ids, os_time, os_event, tmb = processor.load_all()
    if not np.all(np.asarray(task_ids) == 0):
        raise RuntimeError("Single-cohort isolation failed: unexpected task id detected.")

    x = rna.reindex(columns=selected_genes, fill_value=0.0).values.astype(np.float32)
    chunks = []
    autoencoder.eval()
    with torch.no_grad():
        for start in range(0, len(x), 42):
            xb = torch.as_tensor(x[start:start + 42], device=device)
            chunks.append(autoencoder.encoder(xb).cpu().numpy())
    encoded = pd.DataFrame(
        np.vstack(chunks), index=rna.index,
        columns=[f"z{i + 1}" for i in range(latent_dim)],
    )
    return encoded, shared, specific, response, task_ids, os_time, os_event, tmb


def metrics(y_true, probabilities) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= 0.5).astype(int)
    result = {
        "auc": np.nan,
        "auprc": np.nan,
        "accuracy": accuracy_score(y_true, predicted),
        "f1": f1_score(y_true, predicted, zero_division=0),
    }
    if np.unique(y_true).size == 2:
        result["auc"] = roc_auc_score(y_true, probabilities)
        result["auprc"] = average_precision_score(y_true, probabilities)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted).ravel()
        result.update({
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "npv": tn / (tn + fn) if tn + fn else np.nan,
        })
    return result


def make_dataset(data, indices):
    encoded, shared, specific, response, task_ids, os_time, os_event, tmb = data
    return ICIDataset(
        encoded.iloc[indices], shared.iloc[indices], specific.iloc[indices],
        response.iloc[indices], task_ids.iloc[indices], os_time.iloc[indices],
        os_event.iloc[indices], tmb.iloc[indices], [specific.shape[1]],
    )


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, probabilities, labels = 0.0, [], []
    for batch in loader:
        if training:
            optimizer.zero_grad()
        x_shared = torch.cat([batch["rna"], batch["shared"]], dim=1).to(device)
        x_specific = batch["specific"].to(device)
        y = batch["label"].float().to(device)
        with torch.set_grad_enabled(training):
            logits, reg_loss = model(x_shared, x_specific)
            loss = criterion(logits, y) + 0.02 * reg_loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss.item()
        probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy())
        labels.extend(y.detach().cpu().numpy())
    return total_loss / max(len(loader), 1), metrics(labels, probabilities), np.asarray(probabilities)


def run_cohort(
    cohort: str,
    args,
    selected_genes,
    encoder_state,
    device
):
    print(f"\n========== Single-cohort: {cohort} ==========")
    autoencoder = Autoencoder(
        input_dim=len(selected_genes), latent_dim=latent_dim,
        hidden_dims=hidden_dims_list, dropout=0.4, noise_std=noise_std,
        gene_dropout=gene_dropout, logvar_min=logvar_min, logvar_max=logvar_max,
    ).to(device)
    autoencoder.load_state_dict(encoder_state, strict=False)
    data = load_one_cohort(cohort, autoencoder, selected_genes, device)
    y = np.asarray(data[3], dtype=int)
    class_counts = np.bincount(y, minlength=2)
    n_splits = min(args.folds, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError(f"{cohort} has too few samples in one class for stratified CV: {class_counts}")

    experiment_name = (
        "output_single_cohort_model_dnn"
    )
    output_dir = Path(args.output_dir) / experiment_name / cohort
    output_dir.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    records, fold_rows = [], []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        seed_everything(args.seed + fold)
        assert set(np.asarray(data[4].iloc[train_idx], dtype=int)) == {0}
        fold_shared, fold_specific = preprocess_single_cohort_fold(
            data[1], data[2], cohort, train_idx, val_idx
        )
        fold_data = (data[0], fold_shared, fold_specific, *data[3:])
        train_set, val_set = make_dataset(fold_data, train_idx), make_dataset(fold_data, val_idx)
        bs = min(args.batch_size, len(train_set))
        drop_last = len(train_set) > bs and len(train_set) % bs == 1
        generator = torch.Generator().manual_seed(args.seed + fold)
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=drop_last, generator=generator)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        model = SingleCohortDNN(
            shared_input_dim=data[0].shape[1] + data[1].shape[1],
            specific_dim=data[2].shape[1], device=device,
        ).to(device)
        copied = 0
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        positives, negatives = y[train_idx].sum(), len(train_idx) - y[train_idx].sum()
        pos_weight = torch.tensor([max(1.0, min(8.0, negatives / max(positives, 1)))], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best_auc, patience_left = -np.inf, args.patience
        best_path = output_dir / f"best_single_cohort_{cohort}_fold_{fold}.pth"
        for epoch in range(args.epochs):
            train_loss, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_metrics, _ = run_epoch(model, val_loader, device, criterion)
            score = np.nan_to_num(val_metrics["auc"], nan=0.0)
            if score > best_auc + 1e-4:
                best_auc, patience_left = score, args.patience
                torch.save(model.state_dict(), best_path)
            else:
                patience_left -= 1
            scheduler.step()
            if epoch % 10 == 0:
                print(f"{cohort} fold={fold} epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={score:.4f}")
            if patience_left <= 0:
                break
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        _, fold_metrics, probabilities = run_epoch(model, val_loader, device, criterion)
        fold_rows.append({"cohort": cohort, "fold": fold, "n_train": len(train_idx), "n_val": len(val_idx), "expert_tensors_loaded": copied, **fold_metrics})
        for local_i, global_i in enumerate(val_idx):
            records.append({
                "fold": fold, "cohort": cohort, "patientID": data[0].index[global_i],
                "response": int(data[3].iloc[global_i]), "prob": float(probabilities[local_i]),
                "OS": float(data[5].iloc[global_i]), "OS_event": int(data[6].iloc[global_i]),
                "TMB": float(data[7].iloc[global_i]),
            })
    predictions = pd.DataFrame(records)
    fold_metrics_df = pd.DataFrame(fold_rows)
    predictions.to_excel(output_dir / f"single_cohort_{cohort}_val_predictions.xlsx", index=False)
    fold_metrics_df.to_csv(output_dir / f"single_cohort_{cohort}_fold_metrics.csv", index=False)
    summary = {"cohort": cohort, "n_patients": len(y), "n_splits": n_splits}
    for name in ["auc", "auprc", "accuracy", "f1", "sensitivity", "specificity", "ppv", "npv"]:
        summary[f"{name}_mean"] = float(fold_metrics_df[name].mean())
        summary[f"{name}_std"] = float(fold_metrics_df[name].std(ddof=0))
    print(f"Saved {cohort} results to {output_dir}")
    return predictions, fold_metrics_df, summary
def parse_args():
    parser = argparse.ArgumentParser(description="Independent single-cohort DNN ablation")
    parser.add_argument("--cohort", choices=[*COHORTS, "all"], default="all")
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "output"))
    parser.add_argument("--device", default="cuda:7" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--learning-rate", type=float, default=learning_rate_train)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()
def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    print(f"Using device: {device}")
    selected_genes, encoder_state, _ = load_tcga_artifacts()
    cohorts = list(COHORTS) if args.cohort == "all" else [args.cohort]
    all_predictions, all_fold_metrics, summaries = [], [], []
    for cohort in cohorts:
        predictions, fold_metrics, summary = run_cohort(
            cohort, args, selected_genes, encoder_state, device
        )
        all_predictions.append(predictions)
        all_fold_metrics.append(fold_metrics)
        summaries.append(summary)

    combined_dir = Path(args.output_dir) / (
        "output_single_cohort_model_dnn"
    )
    pd.concat(all_predictions, ignore_index=True).to_excel(
        combined_dir / "single_cohort_all_val_predictions.xlsx", index=False
    )
    pd.concat(all_fold_metrics, ignore_index=True).to_csv(
        combined_dir / "single_cohort_all_fold_metrics.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(combined_dir / "single_cohort_summary.csv", index=False)
    with open(combined_dir / "experiment_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    print(f"\nAll requested results saved to: {combined_dir}")

if __name__ == "__main__":
    main()
