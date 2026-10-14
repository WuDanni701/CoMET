"""Shared runner for transcriptomic-only and genomic-only CoMET ablations."""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = REPOSITORY_ROOT / "ablation"
ICIMTL_DIR = REPOSITORY_ROOT / "ICIMTL"
for path in (ABLATION_DIR, ICIMTL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from clinical_preprocessing import get_clinical_paths, get_rna_paths, preprocess_fold
from autoencoder import Autoencoder
from config import (
    SEED, batch_size, expert_dnn_hidden_units, gate_dnn_hidden_units,
    gene_dropout, hidden_dims_list, l2_reg, latent_dim, learning_rate_train,
    logvar_max, logvar_min, noise_std, num_experts,
)
from ICIdataset import ICIDataset, ICIProcessor
from model import ICIMTL
from utils1 import EarlyStopping, ICITransferTrainer


COHORTS = ("DavidA", "DavidLiu", "Ravi", "IMvigor210")
SHARED_COLUMNS = ["Sex_F", "Sex_M", "TMB"]
TASK_SPECIFIC_COLUMNS = [
    ["Age", "FS_Counts", "NEO_Weak", "ITH", "MSKCC_FAVORABLE",
     "MSKCC_INTERMEDIATE", "MSKCC_POOR", "homozygous_ANY_0",
     "homozygous_ANY_1", "Deletion_9p21.3_MUT", "Deletion_9p21.3_WT",
     "Deletion_11q23.1_MUT", "Deletion_11q23.1_WT",
     "Amplification_12q24.32_MUT", "Amplification_12q24.32_WT",
     "Amplification_12q24.32_WUT", "Amplification_6q21_MUT",
     "Amplification_6q21_WT"],
    ["CNA_prop", "heterogeneity", "Purity", "MHC-II", "ECOG_0", "ECOG_1",
     "ECOG_2", "ECOG_3", "Primary_Type_binary_0", "Primary_Type_binary_1",
     "TMB_clonal", "TMB_subclonal"],
    ["Purity", "Smoking_Pack_Years", "Neoantigens", "Smoking_Status_0",
     "Smoking_Status_1", "Smoking_Status_2", "Histology_Harmonized_Adeno",
     "Histology_Harmonized_LC-NE", "Histology_Harmonized_Other",
     "Histology_Harmonized_Squamous", "Prior_Platinum_0", "Prior_Platinum_1",
     "TMB_clonal", "TMB_subclonal"],
    ["Neoantigen_burden", "ECOG_0", "ECOG_1", "ECOG_2",
     "Smoking_Status_CURRENT", "Smoking_Status_NEVER", "Smoking_Status_PREVIOUS",
     "IC.Level_IC0", "IC.Level_IC1", "IC.Level_IC2+", "TC.Level_TC0",
     "TC.Level_TC1", "TC.Level_TC2+", "Lund_MS1a", "Lund_MS1b",
     "Lund_MS2a1", "Lund_MS2a2", "Lund_MS2b1", "Lund_MS2b2.1",
     "Lund_MS2b2.2", "Neo_missing_0", "Neo_missing_1"],
]

TRANSCRIPTOMIC_COLUMNS = {
    "t1__MHC-II",
    *{f"t3__{name}" for name in (
        "Lund_MS1a", "Lund_MS1b", "Lund_MS2a1", "Lund_MS2a2",
        "Lund_MS2b1", "Lund_MS2b2.1", "Lund_MS2b2.2",
    )},
}
GENOMIC_COLUMNS = {
    f"t{task_id}__{name}"
    for task_id, names in {
        0: ["FS_Counts", "NEO_Weak", "ITH", "homozygous_ANY_0", "homozygous_ANY_1",
            "Deletion_9p21.3_MUT", "Deletion_9p21.3_WT", "Deletion_11q23.1_MUT",
            "Deletion_11q23.1_WT", "Amplification_12q24.32_MUT",
            "Amplification_12q24.32_WT", "Amplification_12q24.32_WUT",
            "Amplification_6q21_MUT", "Amplification_6q21_WT"],
        1: ["CNA_prop", "heterogeneity", "Purity", "TMB_clonal", "TMB_subclonal"],
        2: ["Purity", "Neoantigens", "TMB_clonal", "TMB_subclonal"],
        3: ["Neoantigen_burden"],
    }.items()
    for name in names
}


def parse_args(mode: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"CoMET {mode}-only ablation")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--learning-rate", type=float, default=learning_rate_train)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPOSITORY_ROOT / "outputs" / f"ablation_{mode}_only",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(value)


def load_artifacts():
    artifact_dir = REPOSITORY_ROOT / "TCGA_pretraining"
    paths = {
        "genes": artifact_dir / "selected_genes.json",
        "scalers": artifact_dir / "std_scaler.joblib",
        "encoder": artifact_dir / "encoder.pth",
        "shared": artifact_dir / "shared_encoder.pth",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing artifacts:\n  " + "\n  ".join(missing))
    with paths["genes"].open("r", encoding="utf-8") as handle:
        genes = json.load(handle)
    return genes, joblib.load(paths["scalers"]), torch.load(
        paths["encoder"], map_location="cpu", weights_only=True
    ), paths["shared"]


def prepare_data(device: torch.device, encoding_batch_size: int):
    genes, cohort_scalers, encoder_state, shared_checkpoint = load_artifacts()
    rna_paths = [str(path) for path in get_rna_paths()]
    clinical_paths = [str(path) for path in get_clinical_paths()]
    processor = ICIProcessor(rna_paths, clinical_paths, SHARED_COLUMNS, TASK_SPECIFIC_COLUMNS)
    rna, shared, specific, labels, task_ids, os_time, os_event, tmb = processor.load_all()

    scaled_rna = rna.copy()
    task_array = np.asarray(task_ids, dtype=int)
    for task_id, cohort in enumerate(COHORTS):
        positions = np.flatnonzero(task_array == task_id)
        cohort_rna = rna.iloc[positions]
        scaler = cohort_scalers[cohort]["scaler"]
        scaled_rna.iloc[positions, :] = scaler.transform(cohort_rna)
    expression = scaled_rna.reindex(columns=genes, fill_value=0.0).to_numpy(np.float32)

    encoder = Autoencoder(
        input_dim=len(genes), hidden_dims=hidden_dims_list, latent_dim=latent_dim,
        dropout=0.4, noise_std=noise_std, gene_dropout=gene_dropout,
        logvar_min=logvar_min, logvar_max=logvar_max,
    ).to(device)
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(expression), encoding_batch_size):
            batch = torch.as_tensor(expression[start:start + encoding_batch_size], device=device)
            chunks.append(encoder.encoder(batch).cpu().numpy())
    encoded = pd.DataFrame(
        np.concatenate(chunks), index=rna.index,
        columns=[f"z{i + 1}" for i in range(latent_dim)],
    )
    return encoded, shared, specific, labels, task_ids, os_time, os_event, tmb, shared_checkpoint


def apply_feature_mask(mode: str, encoded, shared, specific):
    encoded, shared, specific = encoded.copy(), shared.copy(), specific.copy()
    if mode == "transcriptomic":
        shared.loc[:, :] = 0.0
        keep = TRANSCRIPTOMIC_COLUMNS
    elif mode == "genomic":
        encoded.loc[:, :] = 0.0
        shared.loc[:, [name for name in shared.columns if name != "TMB"]] = 0.0
        keep = GENOMIC_COLUMNS
    else:
        raise ValueError(f"Unknown mode: {mode}")
    missing = sorted(keep.difference(specific.columns))
    if missing:
        raise KeyError(f"Missing {mode} columns: {missing}")
    specific.loc[:, [name for name in specific.columns if name not in keep]] = 0.0
    return encoded, shared, specific


def transfer_shared_encoder(model: ICIMTL, checkpoint: Path, device: torch.device) -> None:
    source = torch.load(checkpoint, map_location=device, weights_only=True)
    target = model.shared_encoder.state_dict()
    gate_mapping = {0: [0], 3: [1], 2: [2], 4: [3]}
    expert_count = 0
    for name, value in source.items():
        targets = [name] if name.startswith("expert_dnn.") else []
        match = re.match(r"gate_dnn\.(\d+)\.(.+)", name)
        if match and int(match.group(1)) in gate_mapping:
            targets += [f"gate_dnn.{task}.{match.group(2)}" for task in gate_mapping[int(match.group(1))]]
        for target_name in targets:
            if target_name in target and target[target_name].shape == value.shape:
                target[target_name] = value.clone()
                expert_count += int(name.startswith("expert_dnn."))
    if expert_count == 0:
        raise RuntimeError("No compatible expert parameters were transferred.")
    model.shared_encoder.load_state_dict(target, strict=False)


def build_model(input_dim: int, task_dims: list[int], device: torch.device) -> ICIMTL:
    return ICIMTL(
        shared_input_dim=input_dim, expert_dnn_hidden_units=expert_dnn_hidden_units,
        num_experts=num_experts, gate_dnn_hidden_units=gate_dnn_hidden_units,
        num_tasks=len(COHORTS), task_feature_dims=task_dims, fusion_dim=128,
        tower_dnn_hidden_units=[64, 32], l2_reg=l2_reg, dnn_activation="relu",
        dropout_rate=0.4, use_bn=True, init_std=0.001, seed=SEED, device=device,
    ).to(device)


def metric_row(scope: str, cohort: str, labels, probabilities) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "scope": scope, "cohort": cohort, "n": len(labels),
        "auc": roc_auc_score(labels, probabilities),
        "auprc": average_precision_score(labels, probabilities),
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def run(mode: str) -> None:
    args = parse_args(mode)
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(device, args.batch_size)
    encoded, shared, specific, labels, task_ids, os_time, os_event, tmb, checkpoint = data
    encoded, shared, specific = apply_feature_mask(mode, encoded, shared, specific)
    y, tasks = np.asarray(labels, int), np.asarray(task_ids, int)
    strata = np.asarray([f"{task}:{label}" for task, label in zip(tasks, y)])
    splitter = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    task_dims = [len(columns) for columns in TASK_SPECIFIC_COLUMNS]
    records = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(y, strata), start=1):
        set_seed(args.seed + fold)
        fold_shared, fold_specific = preprocess_fold(shared, specific, task_ids, train_idx, val_idx)
        train_set = ICIDataset(
            encoded.iloc[train_idx], fold_shared.iloc[train_idx], fold_specific.iloc[train_idx],
            labels.iloc[train_idx], task_ids.iloc[train_idx], os_time.iloc[train_idx],
            os_event.iloc[train_idx], tmb.iloc[train_idx], task_dims,
        )
        val_set = ICIDataset(
            encoded.iloc[val_idx], fold_shared.iloc[val_idx], fold_specific.iloc[val_idx],
            labels.iloc[val_idx], task_ids.iloc[val_idx], os_time.iloc[val_idx],
            os_event.iloc[val_idx], tmb.iloc[val_idx], task_dims,
        )
        generator = torch.Generator().manual_seed(args.seed + fold)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        model = build_model(encoded.shape[1] + shared.shape[1], task_dims, device)
        transfer_shared_encoder(model, checkpoint, device)
        trainer = ICITransferTrainer(model, device, args.learning_rate, freeze_shared=True)
        trainer.update_pos_weights_from_loader(train_loader)
        stopper = EarlyStopping(args.patience, 1e-4, "auc", "max")
        best_state = None

        for epoch in range(args.epochs):
            trainer.train_epoch(train_loader)
            _, metrics = trainer.evaluate(val_loader, return_loss=True)
            mean_auc = float(np.mean([item["auc"] for item in metrics.values()]))
            if stopper.step({"auc": mean_auc}, epoch):
                best_state = copy.deepcopy(model.state_dict())
            elif stopper.should_stop():
                break
            trainer.scheduler.step(epoch)
        if best_state is None:
            raise RuntimeError(f"Fold {fold} did not produce a valid model state.")
        model.load_state_dict(best_state)
        _, metrics = trainer.evaluate(val_loader, return_loss=True)
        for task_id, cohort in enumerate(COHORTS):
            result = metrics[f"task_{task_id}"]
            indices = np.asarray(val_idx)[tasks[val_idx] == task_id]
            for global_index, probability in zip(indices, result["preds"]):
                records.append({
                    "fold": fold, "cohort": cohort,
                    "patient_id": str(encoded.index[global_index]),
                    "label": int(labels.iloc[global_index]),
                    "probability": float(probability),
                })

    predictions = pd.DataFrame(records)
    summaries = [metric_row("overall", "all", predictions.label, predictions.probability)]
    for cohort, frame in predictions.groupby("cohort", sort=False):
        summaries.append(metric_row("cohort", cohort, frame.label, frame.probability))
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "summary_metrics.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))

