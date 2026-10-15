from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMET_DIR = REPOSITORY_ROOT / "CoMET"
if str(COMET_DIR) not in sys.path:
    sys.path.insert(0, str(COMET_DIR))
from autoencoder import Autoencoder
from config import SEED, batch_size, gene_dropout, hidden_dims_list, latent_dim, learning_rate_train, logvar_max, logvar_min, noise_std
POOLED_DIR = Path(__file__).resolve().parent
if str(POOLED_DIR) in sys.path:
    sys.path.remove(str(POOLED_DIR))
sys.path.insert(0, str(POOLED_DIR))

from model import PooledDNN

COHORTS = ("DavidA", "DavidLiu", "Ravi", "IMvigor210")
SHARED_COLUMNS = ("Sex_F", "Sex_M", "TMB")


def parse_args():
    parser = argparse.ArgumentParser(description="Pooled DNN learning")
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "icidata")
    parser.add_argument("--pretraining-dir", type=Path, default=REPOSITORY_ROOT / "TCGA_pretraining")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / "ablation_pooled_dnn")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--learning-rate", type=float, default=learning_rate_train)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value):
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)


def load_data(data_dir, pretraining_dir, device, encode_batch_size):
    selected_genes = json.loads((pretraining_dir / "selected_genes.json").read_text(encoding="utf-8"))
    cohort_scalers = joblib.load(pretraining_dir / "std_scaler.joblib")
    encoder_state = torch.load(pretraining_dir / "encoder.pth", map_location="cpu", weights_only=True)
    encoder = Autoencoder(
        input_dim=len(selected_genes), hidden_dims=hidden_dims_list, latent_dim=latent_dim,
        dropout=0.4, noise_std=noise_std, gene_dropout=gene_dropout,
        logvar_min=logvar_min, logvar_max=logvar_max,
    ).to(device)
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.eval()

    expression_parts, shared_parts, label_parts, cohort_parts, sample_ids = [], [], [], [], []
    for cohort_id, cohort in enumerate(COHORTS):
        rna = pd.read_csv(data_dir / "rna" / f"{cohort}_rna.csv", index_col=0)
        clinical = pd.read_csv(data_dir / "clinical" / f"{cohort}_clinical.csv", index_col=0)
        common = clinical.index.intersection(rna.index).sort_values()
        rna, clinical = rna.loc[common], clinical.loc[common]
        scaler = cohort_scalers[cohort]["scaler"]
        scaled = pd.DataFrame(scaler.transform(rna), index=common, columns=rna.columns)
        expression_parts.append(scaled.loc[:, selected_genes].to_numpy(np.float32))
        shared_parts.append(clinical.loc[:, SHARED_COLUMNS].astype(float))
        label_parts.append(clinical["response"].to_numpy(np.int64))
        cohort_parts.append(np.full(len(common), cohort_id, dtype=np.int64))
        sample_ids.extend([f"{cohort}:{sample_id}" for sample_id in common.astype(str)])

    expression = np.concatenate(expression_parts)
    latent_parts = []
    with torch.no_grad():
        for start in range(0, len(expression), encode_batch_size):
            batch = torch.as_tensor(expression[start:start + encode_batch_size], dtype=torch.float32, device=device)
            latent_parts.append(encoder.encoder(batch).cpu().numpy())
    return np.concatenate(latent_parts), pd.concat(shared_parts, ignore_index=True), np.concatenate(label_parts), np.concatenate(cohort_parts), sample_ids


def preprocess_shared(shared, fit_indices, transform_indices):
    fit, transform = shared.iloc[fit_indices].copy(), shared.iloc[transform_indices].copy()
    imputer, scaler = SimpleImputer(strategy="median"), RobustScaler()
    fit.loc[:, "TMB"] = scaler.fit_transform(imputer.fit_transform(fit[["TMB"]])).ravel()
    transform.loc[:, "TMB"] = scaler.transform(imputer.transform(transform[["TMB"]])).ravel()
    return fit.to_numpy(np.float32), transform.to_numpy(np.float32)


def make_loader(features, labels, batch_size_value, shuffle, seed):
    dataset = TensorDataset(torch.as_tensor(features, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size_value, shuffle=shuffle, generator=torch.Generator().manual_seed(seed))


def make_loss(labels, device):
    positives, negatives = int((labels == 1).sum()), int((labels == 0).sum())
    weight = min(max(negatives / positives, 1.0), 6.0)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.no_grad()
def predict(model, features, batch_size_value, device):
    model.eval()
    probabilities = []
    for start in range(0, len(features), batch_size_value):
        batch = torch.as_tensor(features[start:start + batch_size_value], dtype=torch.float32, device=device)
        probabilities.extend(torch.sigmoid(model(batch)).cpu().numpy())
    return np.asarray(probabilities)


def metric_values(labels, probabilities):
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(labels, probabilities),
        "auprc": average_precision_score(labels, probabilities),
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def select_epoch(encoded_rna, shared, labels, cohort_ids, outer_train, args, device, fold):
    strata = np.asarray([f"{cohort}:{label}" for cohort, label in zip(cohort_ids[outer_train], labels[outer_train])])
    inner_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed + fold)
    inner_train_local, inner_val_local = next(inner_split.split(outer_train, strata))
    inner_train, inner_val = outer_train[inner_train_local], outer_train[inner_val_local]
    train_shared, val_shared = preprocess_shared(shared, inner_train, inner_val)
    x_train = np.hstack([encoded_rna[inner_train], train_shared])
    x_val = np.hstack([encoded_rna[inner_val], val_shared])
    loader = make_loader(x_train, labels[inner_train], args.batch_size, True, args.seed + fold)
    model = PooledDNN(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = make_loss(labels[inner_train], device)
    best_auc, best_epoch, stale = -np.inf, 1, 0
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, loader, optimizer, criterion, device)
        auc = roc_auc_score(labels[inner_val], predict(model, x_val, args.batch_size, device))
        if auc > best_auc + 1e-4:
            best_auc, best_epoch, stale = auc, epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    return best_epoch


def run(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoded_rna, shared, labels, cohort_ids, sample_ids = load_data(args.data_dir, args.pretraining_dir, device, args.batch_size)
    strata = np.asarray([f"{cohort}:{label}" for cohort, label in zip(cohort_ids, labels)])
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    fold_rows, prediction_rows = [], []

    for fold, (train_indices, test_indices) in enumerate(splitter.split(labels, strata), start=1):
        set_seed(args.seed + fold)
        best_epoch = select_epoch(encoded_rna, shared, labels, cohort_ids, train_indices, args, device, fold)
        train_shared, test_shared = preprocess_shared(shared, train_indices, test_indices)
        x_train = np.hstack([encoded_rna[train_indices], train_shared])
        x_test = np.hstack([encoded_rna[test_indices], test_shared])
        loader = make_loader(x_train, labels[train_indices], args.batch_size, True, args.seed + fold)
        model = PooledDNN(x_train.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        criterion = make_loss(labels[train_indices], device)
        for _ in range(best_epoch):
            train_epoch(model, loader, optimizer, criterion, device)

        probabilities = predict(model, x_test, args.batch_size, device)
        fold_metrics = metric_values(labels[test_indices], probabilities)
        fold_rows.append({"fold": fold, "selected_epoch": best_epoch, **fold_metrics})
        torch.save(model.state_dict(), args.output_dir / f"model_fold_{fold}.pth")
        for local_index, global_index in enumerate(test_indices):
            prediction_rows.append({
                "fold": fold, "sample_id": sample_ids[global_index],
                "cohort": COHORTS[int(cohort_ids[global_index])],
                "label": int(labels[global_index]), "probability": float(probabilities[local_index]),
            })
        print(f"Fold {fold}: epoch={best_epoch}, {fold_metrics}")

    fold_frame, prediction_frame = pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)
    fold_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    prediction_frame.to_csv(args.output_dir / "out_of_fold_predictions.csv", index=False)
    for cohort, frame in prediction_frame.groupby("cohort"):
        print(cohort, metric_values(frame["label"], frame["probability"]))


if __name__ == "__main__":
    run(parse_args())
