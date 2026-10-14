import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from utils_notcga import EarlyStopping, ICITrainer
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ABLATION_DIR = REPOSITORY_ROOT / "ablation"
_ICIMTL_DIR = REPOSITORY_ROOT / "ICIMTL"
if str(_ABLATION_DIR) not in sys.path:
    sys.path.insert(0, str(_ABLATION_DIR))
if str(_ICIMTL_DIR) not in sys.path:
    sys.path.insert(0, str(_ICIMTL_DIR))
from clinical_preprocessing import get_clinical_paths, get_rna_paths, preprocess_fold

from autoencoder import Autoencoder
from model import ICIMTL
from ICIdataset import ICIProcessor, ICIDataset
import json
import random
from sklearn.metrics import roc_curve, auc, confusion_matrix, average_precision_score, roc_auc_score,f1_score,accuracy_score
from config import (output_dir_tcga,hidden_dims_list,noise_std,l2_reg,num_experts,batch_size,
                    latent_dim,logvar_max,logvar_min,gene_dropout,expert_dnn_hidden_units,gate_dnn_hidden_units)

SEED=269
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_gene_artifacts(output_dir_tcga):
    with (Path(output_dir_tcga) / "selected_genes.json").open("r", encoding="utf-8") as f:
        selected_genes = json.load(f)
    return selected_genes
def prepare_ici_data(rna_paths, cohort_paths, shared_cols, specific_cols,
                     autoencoder, device, selected_genes, batch_size=50):
    processor = ICIProcessor(rna_paths, cohort_paths, shared_cols, specific_cols)
    all_rna, all_shared, all_specific, all_response, all_task_ids, OSs, OSEs,TMB_origin = processor.load_all()
    missing_genes = [gene for gene in selected_genes if gene not in all_rna.columns]
    if missing_genes:
        raise ValueError(f"{len(missing_genes)} selected genes are missing from the ICB RNA inputs.")
    X_sel = all_rna.loc[:, selected_genes].copy()
    X = X_sel.values.astype(np.float32)
    autoencoder.eval().to(device)
    Z_chunks = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            Z_mu = autoencoder.encoder(xb)
            Z_chunks.append(Z_mu.cpu().numpy())
    Z = np.vstack(Z_chunks)
    z_cols = [f"z{i + 1}" for i in range(Z.shape[1])]
    encoded_rna_df = pd.DataFrame(Z, index=all_rna.index, columns=z_cols)
    return (
        encoded_rna_df,
        all_shared.copy(),
        all_specific.copy(),
        all_response.copy(),
        all_task_ids.copy(),
        OSs.copy(),
        OSEs.copy(),
        TMB_origin.copy()
    )
def AUC_calculator(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    auroc = auc(fpr, tpr)
    specificity_sensitivity_sum = tpr + (1 - fpr)
    ind_max = np.argmax(specificity_sensitivity_sum)
    return auroc, threshold[ind_max]

def compute_all_metrics(y_true,y_prob,threshold = None):
    metrics = {}
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        metrics.update({
            'auc': np.nan, 'auprc': np.nan, 'f1': np.nan, 'accuracy': np.nan,
            'sensitivity': np.nan, 'specificity': np.nan, 'ppv': np.nan, 'npv': np.nan,
            'preds_prob': y_prob, 'labels': y_true, 'fpr': np.array([]), 'tpr': np.array([])
        })
        return metrics
    if threshold is not None:
        threshold_use = threshold
    else:
        auc,threshold_use = AUC_calculator(y_true,y_prob)
    metrics['threshold'] = threshold_use
    y_pred_01 = [int(c >= threshold_use) for c in y_prob]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_01).ravel()
    metrics['f1'] = f1_score(y_true, y_pred_01, zero_division=0)
    metrics['accuracy'] = accuracy_score(y_true, y_pred_01)
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics['auc'] = roc_auc_score(y_true, y_prob)
    metrics['auprc'] = average_precision_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    metrics['preds_prob'] = y_prob
    metrics['labels'] = y_true
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    return metrics

def five_fold_cross_validation():
    #read in data
    output_dir =  "output/ablation_without_pretraining"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cohort_paths = [str(path) for path in get_clinical_paths()]
    rna_paths = [str(path) for path in get_rna_paths()]
    shared_feature_cols = ["Sex_F", "Sex_M", "TMB"]
    task_specific_feature_cols = [
        ["Age",
         "FS_Counts",
         "NEO_Weak",
         "ITH",
         "MSKCC_FAVORABLE",
         "MSKCC_INTERMEDIATE",
         "MSKCC_POOR",
         "homozygous_ANY_0",
         "homozygous_ANY_1",
         "Deletion_9p21.3_MUT",
         "Deletion_9p21.3_WT",
         "Deletion_11q23.1_MUT",

         "Deletion_11q23.1_WT",
         "Amplification_12q24.32_MUT",
         "Amplification_12q24.32_WT",
         "Amplification_12q24.32_WUT",
         "Amplification_6q21_MUT",
         "Amplification_6q21_WT"
         ],
        [
            "CNA_prop",
            "heterogeneity",
            "Purity",
            "MHC-II",
            "ECOG_0",
            "ECOG_1",
            "ECOG_2",
            "ECOG_3",
            "Primary_Type_binary_0",
            "Primary_Type_binary_1",
            "TMB_clonal",
            "TMB_subclonal"
        ],
        [
            "Purity",
            "Smoking_Pack_Years",
            "Neoantigens",
            "Smoking_Status_0",
            "Smoking_Status_1",
            "Smoking_Status_2",
            "Histology_Harmonized_Adeno",
            "Histology_Harmonized_LC-NE",
            "Histology_Harmonized_Other",
            "Histology_Harmonized_Squamous",
            "Prior_Platinum_0",
            "Prior_Platinum_1",
            "TMB_clonal",
            "TMB_subclonal"
        ],
        [
            "Neoantigen_burden",
            "ECOG_0",
            "ECOG_1",
            "ECOG_2",
            "Smoking_Status_CURRENT",
            "Smoking_Status_NEVER",
            "Smoking_Status_PREVIOUS",
            "IC.Level_IC0",
            "IC.Level_IC1",
            "IC.Level_IC2+",
            "TC.Level_TC0",
            "TC.Level_TC1",
            "TC.Level_TC2+",
            "Lund_MS1a",
            "Lund_MS1b",
            "Lund_MS2a1",
            "Lund_MS2a2",
            "Lund_MS2b1",
            "Lund_MS2b2.1",
            "Lund_MS2b2.2",
            "Neo_missing_0",
            "Neo_missing_1"

        ]
    ]
    selected_genes = load_gene_artifacts(output_dir_tcga)
    # autoencoder parameter
    dropout_ae = 0.4
    autoencoder = Autoencoder(
        input_dim=len(selected_genes),
        latent_dim=latent_dim,
        hidden_dims=hidden_dims_list,
        dropout=dropout_ae,
        noise_std=noise_std,
        gene_dropout=gene_dropout,
        logvar_min=logvar_min,
        logvar_max=logvar_max
    ).to(device)

    autoencoder.eval().to(device)
    encoded_rna, shared_feats, specific_feats, responses, task_ids, OSs, OSEs,TMB_origin = prepare_ici_data(
        rna_paths, cohort_paths, shared_feature_cols, task_specific_feature_cols,
        autoencoder, device,
        selected_genes=selected_genes,
        batch_size=42
    )
    all_metrics_icimtl = {0: [], 1: [], 2: [],3:[]}
    icimtl_records = []
    task_id_to_cohort = {0: "DavidA", 1: "DavidLiu", 2: "Ravi",3:"IMvigor210"}
    #（task_id,label）分层
    y=np.asarray(responses).astype(int)
    t=np.asarray(task_ids).astype(int)
    y_start=np.array([f"t{ti}_y{yi}" for ti, yi in zip(t, y)])
    task_feature_dims = [len(cols) for cols in task_specific_feature_cols]
    task_slices = {}
    start_idx = 0
    for i, dim in enumerate(task_feature_dims):
        task_slices[i] = (start_idx, start_idx + dim)
        start_idx += dim
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    #5_cv
    for fold, (train_idx, val_idx) in enumerate(skf.split(y,y_start),0):
        torch.manual_seed(SEED)
        print(f"\n=== Fold {fold} ===")
        for tid in np.unique(t):
            val_mask = (t[val_idx] == tid)
            classes = np.unique(y[val_idx][val_mask])
            if classes.size < 2:
                print(f"[WARN] Fold {fold}: task {tid} 验证集只有单一类别 {classes}.")
        fold_shared_feats, fold_specific_feats = preprocess_fold(
            shared_feats, specific_feats, task_ids, train_idx, val_idx
        )
        train_dataset = ICIDataset(encoded_rna.iloc[train_idx], fold_shared_feats.iloc[train_idx],
                                  fold_specific_feats.iloc[train_idx], responses.iloc[train_idx],
                                  task_ids.iloc[train_idx], OSs.iloc[train_idx], OSEs.iloc[train_idx],TMB_origin.iloc[train_idx],
                                  task_feature_dims)
        val_dataset = ICIDataset(encoded_rna.iloc[val_idx], fold_shared_feats.iloc[val_idx],
                                fold_specific_feats.iloc[val_idx], responses.iloc[val_idx],
                                task_ids.iloc[val_idx], OSs.iloc[val_idx], OSEs.iloc[val_idx],TMB_origin.iloc[val_idx],
                                task_feature_dims)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model = ICIMTL(
            shared_input_dim=encoded_rna.shape[1] + shared_feats.shape[1],
            expert_dnn_hidden_units=expert_dnn_hidden_units,
            num_experts=num_experts,
            gate_dnn_hidden_units=gate_dnn_hidden_units,
            num_tasks=4,
            task_feature_dims=task_feature_dims,
            fusion_dim=128,
            tower_dnn_hidden_units=[64, 32],
            l2_reg=l2_reg,
            dnn_activation='relu',
            dropout_rate=0.4,
            use_bn=True,
            init_std=0.001,
            seed=SEED,
            device=device
        ).to(device)
        print("[No TCGA Transfer] Train shared_encoder, gate, and task heads from scratch.")
        trainer = ICITrainer(model, device, learning_rate=1e-4)
        trainer.update_pos_weights_from_loader(train_loader)
        trainer.task_slices = train_dataset.task_slices
        early_stop = EarlyStopping(patience=13,min_delta=1e-4,monitor="auc",mode="max")
        best_model_path = output_dir / f"best_model_fold_{fold}.pth"
        for epoch in range(50):
            train_loss, train_metrics = trainer.train_epoch(train_loader)
            val_loss, val_metrics = trainer.evaluate(val_loader, return_loss=True)
            auc_values = []
            for task_key in val_metrics:
                if "auc" in val_metrics[task_key]:
                    auc_values.append(val_metrics[task_key]["auc"])
            mean_auc = float(np.mean(auc_values)) if auc_values else 0.0
            if early_stop.step({"auc": mean_auc}, epoch):
                torch.save(model.state_dict(), best_model_path)
            else:
                if early_stop.should_stop():
                    print(f"Early stopping at epoch {epoch} "
                          f"| best epoch = {early_stop.best_epoch}, best AUC = {early_stop.best_value:.4f}")
                    break
            trainer.scheduler.step(epoch)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
                for task_key in val_metrics:
                    metrics = val_metrics[task_key]
                    auc = float(np.nan_to_num(metrics.get('auc', np.nan), nan=0.0))
                    print(f"  {task_key}: AUC={auc:.4f}, ACC={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        mtl_val_loss, mtl_val_metrics = trainer.evaluate(val_loader, return_loss=True)
        for tid in range(4):
            task_key = f"task_{tid}"
            if task_key not in mtl_val_metrics:
                continue
            metrics = mtl_val_metrics[task_key]
            all_metrics = compute_all_metrics(
                metrics["labels"],
                metrics["preds"],
                threshold=0.5
            )
            all_metrics_icimtl[tid].append(all_metrics)
            y_prob_icimtl = np.asarray(metrics["preds"])
            task_val_mask = task_ids.iloc[val_idx] == tid
            val_idx_task = np.asarray(val_idx)[task_val_mask.values]
            if len(val_idx_task) != len(y_prob_icimtl):
                print(f"[WARN] Fold {fold}, task {tid}: sample size mismatch. Skipped record saving.")
                continue
            cohort_name = task_id_to_cohort[tid]
            for i, g_idx in enumerate(val_idx_task):
                icimtl_records.append({
                    "fold": int(fold),
                    "cohort": cohort_name,
                    "patientID": encoded_rna.index[g_idx],
                    "response": int(responses.iloc[g_idx]),
                    "prob": float(y_prob_icimtl[i]),
                    "OS": float(OSs.iloc[g_idx]),
                    "OS_event": int(OSEs.iloc[g_idx]),
                    "TMB": float(TMB_origin.iloc[g_idx])
                })
        icimtl_df = pd.DataFrame(icimtl_records)
        save_path = output_dir / "MTL_without_TCGA_pretraining_val_predictions.xlsx"
        icimtl_df.to_excel(save_path, index=False)
        print("\nSaved out-of-fold predictions:")
        print(f"  {save_path}")
        for task_id in range(4):
            task_name = ["DavidA", "DavidLiu", "Ravi", "IMvigor210"][task_id]
            print(f"\n======= Task {task_id} ({task_name}) Results =======")
            aucs = [m.get("auc", np.nan) for m in all_metrics_icimtl[task_id]]
            auprcs = [m.get("auprc", np.nan) for m in all_metrics_icimtl[task_id]]
            f1s = [m.get("f1", np.nan) for m in all_metrics_icimtl[task_id]]
            print("  --- MTL without TCGA pretraining ---")
            print(f"    AUC:      {np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f}")
            print(f"    AUPRC:    {np.nanmean(auprcs):.4f} ± {np.nanstd(auprcs):.4f}")
            print(f"    F1 Score: {np.nanmean(f1s):.4f} ± {np.nanstd(f1s):.4f}")

if __name__ == "__main__":
    five_fold_cross_validation()
