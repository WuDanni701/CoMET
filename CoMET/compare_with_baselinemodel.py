import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from utils1 import EarlyStopping,ICITransferTrainer
from autoencoder import Autoencoder
from ICIMTL_model import ICIMTL
from ICIdataset import ICIProcessor, ICIDataset
import matplotlib.pyplot as plt
from joblib import load
import os
import json
import re
import random
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, average_precision_score, roc_auc_score,f1_score,accuracy_score
from config import (DATA_ROOT, output_dir_tcga, hidden_dims_list, noise_std, l2_reg, num_experts, learning_rate_train, batch_size,
                    latent_dim, logvar_max, logvar_min, gene_dropout, expert_dnn_hidden_units, gate_dnn_hidden_units,SEED)

plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = ("Arial")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def plot_cv_roc_curves(models_metrics_dict, task_id, task_name):
    model_colors = {
        "CoMET": "#F06F6A",
        "Logistic Regression": "#62A8C4",
        "XGBoost": "#F2A654",
        "Neural Network": "#90BF58"
    }
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['left'].set_linewidth(1.8)
    ax.tick_params(axis='both', width=1.5, length=5, labelsize=14)
    mean_fpr = np.linspace(0, 1, 100)
    for model_name, all_metrics in models_metrics_dict.items():
        color = model_colors.get(model_name, "#999999")
        metrics_list = all_metrics[task_id]
        tprs = []
        aucs = []
        if not metrics_list:
            print(f"No metrics found for {model_name} on task {task_id}")
            continue
        for fold_metrics in metrics_list:
            if 'fpr' in fold_metrics and 'tpr' in fold_metrics and len(fold_metrics['fpr']) > 0:
                interp_tpr = np.interp(mean_fpr, fold_metrics['fpr'], fold_metrics['tpr'])
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(fold_metrics.get('auc', np.nan))
        if not tprs:
            print(f"No valid ROC data to plot for {model_name} on task {task_id}")
            continue
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.nanmean(aucs)
        line_width = 3.2 if model_name == "CoMET" else 2.2
        name1 = {
            "CoMET": "CoMET",
            "Logistic Regression": "LR",
            "XGBoost": "XGB",
            "Neural Network": "NN"
        }.get(model_name,model_name)
        plt.plot(
            mean_fpr,
            mean_tpr,
            label=f'{name1} (AUC = {mean_auc:.3f})',
            color=color,
            linewidth=line_width
        )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle='--',
        color='k',
        linewidth=1.8,
        label='Chance (AUC = 0.50)'
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    plt.title(f'Cross-Validated ROC Curves - ({task_name})', fontsize=18, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    normal_items = [
        (h, l) for h, l in zip(handles, labels)
        if not l.startswith('Chance')
    ]
    chance_items = [
        (h, l) for h, l in zip(handles, labels)
        if l.startswith('Chance')
    ]
    ordered_items = normal_items + chance_items
    handles, labels = zip(*ordered_items)
    legend = plt.legend(
        handles,
        labels,
        loc="lower right",
        fontsize=15,
        title_fontsize=14,
        frameon=True,
        borderpad=1.3,
        labelspacing=1.1,
        handlelength=2.2,
        handletextpad=0.8
    )
    for text in legend.get_texts():
        if text.get_text().startswith("CoMET"):
            text.set_color(model_colors["CoMET"])
            text.set_fontweight("bold")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

def load_tcga_artifacts(output_dir_tcga):
    with open(os.path.join(output_dir_tcga, "selected_genes.json"), "r") as f:
        selected_genes = json.load(f)
    scaler_path = os.path.join(output_dir_tcga, "std_scaler.joblib")
    ss = load(scaler_path)
    encoder_path = os.path.join(output_dir_tcga, "encoder.pth")
    enc_state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    se_path = os.path.join(output_dir_tcga, "shared_encoder.pth")
    return selected_genes, ss, enc_state,se_path
def prepare_ici_data(rna_paths, cohort_paths, shared_cols, specific_cols,
                     autoencoder, device, tcga_scaler, selected_genes, batch_size=50):
    processor = ICIProcessor(rna_paths, cohort_paths, shared_cols, specific_cols)
    all_rna, all_shared, all_specific, all_response, all_task_ids, OSs, OSEs,TMB_origin = processor.load_all()
    task_to_cohort = {0: "DavidA",1: "DavidLiu",2: "Ravi",3: "IMvigor210"}
    all_rna_scaled = all_rna.copy()
    task_array = np.asarray(all_task_ids).astype(int)
    for task_id, cohort_name in task_to_cohort.items():
        mask = (task_array == task_id)
        if mask.sum() == 0:
            continue
        scaler = tcga_scaler[cohort_name]["scaler"]
        cohort_rna = all_rna.loc[mask].copy()
        cohort_scaled = (scaler.transform(cohort_rna))
        cohort_scaled_df = pd.DataFrame(cohort_scaled,index=cohort_rna.index,columns=cohort_rna.columns)
        all_rna_scaled.loc[cohort_rna.index,cohort_rna.columns] = cohort_scaled_df
    genes_in_data = [g for g in selected_genes if g in  all_rna_scaled.columns]
    X_sel =  all_rna_scaled.loc[:, genes_in_data].copy()
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

def preprocess_fold(
        shared_feats, specific_feats, task_ids, train_idx, val_idx,
        shared_continuous_cols, task_specific_continuous_cols,
        task_specific_categorical_groups):
    shared_scaled = shared_feats.copy()
    specific_scaled = specific_feats.copy()
    fitted_preprocessors = {
        "shared": None,
        "specific": {},
        "categorical_modes": {},
    }
    valid_shared_cols = [
        col for col in shared_continuous_cols
        if col in shared_scaled.columns
    ]
    if valid_shared_cols:
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler()
        train_values = imputer.fit_transform(
            shared_feats.iloc[train_idx][valid_shared_cols]
        )
        val_values = imputer.transform(
            shared_feats.iloc[val_idx][valid_shared_cols]
        )
        positions = shared_scaled.columns.get_indexer(valid_shared_cols)
        shared_scaled.iloc[train_idx, positions] = scaler.fit_transform(train_values)
        shared_scaled.iloc[val_idx, positions] = scaler.transform(val_values)
        fitted_preprocessors["shared"] = {
            "columns": valid_shared_cols,
            "imputer": imputer,
            "scaler": scaler,
        }

    task_array = np.asarray(task_ids).astype(int)

    for tid, groups in task_specific_categorical_groups.items():
        task_train_idx = np.asarray(train_idx)[task_array[train_idx] == tid]
        task_val_idx = np.asarray(val_idx)[task_array[val_idx] == tid]
        fitted_preprocessors["categorical_modes"][tid] = {}

        for group_name, columns in groups.items():
            prefixed_cols = [f"t{tid}__{col}" for col in columns]
            absent_cols = [
                col for col in prefixed_cols
                if col not in specific_scaled.columns
            ]
            if absent_cols:
                raise ValueError(
                    f"Missing columns for task {tid}, {group_name}: {absent_cols}"
                )

            train_group = specific_feats.iloc[task_train_idx][prefixed_cols]
            train_all_missing = train_group.isna().all(axis=1)
            train_partial_missing = (
                train_group.isna().any(axis=1) & ~train_all_missing
            )
            if train_partial_missing.any():
                raise ValueError(
                    f"Partially missing one-hot group: task={tid}, group={group_name}"
                )

            observed_train = train_group.loc[~train_all_missing]
            if observed_train.empty:
                raise ValueError(
                    f"No observed training values: task={tid}, group={group_name}"
                )

            mode_col = observed_train.sum(axis=0).idxmax()
            mode_vector = np.zeros(len(prefixed_cols), dtype=float)
            mode_vector[prefixed_cols.index(mode_col)] = 1.0
            positions = specific_scaled.columns.get_indexer(prefixed_cols)

            missing_train_idx = task_train_idx[train_all_missing.to_numpy()]
            if len(missing_train_idx) > 0:
                specific_scaled.iloc[missing_train_idx, positions] = mode_vector

            if len(task_val_idx) > 0:
                val_group = specific_feats.iloc[task_val_idx][prefixed_cols]
                val_all_missing = val_group.isna().all(axis=1)
                val_partial_missing = (
                    val_group.isna().any(axis=1) & ~val_all_missing
                )
                if val_partial_missing.any():
                    raise ValueError(
                        f"Partially missing one-hot group: task={tid}, group={group_name}"
                    )
                missing_val_idx = task_val_idx[val_all_missing.to_numpy()]
                if len(missing_val_idx) > 0:
                    specific_scaled.iloc[missing_val_idx, positions] = mode_vector

            fitted_preprocessors["categorical_modes"][tid][group_name] = {
                "columns": prefixed_cols,
                "mode_column": mode_col,
                "mode_vector": mode_vector,
            }

    for tid, continuous_cols in task_specific_continuous_cols.items():
        prefixed_cols = [
            f"t{tid}__{col}"
            for col in continuous_cols
            if f"t{tid}__{col}" in specific_scaled.columns
        ]
        if not prefixed_cols:
            continue
        task_train_idx = np.asarray(train_idx)[task_array[train_idx] == tid]
        task_val_idx = np.asarray(val_idx)[task_array[val_idx] == tid]
        if len(task_train_idx) == 0:
            raise ValueError(f"No training samples for task {tid}")

        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler()
        train_values = imputer.fit_transform(
            specific_feats.iloc[task_train_idx][prefixed_cols]
        )
        positions = specific_scaled.columns.get_indexer(prefixed_cols)
        specific_scaled.iloc[task_train_idx, positions] = scaler.fit_transform(
            train_values
        )
        if len(task_val_idx) > 0:
            val_values = imputer.transform(
                specific_feats.iloc[task_val_idx][prefixed_cols]
            )
            specific_scaled.iloc[task_val_idx, positions] = scaler.transform(
                val_values
            )
        fitted_preprocessors["specific"][tid] = {
            "columns": prefixed_cols,
            "imputer": imputer,
            "scaler": scaler,
        }

    for name, frame in {
        "train shared": shared_scaled.iloc[train_idx],
        "validation shared": shared_scaled.iloc[val_idx],
        "train specific": specific_scaled.iloc[train_idx],
        "validation specific": specific_scaled.iloc[val_idx],
    }.items():
        bad_cols = frame.columns[frame.isna().any()].tolist()
        if bad_cols:
            raise ValueError(f"NaN remains in {name}: {bad_cols}")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite value remains in {name}")

    return shared_scaled, specific_scaled, fitted_preprocessors

def five_fold_cross_validation(freeze_shared=False):
    #read in data
    CODE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CODE_DIR.parent
    output_dir = PROJECT_ROOT / "output1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cohort_paths = [
        DATA_ROOT / "clinical" / "DavidA_clinical.csv",
        DATA_ROOT / "clinical" / "DavidLiu_clinical.csv",
        DATA_ROOT / "clinical" / "Ravi_clinical.csv",
        DATA_ROOT / "clinical" / "IMvigor210_clinical.csv",
    ]
    rna_paths = [
        DATA_ROOT / "rna" / "DavidA_rna.csv",
        DATA_ROOT / "rna" / "DavidLiu_rna.csv",
        DATA_ROOT / "rna" / "Ravi_rna.csv",
        DATA_ROOT / "rna" / "IMvigor210_rna.csv",
    ]
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
    shared_continuous_cols = ["TMB"]
    task_specific_continuous_cols = {
        0:["Age","FS_Counts","NEO_Weak","ITH"],
        1:["CNA_prop","heterogeneity","Purity","MHC-II","TMB_clonal","TMB_subclonal"],
        2:[ "Purity","Smoking_Pack_Years","Neoantigens","TMB_clonal","TMB_subclonal"],
        3:["Neoantigen_burden"]
    }
    task_specific_categorical_groups = {
        0: {
            "MSKCC": [
                "MSKCC_FAVORABLE", "MSKCC_INTERMEDIATE", "MSKCC_POOR"
            ],
            "Deletion_9p21.3": [
                "Deletion_9p21.3_MUT", "Deletion_9p21.3_WT"
            ],
            "Deletion_11q23.1": [
                "Deletion_11q23.1_MUT", "Deletion_11q23.1_WT"
            ],
            "Amplification_12q24.32": [
                "Amplification_12q24.32_MUT",
                "Amplification_12q24.32_WT",
                "Amplification_12q24.32_WUT",
            ],
            "Amplification_6q21": [
                "Amplification_6q21_MUT", "Amplification_6q21_WT"
            ],
        },
        1: {
            "ECOG": ["ECOG_0", "ECOG_1", "ECOG_2", "ECOG_3"],
        },
    }
    selected_genes, ss, enc_state, se_path = load_tcga_artifacts(output_dir_tcga)
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
    autoencoder.load_state_dict(enc_state, strict=False)
    autoencoder.eval().to(device)
    encoded_rna, shared_feats, specific_feats, responses, task_ids, OSs, OSEs,TMB_origin = prepare_ici_data(
        rna_paths, cohort_paths, shared_feature_cols, task_specific_feature_cols,
        autoencoder, device,
        tcga_scaler=ss,
        selected_genes=selected_genes,
        batch_size=42
    )
    all_metrics_icimtl = {0: [], 1: [], 2: [],3:[]}
    all_metrics_xgboost = {0: [], 1: [], 2: [],3:[]}
    all_metrics_lr = {0: [], 1: [], 2: [],3:[]}
    all_metrics_mlp = {0: [], 1: [], 2: [],3:[]}
    icimtl_records = []
    lr_records = []
    xgb_records = []
    mlp_records = []
    task_id_to_cohort = {0: "DavidA", 1: "DavidLiu", 2: "Ravi",3:"IMvigor210"}
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
        print(f"Fold {fold + 1}/5")
        fold_shared_feats, fold_specific_feats, clinical_preprocessors = (
            preprocess_fold(shared_feats=shared_feats,
            specific_feats=specific_feats,
            task_ids=task_ids,
            train_idx=train_idx,
            val_idx=val_idx,
            shared_continuous_cols=shared_continuous_cols,
            task_specific_continuous_cols=task_specific_continuous_cols,
            task_specific_categorical_groups=task_specific_categorical_groups)
        )
        train_dataset = ICIDataset(encoded_rna.iloc[train_idx], fold_shared_feats.iloc[train_idx],
                                  fold_specific_feats.iloc[train_idx], responses.iloc[train_idx],
                                  task_ids.iloc[train_idx], OSs.iloc[train_idx], OSEs.iloc[train_idx],TMB_origin.iloc[train_idx],
                                  task_feature_dims)
        val_dataset = ICIDataset(encoded_rna.iloc[val_idx], fold_shared_feats.iloc[val_idx],
                                fold_specific_feats.iloc[val_idx], responses.iloc[val_idx],
                                task_ids.iloc[val_idx], OSs.iloc[val_idx], OSEs.iloc[val_idx],TMB_origin.iloc[val_idx],
                                task_feature_dims)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model = ICIMTL(
            shared_input_dim=encoded_rna.shape[1] + fold_shared_feats.shape[1],
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
        if freeze_shared:
            pretrained_state_dict = torch.load(se_path, map_location=device, weights_only=True)
            new_shared_encoder_state_dict = model.shared_encoder.state_dict()
            expert_copied = 0
            gate_copied = 0
            skipped = []
            gate_src2tgt = {
                3: [1],
                2: [2],
                0: [0],
                4:[3]
            }
            for name, param in pretrained_state_dict.items():
                if name.startswith("expert_dnn."):
                    if name in new_shared_encoder_state_dict and \
                            new_shared_encoder_state_dict[name].shape == param.shape:
                        new_shared_encoder_state_dict[name] = param.clone()
                        expert_copied += 1
                    else:
                        skipped.append((
                            "expert",
                            name,
                            tuple(param.shape),
                            tuple(new_shared_encoder_state_dict.get(name, torch.empty(0)).shape)
                        ))
                elif name.startswith("gate_dnn."):
                    m = re.match(r"gate_dnn\.(\d+)\.(.+)", name)
                    if m is None:
                        continue
                    src_tid = int(m.group(1))
                    suffix = m.group(2)
                    if src_tid not in gate_src2tgt:
                        continue
                    for tgt_tid in gate_src2tgt[src_tid]:
                        tgt_name = f"gate_dnn.{tgt_tid}.{suffix}"
                        if tgt_name in new_shared_encoder_state_dict and \
                                new_shared_encoder_state_dict[tgt_name].shape == param.shape:
                            new_shared_encoder_state_dict[tgt_name] = param.clone()
                            gate_copied += 1
                        else:
                            skipped.append((
                                "gate",
                                f"{name} -> {tgt_name}",
                                tuple(param.shape),
                                tuple(new_shared_encoder_state_dict.get(tgt_name, torch.empty(0)).shape)
                            ))
            model.shared_encoder.load_state_dict(new_shared_encoder_state_dict, strict=False)
        trainer = ICITransferTrainer(model, device, learning_rate=learning_rate_train, freeze_shared=freeze_shared)
        trainer.update_pos_weights_from_loader(train_loader)
        trainer.task_slices = train_dataset.task_slices
        early_stop = EarlyStopping(patience=13,min_delta=1e-4,monitor="auc",mode="max")
        best_model_path = os.path.join(output_dir, f"best_model_fold_{fold}.pth")
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
                    break
            trainer.scheduler.step(epoch)
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        mtl_val_loss, mtl_val_metrics = trainer.evaluate(val_loader, return_loss=True)
        for tid in range(4):
            task_key = f"task_{tid}"
            if task_key in mtl_val_metrics:
                metrics = mtl_val_metrics[task_key]
                all_metrics = compute_all_metrics(metrics['labels'], metrics['preds'],threshold=0.5)
                all_metrics_icimtl[tid].append(all_metrics)
                y_prob_icimtl = np.asarray(metrics['preds'])
                task_val_mask = (task_ids.iloc[val_idx] == tid)
                val_idx_task = np.asarray(val_idx)[task_val_mask.values]
                if len(val_idx_task) == len(y_prob_icimtl):
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
        #LR, XGB, MLP
        for tid in range(4):
            task_train_mask = (task_ids.iloc[train_idx] == tid)
            task_val_mask = (task_ids.iloc[val_idx] == tid)
            train_idx_task = train_idx[task_train_mask]
            val_idx_task = val_idx[task_val_mask]
            if len(val_idx_task) == 0 or len(np.unique(responses.iloc[val_idx_task])) < 2:
                print(f"Skipping baseline task {tid} in fold {fold}, insufficient val data.")
                all_metrics_lr[tid].append(compute_all_metrics(np.array([]), np.array([])))
                all_metrics_xgboost[tid].append(compute_all_metrics(np.array([]), np.array([])))
                all_metrics_mlp[tid].append(compute_all_metrics(np.array([]), np.array([])))
                continue
            X_train_rna = encoded_rna.iloc[train_idx_task]
            X_val_rna = encoded_rna.iloc[val_idx_task]
            X_train_shared = fold_shared_feats.iloc[train_idx_task]
            X_val_shared = fold_shared_feats.iloc[val_idx_task]
            start, end = task_slices[tid]
            X_train_specific_task = fold_specific_feats.iloc[
                                    train_idx_task, start:end
                                    ]
            X_val_specific_task = fold_specific_feats.iloc[
                                  val_idx_task, start:end
                                  ]
            X_train_task = pd.concat([X_train_rna, X_train_shared, X_train_specific_task], axis=1)
            X_val_task = pd.concat([X_val_rna, X_val_shared, X_val_specific_task], axis=1)
            y_train_task = responses.iloc[train_idx_task]
            y_val_task = responses.iloc[val_idx_task]
            X_train_task.columns = X_train_task.columns.astype(str)
            X_val_task.columns = X_val_task.columns.astype(str)
            cohort_name = task_id_to_cohort[tid]
            #lr
            lr_model = LogisticRegression(max_iter=500, solver='liblinear',random_state=SEED)
            lr_model.fit(X_train_task, y_train_task)
            y_prob_lr = lr_model.predict_proba(X_val_task)[:, 1]
            all_metrics_lr[tid].append(compute_all_metrics(y_val_task, y_prob_lr, threshold=0.5))
            #xgb
            xgb_model = xgb.XGBClassifier(
                n_estimators=50, max_depth=2,random_state=SEED
            )
            xgb_model.fit(X_train_task, y_train_task)
            y_prob_xgb = xgb_model.predict_proba(X_val_task)[:, 1]
            all_metrics_xgboost[tid].append(compute_all_metrics(y_val_task, y_prob_xgb,threshold=0.5))
            #nn
            nn_model = MLPClassifier(hidden_layer_sizes=(18,9), alpha=4e-3,max_iter=2000,random_state=SEED)
            nn_model.fit(X_train_task, y_train_task)
            y_prob_mlp = nn_model.predict_proba(X_val_task)[:, 1]
            all_metrics_mlp[tid].append(compute_all_metrics(y_val_task, y_prob_mlp,threshold=0.5))
            for i, g_idx in enumerate(val_idx_task):
                base_info = {
                    "fold": int(fold),
                    "cohort": cohort_name,
                    "patientID": encoded_rna.index[g_idx],
                    "response": int(responses.iloc[g_idx]),
                    "OS": float(OSs.iloc[g_idx]),
                    "OS_event": int(OSEs.iloc[g_idx]),
                    "TMB": float(TMB_origin.iloc[g_idx])
                }
                lr_records.append({
                    **base_info,
                    "prob": float(y_prob_lr[i])
                })
                xgb_records.append({
                    **base_info,
                    "prob": float(y_prob_xgb[i])
                })
                mlp_records.append({
                    **base_info,
                    "prob": float(y_prob_mlp[i])
                })
    os.makedirs(output_dir, exist_ok=True)
    icimtl_df = pd.DataFrame(icimtl_records)
    lr_df = pd.DataFrame(lr_records)
    xgb_df = pd.DataFrame(xgb_records)
    mlp_df = pd.DataFrame(mlp_records)
    icimtl_df.to_excel(os.path.join(output_dir, "MTL_val_predictions.xlsx"), index=False)
    lr_df.to_excel(os.path.join(output_dir, "LogisticRegression_val_predictions.xlsx"), index=False)
    xgb_df.to_excel(os.path.join(output_dir, "XGBoost_val_predictions.xlsx"), index=False)
    mlp_df.to_excel(os.path.join(output_dir, "NeuralNetwork_val_predictions.xlsx"), index=False)

    for task_id in range(4):
        task_name = ['DavidA', 'DavidLiu', 'Ravi','IMvigor210'][task_id]
        print(f"\n======= Task {task_id} ({task_name}) Results =======")
        def print_model_metrics(model_name, metrics_dict_list):
            aucs = [m.get('auc', np.nan) for m in metrics_dict_list[task_id]]
            auprcs = [m.get('auprc', np.nan) for m in metrics_dict_list[task_id]]
            f1s = [m.get('f1', np.nan) for m in metrics_dict_list[task_id]]
            print(f"  --- {model_name} ---")
            print(f"    AUC:      {np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f}")
            print(f"    AUPRC:    {np.nanmean(auprcs):.4f} ± {np.nanstd(auprcs):.4f}")
            print(f"    F1 Score: {np.nanmean(f1s):.4f} ± {np.nanstd(f1s):.4f}")
        print_model_metrics("CoMET", all_metrics_icimtl)
        print_model_metrics("Logistic Regression", all_metrics_lr)
        print_model_metrics("XGBoost", all_metrics_xgboost)
        print_model_metrics("Neural Network", all_metrics_mlp)
        models_to_plot = {
            "CoMET": all_metrics_icimtl,
            "Logistic Regression": all_metrics_lr,
            "XGBoost": all_metrics_xgboost,
            "Neural Network": all_metrics_mlp
        }
        plot_cv_roc_curves(models_to_plot, task_id, task_name)
if __name__ == "__main__":
    five_fold_cross_validation(freeze_shared=True)
