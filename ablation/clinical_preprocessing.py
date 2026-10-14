from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_ROOT = PROJECT_ROOT / "icidata"
COHORTS = ("DavidA", "DavidLiu", "Ravi", "IMvigor210")
SHARED_CONTINUOUS_COLUMNS = ["TMB"]
TASK_CONTINUOUS_COLUMNS = {
    0: ["Age", "FS_Counts", "NEO_Weak", "ITH"],
    1: ["CNA_prop", "heterogeneity", "Purity", "MHC-II", "TMB_clonal", "TMB_subclonal"],
    2: ["Purity", "Smoking_Pack_Years", "Neoantigens", "TMB_clonal", "TMB_subclonal"],
    3: ["Neoantigen_burden"],
}
TASK_CATEGORICAL_GROUPS = {
    0: {
        "MSKCC": ["MSKCC_FAVORABLE", "MSKCC_INTERMEDIATE", "MSKCC_POOR"],
        "Deletion_9p21.3": ["Deletion_9p21.3_MUT", "Deletion_9p21.3_WT"],
        "Deletion_11q23.1": ["Deletion_11q23.1_MUT", "Deletion_11q23.1_WT"],
        "Amplification_12q24.32": ["Amplification_12q24.32_MUT", "Amplification_12q24.32_WT", "Amplification_12q24.32_WUT"],
        "Amplification_6q21": ["Amplification_6q21_MUT", "Amplification_6q21_WT"],
    },
    1: {"ECOG": ["ECOG_0", "ECOG_1", "ECOG_2", "ECOG_3"]},
}
def get_clinical_paths():
    return [DATA_ROOT / "clinical" / f"{cohort}_clinical.csv" for cohort in COHORTS]
def get_rna_paths():
    return [DATA_ROOT / "rna" / f"{cohort}_rna.csv" for cohort in COHORTS]
def preprocess_fold(shared, specific, task_ids, train_idx, val_idx):
    shared_out, specific_out = shared.copy(), specific.copy()
    train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
    task_array = np.asarray(task_ids, dtype=int)
    shared_cols = [c for c in SHARED_CONTINUOUS_COLUMNS if c in shared.columns]
    if shared_cols:
        imp, scaler = SimpleImputer(strategy="median"), RobustScaler()
        train_values = imp.fit_transform(shared.iloc[train_idx][shared_cols])
        positions = shared_out.columns.get_indexer(shared_cols)
        shared_out.iloc[train_idx, positions] = scaler.fit_transform(train_values)
        shared_out.iloc[val_idx, positions] = scaler.transform(
            imp.transform(shared.iloc[val_idx][shared_cols])
        )

    for task_id, groups in TASK_CATEGORICAL_GROUPS.items():
        task_train = train_idx[task_array[train_idx] == task_id]
        task_val = val_idx[task_array[val_idx] == task_id]
        if not len(task_train) and not len(task_val):
            continue
        for group_name, names in groups.items():
            columns = [f"t{task_id}__{name}" for name in names]
            missing_columns = [c for c in columns if c not in specific.columns]
            if missing_columns:
                raise ValueError(f"Missing columns for task {task_id}, {group_name}: {missing_columns}")
            train_group = specific.iloc[task_train][columns]
            train_missing = train_group.isna().all(axis=1)
            if (train_group.isna().any(axis=1) & ~train_missing).any():
                raise ValueError(f"Partially missing one-hot group: task={task_id}, group={group_name}")
            observed = train_group.loc[~train_missing]
            if observed.empty:
                raise ValueError(f"No observed training values: task={task_id}, group={group_name}")
            mode_column = observed.sum(axis=0).idxmax()
            mode_vector = np.zeros(len(columns), dtype=float)
            mode_vector[columns.index(mode_column)] = 1.0
            positions = specific_out.columns.get_indexer(columns)
            missing_train_idx = task_train[train_missing.to_numpy()]
            if len(missing_train_idx):
                specific_out.iloc[missing_train_idx, positions] = mode_vector
            if len(task_val):
                val_group = specific.iloc[task_val][columns]
                val_missing = val_group.isna().all(axis=1)
                if (val_group.isna().any(axis=1) & ~val_missing).any():
                    raise ValueError(f"Partially missing one-hot group: task={task_id}, group={group_name}")
                missing_val_idx = task_val[val_missing.to_numpy()]
                if len(missing_val_idx):
                    specific_out.iloc[missing_val_idx, positions] = mode_vector

    for task_id, names in TASK_CONTINUOUS_COLUMNS.items():
        columns = [f"t{task_id}__{name}" for name in names if f"t{task_id}__{name}" in specific.columns]
        if not columns:
            continue
        task_train = train_idx[task_array[train_idx] == task_id]
        task_val = val_idx[task_array[val_idx] == task_id]
        if not len(task_train):
            raise ValueError(f"No training samples for task {task_id}")
        imp, scaler = SimpleImputer(strategy="median"), RobustScaler()
        train_values = imp.fit_transform(specific.iloc[task_train][columns])
        positions = specific_out.columns.get_indexer(columns)
        specific_out.iloc[task_train, positions] = scaler.fit_transform(train_values)
        if len(task_val):
            specific_out.iloc[task_val, positions] = scaler.transform(
                imp.transform(specific.iloc[task_val][columns])
            )

    for label, frame in (("train shared", shared_out.iloc[train_idx]),
                         ("validation shared", shared_out.iloc[val_idx]),
                         ("train specific", specific_out.iloc[train_idx]),
                         ("validation specific", specific_out.iloc[val_idx])):
        bad = frame.columns[frame.isna().any()].tolist()
        if bad:
            raise ValueError(f"NaN remains in {label}: {bad}")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite value remains in {label}")
    return shared_out, specific_out
def preprocess_single_cohort_fold(shared, specific, cohort, train_idx, val_idx):
    global_task_id = COHORTS.index(cohort)
    original_columns = specific.columns.copy()
    global_specific = specific.rename(
        columns=lambda c: f"t{global_task_id}__{c.split('__', 1)[-1]}"
    )
    fake_tasks = np.full(len(shared), global_task_id, dtype=int)
    shared_out, specific_out = preprocess_fold(
        shared, global_specific, fake_tasks, train_idx, val_idx
    )
    specific_out.columns = original_columns
    return shared_out, specific_out
