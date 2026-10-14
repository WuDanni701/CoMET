import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ICIProcessor:
    def __init__(self, rna_paths, cohort_paths, shared_cols, specific_cols):
        self.rna_paths = rna_paths
        self.cohort_paths = cohort_paths
        self.shared_cols = shared_cols
        self.specific_cols = specific_cols
    def load_all(self):
        all_rna, all_shared, all_specific, all_response, all_task_ids, OSs, OSEs,TMB_origin = [], [], [], [], [], [], [],[]
        def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df.columns = df.columns.astype(str).str.strip()
            return df
        prefixed_cols_per_task = []
        for t_id, spec_cols in enumerate(self.specific_cols):
            spec_cols = [str(c).strip() for c in spec_cols]
            prefixed = [f"t{t_id}__{c}" for c in spec_cols]
            prefixed_cols_per_task.append(prefixed)
        specific_all_cols = []
        for cols in prefixed_cols_per_task:
            specific_all_cols.extend(cols)
        for task_id, (rna_path, cohort_path, spec_cols) in enumerate(zip(self.rna_paths, self.cohort_paths, self.specific_cols)):
            rna_df = _clean_cols(pd.read_csv(rna_path, index_col=0))
            cohort_df = _clean_cols(pd.read_csv(cohort_path,index_col=0))
            common = cohort_df.index.intersection(rna_df.index)
            common = common.sort_values()
            cohort_df = cohort_df.loc[common]
            rna_df = rna_df.loc[common]
            shared_features = cohort_df.reindex(columns=self.shared_cols)
            TMB_origin.append(cohort_df.loc[common, "TMB_origin"])
            spec_cols_clean = [str(c).strip() for c in spec_cols]
            specific_raw = cohort_df.reindex(columns=spec_cols_clean, fill_value=0.0).copy()
            cur_prefixed = [f"t{task_id}__{c}" for c in spec_cols_clean]
            specific_raw.columns = cur_prefixed
            spec_full = pd.DataFrame(0.0, index=cohort_df.index, columns=specific_all_cols)
            spec_full.loc[:, cur_prefixed] = specific_raw.values
            response = cohort_df['response']
            OS = cohort_df['OS']
            OSE = cohort_df['OS_Event']
            all_rna.append(rna_df)
            all_shared.append(shared_features)
            all_specific.append(spec_full)
            all_response.append(response)
            all_task_ids.append(pd.Series(np.full(len(response), task_id), index=response.index))
            OSs.append(OS)
            OSEs.append(OSE)
        all_rna = pd.concat(all_rna, axis=0)
        all_shared = pd.concat(all_shared, axis=0,)
        all_specific = pd.concat(all_specific, axis=0)[specific_all_cols]
        all_response = pd.concat(all_response, axis=0)
        all_task_ids = pd.concat(all_task_ids, axis=0)
        OSs = pd.concat(OSs, axis=0)
        OSEs = pd.concat(OSEs, axis=0)
        TMB_origin = pd.concat(TMB_origin, axis=0)
        return all_rna, all_shared, all_specific, all_response, all_task_ids, OSs, OSEs,TMB_origin

class ICIDataset(Dataset):
    def __init__(self, rna_encoded, shared_feats, specific_feats, response, task_ids, OSs, OSEs,TMB_origin,task_feature_dims):
        self.rna = torch.FloatTensor(rna_encoded.values.astype(np.float32))
        self.shared = torch.FloatTensor(shared_feats.values.astype(np.float32))
        self.specific = torch.FloatTensor(specific_feats.values.astype(np.float32))
        self.response = torch.FloatTensor(response.values.astype(np.float32))
        self.task_ids = torch.LongTensor(task_ids.values)
        self.OSs = torch.FloatTensor(OSs.values.astype(np.float32))
        self.OSEs = torch.FloatTensor(OSEs.values.astype(np.float32))
        self.TMB_origin = torch.FloatTensor(TMB_origin.values.astype(np.float32))
        self.task_slices = []
        start = 0
        for dim in task_feature_dims:
            end = start + dim
            self.task_slices.append((start, end))
            start = end
    def __len__(self):
        return len(self.response)
    def __getitem__(self, idx):
        return {
            "rna": self.rna[idx],
            "shared": self.shared[idx],
            "specific": self.specific[idx],
            "label": self.response[idx],
            "task_id": self.task_ids[idx],
            "OS": self.OSs[idx],
            "OSE": self.OSEs[idx],
            "TMB_origin": self.TMB_origin[idx],
        }
