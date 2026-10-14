from pathlib import Path
import os
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_ROOT = PROJECT_ROOT / "icidata"
output_dir_tcga = PROJECT_ROOT / "TCGA_pretraining"
latent_dim = 20
hidden_dims_list = [128, 64]
noise_std = 0.004
gene_dropout = 0.002
logvar_min = -5.0
logvar_max = 2.0
lr_ae= 0.0005
SEED=269
expert_dnn_hidden_units = [64]
gate_dnn_hidden_units = [32]
l2_reg = 1e-3
weight_decay = 2e-4
num_experts = 2
batch_size=45
learning_rate_train=1e-3
