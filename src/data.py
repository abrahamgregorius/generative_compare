# =============================================================================
# src/data.py — standardized MNIST split & loaders (seed 42, 54k/6k/10k, [0,1])
# Shared by train_mlp_proof.py and train_generative_compare.py.
# =============================================================================
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms

SEED = 42
BATCH = 128
VAL_FRAC = 0.10
DATA_ROOT = os.path.join("artifacts_mlp_gen_proof", "data")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(batch_size: int = BATCH):
    """80% train / 10% val / 10% test split with a fixed permutation (seed 42).
    Pixels are scaled to [0,1] (ToTensor), no further normalization."""
    tfm = transforms.ToTensor()
    train_full = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tfm)
    test_ds = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tfm)
    n_val = int(VAL_FRAC * len(train_full))
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(train_full), generator=g).tolist()
    val_ds, trn_ds = Subset(train_full, idx[:n_val]), Subset(train_full, idx[n_val:])
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=0)
    return mk(trn_ds, True), mk(val_ds, False), mk(test_ds, False)
