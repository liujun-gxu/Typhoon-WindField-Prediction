"""Load prepared fixed-t0 NPZ splits without embedding machine-specific paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def split_path(data_dir: Path, horizon_h: int, split: str) -> Path:
    return data_dir / f"fixed_t0_{horizon_h}h_{split}_raw.npz"


def load_split(data_dir: Path, horizon_h: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Read X=(N,4,2,31,31) and Y=(N,2,31,31) from a prepared split."""
    path = split_path(data_dir, horizon_h, split)
    if not path.exists():
        raise FileNotFoundError(f"Prepared split not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if not {"X", "Y"}.issubset(data.files):
            raise KeyError(f"{path} must contain X and Y")
        x, y = data["X"].astype(np.float32, copy=False), data["Y"].astype(np.float32, copy=False)
    if x.ndim != 5 or x.shape[1:] != (4, 2, 31, 31) or y.ndim != 4 or y.shape[1:] != (2, 31, 31):
        raise ValueError(f"Unexpected split shapes X={x.shape}, Y={y.shape}")
    if len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid sample count or non-finite values in prepared split")
    return x, y
