"""Train-only shared per-channel Min-Max normalization for u10/v10 fields."""

from __future__ import annotations

import numpy as np


def fit_train_minmax(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit one u/v range from training X and Y only; never use validation/test data."""
    x_min, x_max = x_train.min(axis=(0, 1, 3, 4)), x_train.max(axis=(0, 1, 3, 4))
    y_min, y_max = y_train.min(axis=(0, 2, 3)), y_train.max(axis=(0, 2, 3))
    channel_min, channel_max = np.minimum(x_min, y_min).astype(np.float32), np.maximum(x_max, y_max).astype(np.float32)
    if not np.isfinite(channel_min).all() or not np.isfinite(channel_max).all() or np.any(channel_max <= channel_min):
        raise ValueError("Invalid training Min-Max statistics")
    return channel_min, channel_max


def normalize_x(x: np.ndarray, channel_min: np.ndarray, channel_max: np.ndarray) -> np.ndarray:
    return ((x - channel_min[None, None, :, None, None]) / (channel_max - channel_min)[None, None, :, None, None]).astype(np.float32)


def normalize_y(y: np.ndarray, channel_min: np.ndarray, channel_max: np.ndarray) -> np.ndarray:
    return ((y - channel_min[None, :, None, None]) / (channel_max - channel_min)[None, :, None, None]).astype(np.float32)


def denormalize_y(y: np.ndarray, channel_min: np.ndarray, channel_max: np.ndarray) -> np.ndarray:
    return y * (channel_max - channel_min)[None, :, None, None] + channel_min[None, :, None, None]
