"""RMSE, MAE, and wind-speed error metrics for u/v local wind fields."""

from __future__ import annotations

import numpy as np


def _validate(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError(f"Expected matching (N,2,H,W) arrays, got {prediction.shape} and {target.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Metrics require finite prediction and target values")
    return prediction.astype(np.float64, copy=False), target.astype(np.float64, copy=False)


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = _validate(prediction, target)
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = _validate(prediction, target)
    return float(np.mean(np.abs(prediction - target)))


def wind_speed_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute wind-speed error, also reported as WS-MAE/AWSE."""
    prediction, target = _validate(prediction, target)
    speed_prediction = np.hypot(prediction[:, 0], prediction[:, 1])
    speed_target = np.hypot(target[:, 0], target[:, 1])
    return float(np.mean(np.abs(speed_prediction - speed_target)))


awse = wind_speed_mae


def all_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {"rmse_m_s": rmse(prediction, target), "mae_m_s": mae(prediction, target), "ws_mae_m_s": wind_speed_mae(prediction, target)}
