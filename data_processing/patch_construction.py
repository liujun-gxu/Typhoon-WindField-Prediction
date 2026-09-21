"""Fixed-t0 patch semantics used to construct supervised local wind-field samples."""

from __future__ import annotations

import numpy as np


def extract_centered_patch(field: np.ndarray, center_y: int, center_x: int, patch_size: int = 31) -> np.ndarray:
    """Extract a Cx31x31 patch from a CxHxW wind field with strict boundary checks."""
    if field.ndim != 3 or field.shape[0] != 2:
        raise ValueError(f"Expected field shape (2,H,W), got {field.shape}")
    half = patch_size // 2
    y0, y1, x0, x1 = center_y - half, center_y + half + 1, center_x - half, center_x + half + 1
    if y0 < 0 or x0 < 0 or y1 > field.shape[1] or x1 > field.shape[2]:
        raise ValueError("Patch crosses the supplied field boundary; provide a stitched/padded domain first.")
    patch = field[:, y0:y1, x0:x1]
    if patch.shape != (2, patch_size, patch_size) or not np.isfinite(patch).all():
        raise ValueError("Invalid extracted patch")
    return patch.astype(np.float32, copy=False)


def build_fixed_t0_sample(
    input_fields: list[tuple[np.ndarray, int, int]], future_field: np.ndarray, t0_center_y: int, t0_center_x: int, patch_size: int = 31
) -> tuple[np.ndarray, np.ndarray]:
    """Build X and Y using the corrected definition Y_H=W(t0+H, C(t0)).

    ``input_fields`` must contain ``(field, center_y, center_x)`` for the four
    historical fields at t0-3, t0-2, t0-1, and t0.
    The future target is deliberately extracted around the *t0* center, not the
    future best-track center.
    """
    if len(input_fields) != 4:
        raise ValueError("Exactly four historical hourly fields are required")
    x = np.stack([extract_centered_patch(field, center_y, center_x, patch_size) for field, center_y, center_x in input_fields])
    y = extract_centered_patch(future_field, t0_center_y, t0_center_x, patch_size)
    return x, y
