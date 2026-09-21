"""Draw manuscript Figure 4 from seed-0 independent-2023 prediction arrays.

Each run reads the corrected fixed-t0 main-experiment ``prediction`` and
``target`` arrays directly, then recomputes the unsmoothed 31 x 31 maps.  It
does not read pre-aggregated maps or construct distributions from summaries.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


HORIZONS = (6, 12, 24)
METRICS = (
    "u_signed_bias_m_s",
    "v_signed_bias_m_s",
    "vector_component_mae_m_s",
    "wind_speed_mae_m_s",
)
TITLES = {
    "u_signed_bias_m_s": "Signed u-component bias",
    "v_signed_bias_m_s": "Signed v-component bias",
    "vector_component_mae_m_s": "Vector-component MAE",
    "wind_speed_mae_m_s": "WS-MAE",
}
COLORBARS = {
    "u_signed_bias_m_s": "u bias (m$\\cdot$s$^{-1}$)",
    "v_signed_bias_m_s": "v bias (m$\\cdot$s$^{-1}$)",
    "vector_component_mae_m_s": "Vector-component MAE (m$\\cdot$s$^{-1}$)",
    "wind_speed_mae_m_s": "WS-MAE (m$\\cdot$s$^{-1}$)",
}
EXPECTED_DOMAIN_MEANS = {
    6: (0.0934, 0.4710, 2.2445, 2.0534, 490),
    12: (-0.0583, 0.0926, 2.8084, 2.5778, 480),
    24: (-0.2350, 0.3899, 3.6807, 3.3671, 456),
}


def calculate_maps_from_predictions(results_dir: Path) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, Path]]:
    """Recompute maps from the seed-0 independent-2023 prediction/target arrays."""
    all_maps: dict[int, dict[str, np.ndarray]] = {}
    source_paths: dict[int, Path] = {}
    for horizon in HORIZONS:
        path = results_dir / f"{horizon}h" / "test_predictions.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing seed-0 prediction result: {path}")
        with np.load(path, allow_pickle=False) as data:
            missing = {"prediction", "target"}.difference(data.files)
            if missing:
                raise KeyError(f"{path} is missing arrays: {sorted(missing)}")
            prediction = data["prediction"].astype(np.float64, copy=False)
            target = data["target"].astype(np.float64, copy=False)

        expected_samples = EXPECTED_DOMAIN_MEANS[horizon][4]
        expected_shape = (expected_samples, 2, 31, 31)
        if prediction.shape != expected_shape or target.shape != expected_shape:
            raise ValueError(f"Unexpected prediction/target shape in {path}: {prediction.shape} / {target.shape}")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError(f"Non-finite prediction/target values in {path}")

        error = prediction - target
        predicted_speed = np.hypot(prediction[:, 0], prediction[:, 1])
        target_speed = np.hypot(target[:, 0], target[:, 1])
        matrices = {
            "u_signed_bias_m_s": error[:, 0].mean(axis=0),
            "v_signed_bias_m_s": error[:, 1].mean(axis=0),
            "vector_component_mae_m_s": ((np.abs(error[:, 0]) + np.abs(error[:, 1])) / 2.0).mean(axis=0),
            "wind_speed_mae_m_s": np.abs(predicted_speed - target_speed).mean(axis=0),
        }
        actual_means = tuple(float(matrices[metric].mean()) for metric in METRICS)
        expected_means = EXPECTED_DOMAIN_MEANS[horizon][:4]
        if not np.allclose(actual_means, expected_means, rtol=0.0, atol=5e-5):
            raise ValueError(
                f"Formal domain-mean check failed for {horizon} h: "
                f"observed={actual_means}, expected={expected_means}"
            )
        for metric, matrix in matrices.items():
            if matrix.shape != (31, 31) or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid recomputed {metric} map in {path}: {matrix.shape}")
        all_maps[horizon] = matrices
        source_paths[horizon] = path
    return all_maps, source_paths


def column_norms(all_maps: dict[int, dict[str, np.ndarray]]) -> dict[str, object]:
    """Make a single comparison scale for all three horizons in each column."""
    norms: dict[str, object] = {}
    for metric in METRICS:
        values = np.concatenate([all_maps[horizon][metric].ravel() for horizon in HORIZONS])
        if "signed" in metric:
            limit = float(np.max(np.abs(values)))
            norms[metric] = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        else:
            norms[metric] = plt.Normalize(vmin=0.0, vmax=float(np.max(values)))
    return norms


def domain_means(maps: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    """Return the four domain means in the order used for manuscript checks."""
    return tuple(float(maps[metric].mean()) for metric in METRICS)  # type: ignore[return-value]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    output_dir = root / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_maps, source_paths = calculate_maps_from_predictions(results_dir)
    norms = column_norms(all_maps)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    # Wider than the prior export, while preserving square 31 x 31 map panels.
    figure = plt.figure(figsize=(18.5, 10.4), facecolor="white")
    grid = figure.add_gridspec(
        3,
        8,
        width_ratios=(1.0, 0.065, 1.0, 0.065, 1.0, 0.065, 1.0, 0.065),
        left=0.068,
        right=0.987,
        bottom=0.070,
        top=0.947,
        wspace=0.28,
        hspace=0.31,
    )
    axes = np.empty((3, 4), dtype=object)
    images = {}
    for row, horizon in enumerate(HORIZONS):
        for column, metric in enumerate(METRICS):
            axis = figure.add_subplot(grid[row, 2 * column])
            axes[row, column] = axis
            cmap = "RdBu_r" if "signed" in metric else "viridis"
            image = axis.imshow(
                all_maps[horizon][metric],
                origin="upper",
                cmap=cmap,
                norm=norms[metric],
                interpolation="nearest",
                aspect="equal",
            )
            images[metric] = image
            if row == 0:
                axis.set_title(TITLES[metric], pad=9, fontsize=11, fontweight="semibold")
            axis.set_xticks((0, 15, 30))
            axis.set_yticks((0, 15, 30))
            axis.set_xlabel("Grid x index")
            axis.set_ylabel("Grid y index")
            axis.tick_params(length=3, width=0.7)
            for spine in axis.spines.values():
                spine.set_linewidth(0.7)

        axes[row, 0].annotate(
            f"{horizon} h",
            xy=(-0.37, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            rotation=90,
            fontsize=12,
            fontweight="bold",
        )

    for column, metric in enumerate(METRICS):
        colorbar = figure.colorbar(
            images[metric],
            cax=figure.add_subplot(grid[:, 2 * column + 1]),
        )
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.yaxis.set_label_position("left")
        colorbar.set_label(COLORBARS[metric], rotation=90, labelpad=10)
        colorbar.ax.tick_params(labelsize=8.5, length=3, width=0.7)

    output_paths = []
    for extension, save_kwargs in {"png": {"dpi": 800}, "svg": {}, "pdf": {}}.items():
        output_path = output_dir / f"Figure4_spatial_error_maps.{extension}"
        figure.savefig(output_path, bbox_inches="tight", facecolor="white", **save_kwargs)
        output_paths.append(output_path)
    plt.close(figure)
    for horizon in HORIZONS:
        actual = domain_means(all_maps[horizon])
        expected = EXPECTED_DOMAIN_MEANS[horizon][:4]
        matches_target = np.allclose(actual, expected, rtol=0.0, atol=5e-5)
        print(f"{horizon} h source: {source_paths[horizon]}")
        print(
            f"{horizon} h self-check (u-bias / v-bias / component MAE / WS-MAE): "
            f"{actual[0]:.4f} / {actual[1]:.4f} / {actual[2]:.4f} / {actual[3]:.4f}; "
            f"target match: {'yes' if matches_target else 'no'}"
        )
    print("Formal domain-mean checks passed for 6 h, 12 h, and 24 h.")
    print("Saved paths:", *(path.resolve() for path in output_paths), sep="\n")


if __name__ == "__main__":
    main()
