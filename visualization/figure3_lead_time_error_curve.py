"""Draw Figure 2: lead-time-dependent errors on the independent 2023 test set.

Figure caption (not rendered in the figure):
Figure 2. Lead-time-dependent changes in RMSE, MAE, and WS-MAE on the
independent 2023 test set. Error bars denote ±1 sample standard deviation
across three random seeds.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LEAD_TIMES = np.array([6, 12, 24])
METRICS = {
    "RMSE": {
        "mean": np.array([3.1126, 4.0568, 5.3105]),
        "std": np.array([0.0299, 0.0979, 0.1756]),
        "color": "#1f77b4",
        "marker": "o",
    },
    "MAE": {
        "mean": np.array([2.2412, 2.8681, 3.8030]),
        "std": np.array([0.0109, 0.0688, 0.2000]),
        "color": "#ff7f0e",
        "marker": "s",
    },
    "WS-MAE": {
        "mean": np.array([2.0676, 2.6306, 3.5797]),
        "std": np.array([0.0126, 0.0922, 0.4543]),
        "color": "#2ca02c",
        "marker": "^",
    },
}


def validate_data() -> None:
    """Check that the manuscript summary data have the expected shape."""
    assert tuple(METRICS) == ("RMSE", "MAE", "WS-MAE")
    assert np.array_equal(LEAD_TIMES, np.array([6, 12, 24]))
    for values in METRICS.values():
        assert values["mean"].shape == LEAD_TIMES.shape
        assert values["std"].shape == LEAD_TIMES.shape
        assert np.all(values["std"] >= 0)


def main() -> None:
    validate_data()

    root = Path(__file__).resolve().parents[1]
    output_dir = root / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    figure, axis = plt.subplots(figsize=(6.6, 4.5), facecolor="white")
    axis.set_facecolor("white")

    for metric, values in METRICS.items():
        axis.errorbar(
            LEAD_TIMES,
            values["mean"],
            yerr=values["std"],
            label=metric,
            color=values["color"],
            marker=values["marker"],
            markersize=6.5,
            linewidth=1.8,
            elinewidth=1.15,
            capsize=4,
            capthick=1.15,
            markeredgewidth=0.7,
            markeredgecolor="white",
            zorder=3,
        )

    axis.set_xlabel("Lead time (h)")
    axis.set_ylabel(r"Error (m$\cdot$s$^{-1}$)")
    axis.set_xticks(LEAD_TIMES)
    axis.set_xlim(4.5, 25.5)
    axis.set_ylim(1.5, 6.5)
    axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#BFBFBF")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(width=0.8, length=3.5)

    figure.tight_layout(pad=0.7)
    output_paths = {
        "png": output_dir / "Figure2_lead_time_error_curve.png",
        "svg": output_dir / "Figure2_lead_time_error_curve.svg",
        "pdf": output_dir / "Figure2_lead_time_error_curve.pdf",
    }
    figure.savefig(output_paths["png"], dpi=800, facecolor="white", bbox_inches="tight")
    figure.savefig(output_paths["svg"], facecolor="white", bbox_inches="tight")
    figure.savefig(output_paths["pdf"], facecolor="white", bbox_inches="tight")
    plt.close(figure)

    for output_path in output_paths.values():
        print(output_path.resolve())


if __name__ == "__main__":
    main()
