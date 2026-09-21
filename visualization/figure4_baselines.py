"""Draw Figure 3: comparison of STL-Net with reference baselines.

The STL-Net values below are the formal three-seed mean and *sample* standard
deviation supplied for the manuscript.  They are deliberately defined here,
rather than read from ``main_vs_baselines.csv``, whose STL-Net column is a
single seed=0 result.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LEAD_TIMES = ("6 h", "12 h", "24 h")
METHODS = ("STL-Net", "Persistence", "Climatology")

# Formal manuscript results: STL-Net has mean +/- sample standard deviation.
RESULTS = {
    "RMSE": {
        "STL-Net": (3.1126, 4.0568, 5.3105),
        "Persistence": (3.8111, 5.8011, 7.9596),
        "Climatology": (4.9896, 5.5752, 5.9995),
    },
    "MAE": {
        "STL-Net": (2.2412, 2.8681, 3.8030),
        "Persistence": (2.3590, 3.6950, 5.4433),
        "Climatology": (3.9357, 4.2890, 4.5415),
    },
    "WS-MAE": {
        "STL-Net": (2.0676, 2.6306, 3.5797),
        "Persistence": (1.6354, 2.4040, 3.4659),
        "Climatology": (4.6800, 4.9968, 5.3539),
    },
}
STL_NET_STD = {
    "RMSE": (0.0299, 0.0979, 0.1756),
    "MAE": (0.0109, 0.0688, 0.2000),
    "WS-MAE": (0.0126, 0.0922, 0.4543),
}

COLORS = {
    "STL-Net": "#1f77b4",
    "Persistence": "#ff7f0e",
    "Climatology": "#2ca02c",
}


def validate_data() -> None:
    """Guard the manuscript values and plot structure against accidental edits."""
    assert tuple(RESULTS) == ("RMSE", "MAE", "WS-MAE")
    assert sum(len(RESULTS[metric][method]) for metric in RESULTS for method in METHODS) == 27
    assert sum(len(STL_NET_STD[metric]) for metric in STL_NET_STD) == 9
    assert RESULTS["WS-MAE"]["Persistence"][2] == 3.4659
    assert RESULTS["WS-MAE"]["STL-Net"][2] == 3.5797
    assert RESULTS["WS-MAE"]["Persistence"][2] < RESULTS["WS-MAE"]["STL-Net"][2]


def main() -> None:
    validate_data()

    script_path = Path(__file__).resolve()
    output_dir = script_path.parent.parent / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    # A wider canvas leaves adequate room for four-decimal annotations while
    # retaining the three-panel layout used in the manuscript.
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.5), facecolor="white")
    x = np.arange(len(LEAD_TIMES), dtype=float)
    bar_width = 0.21
    offsets = (-bar_width - 0.035, 0.0, bar_width + 0.035)
    panel_labels = ("(a)", "(b)", "(c)")
    legend_handles = []

    for panel_index, (ax, metric) in enumerate(zip(axes, RESULTS)):
        upper_values = np.asarray(RESULTS[metric]["STL-Net"]) + np.asarray(STL_NET_STD[metric])
        all_values = np.concatenate([np.asarray(RESULTS[metric][method]) for method in METHODS])
        y_max = max(float(all_values.max()), float(upper_values.max()))
        y_limit = y_max * 1.20
        label_offset = y_limit * 0.018

        for method_index, method in enumerate(METHODS):
            values = np.asarray(RESULTS[metric][method])
            positions = x + offsets[method_index]
            kwargs = {
                "width": bar_width,
                "color": COLORS[method],
                "edgecolor": "black",
                "linewidth": 0.65,
                "zorder": 3,
                "label": "STL-Net (mean $\\pm$ std)" if method == "STL-Net" else method,
            }
            if method == "STL-Net":
                bars = ax.bar(
                    positions,
                    values,
                    yerr=STL_NET_STD[metric],
                    error_kw={"ecolor": "black", "elinewidth": 0.9, "capsize": 3, "capthick": 0.9},
                    **kwargs,
                )
            else:
                bars = ax.bar(positions, values, **kwargs)

            if panel_index == 0:
                legend_handles.append(bars)

            errors = np.asarray(STL_NET_STD[metric]) if method == "STL-Net" else np.zeros_like(values)
            for bar, value, error in zip(bars, values, errors):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + error + label_offset,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    clip_on=False,
                    zorder=4,
                )

        ax.set_xlim(-0.62, len(LEAD_TIMES) - 0.38)
        ax.set_ylim(0, y_limit)
        ax.set_xticks(x)
        ax.set_xticklabels(LEAD_TIMES)
        ax.set_xlabel("Lead time")
        ax.set_ylabel(f"{metric} (m$\\cdot$s$^{{-1}}$)")
        ax.grid(axis="y", color="#C8C8C8", linewidth=0.6, alpha=0.65, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(width=0.8, length=3.5)
        ax.text(
            0.015,
            0.975,
            f"{panel_labels[panel_index]} {metric}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
        )

    fig.legend(
        legend_handles,
        (r"STL-Net (mean $\pm$ std)", "Persistence", "Climatology"),
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
        handlelength=1.5,
        columnspacing=2.5,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.20, wspace=0.29)

    output_paths = []
    for extension, save_kwargs in {
        "png": {"dpi": 800},
        "svg": {},
        "pdf": {},
    }.items():
        output_path = output_dir / f"Figure3_baseline_comparison.{extension}"
        fig.savefig(
            output_path,
            bbox_inches="tight",
            facecolor="white",
            **save_kwargs,
        )
        output_paths.append(output_path)
    plt.close(fig)

    for output_path in output_paths:
        print(output_path.resolve())


if __name__ == "__main__":
    main()
