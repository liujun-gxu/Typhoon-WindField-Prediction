"""Draw the two-panel main-text version of Figure 5: corrected fixed-t0 target definition.

The layout deliberately uses the rounded pastel modules, restrained arrows,
and sans-serif workflow-diagram language used for the manuscript Figure 1.
All positions are schematic and have no geographical meaning.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


INK = "#222222"
EDGE = "#5B5B5B"
ARROW = "#5D6268"
BLUE = "#B9D8F1"
BLUE_LIGHT = "#E2F0FB"
BLUE_DARK = "#4F87B4"
GREEN = "#CBE6CE"
GREEN_LIGHT = "#E9F5EA"
GREEN_DARK = "#4A936A"
YELLOW = "#FFF0BD"


def rounded_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    facecolor: str,
    *,
    edgecolor: str = EDGE,
    linewidth: float = 1.15,
    radius: float = 0.025,
    zorder: int = 1,
) -> FancyBboxPatch:
    """Add a Figure-1-style rounded workflow module."""
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    axis.add_patch(box)
    return box


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, dashed: bool = False, color: str = ARROW) -> None:
    """Add a restrained directional flow arrow."""
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10.5,
            linewidth=1.25,
            linestyle="--" if dashed else "-",
            color=color,
            shrinkA=2,
            shrinkB=3,
            zorder=5,
        )
    )


def setup_panel(axis: plt.Axes, label: str, title: str, fill: str, border: str) -> None:
    """Set up an equal-width rounded outer panel."""
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()
    rounded_box(axis, (0.02, 0.025), 0.96, 0.95, fill, edgecolor=border, linewidth=1.25, radius=0.03, zorder=0)
    axis.text(0.065, 0.925, label, ha="left", va="center", fontsize=11.7, fontweight="bold", color=INK)
    axis.text(0.135, 0.925, title, ha="left", va="center", fontsize=11.7, fontweight="bold", color=INK)
    axis.plot((0.06, 0.94), (0.878, 0.878), color=border, linewidth=0.9, alpha=0.75, zorder=1)


def patch_grid(
    axis: plt.Axes,
    center: tuple[float, float],
    size: float,
    color: str,
    *,
    label: str | None = "31 × 31",
    alpha: float = 0.20,
    label_offset: float = 0.024,
) -> None:
    """Draw a compact regular-grid patch module (a 31×31 schematic)."""
    x, y = center
    x0, y0 = x - size / 2, y - size / 2
    axis.add_patch(Rectangle((x0, y0), size, size, facecolor="white", edgecolor=color, linewidth=1.0, alpha=alpha, zorder=3))
    for fraction in (0.2, 0.4, 0.6, 0.8):
        axis.plot((x0 + fraction * size, x0 + fraction * size), (y0, y0 + size), color=color, linewidth=0.45, alpha=0.75, zorder=4)
        axis.plot((x0, x0 + size), (y0 + fraction * size, y0 + fraction * size), color=color, linewidth=0.45, alpha=0.75, zorder=4)
    axis.add_patch(Circle((x, y), 0.0085, facecolor=color, edgecolor="white", linewidth=0.45, zorder=6))
    if label:
        axis.text(x, y0 - label_offset, label, ha="center", va="top", fontsize=7.6, color=color, zorder=7)


def center_marker(axis: plt.Axes, xy: tuple[float, float], color: str, label: str, *, weak: bool = False) -> None:
    """Draw a storm center as a small workflow-diagram node."""
    x, y = xy
    radius = 0.012 if not weak else 0.010
    axis.add_patch(Circle((x, y), radius, facecolor=color, edgecolor="white", linewidth=0.55, alpha=0.72 if weak else 1.0, zorder=7))
    axis.text(x, y + 0.031, label, ha="center", va="bottom", fontsize=8.2, color=ARROW if weak else INK, zorder=8)


def historical_panel(axis: plt.Axes) -> None:
    setup_panel(axis, "(a)", "Historical input sequence", BLUE_LIGHT, BLUE_DARK)
    axis.text(0.50, 0.830, "Historical storm trajectory", ha="center", va="center", fontsize=8.7, color=ARROW, fontstyle="italic")
    positions = ((0.15, 0.715), (0.37, 0.670), (0.59, 0.625), (0.81, 0.580))
    time_labels = (r"$t_0-3$ h", r"$t_0-2$ h", r"$t_0-1$ h", r"$t_0$")
    center_labels = (r"$C(t_0-3\,\mathrm{h})$", r"$C(t_0-2\,\mathrm{h})$", r"$C(t_0-1\,\mathrm{h})$", r"$C(t_0)$")
    for start, end in zip(positions[:-1], positions[1:]):
        arrow(axis, start, end, color=BLUE_DARK)

    for (x, y), time_label, center_label in zip(positions, time_labels, center_labels):
        rounded_box(axis, (x - 0.092, 0.395), 0.184, 0.168, "#D6E9F8", edgecolor=BLUE_DARK, linewidth=0.9, radius=0.018, zorder=2)
        axis.text(x, 0.544, time_label, ha="center", va="center", fontsize=8.3, fontweight="bold", color=INK, zorder=7)
        patch_grid(axis, (x, 0.468), 0.105, BLUE_DARK, label=None)
        # Fixed caption area: grid -> whitespace -> patch type -> resolution.
        axis.text(x, 0.375, "input patch", ha="center", va="top", fontsize=7.4, color=INK, zorder=7)
        axis.text(x, 0.345, "31 × 31", ha="center", va="top", fontsize=7.4, color=BLUE_DARK, zorder=7)
        center_marker(axis, (x, y), BLUE_DARK, center_label)
        axis.plot((x, x), (y - 0.014, 0.522), color=BLUE_DARK, linewidth=0.78, linestyle=(0, (3, 2)), zorder=3)

    rounded_box(axis, (0.10, 0.205), 0.80, 0.105, "#EDF6FD", edgecolor="#9CBFE0", linewidth=0.8, radius=0.018, zorder=2)
    axis.text(
        0.50,
        0.257,
        "Historical inputs follow the available storm center at each time.",
        ha="center",
        va="center",
        fontsize=8.6,
        color=INK,
        zorder=7,
    )
    axis.text(0.50, 0.123, r"$X=\{W(t,C(t))\}$", ha="center", va="center", fontsize=10.3, color=INK)


def correct_target_panel(axis: plt.Axes) -> None:
    setup_panel(axis, "(b)", r"Fixed-$t_0$ target", GREEN_LIGHT, GREEN_DARK)
    rounded_box(axis, (0.10, 0.535), 0.80, 0.205, YELLOW, edgecolor="#9C8A58", linewidth=0.95, radius=0.022, zorder=2)
    axis.text(0.50, 0.706, r"Future wind field at $t_0+H$", ha="center", va="center", fontsize=10.0, fontweight="bold", color=INK)
    axis.text(0.50, 0.672, "ERA5 values at the forecast valid time", ha="center", va="center", fontsize=7.7, color=ARROW)

    initial = (0.27, 0.590)
    future = (0.74, 0.605)
    center_marker(axis, initial, GREEN_DARK, r"$C(t_0)$")
    center_marker(axis, future, "#B58B55", r"$C(t_0+H)$", weak=True)
    arrow(axis, initial, future, dashed=True, color="#8A8F94")
    axis.text(0.74, 0.550, "storm moved", ha="center", va="center", fontsize=7.2, color="#8A8F94")

    target_center = (initial[0], 0.390)
    rounded_box(axis, (0.115, 0.300), 0.31, 0.155, "#D8EED9", edgecolor=GREEN_DARK, linewidth=1.0, radius=0.018, zorder=2)
    patch_grid(axis, target_center, 0.130, GREEN_DARK, label=None)
    # Fixed caption area separated from both the patch module and formula box.
    axis.text(target_center[0], 0.282, "target patch", ha="center", va="top", fontsize=7.6, color=GREEN_DARK, zorder=7)
    axis.text(target_center[0], 0.252, "31 × 31", ha="center", va="top", fontsize=7.6, color=GREEN_DARK, zorder=7)
    axis.plot((initial[0], initial[0]), (initial[1] - 0.014, 0.455), color=GREEN_DARK, linewidth=0.95, linestyle=(0, (3, 2)), zorder=4)
    axis.text(0.50, 0.432, r"target window anchored at $C(t_0)$", ha="left", va="center", fontsize=8.3, color=GREEN_DARK)
    arrow(axis, (0.44, 0.425), (0.355, 0.405), color=GREEN_DARK)

    rounded_box(axis, (0.17, 0.115), 0.66, 0.085, "#D7ECD9", edgecolor=GREEN_DARK, linewidth=0.85, radius=0.018, zorder=2)
    axis.text(0.50, 0.157, r"$Y_H=W(t_0+H,C(t_0))$", ha="center", va="center", fontsize=12.0, color=INK)
    axis.text(
        0.50,
        0.061,
        r"Future wind field is used; the target window remains anchored at $C(t_0)$.",
        ha="center",
        va="center",
        fontsize=8.3,
        color=INK,
    )



def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Calibri", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    # Main-text layout: retain the original (a)/(b) panels, enlarge them, and
    # omit the former future-center (c) panel without leaving empty space.
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 6.85), facecolor="white", gridspec_kw={"wspace": 0.07})
    historical_panel(axes[0])
    correct_target_panel(axes[1])

    output_paths = {
        "png": output_dir / "Figure5_fixed_t0_target_definition_maintext.png",
        "svg": output_dir / "Figure5_fixed_t0_target_definition_maintext.svg",
        "pdf": output_dir / "Figure5_fixed_t0_target_definition_maintext.pdf",
    }
    figure.savefig(output_paths["png"], dpi=800, bbox_inches="tight", facecolor="white")
    figure.savefig(output_paths["svg"], bbox_inches="tight", facecolor="white")
    figure.savefig(output_paths["pdf"], bbox_inches="tight", facecolor="white")
    plt.close(figure)
    for output_path in output_paths.values():
        print(output_path.resolve())


if __name__ == "__main__":
    main()
