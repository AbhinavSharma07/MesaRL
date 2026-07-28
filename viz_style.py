"""Shared matplotlib styling (validated categorical palette, thin recessive
chrome) so every plot in this project renders as one consistent system."""

import matplotlib

matplotlib.use("Agg")  # headless: every plot in this project is saved to a file, never shown interactively
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots (light mode), used in fixed order -- never cycled/reassigned.
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
SERIES_AQUA = "#1baf7a"
SERIES_YELLOW = "#eda100"
SERIES_VIOLET = "#4a3aa7"
SERIES_RED = "#e34948"

STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"


def new_figure(figsize=(7, 4.5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def style_axes(ax) -> None:
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color(BASELINE)
        ax.spines[spine_name].set_linewidth(1.0)

    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    ax.title.set_color(INK_PRIMARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
