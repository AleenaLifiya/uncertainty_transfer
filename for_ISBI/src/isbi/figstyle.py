# PROVENANCE: copied verbatim from IR3C_Brats_XAI_MI/scripts/figstyle.py (sha256 347a6d3eb02f5cce…) on 2026-10-16 for for_ISBI; project journal figure standard.
#!/usr/bin/env python
"""Shared figure style for IR3C_Brats_XAI_MI — journal quality, applied by construction.

Decisions here are the ones the dataviz method fixes BEFORE any chart code:

  form      each figure states the job it does in its own docstring;
  colour    Okabe-Ito, assigned in FIXED order and never cycled. Computed, not eyeballed:
            blue/vermillion separate at dE 22.3 (protan) and 28.0 (deutan), far above the
            8.0 CVD target. green/purple is the weak pair at 7.9 so they are never placed
            adjacent, and any figure needing >=3 series adds a second encoding (marker or
            hatch) so identity is never colour-alone;
  marks     2px lines, >=5pt markers, thin bars, recessive axes (no top/right spine),
            a light y-grid only;
  text      serif to match the LaTeX body; text wears ink colours, never the series colour;
  size      SINGLE = 3.4in and DOUBLE = 7.0in wide, so the figure is laid out at the size it
            will be printed rather than scaled afterwards.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Okabe-Ito, fixed order -------------------------------------------------------
BLUE, VERM, GREEN, PURPLE, ORANGE, SKY, YELLOW = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442")
SERIES = [BLUE, VERM, GREEN, PURPLE, ORANGE]          # never cycled past this
MARKERS = ["o", "s", "^", "D", "v"]                   # secondary encoding
INK, INK2, MUTED, GRID = "#1a1a1a", "#4b5563", "#6b7280", "#d9dde3"
SINGLE, DOUBLE = 3.4, 7.0                             # inches

_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 1.0,
    "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.columnspacing": 1.1,
    "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "text.color": INK, "axes.labelcolor": INK,
}


def use():
    plt.rcParams.update(_RC)


def panel(ax, letter, dx=-0.16, dy=1.04):
    """Panel label in ink, outside the axes, never over the data."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left", color=INK)


def finish(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", color=INK)
    return ax


def titleblock(fig, title, subtitle=None, axes_top=0.80, pad_pt=4.0):
    """Title and subtitle in a reserved band, wrapped to the figure width.

    Two failures this guards against, both seen in this project:
      * offsetting the subtitle by a FRACTION of figure height, which collapses on a short
        wide figure -- the gap is measured in POINTS instead;
      * leaving the subtitle as one long line, which under a tight bounding box stretches
        the whole figure sideways and leaves the axes stranded in the left half. It is
        wrapped to the figure's own width.
    """
    import textwrap
    w_in, h_in = fig.get_size_inches()
    # ~2.1 characters per point of width at 9pt serif; conservative so it never overruns
    n_title = max(28, int(w_in * 72 / (9 * 0.50)))
    n_sub = max(34, int(w_in * 72 / (7 * 0.50)))
    tl = textwrap.wrap(title, n_title) or [title]
    sl = textwrap.wrap(subtitle, n_sub) if subtitle else []

    # The band is MEASURED from the wrapped line count, not taken on the caller's word.
    # axes_top is only a ceiling: a subtitle that wrapped to two lines used to run straight
    # over the panel labels because the caller's constant still assumed one.
    need = (9 * 1.3 * len(tl) + (7 * 1.35 * len(sl) + pad_pt if sl else 0)
            + 2 * pad_pt) / (h_in * 72.0)
    fig.subplots_adjust(top=min(axes_top, 1.0 - need))

    fig.text(0.0, 1.0, "\n".join(tl), ha="left", va="top", fontsize=9, color=INK,
             transform=fig.transFigure, linespacing=1.3)
    if sl:
        dy = (9 * 1.3 * len(tl) + pad_pt) / (h_in * 72.0)
        fig.text(0.0, 1.0 - dy, "\n".join(sl), ha="left", va="top", fontsize=7,
                 color=MUTED, transform=fig.transFigure, linespacing=1.35)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("  " + path.split("/")[-1])
