"""Figure style for the 4-page ISBI manuscript.

Adapted from scripts/original_copies/figstyle.py (the for_ISBI journal figure standard): same Okabe-Ito
colours in fixed roles, recessive axes and y-grid. Changes for the manuscript: Times-compatible serif
(Nimbus Roman + STIX math) to match the spconf body text, sizes set to the printed column width
(single column 3.39 in, double 7.0 in) so the PDF is placed at 100 % and 7 pt is the smallest text.
Colour roles (identical in the TikZ method figure, see figures/fig_method.tex):
  ADAPTER  vermillion  reconstruction needs the interpolation adapter (SAINT, I3Net)
  NATIVE   blue        reconstruction takes acquired slices at their true positions (linear, ArSSR)
  marker   circle = learned (MC-dropout + scale head), square = post-hoc error-scale model
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, VERM, GREEN, PURPLE, ORANGE = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"
ADAPTER, NATIVE = VERM, BLUE
INK, INK2, MUTED, GRID = "#1a1a1a", "#4b5563", "#6b7280", "#d9dde3"
SINGLE, DOUBLE = 3.39, 7.0

_RC = {
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "x",
    "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 2.5, "ytick.major.size": 0, "xtick.major.width": 0.6,
    "lines.linewidth": 1.0, "lines.markersize": 4,
    "legend.frameon": False, "legend.handlelength": 1.2, "legend.columnspacing": 0.9,
    "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
    "pdf.fonttype": 42,
    "text.color": INK, "axes.labelcolor": INK,
}


def use():
    plt.rcParams.update(_RC)


def save(fig, stem):
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    print("wrote", stem)
