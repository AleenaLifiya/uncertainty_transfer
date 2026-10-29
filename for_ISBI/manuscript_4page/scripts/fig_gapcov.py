"""Fig. 3 (ISBI_paper): 90 % coverage against bracketing-gap length -- the 'metric vs shift intensity' view of
Ovadia et al. (NeurIPS 2019), with physical gap length as the shift intensity.

Rows: reconstruction (SAINT-P adapter, ArSSR-P native, linear native). Columns: BraTS x4, BraTS x8, IXI-tiny x4, IXI-tiny x8.
Lines: learned (pink), post-hoc without geometry (grey), post-hoc with geometry (blue). All TEST patterns pooled; for each
seed, per-patient coverage in the gap bin is averaged over patients (bins with >= 20 patients); line = mean over the three
fine-tuning seeds, band = min--max over seeds (linear: single run). Grey ticks: the gap lengths present in uniform FIT/CAL
data (x4 and x8 strides). Reads exp_seeds/results/<dataset>_s<seed>_<method>/gapbin_case.csv (metrics_seed.py).
"""

from project_paths import root_path
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import msstyle as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RES = root_path('for_ISBI/exp_seeds/results')
OUT = root_path('for_ISBI/ISBI_paper/figures/fig_gapcov')
S.use()
COL = {"learned": "#CC79A7", "learned_geo": "#7B2D8E", "ph_nogeo": "#8A8F98", "ph_geo": "#0072B2"}
LAB = {"learned": "learned", "learned_geo": "learned + geometry", "ph_nogeo": "post-hoc", "ph_geo": "post-hoc + geometry"}
MODELS = [("SAINT_P", "SAINT-P"), ("ArSSR", "ArSSR-P"), ("linear", "Linear")]
PANELS = [("brats", 8, "BraTS, ×8"), ("ixi", 8, "IXI-tiny, ×8")]
CALGAPS = {"brats": (4.0, 8.0), "ixi": (8.72, 17.44)}


def mid(b):
    lo, hi = (float(v) for v in b.split("-"))
    return 0.5 * (lo + hi)


rows = []
for d in ("brats", "ixi"):
    for m in ("SAINT_P", "ArSSR", "linear", "SAINT_PG", "ArSSR_PG"):
        for s in ((42, 43, 44) if m != "linear" else (42,)):
            g = pd.read_csv(os.path.join(RES, f"{d}_s{s}_{m}", "gapbin_case.csv"))
            pp = g.groupby(["arm", "R", "gap_bin", "patient"], as_index=False).cov90.mean()
            agg = pp.groupby(["arm", "R", "gap_bin"]).agg(cov=("cov90", "mean"), n=("patient", "nunique")).reset_index()
            agg["dataset"], agg["method"], agg["seed"] = d, m, s
            rows.append(agg)
A = pd.concat(rows)
A = A[A.n >= 20]
A["x"] = A.gap_bin.map(mid)
A["method"] = A["method"].replace({"SAINT_PG": "SAINT_P", "ArSSR_PG": "ArSSR"})
fig, axs = plt.subplots(3, 2, figsize=(S.SINGLE, 3.0), sharey=True)
for r, (m, mlab) in enumerate(MODELS):
    for c, (d, R, title) in enumerate(PANELS):
        ax = axs[r, c]
        ax.axhline(0.9, color=S.INK2, lw=0.6, ls=(0, (3, 2)), zorder=0)
        for gx in CALGAPS[d]:
            ax.axvline(gx, color="#D1D5DB", lw=2.0, zorder=0)
        t = A[(A.dataset == d) & (A.method == m) & (A.R == R)]
        for arm in ("learned", "learned_geo", "ph_nogeo", "ph_geo"):
            u = t[t.arm == arm]
            if u.empty:
                continue
            q = u.groupby("x")["cov"].agg(["mean", "min", "max"]).reset_index().sort_values("x")
            ax.fill_between(q.x, q["min"], q["max"], color=COL[arm], alpha=0.40, lw=0)
            ax.plot(q.x, q["mean"], color=COL[arm], marker="o", ms=2.6, lw=1.1)
        ax.set_xscale("log")
        ax.set_ylim(0.35, 1.0)
        ax.set_yticks([0.4, 0.6, 0.8, 0.9, 1.0])
        ax.set_yticklabels(["0.4", "0.6", "0.8", "", "1.0"])
        ax.grid(axis="y", color=S.GRID, lw=0.4)
        ax.set_xticks({"brats": [3, 6, 12, 20], "ixi": [5, 10, 20, 40]}[d])
        ax.set_xticklabels([str(v) for v in {"brats": [3, 6, 12, 20], "ixi": [5, 10, 20, 40]}[d]])
        ax.minorticks_off()
        ax.tick_params(axis="both", labelsize=6.8, pad=1.5)
        if r == 0:
            ax.set_title(title, fontsize=7.5, pad=3)
        if r < 2:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel(mlab + "\ncoverage", fontsize=7.0, linespacing=1.1)
fig.supxlabel("gap length (mm, log scale)", fontsize=7.2, y=0.01)
h = [Line2D([], [], color=COL[a], marker="o", ms=3, lw=1.1, label=LAB[a]) for a in ("learned", "learned_geo", "ph_nogeo", "ph_geo")]
h += [Line2D([], [], color=S.INK2, lw=0.6, ls=(0, (3, 2)), label="nominal 0.90"),
      Line2D([], [], color="#D1D5DB", lw=2.0, label="gaps in FIT/CAL")]
fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.01), fontsize=6.5, frameon=False, columnspacing=0.8)
fig.subplots_adjust(left=0.17, right=0.985, top=0.82, bottom=0.11, hspace=0.14, wspace=0.08)
plt.rcParams["savefig.bbox"] = "standard"
fig.savefig(OUT + ".pdf", bbox_inches=None)
fig.savefig(root_path('for_ISBI/manuscript_4page/figures/fig_gapcov_preview.png'), dpi=300, bbox_inches=None)
print("wrote", OUT)
