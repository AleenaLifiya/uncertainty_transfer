"""Figures for the geometry story (reads results/geometry_story/*.csv only).

fig_geo_main     clustered 90 % coverage of every arm (x4, x8), grouped by access to gap geometry
fig_geo_2x2      SAINT 2x2: coverage change uniform->clustered for {geometry no/yes} x {irregular training no/yes}
fig_geo_patterns post-hoc (gap-conditional) coverage by pattern: native vs adapter reconstructions
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from isbi import common as C  # noqa: E402
from isbi import figstyle as FS  # noqa: E402

FS.use()
S = pd.read_csv(os.path.join(C.ISBI, "results", "geometry_story", "arms_summary.csv"))
PP = pd.read_csv(os.path.join(C.ISBI, "results", "geometry_story", "arms_patient.csv"))
FIG = os.path.join(C.ISBI, "figures")
NOGEO = ["SAINT-P learned", "I3Net-P learned", "SAINT exposure learned", "I3Net exposure learned",
         "SAINT-P constant width", "SAINT-P post-hoc, no geometry", "I3Net-P post-hoc, no geometry",
         "linear post-hoc, no geometry"]
GEO = ["ArSSR-P learned", "SAINT-PG learned", "SAINT-PGX learned", "SAINT-P post-hoc", "I3Net-P post-hoc",
       "SAINT-P post-hoc, gap-conditional", "I3Net-P post-hoc, gap-conditional",
       "ArSSR-P post-hoc, gap-conditional", "linear post-hoc, gap-conditional"]
COL = {"adapter": FS.VERM, "native": FS.BLUE}
MK = {"learned": "o", "post-hoc": "s", "post-hoc (gap-cond.)": "D", "constant": "^"}
LAB = {"SAINT-PG learned": "SAINT-PG learned (geometry head)", "SAINT-PGX learned": "SAINT-PGX learned (+ irregular training)",
       "ArSSR-P learned": "ArSSR-P learned (position-in-gap only)"}

# ------------------------------------------------------------------ main figure
fig, axs = plt.subplots(1, 2, figsize=(FS.DOUBLE, 3.9), sharey=True)
order = NOGEO + GEO
ys = {a: i + 1.0 + (1.6 if a in GEO else 0) for i, a in enumerate(order)}
for ax, R in zip(axs, C.STRIDES):
    t = S[S.R == R].set_index("arm")
    for a in order:
        r = t.loc[a]
        y = ys[a]
        ax.errorbar(r.cov90_clustered, y, xerr=[[r.cov90_clustered - r.clustered_lo], [r.clustered_hi - r.cov90_clustered]],
                    color=COL[r.input], marker=MK[r.uq], ms=4.5, capsize=1.8, lw=1.1, ls="none")
        ax.plot(r.cov90_uniform, y, marker="|", color=FS.MUTED, ms=7, mew=1.2)
    ax.axvline(0.9, color=FS.INK2, lw=0.8, ls=(0, (4, 2)))
    ax.axhline(len(NOGEO) + 1.3, color=FS.GRID, lw=1.0)
    ax.set_xlim(0.5, 0.97)
    ax.set_xlabel(f"90% coverage on clustered sampling (×{R})")
    ax.grid(axis="x", color=FS.GRID)
    ax.set_title(f"×{R}" + ("  (primary)" if R == 4 else "  (confirmation)"), loc="left")
axs[0].set_yticks([ys[a] for a in order])
axs[0].set_yticklabels([LAB.get(a, a) for a in order], fontsize=6.6)
axs[0].invert_yaxis()
for ax in axs:
    ax.text(0.505, 0.1, "no gap geometry", fontsize=7, color=FS.INK2, fontweight="bold", va="center")
    ax.text(0.505, len(NOGEO) + 1.75, "uses gap geometry", fontsize=7, color=FS.INK2, fontweight="bold", va="center")
from matplotlib.lines import Line2D  # noqa: E402
h = [Line2D([], [], color=FS.VERM, marker="o", ls="none", label="adapter-based (SAINT, I3Net)"),
     Line2D([], [], color=FS.BLUE, marker="o", ls="none", label="native irregular input (ArSSR, linear)"),
     Line2D([], [], color=FS.MUTED, marker="|", ls="none", ms=8, label="same arm on uniform")]
h += [Line2D([], [], color=FS.INK2, marker=v, ls="none", label=k) for k, v in MK.items()]
fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.55, -0.02))
fig.tight_layout(rect=(0, 0.09, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig_geo_main.{ext}"))
plt.close(fig)

# ------------------------------------------------------------------ 2x2
cells = [("SAINT-P learned", "neither"), ("SAINT exposure learned", "irregular training"),
         ("SAINT-PG learned", "geometry"), ("SAINT-PGX learned", "both")]
fig, ax = plt.subplots(figsize=(FS.SINGLE * 1.05, 2.3))
for j, R in enumerate(C.STRIDES):
    t = S[S.R == R].set_index("arm")
    for i, (a, lab) in enumerate(cells):
        r = t.loc[a]
        x = i + (j - 0.5) * 0.3
        ax.bar(x, r.dcov90, width=0.28, color=[FS.SKY, FS.BLUE][j], label=f"×{R}" if i == 0 else None)
        ax.errorbar(x, r.dcov90, yerr=[[r.dcov90 - r.dcov90_lo], [r.dcov90_hi - r.dcov90]], color=FS.INK, capsize=1.5, lw=0.8)
ax.axhline(0, color=FS.INK2, lw=0.8)
ax.set_xticks(range(4))
ax.set_xticklabels(["neither", "irregular\ntraining", "geometry", "both"], fontsize=7)
ax.set_ylabel("change in 90% coverage\n(clustered − uniform)")
ax.set_title("SAINT, same mean", loc="left", fontsize=8)
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig_geo_2x2.{ext}"))
plt.close(fig)

# ------------------------------------------------------------------ patterns: native vs adapter
arms = [("linear post-hoc, gap-conditional", "linear", FS.BLUE, "o"), ("ArSSR-P post-hoc, gap-conditional", "ArSSR-P", FS.SKY, "s"),
        ("I3Net-P post-hoc, gap-conditional", "I3Net-P", FS.ORANGE, "^"), ("SAINT-P post-hoc, gap-conditional", "SAINT-P", FS.VERM, "D")]
fig, axs = plt.subplots(1, 2, figsize=(FS.DOUBLE * 0.7, 2.3), sharey=True)
pats = ["uniform", "jittered", "clustered"]
for ax, R in zip(axs, C.STRIDES):
    for k, (a, lab, col, mk) in enumerate(arms):
        g = PP[(PP.arm == a) & (PP.R == R)]
        m = g.groupby("pattern").cov90.mean().reindex(pats)
        se = g.groupby("pattern").cov90.std().reindex(pats) / np.sqrt(g.patient.nunique())
        xs = np.arange(3) + (k - 1.5) * 0.06
        ax.errorbar(xs, m, yerr=1.96 * se, color=col, marker=mk, ms=4, capsize=1.5, lw=1.2, label=lab)
    ax.axhline(0.9, color=FS.INK2, lw=0.8, ls=(0, (4, 2)))
    ax.set_xticks(range(3)); ax.set_xticklabels(pats)
    ax.set_title(f"post-hoc, gap-conditional  ×{R}", loc="left", fontsize=8)
axs[0].set_ylabel("90% coverage")
axs[1].legend(fontsize=6.5, loc="lower left")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig_geo_patterns.{ext}"))
plt.close(fig)
print("ok")

# ------------------------------------------------------------------ slide version of the main figure (x4, fewer rows, larger type)
SL = [("SAINT-P learned", "SAINT-P learned"), ("I3Net-P learned", "I3Net-P learned"),
      ("SAINT exposure learned", "SAINT + irregular training"), ("I3Net exposure learned", "I3Net + irregular training"),
      ("SAINT-P post-hoc, no geometry", "post-hoc, no geometry (SAINT)"), (None, None),
      ("ArSSR-P learned", "ArSSR-P learned (native)"), ("SAINT-PG learned", "SAINT-PG (geometry head)"),
      ("SAINT-PGX learned", "SAINT-PGX (+ irregular training)"), ("SAINT-P post-hoc", "post-hoc (SAINT)"),
      ("I3Net-P post-hoc", "post-hoc (I3Net)"), ("ArSSR-P post-hoc, gap-conditional", "post-hoc (ArSSR), gap-cond."),
      ("linear post-hoc, gap-conditional", "post-hoc (linear), gap-cond.")]
fig, ax = plt.subplots(figsize=(4.4, 3.5))
t = S[S.R == 4].set_index("arm")
yt, yl = [], []
for i, (a, lab) in enumerate(SL):
    if a is None:
        ax.axhline(i, color=FS.GRID, lw=1.0)
        continue
    r = t.loc[a]
    ax.errorbar(r.cov90_clustered, i, xerr=[[r.cov90_clustered - r.clustered_lo], [r.clustered_hi - r.cov90_clustered]],
                color=COL[r.input], marker=MK[r.uq], ms=5.5, capsize=2, lw=1.3, ls="none")
    ax.plot(r.cov90_uniform, i, marker="|", color=FS.MUTED, ms=9, mew=1.4)
    yt.append(i); yl.append(lab)
ax.axvline(0.9, color=FS.INK2, lw=0.9, ls=(0, (4, 2)))
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0.5, 0.95)
ax.set_xlabel("90% coverage, clustered, ×4", fontsize=8)
ax.grid(axis="x", color=FS.GRID)
ax.text(0.505, -0.75, "no gap geometry", fontsize=8, fontweight="bold", color=FS.INK2)
ax.text(0.505, 5.0, "with gap geometry", fontsize=8, fontweight="bold", color=FS.INK2, va="center")
h = [Line2D([], [], color=FS.VERM, marker="o", ls="none", label="adapter-based"),
     Line2D([], [], color=FS.BLUE, marker="o", ls="none", label="native input"),
     Line2D([], [], color=FS.MUTED, marker="|", ls="none", ms=9, label="uniform")]
ax.legend(handles=h, fontsize=7, loc="lower left", frameon=True, framealpha=0.9)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig_geo_slide.{ext}"))
plt.close(fig)
