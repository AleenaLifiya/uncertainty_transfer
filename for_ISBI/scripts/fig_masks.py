"""Sampling-pattern strip figure (one SEL patient's actual masks; masks never depend on images)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from isbi import common as C, figstyle as FS
FS.use()
roles = C.load_roles(); p = roles["SEL"][4]
fig, axs = plt.subplots(2, 1, figsize=(3.6, 1.7), sharex=True)
for ax, R in zip(axs, C.STRIDES):
    for i, pat in enumerate(C.PATTERNS):
        m = C.make_mask(pat, R, p, 0); st = C.mask_stats(m)
        y = 2 - i
        ax.hlines(y, 0, C.D - 1, color=FS.GRID, lw=5)
        ax.vlines(m, y - 0.32, y + 0.32, color=[FS.BLUE, FS.GREEN, FS.VERM][i], lw=0.9)
        ax.text(C.D + 2, y, f"max gap {st['max_gap_mm']:.0f} mm", va="center", fontsize=6, color=FS.INK2)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(C.PATTERNS, fontsize=6.5)
    ax.set_ylim(-0.6, 3.0); ax.grid(False); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title(f"×{R}: K = {len(C.uniform_mask(R))} acquired of {C.D} slices (1 mm)", loc="left", fontsize=7)
axs[-1].set_xlabel("slice index z", fontsize=7); axs[-1].set_xlim(-2, C.D + 40); axs[-1].set_xticks([0, 50, 100, 152])
fig.tight_layout(h_pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(C.ISBI, "figures", f"fig_masks.{ext}"))
