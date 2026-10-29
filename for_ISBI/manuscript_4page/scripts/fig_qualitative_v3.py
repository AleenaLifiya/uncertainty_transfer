"""Fig. 2 (v3): qualitative comparison for the matched setup, clustered x4, protocol-selected median TEST patient, coronal.
Rows: SAINT-P (adapter), ArSSR-P (native, uniform-trained), linear. Columns: reconstruction, |error|, learned 90 % half-width,
post-hoc 90 % half-width without geometry, with geometry. |error| and half-widths share one colour scale (a voxel is
covered when |error| <= half-width). Input and ground truth in the first column. Acquired slices: dark grey lines in the
maps, red ticks beside the input. Reads exp_uniform_arssr/results/qual/planes_clustered.npz (qual_v3.py).
Also writes the thumbnails used by Fig. 1 (SAINT-P, clustered).
"""

from project_paths import root_path
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import msstyle as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402

ISBI = root_path('for_ISBI')
FIG = os.path.join(ISBI, "manuscript_4page", "figures")
PAPER_FIG = os.path.join(ISBI, "ISBI_paper", "figures")
TH = os.path.join(FIG, "thumbs")
os.makedirs(TH, exist_ok=True)
S.use()
z = dict(np.load(os.path.join(ISBI, "exp_uniform_arssr", "results", "qual", "planes_clustered.npz")))
VMAX = 0.20
CMAP = plt.get_cmap("magma").copy()
CMAP.set_bad("black")
GREY = plt.get_cmap("gray")
fg, acq = z["fg"].astype(bool), z["acq"].astype(bool)
h, w = z["gt"].shape
MODELS = [("SAINT_P", "SAINT-P\n(adapter)"), ("ArSSR_P", "ArSSR-P\n(native)"), ("linear", "Linear\n(native)")]
COLS = [("mean", "Reconstruction $\\bar y$"), ("err", "$|y-\\bar y|$"), ("learned", "Learned"),
        ("ph_nogeo", "Post-hoc, no geometry"), ("ph_geo", "Post-hoc + geometry")]


def mmap(a):
    a = np.array(a, float)
    a[~fg] = np.nan
    a[acq & fg] = np.nan
    return a


def overlay(ax):
    ov = np.zeros(fg.shape + (4,))
    ov[acq & fg] = (0.33, 0.34, 0.37, 1.0)
    ax.imshow(ov, interpolation="nearest")


def clean(ax):
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)


fig_w = 6.3
lm, gap_c0, rg = 0.06, 0.40, 0.10                        # left margin, gap after column 0 (row labels), row gap
pw = (fig_w - lm - gap_c0 - 0.05) / 6
ph = pw * h / w
top, bot = 0.19, 0.34
fig_h = top + 3 * ph + 2 * rg + bot
fig = plt.figure(figsize=(fig_w, fig_h))


def ax_at(r, c):
    x0 = lm + c * pw + (gap_c0 if c > 0 else 0)
    y0 = fig_h - top - (r + 1) * ph - r * rg
    return fig.add_axes([x0 / fig_w, y0 / fig_h, (pw - 0.035) / fig_w, (ph - 0.035) / fig_h])


# column 0: input and ground truth
ax = ax_at(0, 0)
ax.imshow(z["sparse"], cmap=GREY, vmin=0, vmax=1, interpolation="nearest")
rows_acq = np.where(acq.any(1))[0]
ax.plot(np.full(len(rows_acq), -3.0), rows_acq, ls="none", marker="_", ms=3.0, mew=0.9, color="#C0392B", clip_on=False)
clean(ax)
ax.set_title("Input $x$", fontsize=7.5, pad=2.5)
ax = ax_at(1, 0)
ax.imshow(z["gt"], cmap=GREY, vmin=0, vmax=1, interpolation="nearest")
clean(ax)
ax.set_title("Ground truth $y$", fontsize=7.5, pad=2.5)
for r, (m, lab) in enumerate(MODELS):
    for c, (key, title) in enumerate(COLS, start=1):
        ax = ax_at(r, c)
        k = f"{m}__{key}"
        if key == "mean":
            ax.imshow(z[k], cmap=GREY, vmin=0, vmax=1, interpolation="nearest")
        elif k in z:
            ax.imshow(mmap(z[k]), cmap=CMAP, vmin=0, vmax=VMAX, interpolation="nearest")
            overlay(ax)
        else:
            ax.set_facecolor("white")
            ax.text(0.5, 0.5, "no learned\nuncertainty\n(no network)", transform=ax.transAxes, ha="center", va="center", fontsize=7, color=S.MUTED, linespacing=1.3)
            for s_ in ax.spines.values():
                s_.set_visible(True); s_.set_color(S.GRID)
        clean(ax) if k in z or key == "mean" else (ax.set_xticks([]), ax.set_yticks([]))
        if r == 0:
            ax.set_title(title, fontsize=7.5, pad=2.5)
        if c == 1:
            ax.text(-0.06, 0.5, f"{lab}\n{float(z[m + '__psnr']):.1f} dB", transform=ax.transAxes, rotation=90,
                    ha="right", va="center", fontsize=7.2, linespacing=1.15)
cx0 = lm + gap_c0 + 2 * pw
cax = fig.add_axes([cx0 / fig_w, 0.17 / fig_h, (4 * pw - 0.035) / fig_w, 0.065 / fig_h])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, VMAX), cmap=CMAP), cax=cax, orientation="horizontal")
cb.set_ticks([0, 0.05, 0.10, 0.15, 0.20])
cb.ax.tick_params(labelsize=6.8, length=2, pad=1)
cb.outline.set_linewidth(0.4)
fig.text((cx0 - 0.06) / fig_w, 0.20 / fig_h, "$|y-\\bar y|$, 90% half-width", ha="right", va="center", fontsize=7)
plt.rcParams["savefig.bbox"] = "standard"
fig.savefig(os.path.join(FIG, "fig_qualitative.pdf"), bbox_inches=None)
fig.savefig(os.path.join(PAPER_FIG, "fig_qualitative.pdf"), bbox_inches=None)
fig.savefig(os.path.join(FIG, "fig_qualitative.png"), dpi=300, bbox_inches=None)
plt.close(fig)
print("fig_qualitative", fig_w, round(fig_h, 2))

# thumbnails for Fig. 1 (SAINT-P, clustered)
plt.imsave(os.path.join(TH, "input.png"), z["sparse"], cmap=GREY, vmin=0, vmax=1)
plt.imsave(os.path.join(TH, "mean.png"), z["SAINT_P__mean"], cmap=GREY, vmin=0, vmax=1)
plt.imsave(os.path.join(TH, "learned.png"), np.ma.masked_invalid(mmap(z["SAINT_P__learned"])), cmap=CMAP, vmin=0, vmax=VMAX)
plt.imsave(os.path.join(TH, "posthoc.png"), np.ma.masked_invalid(mmap(z["SAINT_P__ph_geo"])), cmap=CMAP, vmin=0, vmax=VMAX)
print("thumbs")
