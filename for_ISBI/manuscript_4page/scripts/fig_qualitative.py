"""Fig. 2 (qualitative) + image thumbnails for Fig. 1, from SAVED planes only (runs/qual/planes_*.npz).

The planes were produced by scripts/qualitative.py (copy in original_copies/) for the protocol-selected TEST patients
(median and 10th-percentile SAINT-P PSNR at x4 uniform; realisation r0). Maps are SAINT-P: calibrated learned 90 %
interval (mixture quantiles, lambda_0.9 from CAL) and post-hoc interval (kappa_0.9 * b_hat), both from uniform CAL.
We show the 90 % HALF-width so that it is on the same scale as |error|: a voxel is covered when |error| <= half-width
(exactly for the symmetric post-hoc interval, approximately for the learned mixture interval).
Same patient, view, colour limits and masks across columns; background and acquired slices shown in light grey.
"""

from project_paths import root_path
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import msstyle as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402

ISBI = root_path('for_ISBI')
QD = os.path.join(ISBI, "runs", "qual")
FIG = os.path.join(ISBI, "manuscript_4page", "figures")
TH = os.path.join(FIG, "thumbs")
os.makedirs(TH, exist_ok=True)
S.use()
VIEW = "coronal"
VMAX = 0.20
CMAP = plt.get_cmap("magma").copy()
CMAP.set_bad("black")
GREY = plt.get_cmap("gray").copy()


def load(pattern, tag="median"):
    z = np.load(os.path.join(QD, f"planes_{tag}_{pattern}.npz"), allow_pickle=True)
    d = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(VIEW + "__")}
    d["meta"] = json.loads(str(z["meta"]))
    return d


def masked(d, key, half=False):
    a = np.array(d[key], float)
    if half:
        a = a / 2.0
    a[~d["fg"].astype(bool)] = np.nan
    a[d["acq"].astype(bool) & d["fg"].astype(bool)] = np.nan
    return a


# ------------------------------------------------------------------ qualitative figure
pats = ["uniform", "clustered"]
cols = [("gt", "Ground truth", "img"), ("mean", "SAINT-P mean $\\bar y$", "img"), ("err", "$|y-\\bar y|$", "map"),
        ("lw", "Learned half-width", "maph"), ("pw", "Post-hoc half-width", "maph")]
D = {p: load(p) for p in pats}
h, w = D["uniform"]["gt"].shape
fig_w = S.DOUBLE
panel_w = (fig_w - 0.55) / 5
fig_h = 2 * panel_w * h / w + 0.42
fig = plt.figure(figsize=(fig_w, fig_h))
left, cbw = 0.30, 0.25
for r, p in enumerate(pats):
    d = D[p]
    for c, (key, title, kind) in enumerate(cols):
        x0 = (left + c * panel_w) / fig_w
        y0 = 1 - (0.17 + (r + 1) * panel_w * h / w) / fig_h
        ax = fig.add_axes([x0, y0, (panel_w - 0.03) / fig_w, (panel_w * h / w - 0.03) / fig_h])
        if kind == "img":
            im = np.array(d[key], float)
            if key == "mean":
                im[d["acq"].astype(bool)] = d["gt"][d["acq"].astype(bool)]
            ax.imshow(im, cmap=GREY, vmin=0, vmax=1, interpolation="nearest")
            acq_rows = np.where(d["acq"].any(1))[0]
            if key == "gt": ax.plot(np.full(len(acq_rows), -3), acq_rows, ls="none", marker="_", ms=3.2, mew=0.9,
                    color="#C0392B", clip_on=False)
        else:
            ax.imshow(masked(d, key, half=(kind == "maph")), cmap=CMAP, vmin=0, vmax=VMAX, interpolation="nearest")
            ov = np.zeros(d["fg"].shape + (4,))
            ov[d["acq"].astype(bool) & d["fg"].astype(bool)] = (0.33, 0.34, 0.37, 1.0)
            ax.imshow(ov, interpolation="nearest")
        ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(h - 0.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(False)
        if r == 0:
            ax.set_title(title, fontsize=7.5, pad=2.5)
    psnr = d["meta"]["psnr"]["SAINT_P"]
    fig.text(0.13 / fig_w, 1 - (0.17 + (r + 0.5) * panel_w * h / w) / fig_h, f"{p}\n{psnr:.1f} dB", rotation=90,
             ha="center", va="center", fontsize=7.5)
cax = fig.add_axes([(left + 2 * panel_w) / fig_w, 0.12 / fig_h, (3 * panel_w - 0.03) / fig_w, 0.06 / fig_h])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, VMAX), cmap=CMAP), cax=cax, orientation="horizontal")
cb.set_ticks([0, 0.05, 0.10, 0.15, 0.20])
cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
cb.outline.set_linewidth(0.4)
fig.text((left + 2 * panel_w - 0.05) / fig_w, 0.15 / fig_h, "$|y-\\bar y|$ and 90% half-width", ha="right", va="center", fontsize=7)
plt.rcParams["savefig.bbox"] = "standard"
fig.savefig(os.path.join(FIG, "fig_qualitative.pdf"), bbox_inches=None)
fig.savefig(os.path.join(FIG, "fig_qualitative.png"), dpi=300, bbox_inches=None)
plt.close(fig)
print("wrote fig_qualitative", fig_w, round(fig_h, 2))

# ------------------------------------------------------------------ thumbnails for the model figure (clustered case)
d = D["clustered"]
acq = d["acq"].astype(bool)
sparse = np.where(acq, d["gt"], 0.0)


def save_img(a, name, cmap, vmin, vmax):
    plt.imsave(os.path.join(TH, name + ".png"), np.ma.masked_invalid(a), cmap=cmap, vmin=vmin, vmax=vmax)


save_img(sparse, "input", GREY, 0, 1)
save_img(np.where(acq, d["gt"], d["mean"]), "mean", GREY, 0, 1)
save_img(masked(d, "lw", True), "learned", CMAP, 0, VMAX)
save_img(masked(d, "pw", True), "posthoc", CMAP, 0, VMAX)
print("wrote thumbnails")
