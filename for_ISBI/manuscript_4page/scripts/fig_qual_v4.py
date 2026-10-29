"""Fig. 2 v4 (2026-10-29): qualitative spatial allocation of uncertainty, after QUALITATIVE_FIGURE_RECOMMENDATION.md.
Protocol-selected median BraTS TEST patient (BraTS-GLI-01739; rule in PROTOCOL.md), x4 clustered r0, seed 42, coronal plane
through the brain centre of mass (fixed before inspecting methods).
Context column: uniform input, clustered input (longest gap in view bracketed), ground truth.
Rows SAINT-P / ArSSR-P / linear; columns: reconstruction, |error|, full 90 % interval WIDTH (clipped endpoints hi - lo) for
learned, learned + geometry, post hoc, post hoc + geometry; labels = PATIENT-WIDE coverage on the 40k-voxel UQ subset
(exp_seeds/results/brats_s42_*/case.csv). Right: profile along the through-plane axis at the fixed centre column: ground truth,
reconstruction and the post-hoc 90 % intervals without (grey) and with (blue) geometry, from exact endpoints.
Acquired slices: neutral grey in the maps (copied back, not scored). Inputs: exp_uniform_arssr/results/qual/planes4_clustered.npz
(qual_v4.py) and planes_uniform.npz (qual_v3.py; same crop).
"""

from project_paths import root_path
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import msstyle as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402

I = root_path('for_ISBI')
Q = os.path.join(I, "exp_uniform_arssr", "results", "qual")
OUTP = os.path.join(I, "ISBI_paper", "figures", "fig_qualitative.pdf")
S.use()
z = dict(np.load(os.path.join(Q, "planes4_clustered.npz")))
zu = dict(np.load(os.path.join(Q, "planes_uniform.npz")))
fg, acq = z["fg"].astype(bool), z["acq"].astype(bool)
h, w = z["gt"].shape
EMAP = plt.get_cmap("magma").copy(); EMAP.set_bad("black")
WMAP = plt.get_cmap("viridis").copy(); WMAP.set_bad("black")
EMAX, WMAX = 0.20, 0.35
GREY = plt.get_cmap("gray")
PAT = "BraTS-GLI-01739"


def cov(method, arm):
    c = pd.read_csv(os.path.join(I, "exp_seeds", "results", f"brats_s42_{method}", "case.csv"))
    c = c[(c.patient == PAT) & (c.R == 4) & (c.pattern == "clustered") & (c.rep == 0) & (c.arm == arm)]
    return float(c.cov90.iloc[0])


ROWS = [("SAINT_P", "SAINT-P", "SAINT_P", "SAINT_PG"), ("ArSSR_P", "ArSSR-P", "ArSSR", "ArSSR_PG"), ("linear", "Linear", "linear", None)]
ARMS = [("learned", "learned"), ("learned_geo", "learned\n+ geometry"), ("ph_nogeo", "post hoc"), ("ph_geo", "post hoc\n+ geometry")]


def mask(a):
    a = np.array(a, float)
    a[~fg | acq] = np.nan
    return a


# ------------------------------------------------------------------ layout (inches)
fig_w = S.DOUBLE
pw = 0.66                      # map panel width
ph = pw * h / w
x0, gap_c, gap_r = 0.05, 0.36, 0.07
top, bot = 0.52, 0.55
prof_w = fig_w - (x0 + 7 * pw + gap_c + 0.40) - 0.04
fig_h = top + 3 * ph + 2 * gap_r + bot
fig = plt.figure(figsize=(fig_w, fig_h))


def ax_at(r, c, width=pw):
    if c == "prof":
        left = x0 + 7 * pw + gap_c + 0.40
    else:
        left = x0 + c * pw + (gap_c if c > 0 else 0)
    y0 = fig_h - top - (r + 1) * ph - r * gap_r
    return fig.add_axes([left / fig_w, y0 / fig_h, (width - 0.03) / fig_w, (ph - 0.03) / fig_h])


def clean(ax):
    ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)


def overlay(ax):
    ov = np.zeros(fg.shape + (4,))
    ov[acq & fg] = (0.42, 0.43, 0.46, 1.0)
    ax.imshow(ov, interpolation="nearest")


# fixed profile column: centre of the brain bounding box in this plane (anatomy only)
cols = np.where(fg.any(0))[0]
PC = int(round(0.5 * (cols.min() + cols.max())))
rows_acq = np.where(acq.any(1))[0]
# longest gap (in view) between consecutive acquired rows
d = np.diff(rows_acq)
brain_rows = np.where(fg[:, PC])[0]
cmid = 0.5 * (brain_rows.min() + brain_rows.max())
cand = np.where(d >= d.max() - 2)[0]                 # near-maximal gaps (clusters vary by +-1 slice)
j = int(cand[np.argmin(np.abs(0.5 * (rows_acq[cand] + rows_acq[cand + 1]) - cmid))])
G0, G1 = rows_acq[j], rows_acq[j + 1]
W0, W1 = max(0, G0 - 22), min(h - 1, G1 + 22)

# context column
uacq = zu["acq"].astype(bool)
for r, (img, lab, ticks) in enumerate(((zu["sparse"], "uniform input", np.where(uacq.any(1))[0]),
                                       (z["sparse"], "clustered input", rows_acq), (z["gt"], "ground truth", None))):
    ax = ax_at(r, 0)
    ax.imshow(img, cmap=GREY, vmin=0, vmax=1, interpolation="nearest")
    if ticks is not None:
        ax.plot(np.full(len(ticks), -3.0), ticks, ls="none", marker="_", ms=2.6, mew=0.8, color="#C0392B", clip_on=False)
    if r == 1:
        ax.axhspan(G0, G1, color="#38BDF8", alpha=0.25, lw=0)
    clean(ax)
    ax.set_title(lab, fontsize=7, pad=2)

# map grid
for r, (k, lab, cm, cmg) in enumerate(ROWS):
    ax = ax_at(r, 1)
    ax.imshow(z[f"{k}__mean"], cmap=GREY, vmin=0, vmax=1, interpolation="nearest")
    ax.plot([PC, PC], [W0, W1], color="white", lw=0.6, ls=(0, (2, 1.5)))   # profile window only
    ax.axhspan(G0, G1, color="#38BDF8", alpha=0.25, lw=0)  # long gap, same band as clustered input and profile
    clean(ax)
    ax.text(-0.06, 0.5, f"{lab}\n{float(z[k + '__psnr']):.1f} dB", transform=ax.transAxes, rotation=90, ha="right",
            va="center", fontsize=7, linespacing=1.15)
    if r == 0:
        ax.set_title("reconstruction", fontsize=7, pad=2)
    ax = ax_at(r, 2)
    ax.imshow(mask(z[f"{k}__err"]), cmap=EMAP, vmin=0, vmax=EMAX, interpolation="nearest")
    overlay(ax); clean(ax)
    if r == 0:
        ax.set_title("$|y-\\bar y|$", fontsize=7, pad=2)
    for c, (arm, alab) in enumerate(ARMS, start=3):
        ax = ax_at(r, c)
        key_lo, key_hi = f"{k}__{arm}_lo", f"{k}__{arm}_hi"
        if key_lo in z:
            ax.imshow(mask(z[key_hi] - z[key_lo]), cmap=WMAP, vmin=0, vmax=WMAX, interpolation="nearest")
            overlay(ax); clean(ax)
            src = (cmg if arm == "learned_geo" else cm)
            ax.text(0.04, 0.03, f"cov {100 * cov(src, arm):.0f}%", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3,
                    color="white")
        else:
            clean(ax)
            ax.text(0.5, 0.5, "no network", transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color=S.MUTED)
        if r == 0:
            ax.set_title(alab, fontsize=7, pad=2, linespacing=1.0)
    # profile
    ax = ax_at(r, "prof", width=prof_w)
    zz = np.arange(h)
    gtp = z["gt"][:, PC]
    xp = z[f"{k}__mean"][:, PC]
    lo, hi = z[f"{k}__ph_geo_lo"][:, PC], z[f"{k}__ph_geo_hi"][:, PC]
    ax.fill_between(zz, lo, hi, color="#0072B2", alpha=0.28, lw=0)
    lo, hi = z[f"{k}__ph_nogeo_lo"][:, PC], z[f"{k}__ph_nogeo_hi"][:, PC]
    ax.plot(zz, lo, color="#6B7280", lw=0.6, ls=(0, (1.5, 1)))
    ax.plot(zz, hi, color="#6B7280", lw=0.6, ls=(0, (1.5, 1)))
    ax.plot(zz, gtp, color=S.INK, lw=0.8)
    ax.plot(zz, xp, color="#D55E00", lw=0.7, ls=(0, (2, 1)))
    ax.axvspan(G0, G1, color="#38BDF8", alpha=0.12, lw=0, zorder=0)
    for a_ in rows_acq:
        if W0 <= a_ <= W1:
            ax.axvline(a_, color="#C0392B", lw=0.5, alpha=0.7, zorder=0)
    ax.set_xlim(W0, W1); ax.set_ylim(-0.02, 1.0)
    ax.set_yticks([0, 0.5, 1.0]); ax.tick_params(labelsize=6, pad=1, length=1.5)
    ax.grid(False)
    ax.set_ylabel("intensity", fontsize=6.3, labelpad=1)
    if r < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("position (mm); shaded: long gap", fontsize=6.5, labelpad=1)
    if r == 0:
        ax.set_title("intensity along the dashed line", fontsize=7, pad=2)
# group headers + separator: the context column is reference only; every map and profile is from the CLUSTERED input
from matplotlib.lines import Line2D as _L  # noqa: E402
yh = (fig_h - 0.13) / fig_h
gx0, gx1 = x0 + pw + gap_c, x0 + 7 * pw + gap_c - 0.03
pr0 = x0 + 7 * pw + gap_c + 0.40
fig.text((x0 + 0.5 * pw) / fig_w, yh, "inputs", ha="center", va="center", fontsize=7.3, weight="bold", color=S.INK2)
fig.text(0.5 * (gx0 + pr0 + prof_w) / fig_w, yh, "all reconstructions, uncertainty maps and profiles: from the clustered input",
         ha="center", va="center", fontsize=7.3, weight="bold", color=S.INK)
for xa, xb in ((x0, x0 + pw - 0.03), (gx0, pr0 + prof_w - 0.03)):
    fig.add_artist(_L([xa / fig_w, xb / fig_w], [(fig_h - 0.22) / fig_h] * 2, color=S.INK2, lw=0.6))
xs = (x0 + pw + 0.05) / fig_w
fig.add_artist(_L([xs, xs], [(bot - 0.05) / fig_h, (fig_h - 0.08) / fig_h], color=S.INK2, lw=0.8))
# colourbars + profile legend
cx0 = x0 + 2 * pw + gap_c
cax = fig.add_axes([cx0 / fig_w, 0.30 / fig_h, (pw - 0.05) / fig_w, 0.05 / fig_h])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, EMAX), cmap=EMAP), cax=cax, orientation="horizontal")
cb.set_ticks([0, 0.1, 0.2]); cb.set_ticklabels(["0", "0.1", "0.2"]); cb.ax.tick_params(labelsize=6, length=1.5, pad=1); cb.outline.set_linewidth(0.3)
cb.set_label("|error|", fontsize=6.5, labelpad=1)
cax = fig.add_axes([(cx0 + pw + 0.12) / fig_w, 0.30 / fig_h, (4 * pw - 0.17) / fig_w, 0.05 / fig_h])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, WMAX), cmap=WMAP), cax=cax, orientation="horizontal")
cb.set_ticks([0, 0.1, 0.2, 0.3]); cb.ax.tick_params(labelsize=6, length=1.5, pad=1); cb.outline.set_linewidth(0.3)
cb.set_label("90% interval width", fontsize=6.5, labelpad=1)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
hnd = [Line2D([], [], color=S.INK, lw=0.8, label="truth"), Line2D([], [], color="#D55E00", lw=0.7, ls=(0, (2, 1)), label="recon."),
       Line2D([], [], color="#6B7280", lw=0.6, ls=(0, (1.5, 1)), label="post hoc"), Patch(color="#0072B2", alpha=0.28, label="+ geometry")]
fig.legend(handles=hnd, loc="lower left", bbox_to_anchor=((x0 + 7 * pw + gap_c + 0.15) / fig_w, 0.0), ncol=2, fontsize=6,
           frameon=False, handlelength=1.3, columnspacing=0.6, handletextpad=0.3)
plt.rcParams["savefig.bbox"] = "standard"
fig.savefig(OUTP, bbox_inches=None)
fig.savefig(os.path.join(I, "manuscript_4page", "figures", "fig_qual_v4_preview.png"), dpi=300, bbox_inches=None)
print("wrote", OUTP, round(fig_h, 2), "profile col", PC, "gap rows", G0, G1)
