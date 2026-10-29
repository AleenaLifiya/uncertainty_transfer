"""Qualitative panels + NIfTI for the rule-selected TEST cases (PROTOCOL.md), and diagram textures.

    python scripts/qualitative.py            # selection rule -> runs/qual/, figures/qual_*.{png,pdf}, nifti/
    python scripts/qualitative.py --texture  # one SEL case -> runs/qual/sel_texture.npz (diagram textures)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from isbi import common as C  # noqa: E402
from isbi import evalcore as E  # noqa: E402
from isbi import recon as RC  # noqa: E402
from isbi import uq as U  # noqa: E402
from comparisons import protocol as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--texture", action="store_true")
a = ap.parse_args()
torch.backends.cudnn.deterministic = True
QD = os.path.join(C.ISBI, "runs", "qual")
FIG = os.path.join(C.ISBI, "figures")
NII = os.path.join(C.ISBI, "results", "nifti")
for d in (QD, FIG, NII):
    os.makedirs(d, exist_ok=True)
roles = C.load_roles()
proto = json.load(open(os.path.join(C.ISBI, "configs", "protocol_frozen.json")))
models = E.Models(proto["selected_checkpoints"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})


def full_case(p, pattern, R, rep):
    """Re-run the case exactly as in evaluation (same seed) and keep full volumes + MC samples."""
    scan = roles["scan_of"][p]
    gt = P.prepare_one(C.scan_path(scan))["vol01"]
    ret = C.make_mask(pattern, R, p, rep)
    vin, G = RC.adapter_input(gt, ret, R)
    seed = C._h("isbi-mc", p, C.cond_key(pattern, R, rep))
    sag, cor = RC.saint_marginals(models.saint_net, vin, G, R, models.chunk)
    mu, b = RC.saint_mc(models.saint_p, sag, cor, proto["T"], seed)
    return gt, ret, sag, cor, mu, b, vin


if a.texture:
    p = roles["SEL"][4]
    gt, ret, sag, cor, mu, b, _ = full_case(p, "clustered", 4, 0)
    mean, epi, alea = RC.moments(mu, b)
    np.savez(os.path.join(QD, "sel_texture.npz"), sag=sag, cor=cor, mean=mean.cpu().numpy(),
             epi=epi.cpu().numpy(), alea=alea.cpu().numpy())
    print("texture case", p)
    raise SystemExit

# ------------------------------------------------------------------ selection rule (PROTOCOL.md)
rec = pd.read_csv(os.path.join(C.ISBI, "results", "test_recon_metrics.csv"))
u = rec[(rec.method == "SAINT_P") & (rec.R == 4) & (rec.pattern == "uniform")].sort_values("psnr").reset_index(drop=True)
pick = {"median": u.patient.iloc[len(u) // 2], "difficult": u.patient.iloc[int(np.floor(0.10 * (len(u) - 1)))]}
json.dump(dict(rule="rank TEST by SAINT_P PSNR at x4 uniform; median and 10th percentile", picks=pick,
               psnr={k: float(u[u.patient == v].psnr.iloc[0]) for k, v in pick.items()}),
          open(os.path.join(QD, "selection.json"), "w"), indent=1)
glms = pickle.load(open(os.path.join(C.ISBI, "results", "posthoc_models.pkl"), "rb"))
cal = pd.read_csv(os.path.join(C.ISBI, "results", "calibration.csv"))
lam = {(r.R, r.arm, r.alpha): r.lam for r in cal.itertuples()}


def posthoc_map(x, lin, gt, ret, planes, method):
    """b_hat on the displayed planes only (all voxels of missing slices in those planes)."""
    geo = C.gap_geometry(ret)
    H, W, D = gt.shape
    out = {}
    for name, (ax, idx) in planes.items():
        grid = np.zeros((H, W, D), bool)
        if ax == 0:
            grid[idx, :, :] = True
        elif ax == 1:
            grid[:, idx, :] = True
        else:
            grid[:, :, idx] = True
        grid[:, :, ret] = False
        sub = np.nonzero(grid)
        bh = np.full((H, W, D), np.nan, np.float32)
        bh[sub] = glms[method].predict(E.features(x, lin, gt, ret, geo, sub))
        out[name] = bh
    return out


def affine_of(scan):
    prep = P.prepare_one(C.scan_path(scan))
    T = np.eye(4)
    T[2, 3] = float(prep["depth_start"])
    return np.asarray(prep["affine"], float) @ T


GRAY = "#8a8f98"
COLS = [("acquired", "acquired", "gray", (0, 1)), ("gt", "ground truth", "gray", (0, 1)),
        ("linear", "linear", "gray", (0, 1)), ("mean", "SAINT-P mean", "gray", (0, 1)),
        ("err", "|error|", "magma", (0, 0.15)), ("sd", "learned σ (total)", "magma", (0, 0.15)),
        ("phk", "post-hoc κ₉₀·b̂", "magma", (0, 0.15)), ("lw", "learned 90% width", "viridis", (0, 0.4)),
        ("pw", "post-hoc 90% width", "viridis", (0, 0.4)), ("epi", "epistemic σ (own scale)", "cividis", None)]


def load_planes(tag, pattern):
    z = np.load(os.path.join(QD, f"planes_{tag}_{pattern}.npz"))
    meta = json.loads(str(z["meta"]))
    d = {}
    for k in z.files:
        if k == "meta":
            continue
        n, f = k.split("__")
        d.setdefault(n, {})[f] = z[k]
    return d, meta


def show(ax, d, key, cmap, rng):
    img = d[key].astype(float).copy()
    m = np.ma.masked_array(img, mask=False)
    if key not in ("acquired", "gt", "linear", "mean"):
        m = np.ma.masked_array(img, mask=(~d["fg"]) | d["acq"])     # outside head / acquired rows: gray
    cm_ = matplotlib.colormaps[cmap].copy()
    cm_.set_bad(GRAY)
    ax.imshow(m, cmap=cm_, vmin=rng[0], vmax=rng[1], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_case(tag, p, pattern):
    d, meta = load_planes(tag, pattern)
    fig, axs = plt.subplots(3, len(COLS), figsize=(10.4, 3.9), gridspec_kw=dict(wspace=0.04, hspace=0.06))
    for r_, name in enumerate(("coronal", "sagittal", "axial")):
        for c_, (key, title, cmap, rng) in enumerate(COLS):
            rng = rng or (0, meta["epi_vmax"])
            show(axs[r_, c_], d[name], key, cmap, rng)
            if r_ == 0:
                axs[r_, c_].set_title(title, fontsize=6.8)
            if c_ == 0:
                axs[r_, c_].set_ylabel(name, fontsize=7)
    axs[2, -1].set_xlabel(f"0–{meta['epi_vmax']:.4f}", fontsize=6)
    axs[2, 0].set_xlabel("axial slice not acquired", fontsize=6)
    fig.suptitle(f"{tag} case {p}  ·  ×4 {pattern}  ·  SAINT-P PSNR {meta['psnr']['SAINT_P']:.2f} dB  ·  "
                 f"error, σ, κ·b̂ share 0–0.15; widths 0–0.4; gray = outside head / acquired slice", fontsize=7.5)
    base = os.path.join(FIG, f"qual_{tag}_{pattern}_x4")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def slide_figure(tag, p):
    """Same patient, uniform vs clustered, coronal + missing axial; columns chosen for the slide."""
    keys = [("acquired", "acquired", "gray", (0, 1)), ("gt", "ground truth", "gray", (0, 1)),
            ("mean", "SAINT-P mean", "gray", (0, 1)), ("err", "|error|", "magma", (0, 0.15)),
            ("lw", "learned\n90% width", "viridis", (0, 0.4)), ("pw", "post-hoc\n90% width", "viridis", (0, 0.4))]
    fig, axs = plt.subplots(4, len(keys), figsize=(7.4, 4.9), gridspec_kw=dict(wspace=0.03, hspace=0.05))
    r_ = 0
    for pattern in ("uniform", "clustered"):
        d, meta = load_planes(tag, pattern)
        for name in ("coronal", "axial"):
            for c_, (key, title, cmap, rng) in enumerate(keys):
                show(axs[r_, c_], d[name], key, cmap, rng)
                if r_ == 0:
                    axs[r_, c_].set_title(title, fontsize=8)
            axs[r_, 0].set_ylabel(f"{pattern}\n{name}", fontsize=7.5)
            r_ += 1
    for ax_, (lab, cmap, rng) in zip((axs[3, 3], axs[3, 4]), (("|error| 0–0.15", "magma", (0, .15)), ("width 0–0.4", "viridis", (0, .4)))):
        ax_.set_xlabel(lab, fontsize=6.5)
    fig.savefig(os.path.join(FIG, f"qual_slide_{tag}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, f"qual_slide_{tag}.pdf"), bbox_inches="tight")
    plt.close(fig)


if os.environ.get("QUAL_PLOT_ONLY"):
    for tag, p in pick.items():
        for pattern in ("uniform", "clustered"):
            plot_case(tag, p, pattern)
        slide_figure(tag, p)
    raise SystemExit

summary = []
for tag, p in pick.items():
    scan = roles["scan_of"][p]
    for pattern in ("uniform", "clustered"):
        R = 4
        out, rows, V = E.run_case(models, scan, p, pattern, R, 0, proto["T"], keep_volumes=True)
        gt, ret, region = V["gt"], V["ret"], V["region"]
        x = V["recons"]["SAINT_P"]
        lin = V["recons"]["linear"]
        sd_learn = np.sqrt(V["maps"]["SAINT_P__epi"] + V["maps"]["SAINT_P__alea"])
        sd_epi = np.sqrt(V["maps"]["SAINT_P__epi"])
        err = np.abs(gt - x)
        err[:, :, ret] = np.nan
        # display planes: coronal and sagittal through the brain centre-of-mass, one missing axial slice
        com = np.array(np.nonzero(gt > 0)).mean(1).astype(int)
        miss = [z for z in range(C.D) if z not in set(ret)]
        geo = C.gap_geometry(ret)
        zmid = max(miss, key=lambda z: (geo["near_mm"][z], -abs(z - com[2])))   # farthest from acquired, near centre
        planes = {"coronal": (1, com[1]), "sagittal": (0, com[0]), "axial": (2, zmid)}
        bh = posthoc_map(x, lin, gt, ret, planes, "SAINT_P")
        # learned 90% interval width (calibrated) and post-hoc width, on the planes
        g90l = lam[(R, "SAINT_P|learned_cal", 0.9)]
        k90 = lam[(R, "SAINT_P|posthoc", 0.9)]
        mu_t, b_t = full_case(p, pattern, R, 0)[4:6]

        def sl(v, name):
            ax, i = planes[name]
            s = v[i] if ax == 0 else (v[:, i] if ax == 1 else v[:, :, i])
            return np.rot90(s) if ax != 2 else np.rot90(s)

        def learned_width(name):
            ax, i = planes[name]
            m = mu_t[:, i] if ax == 0 else (mu_t[:, :, i] if ax == 1 else mu_t[:, :, :, i])
            bb = b_t[:, i] if ax == 0 else (b_t[:, :, i] if ax == 1 else b_t[:, :, :, i])
            shp = m.shape[1:]
            lo = U.mix_quantile(m.reshape(m.shape[0], -1).cpu(), bb.reshape(bb.shape[0], -1).cpu(), 0.05, g90l)
            hi = U.mix_quantile(m.reshape(m.shape[0], -1).cpu(), bb.reshape(bb.shape[0], -1).cpu(), 0.95, g90l)
            w = (hi.clamp(0, 1) - lo.clamp(0, 1)).numpy().reshape(shp)
            return w

        spm = np.zeros_like(gt)
        spm[:, :, ret] = gt[:, :, ret]
        acqv = np.zeros(gt.shape, bool)
        acqv[:, :, ret] = True
        cache = {}
        for name in planes:
            fg = sl(gt, name) > 0
            ys, xs = np.nonzero(fg)
            box = (max(ys.min() - 4, 0), ys.max() + 5, max(xs.min() - 4, 0), xs.max() + 5)
            cr = lambda a: a[box[0]:box[1], box[2]:box[3]]  # noqa: E731
            xx = sl(x, name)
            h = sl(bh[name] * k90, name)
            arr = dict(acquired=sl(spm, name), gt=sl(gt, name), linear=sl(lin, name), mean=xx,
                       err=sl(np.nan_to_num(err), name), sd=sl(sd_learn, name), phk=np.nan_to_num(h),
                       lw=np.rot90(learned_width(name)), pw=np.clip(xx + np.nan_to_num(h), 0, 1) - np.clip(xx - np.nan_to_num(h), 0, 1),
                       epi=sl(sd_epi, name), fg=fg, acq=sl(acqv, name))
            cache[name] = {k: cr(v) for k, v in arr.items()}
        cache_meta = dict(psnr={r["method"]: r["psnr"] for r in rows}, epi_vmax=float(np.quantile(sd_epi[region], 0.99)),
                          kappa90=k90, lambda90=g90l, zmid=int(zmid))
        np.savez(os.path.join(QD, f"planes_{tag}_{pattern}.npz"),
                 **{f"{n}__{k}": v for n, d in cache.items() for k, v in d.items()}, meta=json.dumps(cache_meta))
        plot_case(tag, p, pattern)
        # NIfTI (original scanner geometry incl. depth crop offset)
        aff = affine_of(scan)
        nd = os.path.join(NII, f"{p}_x4_{pattern}")
        os.makedirs(nd, exist_ok=True)
        acqm = np.zeros(gt.shape, np.uint8)
        acqm[:, :, ret] = 1
        vols = dict(gt=gt, acquired_mask=acqm, linear=lin, bicubic=V["recons"]["bicubic"], SAINT_P_mean=x,
                    SAINT_det=V["recons"]["SAINT_det"], I3Net_P_mean=V["recons"]["I3Net_P"],
                    SAINT_P_sigma_total=sd_learn, SAINT_P_sigma_epistemic=sd_epi,
                    SAINT_P_sigma_aleatoric=np.sqrt(V["maps"]["SAINT_P__alea"]),
                    I3Net_P_sigma_total=np.sqrt(V["maps"]["I3Net_P__epi"] + V["maps"]["I3Net_P__alea"]),
                    abs_error_SAINT_P=np.nan_to_num(err), eval_region=region.astype(np.uint8))
        for k, v in vols.items():
            img = nib.Nifti1Image(np.asarray(v, np.float32 if v.dtype != np.uint8 else np.uint8), aff)
            img.header.set_xyzt_units("mm")
            nib.save(img, os.path.join(nd, f"{k}.nii.gz"))
        summary.append(dict(tag=tag, patient=p, pattern=pattern, **{r["method"]: r["psnr"] for r in rows}))
        print(summary[-1], flush=True)
pd.DataFrame(summary).to_csv(os.path.join(QD, "qual_cases.csv"), index=False)
for tag, p in pick.items():
    slide_figure(tag, p)
