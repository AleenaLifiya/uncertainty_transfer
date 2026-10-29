"""Fig. 2 (ISBI_paper): reliability diagrams under clustered x4 sampling (Ovadia et al. 2019-style calibration view; the
binned quantities are those of ENCE, Levi et al. 2022). Seed 42, all clustered TEST cases pooled per arm.
For each arm, sigma^2 = variance of the CALIBRATED predictive distribution (as in exp_seeds/scripts/metrics_seed.py):
  learned / learned+geometry: Var_t(mu_t) + mean_t 2 (lambda b_t)^2, lambda calibrated on CAL uniform (same R);
  post hoc +- geometry: 2 (kappa b_hat / ln 10)^2, main-effects Laplace GLMs fitted on FIT uniform, kappa on CAL uniform.
Voxels are split into 15 equal-mass bins of sigma; each point = (RMV, RMSE) of a bin; diagonal = perfect calibration.
Panels: BraTS SAINT-P, ArSSR-P, linear | IXI-tiny SAINT-P, ArSSR-P, linear.
"""

from project_paths import root_path
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import msstyle as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

I = root_path('for_ISBI')
EPS, ALPHA, LN10, R, NB = 1e-3, 0.1, float(np.log(10.0)), 4, 15
torch.set_num_threads(8)
S.use()
COL = {"learned": "#CC79A7", "learned_geo": "#7B2D8E", "ph_nogeo": "#8A8F98", "ph_geo": "#0072B2"}
LAB = {"learned": "learned", "learned_geo": "learned + geometry", "ph_nogeo": "post-hoc", "ph_geo": "post-hoc + geometry"}
BASE = {"brats": os.path.join(I, "runs", "eval"), "ixi": os.path.join(I, "exp_ixi", "runs", "eval")}
NAT = {"brats": os.path.join(I, "exp_uniform_arssr", "runs", "eval_native", "arssr"),
       "ixi": os.path.join(I, "exp_ixi", "runs", "eval_native", "arssr")}


def load(d, m, role, key, pg=False):
    z = np.load(os.path.join(BASE[d], role, "cases", key))
    c = dict(y=z["y"], R=int(z["R"]), pattern=z["pattern"].item())
    if m == "SAINT_P":
        c.update(x=z["SAINT_P__x"], F=z["SAINT_P__feat"], mu=z["SAINT_P__mu_t"], b=z["SAINT_P__b_t"])
    elif m == "ArSSR":
        n = np.load(os.path.join(NAT[d], role, "cases", key))
        c.update(x=n["NAT__x"], F=n["NAT__feat"], mu=n["NAT__mu_t"], b=n["NAT__b_t"])
    else:
        c.update(x=z["linear__x"], F=z["linear__feat"])
    if pg:
        arch = "saint" if m == "SAINT_P" else "arssr"
        g = np.load(os.path.join(I, "exp_pg", "runs", f"{d}_{arch}_s42", "eval", role, key))
        c.update(mu_g=g["PG__mu_t"], b_g=g["PG__b_t"])
    return c


def keys(d, role, pat=None, all_r=False):
    ks = sorted(os.path.basename(f) for f in glob.glob(os.path.join(BASE[d], role, "cases", "*.npz")))
    return [k for k in ks if (all_r or f"_x{R}_" in k) and (pat is None or f"_{pat}_" in k)]


def phi(F, geo):
    cols = [F[:, 0], np.log(F[:, 1] + EPS), np.log(F[:, 2] + EPS), np.log(F[:, 3] + EPS)]
    if geo:
        cols += [np.log(F[:, 4]), np.log(F[:, 5]), F[:, 6]]
    return np.stack(cols, 1).astype(np.float64)


def fit_glm(X, e):
    m, s = X.mean(0), X.std(0) + 1e-8
    Xt = torch.from_numpy((X - m) / s)
    aa = torch.from_numpy(np.abs(e).astype(np.float64))
    th = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    c0 = torch.tensor([float(np.log(np.mean(np.abs(e)) + 1e-6))], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([th, c0], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        lb = c0 + Xt @ th
        loss = (lb + aa * torch.exp(-lb)).mean() + 1e-4 * (th ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    th_, c_ = th.detach().numpy(), float(c0.detach())
    return lambda X_: np.exp(c_ + ((X_ - m) / s) @ th_)


def calib(vals, c=0.9):
    srt = [np.sort(v) for v in vals]
    f = lambda k: np.mean([np.searchsorted(v, k, side="right") / len(v) for v in srt])  # noqa: E731
    lo, hi = np.log(1e-4), np.log(1e4)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(np.exp(mid)) < c else (lo, mid)
    return float(np.exp(0.5 * (lo + hi)))


def lam_thr(y, mu, b):
    y, mu, b = (torch.from_numpy(np.asarray(v, np.float64)) for v in (y, mu, b))

    def cov(lam):
        z = (y[None] - mu) / (lam * b)
        F = torch.where(z < 0, 0.5 * torch.exp(z.clamp(max=0)), 1 - 0.5 * torch.exp(-z.clamp(min=0))).mean(0)
        return (F >= ALPHA / 2) & (F <= 1 - ALPHA / 2)
    lo = torch.full(y.shape, np.log(1e-3), dtype=torch.float64)
    hi = torch.full(y.shape, np.log(1e3), dtype=torch.float64)
    c_lo, c_hi = cov(torch.exp(lo)), cov(torch.exp(hi))
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        cm = cov(torch.exp(mid))
        hi, lo = torch.where(cm, mid, hi), torch.where(cm, lo, mid)
    t = torch.exp(hi).numpy()
    t[c_lo.numpy()] = 0
    t[~c_hi.numpy()] = np.inf
    return t


def curves(d, m):
    prob = m != "linear"
    fit = [load(d, m, "FIT", k) for k in keys(d, "FIT", all_r=True)]          # FIT pools uniform x4 + x8 (as in the paper)
    F = np.concatenate([c["F"] for c in fit])
    e = np.concatenate([c["y"] - c["x"] for c in fit])
    G = {a: (fit_glm(phi(F, g), e), g) for a, g in (("ph_nogeo", False), ("ph_geo", True))}
    cal = [load(d, m, "CAL", k, pg=prob) for k in keys(d, "CAL")]
    K = {a: calib([np.abs(c["y"] - c["x"]) / g(phi(c["F"], geo)) for c in cal]) for a, (g, geo) in G.items()}
    L = {}
    if prob:
        L["learned"] = calib([lam_thr(c["y"], c["mu"], c["b"]) for c in cal])
        L["learned_geo"] = calib([lam_thr(c["y"], c["mu_g"], c["b_g"]) for c in cal])
    pool = {a: ([], []) for a in list(G) + list(L)}
    for k in keys(d, "TEST", "clustered"):
        c = load(d, m, "TEST", k, pg=prob)
        e2 = (c["y"] - c["x"]) ** 2
        for a, (g, geo) in G.items():
            pool[a][0].append(2 * (K[a] * g(phi(c["F"], geo)) / LN10) ** 2)
            pool[a][1].append(e2)
        if prob:
            for a, (mu, b) in (("learned", (c["mu"], c["b"])), ("learned_geo", (c["mu_g"], c["b_g"]))):
                pool[a][0].append(mu.var(0) + (2 * (L[a] * b) ** 2).mean(0))
                pool[a][1].append(e2)
    out = {}
    for a, (s2, e2) in pool.items():
        s2, e2 = np.concatenate(s2), np.concatenate(e2)
        o = np.argsort(s2, kind="stable")
        bins = np.array_split(o, NB)
        out[a] = (np.array([np.sqrt(s2[ix].mean()) for ix in bins]), np.array([np.sqrt(e2[ix].mean()) for ix in bins]))
    print(d, m, "kappa", {a: round(v, 3) for a, v in K.items()}, "lambda", {a: round(v, 3) for a, v in L.items()}, flush=True)
    return out


PANELS = [("brats", "SAINT_P", "BraTS · SAINT-P"), ("brats", "ArSSR", "BraTS · ArSSR-P"), ("brats", "linear", "BraTS · linear"),
          ("ixi", "SAINT_P", "IXI-tiny · SAINT-P"), ("ixi", "ArSSR", "IXI-tiny · ArSSR-P"), ("ixi", "linear", "IXI-tiny · linear")]
CACHE = os.path.join(I, "manuscript_4page", "figures", "reliability_cache.npz")
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    DATA = z["data"].item()
else:
    DATA = {f"{d}|{m}": curves(d, m) for d, m, _ in PANELS}
    np.savez(CACHE, data=np.array(DATA, dtype=object))
fig, axs = plt.subplots(1, 6, figsize=(S.DOUBLE, 1.62), sharex=True, sharey=True)
for ax, (d, m, title) in zip(axs, PANELS):
    lim = (0.018, 0.30)
    ax.plot(lim, lim, color=S.INK2, lw=0.6, ls=(0, (3, 2)), zorder=0)
    for a in ("learned", "learned_geo", "ph_nogeo", "ph_geo"):
        if a in DATA[f"{d}|{m}"]:
            x, y = DATA[f"{d}|{m}"][a]
            ax.plot(x, y, color=COL[a], marker="o", ms=2.0, lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xticks([0.02, 0.05, 0.1, 0.2])
    ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2"])
    ax.set_yticks([0.02, 0.05, 0.1, 0.2])
    ax.set_yticklabels(["0.02", "0.05", "0.1", "0.2"])
    ax.minorticks_off()
    ax.grid(True, color=S.GRID, lw=0.4)
    ax.tick_params(labelsize=6.3, pad=1.2)
    ax.set_title(title, fontsize=7, pad=2.5)
    ax.set_aspect("equal")
axs[0].set_ylabel("observed RMSE", fontsize=7.2)
fig.text(0.5, 0.02, "predicted RMV (root mean predictive variance per bin)", ha="center", fontsize=7.2)
h = [Line2D([], [], color=COL[a], marker="o", ms=2.5, lw=1.0, label=LAB[a]) for a in ("learned", "learned_geo", "ph_nogeo", "ph_geo")]
h.append(Line2D([], [], color=S.INK2, lw=0.6, ls=(0, (3, 2)), label="perfect calibration"))
fig.legend(handles=h, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02), fontsize=6.8, frameon=False)
fig.subplots_adjust(left=0.055, right=0.995, top=0.80, bottom=0.22, wspace=0.12)
plt.rcParams["savefig.bbox"] = "standard"
fig.savefig(os.path.join(I, "ISBI_paper", "figures", "fig_reliability.pdf"), bbox_inches=None)
fig.savefig(os.path.join(I, "manuscript_4page", "figures", "fig_reliability_preview.png"), dpi=300, bbox_inches=None)
print("wrote fig_reliability")
