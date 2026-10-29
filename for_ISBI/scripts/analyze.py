"""Post-hoc fitting (FIT), calibration (CAL, uniform), and TEST metrics.

    python scripts/analyze.py --stage fit       -> results/posthoc_cv.csv, results/posthoc_models.pkl
    python scripts/analyze.py --stage calib     -> results/calibration.csv
    python scripts/analyze.py --stage test      -> results/test_case_metrics.csv, curves, recon tables
    python scripts/analyze.py --stage mcconv    -> results/mc_convergence.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch

from isbi import common as C
from isbi import uq as U

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True)
args = ap.parse_args()
EV = os.environ.get("ISBI_EV", os.path.join(C.ISBI, "runs", "eval"))
RES = os.environ.get("ISBI_RES", os.path.join(C.ISBI, "results"))
os.makedirs(RES, exist_ok=True)
torch.set_num_threads(16)

ARCHS = ["SAINT", "I3Net"]
MEANS = ["SAINT_P", "I3Net_P", "SAINT_det", "I3Net_det", "linear", "bicubic"]


def cases(role):
    for f in sorted(glob.glob(os.path.join(EV, role, "cases", "*.npz"))):
        yield dict(np.load(f))


def arms():
    """(arm name, mean method, interval type)."""
    out = []
    for a in ARCHS:
        out += [(f"{a}_P|learned_raw", f"{a}_P", "learned_raw"), (f"{a}_P|learned_cal", f"{a}_P", "learned_cal"),
                (f"{a}_P|posthoc", f"{a}_P", "posthoc"), (f"{a}_P|const", f"{a}_P", "const"),
                (f"{a}_det|posthoc", f"{a}_det", "posthoc"), (f"{a}_det|const", f"{a}_det", "const")]
    for m in ("linear", "bicubic"):
        out += [(f"{m}|posthoc", m, "posthoc"), (f"{m}|const", m, "const")]
    return out


# ----------------------------------------------------------------------------- interval primitives
def cover_learned(c, m, alpha, lam):
    y = torch.from_numpy(c["y"])
    F = U.mix_cdf(y, torch.from_numpy(c[f"{m}__mu_t"]), torch.from_numpy(c[f"{m}__b_t"]), lam).numpy()
    return (F >= (1 - alpha) / 2) & (F <= (1 + alpha) / 2)


def width_learned(c, m, alpha, lam):
    mu, b = torch.from_numpy(c[f"{m}__mu_t"]), torch.from_numpy(c[f"{m}__b_t"])
    lo = U.mix_quantile(mu, b, (1 - alpha) / 2, lam).numpy()
    hi = U.mix_quantile(mu, b, (1 + alpha) / 2, lam).numpy()
    return np.clip(hi, 0, 1) - np.clip(lo, 0, 1)


def sym_interval(x, half):
    return np.clip(x + half, 0, 1) - np.clip(x - half, 0, 1)


def load_posthoc():
    return pickle.load(open(os.path.join(RES, "posthoc_models.pkl"), "rb"))


def scale_of(c, m, glms):
    return glms[m].predict(c[f"{m}__feat"])


# ============================================================================= FIT
if args.stage == "fit":
    data = {m: [] for m in MEANS}
    for c in cases("FIT"):
        for m in MEANS:
            data[m].append((c["patient"].item(), int(c["R"]), c[f"{m}__feat"], c["y"] - c[f"{m}__x"]))
    configs = [dict(ridge=r, interactions=i) for r in (1e-4, 1e-2) for i in (False, True)]
    rows = []
    for m in MEANS:
        pats = sorted({d[0] for d in data[m]})
        folds = np.array_split(np.random.RandomState(0).permutation(pats), 5)
        for k, cfg in enumerate(configs):
            nll = []
            for fo in folds:
                tr = [d for d in data[m] if d[0] not in set(fo)]
                te = [d for d in data[m] if d[0] in set(fo)]
                g = U.LaplaceGLM(**cfg).fit(np.concatenate([d[2] for d in tr]), np.concatenate([d[3] for d in tr]))
                nll += [U.laplace_nll(d[3], g.predict(d[2])) for d in te]
            rows.append(dict(method=m, config=k, **cfg, cv_nll=float(np.mean(nll))))
            print(rows[-1], flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(os.path.join(RES, "posthoc_cv.csv"), index=False)
    best = int(cv.groupby("config").cv_nll.mean().idxmin())         # ONE common design for all methods
    cfg = configs[best]
    glms = {m: U.LaplaceGLM(**cfg).fit(np.concatenate([d[2] for d in data[m]]),
                                       np.concatenate([d[3] for d in data[m]])) for m in MEANS}
    pickle.dump(glms, open(os.path.join(RES, "posthoc_models.pkl"), "wb"))
    json.dump(dict(selected_config=cfg, n_fit_patients=len({d[0] for d in data[MEANS[0]]}),
                   coefficients={m: dict(c=g.c, theta=g.th.tolist()) for m, g in glms.items()}),
              open(os.path.join(RES, "posthoc_selected.json"), "w"), indent=1)
    print("selected", cfg)

# ============================================================================= CALIB
if args.stage == "calib":
    glms = load_posthoc()
    cal = list(cases("CAL"))
    rows = []
    for R in C.STRIDES:
        cs = [c for c in cal if int(c["R"]) == R and c["pattern"].item() == "uniform"]
        for arm, m, kind in arms():
            if kind == "learned_raw":
                continue
            for a in U.LEVELS:
                if kind == "learned_cal":
                    fn = lambda c, lam: cover_learned(c, m, a, lam).mean()  # noqa: E731
                elif kind == "posthoc":
                    pre = {id(c): (np.abs(c["y"] - c[f"{m}__x"]), scale_of(c, m, glms)) for c in cs}
                    fn = lambda c, lam, pre=pre: (pre[id(c)][0] <= lam * pre[id(c)][1]).mean()  # noqa: E731
                else:
                    fn = lambda c, lam: (np.abs(c["y"] - c[f"{m}__x"]) <= lam).mean()  # noqa: E731
                lo, hi = (1e-2, 1e2) if kind != "const" else (1e-5, 1.0)
                lam = U.calibrate_scalar(fn, cs, a, lo, hi)
                rows.append(dict(R=R, arm=arm, mean=m, kind=kind, alpha=a, lam=lam, n_cal=len(cs)))
                print(rows[-1], flush=True)
        # error-detection threshold: 90th percentile of |e| on CAL uniform, per mean method
        for m in MEANS:
            e = np.concatenate([np.abs(c["y"] - c[f"{m}__x"]) for c in cs])
            rows.append(dict(R=R, arm=f"{m}|tau90", mean=m, kind="tau", alpha=0.9, lam=float(np.quantile(e, 0.9)),
                             n_cal=len(cs)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "calibration.csv"), index=False)

# ============================================================================= TEST / MCCONV
if args.stage in ("test", "sel"):
    role = "TEST" if args.stage == "test" else "SEL"
    glms = load_posthoc()
    cal = pd.read_csv(os.path.join(RES, "calibration.csv"))
    lam = {(r.R, r.arm, r.alpha): r.lam for r in cal.itertuples()}
    rows, curves, gb = [], [], []
    BINS = [(2, 4), (5, 8), (9, 12), (13, 16), (17, 30)]                 # bracketing gap length (mm)
    for c in cases(role):
        R, pat, rep, p = int(c["R"]), c["pattern"].item(), int(c["rep"]), c["patient"].item()
        base = dict(patient=p, pattern=pat, R=R, rep=rep)
        gap, near = c["gap_mm"], c["near_mm"]
        # ---------------- intervals
        for arm, m, kind in arms():
            x = c[f"{m}__x"]
            e = c["y"] - x
            r = dict(base, arm=arm, mean=m, kind=kind, mae=float(np.abs(e).mean()))
            bh = scale_of(c, m, glms) if kind == "posthoc" else None
            for a in U.LEVELS:
                if kind == "learned_raw":
                    cov, w = cover_learned(c, m, a, 1.0), width_learned(c, m, a, 1.0)
                elif kind == "learned_cal":
                    l_ = lam[(R, arm, a)]
                    cov, w = cover_learned(c, m, a, l_), width_learned(c, m, a, l_)
                elif kind == "posthoc":
                    h = lam[(R, arm, a)] * bh
                    cov, w = np.abs(e) <= h, sym_interval(x, h)
                else:
                    h = lam[(R, arm, a)]
                    cov, w = np.abs(e) <= h, sym_interval(x, h)
                r[f"cov{int(a * 100)}"] = float(cov.mean())
                r[f"width{int(a * 100)}"] = float(w.mean())
            rows.append(r)
        # ---------------- ranking scores
        rng = np.random.RandomState(C._h("isbi-random", p, pat, R, rep))
        for m in MEANS:
            x = c[f"{m}__x"]
            e = c["y"] - x
            scores = {"posthoc": scale_of(c, m, glms), "near_mm": near + 1e-6 * rng.rand(len(e)),
                      "gap_mm": gap + 1e-6 * rng.rand(len(e)), "random": rng.rand(len(e))}
            if m.endswith("_P"):
                scores["learned_total"] = c[f"{m}__epi"] + c[f"{m}__alea"]
                scores["learned_alea"] = c[f"{m}__alea"]
                scores["learned_epi"] = c[f"{m}__epi"]
            tau = lam[(R, f"{m}|tau90", 0.9)]
            pos = np.abs(e) > tau
            for sname, s in scores.items():
                au, sp_curve, or_curve = U.ause(e, s)
                mae0 = float(np.abs(e).mean())
                rows.append(dict(base, arm=f"{m}#{sname}", mean=m, kind="rank", score=sname, mae=mae0,
                                 spearman=U.spearman(np.abs(e), s), strat_spearman=U.strat_spearman(e, s, gap, near),
                                 ause=au, ause_norm=au / mae0, auprc=U.auprc(pos, s), prevalence=float(pos.mean())))
                curves.append(dict(base, mean=m, score=sname, **{f"f{f:.2f}": v for f, v in zip(U.FR, sp_curve)}))
            curves.append(dict(base, mean=m, score="oracle", **{f"f{f:.2f}": v for f, v in zip(U.FR, or_curve)}))
        # ---------------- by gap length (per bin: MAE, calibrated 90% coverage, within-bin ranking)
        for a_ in ARCHS:
            m = f"{a_}_P"
            e = c["y"] - c[f"{m}__x"]
            Fl = cover_learned(c, m, 0.9, lam[(R, f"{m}|learned_cal", 0.9)])
            ph = np.abs(e) <= lam[(R, f"{m}|posthoc", 0.9)] * scale_of(c, m, glms)
            sl = c[f"{m}__epi"] + c[f"{m}__alea"]
            sp = scale_of(c, m, glms)
            for lo_, hi_ in BINS:
                k = (gap >= lo_) & (gap <= hi_)
                if k.sum() < 200:
                    continue
                gb.append(dict(base, mean=m, gap_bin=f"{lo_}-{hi_}", n=int(k.sum()), mae=float(np.abs(e[k]).mean()),
                               cov90_learned_cal=float(Fl[k].mean()), cov90_posthoc=float(ph[k].mean()),
                               spearman_learned=U.spearman(np.abs(e[k]), sl[k]),
                               spearman_posthoc=U.spearman(np.abs(e[k]), sp[k])))
        print(p, pat, R, rep, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{role.lower()}_case_metrics.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(RES, f"{role.lower()}_sparsification_curves.csv"), index=False)
    pd.DataFrame(gb).to_csv(os.path.join(RES, f"{role.lower()}_gapbins.csv"), index=False)
    rr = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(EV, role, "rows", "*.csv"))])
    rr.to_csv(os.path.join(RES, f"{role.lower()}_recon_metrics.csv"), index=False)

if args.stage == "mcconv":
    rows = []
    for c in cases("MCCONV"):
        for m in ("SAINT_P", "I3Net_P"):
            mu, b = c[f"{m}__mu_t"], c[f"{m}__b_t"]
            ref_var = mu.var(0) + (2 * b ** 2).mean(0)
            yt = torch.from_numpy(c["y"])
            Fref = U.mix_cdf(yt, torch.from_numpy(mu), torch.from_numpy(b)).numpy()
            for T in (5, 10, 20, 50, 100):
                v = mu[:T].var(0) + (2 * b[:T] ** 2).mean(0)
                F = U.mix_cdf(yt, torch.from_numpy(mu[:T]), torch.from_numpy(b[:T])).numpy()
                e = np.abs(c["y"] - mu[:T].mean(0))
                rows.append(dict(patient=c["patient"].item(), pattern=c["pattern"].item(), method=m, T=T,
                                 rel_err_total_var=float(np.median(np.abs(v - ref_var) / ref_var)),
                                 epi_frac=float((mu[:T].var(0) / v).mean()),
                                 rank_corr_vs_T100=U.spearman(v, ref_var), spearman_err=U.spearman(e, v),
                                 cov90=float(((F >= .05) & (F <= .95)).mean()),
                                 cov90_T100=float(((Fref >= .05) & (Fref <= .95)).mean())))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RES, "mc_convergence.csv"), index=False)
    print(d.groupby(["method", "pattern", "T"]).mean(numeric_only=True).round(4).to_string())
