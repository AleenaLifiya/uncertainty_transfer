"""CPU analyses from SAVED evaluation outputs only (no training, no inference). Added 2026-10-22.

  (1) post-hoc variants per reconstruction: 'ph'      = main-study design (uq.LaplaceGLM, ridge 1e-4, products),
                                            'ph_geo'  = main effects WITH geometry   (= ablation_posthoc.py 'full'),
                                            'ph_nogeo'= main effects WITHOUT geometry (= ablation_posthoc.py 'no_geom');
      plus the learned (calibrated) intervals for probabilistic models.
  (2) coverage / width / |error| / interval score by bracketing-gap length (mm), per R, patterns pooled and per pattern.
  (3) FIT/CAL sensitivity: 100 random re-splits of the 60 FIT+CAL patients into 30/30 (seed = split index); per split the
      post-hoc GLM ('ph') is refitted on the new FIT half and kappa / lambda recalibrated on the new CAL half (uniform data
      only, as in the study); TEST coverage is recomputed. Learned lambda uses per-voxel coverage thresholds (the smallest
      lambda at which the voxel is inside the central interval), so recalibration needs no MC re-evaluation.
Reads runs/eval/{FIT,CAL,TEST}/cases, runs/eval_native/arssr/... (mixed ArSSR-P) and, if present,
exp_uniform_arssr/runs/eval_native/arssr/... (uniform ArSSR-P-U). Writes exp_uniform_arssr/results/<tag>/.
Consistency checks against results/ are printed and asserted where the definitions coincide.

    python analysis_cpu.py --methods SAINT_P,I3Net_P,linear,ArSSR_mixed [--resplits 100] --tag base
    python analysis_cpu.py --methods ArSSR_uniform --tag uniform
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import torch

from isbi import common as C
from isbi import uq as U

ap = argparse.ArgumentParser()
ap.add_argument("--methods", required=True)
ap.add_argument("--resplits", type=int, default=100)
ap.add_argument("--tag", required=True)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--rs_kind", default="ph_geo", choices=["ph", "ph_geo"], help="post-hoc design used in the FIT/CAL re-splits")
a = ap.parse_args()
torch.set_num_threads(a.threads)
EV = os.environ.get("ISBI_EV", os.path.join(C.ISBI, "runs", "eval"))   # IXI: exp_ixi/runs/eval
NAT = {"ArSSR_mixed": os.path.join(C.ISBI, "runs", "eval_native", "arssr"),
       "ArSSR_uniform": os.environ.get("ISBI_NAT_UNIFORM", os.path.join(C.ISBI, "exp_uniform_arssr", "runs", "eval_native", "arssr"))}
OUT = os.path.join(os.environ.get("ISBI_OUT_BASE", os.path.join(C.ISBI, "exp_uniform_arssr", "results")), a.tag)
os.makedirs(OUT, exist_ok=True)
METHODS = a.methods.split(",")
PROB = {"SAINT_P", "I3Net_P", "ArSSR_mixed", "ArSSR_uniform"}
CVAL, ALPHA = 0.9, 0.1
EPS = U.EPS
BINS = [(2, 3), (4, 4), (5, 6), (7, 8), (9, 12), (13, 16), (17, 22)]
if os.environ.get("ISBI_BINS") == "ixi":          # IXI gaps are multiples of 2.18 mm (4.4 ... 43.6)
    BINS = [(4, 5), (6, 7), (8, 9), (10, 14), (15, 20), (21, 30), (31, 45)]
cal_tab = pd.read_csv(os.path.join(C.ISBI, "results", "calibration.csv"))
if "ISBI_EV" in os.environ:                      # other dataset: no study constants to compare against
    cal_tab = cal_tab.iloc[0:0]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def keys(role):
    return sorted(os.path.basename(f) for f in glob.glob(os.path.join(EV, role, "cases", "*.npz")))


def load_case(role, k, m, need_mc):
    z = np.load(os.path.join(EV, role, "cases", k))
    c = dict(patient=z["patient"].item(), pattern=z["pattern"].item(), R=int(z["R"]), rep=int(z["rep"]),
             y=z["y"], gap=z["gap_mm"], near=z["near_mm"])
    if m in NAT:
        n = np.load(os.path.join(NAT[m], role, "cases", k))
        c["x"], c["F"] = n["NAT__x"], n["NAT__feat"]
        if need_mc:
            c["mu"], c["b"] = n["NAT__mu_t"], n["NAT__b_t"]
    else:
        c["x"], c["F"] = z[f"{m}__x"], z[f"{m}__feat"]
        if need_mc and m in PROB:
            c["mu"], c["b"] = z[f"{m}__mu_t"], z[f"{m}__b_t"]
    return c


# ------------------------------------------------------------------ designs
def phi_me(F, geo=True):
    cols = [F[:, 0], np.log(F[:, 1] + EPS), np.log(F[:, 2] + EPS), np.log(F[:, 3] + EPS)]
    if geo:
        cols += [np.log(F[:, 4]), np.log(F[:, 5]), F[:, 6]]
    return np.stack(cols, 1).astype(np.float64)


class GLMX(U.LaplaceGLM):
    """Copy of ablation_posthoc.GLM: LaplaceGLM on a precomputed design matrix (mean NLL + ridge)."""

    def fit_X(self, X, e):
        self.m, self.s = X.mean(0), X.std(0) + 1e-8
        Xt = torch.from_numpy((X - self.m) / self.s)
        aa = torch.from_numpy(np.abs(e).astype(np.float64))
        th = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        c0 = torch.tensor([float(np.log(np.mean(np.abs(e)) + 1e-6))], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th, c0], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            lb = c0 + Xt @ th
            loss = (lb + aa * torch.exp(-lb)).mean() + self.ridge * (th ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.th, self.c = th.detach().numpy(), float(c0.detach())
        return self

    def predict_X(self, X):
        return np.exp(self.c + ((X - self.m) / self.s) @ self.th).astype(np.float32)


def fit_variants(cases):
    F = np.concatenate([c["F"] for c in cases])
    e = np.concatenate([c["y"] - c["x"] for c in cases])
    g = dict(ph=U.LaplaceGLM(1e-4, True).fit(F, e),
             ph_geo=GLMX(ridge=1e-4).fit_X(phi_me(F, True), e),
             ph_nogeo=GLMX(ridge=1e-4).fit_X(phi_me(F, False), e))
    return g


def predict(g, kind, F):
    if kind == "ph":
        return g["ph"].predict(F)
    return g[kind].predict_X(phi_me(F, kind == "ph_geo"))


# ------------------------------------------------------------------ learned: per-voxel lambda thresholds
def mix_cdf_vec(y, mu, b, lam):
    z = (y[None] - mu) / (lam[None] * b)
    return torch.where(z < 0, 0.5 * torch.exp(z.clamp(max=0)), 1 - 0.5 * torch.exp(-z.clamp(min=0))).mean(0)


def lam_thresholds(c, lo=1e-3, hi=1e3, iters=40):
    """Smallest lambda (log-bisection) at which y lies inside the central 90 % mixture interval; 0 if covered at lo,
    inf if not covered at hi."""
    y, mu, b = (torch.from_numpy(np.asarray(v, np.float64)) for v in (c["y"], c["mu"], c["b"]))
    cov = lambda lam: ((lambda F: (F >= ALPHA / 2) & (F <= 1 - ALPHA / 2))(mix_cdf_vec(y, mu, b, lam)))  # noqa: E731
    n = y.shape[0]
    l, h = torch.full((n,), np.log(lo), dtype=torch.float64), torch.full((n,), np.log(hi), dtype=torch.float64)
    c_lo, c_hi = cov(torch.exp(l)), cov(torch.exp(h))
    for _ in range(iters):
        mid = 0.5 * (l + h)
        cm = cov(torch.exp(mid))
        h = torch.where(cm, mid, h)
        l = torch.where(cm, l, mid)
    t = torch.exp(h).numpy()
    t[c_lo.numpy()] = 0.0
    t[~c_hi.numpy()] = np.inf
    return t.astype(np.float32)


def calibrate(thr_by_patient, c=CVAL):
    """thr_by_patient: list of 1-D threshold arrays (one per patient); find scalar s with mean_p mean(thr<=s) = c."""
    srt = [np.sort(t) for t in thr_by_patient]
    f = lambda s: np.mean([np.searchsorted(t, s, side="right") / len(t) for t in srt])  # noqa: E731
    lo, hi = np.log(1e-4), np.log(1e4)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(np.exp(mid)) < c:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def mix_q(mu, b, q, lam):
    return U.mix_quantile(torch.from_numpy(mu), torch.from_numpy(b), q, lam).numpy()


# ------------------------------------------------------------------ main
rows, bins_rows, consist = [], [], []
resplit_rows = []
rng_master = np.random.RandomState(20261022)
fitcal_patients = (sorted(set(C.load_roles()["FIT"]) | set(C.load_roles()["CAL"])) if "ISBI_EV" not in os.environ else
                   sorted({k.split("__")[0] for r_ in ("FIT", "CAL") for k in os.listdir(os.path.join(EV, r_, "cases"))}))
splits = []
for s in range(a.resplits):
    perm = np.random.RandomState(s).permutation(fitcal_patients)
    splits.append((set(perm[:30]), set(perm[30:])))

for m in METHODS:
    t0 = time.time()
    need_mc = m in PROB
    # FIT and CAL pooled (uniform only, both R) -- the study's FIT/CAL roles plus their union for re-splits
    FC = {}
    for role in ("FIT", "CAL"):
        for k in keys(role):
            c = load_case(role, k, m, need_mc)
            c["role"] = role
            if need_mc:
                c["thr"] = lam_thresholds(c)
                del c["mu"], c["b"]
            FC[(role, k)] = c
    fit = [c for c in FC.values() if c["role"] == "FIT"]
    cal = [c for c in FC.values() if c["role"] == "CAL"]
    log(f"{m}: FIT {len(fit)} CAL {len(cal)} loaded ({time.time() - t0:.0f}s)")
    G = fit_variants(fit)
    kap, lam = {}, {}
    for R in C.STRIDES:
        cr = [c for c in cal if c["R"] == R]
        for kind in ("ph", "ph_geo", "ph_nogeo"):
            kap[(kind, R)] = calibrate([np.abs(c["y"] - c["x"]) / predict(G, kind, c["F"]) for c in cr])
        if need_mc:
            lam[R] = calibrate([c["thr"] for c in cr])
    # consistency with the study's frozen constants
    for R in C.STRIDES:
        ref = cal_tab[(cal_tab.arm == f"{m}|posthoc") & (cal_tab.R == R) & (cal_tab.alpha == 0.9)]
        if len(ref):
            consist.append(dict(method=m, R=R, what="kappa_ph", ours=kap[("ph", R)], study=float(ref.lam.iloc[0])))
        ref = cal_tab[(cal_tab.arm == f"{m}|learned_cal") & (cal_tab.R == R) & (cal_tab.alpha == 0.9)]
        if len(ref) and need_mc:
            consist.append(dict(method=m, R=R, what="lambda", ours=lam[R], study=float(ref.lam.iloc[0])))
    log(f"{m}: kappa {[(k, round(v, 3)) for k, v in kap.items()]} lambda {lam}")

    # ---------------- re-split models (post-hoc 'ph' + learned lambda)
    rs_models = []
    for fset, cset in splits:
        cases_f = [c for c in FC.values() if c["patient"] in fset]
        cases_c = [c for c in FC.values() if c["patient"] in cset]
        Ff = np.concatenate([c["F"] for c in cases_f])
        ef = np.concatenate([c["y"] - c["x"] for c in cases_f])
        g = dict(ph=U.LaplaceGLM(1e-4, True).fit(Ff, ef)) if a.rs_kind == "ph" else dict(ph_geo=GLMX(ridge=1e-4).fit_X(phi_me(Ff, True), ef))
        kk, ll = {}, {}
        for R in C.STRIDES:
            cr = [c for c in cases_c if c["R"] == R]
            kk[R] = calibrate([np.abs(c["y"] - c["x"]) / predict(g, a.rs_kind, c["F"]) for c in cr])
            if need_mc:
                ll[R] = calibrate([c["thr"] for c in cr])
        rs_models.append((g, kk, ll))
    log(f"{m}: {len(rs_models)} re-split models ({time.time() - t0:.0f}s)")
    del FC, fit, cal

    # ---------------- TEST pass (streaming)
    for k in keys("TEST"):
        c = load_case("TEST", k, m, need_mc)
        R, y, x = c["R"], c["y"], c["x"]
        e = np.abs(y - x)
        base = dict(method=m, patient=c["patient"], pattern=c["pattern"], R=R, rep=c["rep"])
        arms = {}
        for kind in ("ph", "ph_geo", "ph_nogeo"):
            h = kap[(kind, R)] * predict(G, kind, c["F"])
            arms[kind] = (x - h, x + h)
        if need_mc:
            arms["learned"] = (mix_q(c["mu"], c["b"], ALPHA / 2, lam[R]), mix_q(c["mu"], c["b"], 1 - ALPHA / 2, lam[R]))
            thr = lam_thresholds(c)
        for arm, (lo, hi) in arms.items():
            lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
            w = hi - lo
            covd = (y >= lo) & (y <= hi)
            isc = w + (2 / ALPHA) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
            rows.append(dict(base, arm=arm, cov90=covd.mean(), width90=w.mean(), is90=isc.mean(), mae=e.mean()))
            for b0, b1 in BINS:
                sel = (c["gap"] >= b0) & (c["gap"] <= b1)
                if sel.sum() < 100:
                    continue
                bins_rows.append(dict(base, arm=arm, gap_bin=f"{b0}-{b1}", n=int(sel.sum()), cov90=covd[sel].mean(),
                                      half_width=0.5 * w[sel].mean(), mae=e[sel].mean(), e_q90=np.quantile(e[sel], 0.9),
                                      is90=isc[sel].mean()))
        # re-splits: coverage only
        for si, (g, kk, ll) in enumerate(rs_models):
            bh = predict(g, a.rs_kind, c["F"])
            r = dict(base, split=si, cov90_ph=float((e <= kk[R] * bh).mean()))
            if need_mc:
                r["cov90_learned"] = float((thr <= ll[R]).mean())
            resplit_rows.append(r)
    log(f"{m}: TEST done ({time.time() - t0:.0f}s)")

d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "case_metrics.csv"), index=False)
pd.DataFrame(bins_rows).to_csv(os.path.join(OUT, "gapbin_case.csv"), index=False)
pd.DataFrame(resplit_rows).to_csv(os.path.join(OUT, "resplit_case.csv"), index=False)
pd.DataFrame(consist).to_csv(os.path.join(OUT, "consistency.csv"), index=False)

# ------------------------------------------------------------------ summaries (patient = unit, realisations averaged)
rng = np.random.RandomState(0)


def boot(v, n=10000):
    v = np.asarray(v)
    bs = v[rng.randint(0, len(v), (n, len(v)))].mean(1)
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


pp = d.groupby(["method", "arm", "R", "pattern", "patient"], as_index=False)[["cov90", "width90", "is90", "mae"]].mean()
summ = []
for (m, arm, R), g in pp.groupby(["method", "arm", "R"]):
    w = g.pivot(index="patient", columns="pattern", values="cov90")
    wd = g.pivot(index="patient", columns="pattern", values="width90")
    isv = g.pivot(index="patient", columns="pattern", values="is90")
    dd = (w["clustered"] - w["uniform"]).values
    mu_, lo_, hi_ = boot(dd)
    summ.append(dict(method=m, arm=arm, R=R, n=len(w), cov_U=w["uniform"].mean(), cov_J=w["jittered"].mean(),
                     cov_C=w["clustered"].mean(), dcov=mu_, dcov_lo=lo_, dcov_hi=hi_,
                     width_U=wd["uniform"].mean(), width_C=wd["clustered"].mean(),
                     width_ratio=(wd["clustered"] / wd["uniform"]).mean(), is_U=isv["uniform"].mean(),
                     is_C=isv["clustered"].mean()))
S = pd.DataFrame(summ)
S.to_csv(os.path.join(OUT, "summary.csv"), index=False)

gb = pd.DataFrame(bins_rows)
gpp = gb.groupby(["method", "arm", "R", "gap_bin", "patient"], as_index=False)[["cov90", "half_width", "mae", "e_q90", "is90"]].mean()
gs = []
for (m, arm, R, b), g in gpp.groupby(["method", "arm", "R", "gap_bin"]):
    cv = boot(g.cov90.values)
    gs.append(dict(method=m, arm=arm, R=R, gap_bin=b, n_patients=len(g), cov90=cv[0], cov90_lo=cv[1], cov90_hi=cv[2],
                   half_width=g.half_width.mean(), mae=g.mae.mean(), e_q90=g.e_q90.mean(),
                   hw_over_eq90=(g.half_width / g.e_q90).mean(), is90=g.is90.mean()))
pd.DataFrame(gs).to_csv(os.path.join(OUT, "gapbin_summary.csv"), index=False)

rs = pd.DataFrame(resplit_rows)
rp = rs.groupby(["method", "R", "pattern", "patient", "split"], as_index=False).mean(numeric_only=True)
rsum = []
for (m, R, s_), g in rp.groupby(["method", "R", "split"]):
    for col in [c_ for c_ in ("cov90_ph", "cov90_learned") if c_ in g and g[c_].notna().any()]:
        w = g.pivot(index="patient", columns="pattern", values=col)
        rsum.append(dict(method=m, R=R, split=s_, arm=col.replace("cov90_", ""), cov_U=w["uniform"].mean(),
                         cov_J=w["jittered"].mean(), cov_C=w["clustered"].mean(),
                         dcov=(w["clustered"] - w["uniform"]).mean()))
RS = pd.DataFrame(rsum)
RS.to_csv(os.path.join(OUT, "resplit_splits.csv"), index=False)
RQ = RS.groupby(["method", "R", "arm"]).agg(
    **{f"{c_}_{q}": (c_, f) for c_ in ("cov_U", "cov_C", "dcov") for q, f in
       (("mean", "mean"), ("sd", "std"), ("p2.5", lambda v: np.percentile(v, 2.5)), ("p97.5", lambda v: np.percentile(v, 97.5)))})
RQ.to_csv(os.path.join(OUT, "resplit_summary.csv"))
pd.set_option("display.width", 250)
print(pd.DataFrame(consist).round(4).to_string())
print(S.round(3).to_string())
print(RQ.round(3).to_string())
log("done")
