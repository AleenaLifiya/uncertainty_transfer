"""Gap-conditional calibration of the post-hoc intervals (exploratory, added 2026-10-20 at the user's request).

Same post-hoc GLMs as the main analysis (results/posthoc_models.pkl; ArSSR-P refitted on FIT exactly as in
analyze_native.py). Calibration uses ONLY uniform CAL data, pooled over x4 (gap 4 mm) and x8 (gap 8 mm):
  kappa_90(4 mm), kappa_90(8 mm)  from the patient-mean coverage target on each, then
  kappa(g) = kappa(4) * (g / 4) ** beta, beta = log(kappa(8)/kappa(4)) / log 2     (log-log line through the 2 points)
and every TEST voxel uses kappa at its own bracketing-gap length (x4 and x8 alike). Compared with the frozen global
per-R kappa. No irregular data is used for fitting or calibration.
Outputs results/gapcal/{summary,dcov90}.csv
"""
import glob
import os
import pickle

import numpy as np
import pandas as pd

from isbi import common as C
from isbi import uq as U

EV = os.path.join(C.ISBI, "runs", "eval")
EN = os.path.join(C.ISBI, "runs", "eval_native", "arssr")
OUT = os.path.join(C.ISBI, "results", "gapcal")
os.makedirs(OUT, exist_ok=True)
glms = pickle.load(open(os.path.join(C.ISBI, "results", "posthoc_models.pkl"), "rb"))
cal_tab = pd.read_csv(os.path.join(C.ISBI, "results", "calibration.csv"))


def load(role, native=False):
    out = []
    for f in sorted(glob.glob(os.path.join(EV, role, "cases", "*.npz"))):
        c = dict(np.load(f))
        if native:
            g = os.path.join(EN, role, "cases", os.path.basename(f))
            if not os.path.exists(g):
                continue
            c.update(dict(np.load(g)))
        out.append(c)
    return out


def x_feat(c, m):
    return (c["NAT__x"], c["NAT__feat"]) if m == "ArSSR_P" else (c[f"{m}__x"], c[f"{m}__feat"])


fit, cal, test = load("FIT", True), load("CAL", True), load("TEST", True)
glms["ArSSR_P"] = U.LaplaceGLM(1e-4, True).fit(np.concatenate([c["NAT__feat"] for c in fit]),
                                               np.concatenate([c["y"] - c["NAT__x"] for c in fit]))
rows = []
for m in ("SAINT_P", "I3Net_P", "ArSSR_P", "linear"):
    pre = {R: [] for R in C.STRIDES}
    for c in cal:
        x, F = x_feat(c, m)
        pre[int(c["R"])].append((np.abs(c["y"] - x), glms[m].predict(F)))
    k = {R: U.calibrate_scalar(lambda q, l: (q[0] <= l * q[1]).mean(), pre[R], 0.9, 1e-2, 1e2) for R in C.STRIDES}
    beta = np.log(k[8] / k[4]) / np.log(2.0)
    kappa_g = lambda g: k[4] * (np.maximum(g, 1.0) / 4.0) ** beta  # noqa: E731
    glob_k = {R: (cal_tab[(cal_tab.arm == f"{m}|posthoc") & (cal_tab.R == R) & (cal_tab.alpha == 0.9)].lam.iloc[0]
                  if m != "ArSSR_P" else k[R]) for R in C.STRIDES}
    for c in test:
        R = int(c["R"])
        x, F = x_feat(c, m)
        e = np.abs(c["y"] - x)
        b = glms[m].predict(F)
        for kind, h in (("global", glob_k[R] * b), ("gap_conditional", kappa_g(c["gap_mm"]) * b)):
            rows.append(dict(mean=m, kind=kind, patient=c["patient"].item(), pattern=c["pattern"].item(), R=R,
                             cov90=float((e <= h).mean()),
                             width90=float((np.clip(x + h, 0, 1) - np.clip(x - h, 0, 1)).mean()),
                             kappa4=k[4], kappa8=k[8], beta=beta))
    print(m, "kappa(4mm)=%.3f kappa(8mm)=%.3f beta=%.3f" % (k[4], k[8], beta), flush=True)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "case_metrics.csv"), index=False)
pp = d.groupby(["mean", "kind", "R", "pattern", "patient"], as_index=False)[["cov90", "width90"]].mean()
summ = pp.groupby(["mean", "kind", "R", "pattern"])[["cov90", "width90"]].mean()
summ.to_csv(os.path.join(OUT, "summary.csv"))
rng = np.random.RandomState(0)
out = []
for (m, kind, R), g in pp.groupby(["mean", "kind", "R"]):
    w = g.pivot(index="patient", columns="pattern", values="cov90")
    dd = (w["clustered"] - w["uniform"]).values
    bs = dd[rng.randint(0, len(dd), (10000, len(dd)))].mean(1)
    out.append(dict(mean=m, kind=kind, R=R, cov90_uniform=w["uniform"].mean(), cov90_jittered=w["jittered"].mean(),
                    cov90_clustered=w["clustered"].mean(), dcov90=dd.mean(),
                    ci=f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]"))
out = pd.DataFrame(out)
out.to_csv(os.path.join(OUT, "dcov90.csv"), index=False)
pd.set_option("display.width", 200)
print(out.round(3).to_string())
