"""Manuscript-only summary (no training, no inference): 90 % interval width and interval score per arm,
recomputed from the SAVED TEST voxel outputs of the for_ISBI study.

Reads (read-only):  runs/eval/{CAL,FIT,TEST}/cases/*.npz, runs/eval_geo/{CAL,TEST}/*.npz,
                    runs/eval_native/arssr/{FIT,CAL,TEST}/cases/*.npz,
                    results/posthoc_models.pkl, results/calibration.csv, results/geo/calibration.csv,
                    results/gapcal/case_metrics.csv (kappa_4, kappa_8, beta)
Writes:             manuscript_4page/data/interval_scores_{case,patient,summary}.csv

Interval score at nominal coverage c (Gneiting & Raftery 2007), with a = 1 - c:
    IS_c = (u - l) + (2/a) (l - y)_+ + (2/a) (y - u)_+          (lower is better)
Intervals are intersected with the intensity support [0, 1] exactly as the study's reported widths.
Coverage is recomputed as a consistency check against results/geometry_story/arms_summary.csv.
"""

from project_paths import root_path
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

ISBI = root_path('for_ISBI')
sys.path.insert(0, os.path.join(ISBI, "src"))
from isbi import uq as U  # noqa: E402

torch.set_num_threads(8)
OUT = os.path.join(ISBI, "manuscript_4page", "data")
os.makedirs(OUT, exist_ok=True)
C90, A = 0.9, 0.1
EV = os.path.join(ISBI, "runs", "eval")
EG = os.path.join(ISBI, "runs", "eval_geo")
EN = os.path.join(ISBI, "runs", "eval_native", "arssr")

glms = pickle.load(open(os.path.join(ISBI, "results", "posthoc_models.pkl"), "rb"))
cal = pd.read_csv(os.path.join(ISBI, "results", "calibration.csv"))
lam = {(r.R, r.arm): r.lam for r in cal[cal.alpha == C90].itertuples()}
geo_cal = pd.read_csv(os.path.join(ISBI, "results", "geo", "calibration.csv"), index_col=0).iloc[:, 0]
lam_pg = {R: float(geo_cal[f"x{R}_0.9"]) for R in (4, 8)}
gc = pd.read_csv(os.path.join(ISBI, "results", "gapcal", "case_metrics.csv"))
gpar = gc.groupby("mean")[["kappa4", "kappa8", "beta"]].first()


def load(role, native=False):
    out = []
    for f in sorted(glob.glob(os.path.join(EV, role, "cases", "*.npz"))):
        c = dict(np.load(f))
        if native:
            g = os.path.join(EN, role, "cases", os.path.basename(f))
            c.update(dict(np.load(g)))
        out.append(c)
    return out


# ArSSR-P: post-hoc GLM refitted on FIT and lambda/kappa calibrated on CAL exactly as analyze_native.py / gapcal_posthoc.py
fit_n, cal_n = load("FIT", True), load("CAL", True)
glms["ArSSR_P"] = U.LaplaceGLM(1e-4, True).fit(np.concatenate([c["NAT__feat"] for c in fit_n]),
                                               np.concatenate([c["y"] - c["NAT__x"] for c in fit_n]))


def cov_l(c, pre, l_):
    F = U.mix_cdf(torch.from_numpy(c["y"]), torch.from_numpy(c[pre + "mu_t"]), torch.from_numpy(c[pre + "b_t"]), l_).numpy()
    return (F >= A / 2) & (F <= 1 - A / 2)


lam_ar, kap_ar = {}, {}
for R in (4, 8):
    cs = [c for c in cal_n if int(c["R"]) == R]
    lam_ar[R] = U.calibrate_scalar(lambda c, l_: cov_l(c, "NAT__", l_).mean(), cs, C90, 1e-2, 1e2)
    pre = [(np.abs(c["y"] - c["NAT__x"]), glms["ArSSR_P"].predict(c["NAT__feat"])) for c in cs]
    kap_ar[R] = U.calibrate_scalar(lambda q, l_: (q[0] <= l_ * q[1]).mean(), pre, C90, 1e-2, 1e2)
del fit_n, cal_n
print("ArSSR lambda", lam_ar, "kappa", kap_ar, flush=True)


def score(y, lo, hi):
    lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    w = hi - lo
    pen = (2 / A) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    return dict(cov90=float(((y >= lo) & (y <= hi)).mean()), width90=float(w.mean()), is90=float((w + pen).mean()))


def learned(y, mu, b, l_):
    mu, b = torch.from_numpy(mu), torch.from_numpy(b)
    lo = U.mix_quantile(mu, b, A / 2, l_).numpy()
    hi = U.mix_quantile(mu, b, 1 - A / 2, l_).numpy()
    return lo, hi


rows = []
files = sorted(glob.glob(os.path.join(EV, "TEST", "cases", "*.npz")))
for k, f in enumerate(files):
    c = dict(np.load(f))
    key = os.path.basename(f)
    c.update(dict(np.load(os.path.join(EN, "TEST", "cases", key))))
    c.update(dict(np.load(os.path.join(EG, "TEST", key))))
    R, y, gap = int(c["R"]), c["y"], c["gap_mm"]
    base = dict(patient=c["patient"].item(), pattern=c["pattern"].item(), R=R, rep=int(c["rep"]))
    arms = {}
    for a in ("SAINT_P", "I3Net_P"):
        arms[f"{a}|learned_cal"] = learned(y, c[f"{a}__mu_t"], c[f"{a}__b_t"], lam[(R, f"{a}|learned_cal")])
    arms["SAINT_PG|learned_cal"] = learned(y, c["SAINT_PG__mu_t"], c["SAINT_PG__b_t"], lam_pg[R])
    arms["ArSSR_P|learned_cal"] = learned(y, c["NAT__mu_t"], c["NAT__b_t"], lam_ar[R])
    for m in ("SAINT_P", "I3Net_P", "linear", "ArSSR_P"):
        x, F = (c["NAT__x"], c["NAT__feat"]) if m == "ArSSR_P" else (c[f"{m}__x"], c[f"{m}__feat"])
        bh = glms[m].predict(F)
        kg = kap_ar[R] if m == "ArSSR_P" else lam[(R, f"{m}|posthoc")]
        arms[f"{m}|posthoc"] = (x - kg * bh, x + kg * bh)
        p = gpar.loc[m]
        kc = p.kappa4 * (np.maximum(gap, 1.0) / 4.0) ** p.beta
        arms[f"{m}|posthoc_gapcond"] = (x - kc * bh, x + kc * bh)
    for arm, (lo, hi) in arms.items():
        rows.append(dict(base, arm=arm, **score(y, lo, hi)))
    if k % 50 == 0:
        print(k, len(files), flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "interval_scores_case.csv"), index=False)
pp = d.groupby(["arm", "R", "pattern", "patient"], as_index=False)[["cov90", "width90", "is90"]].mean()
pp.to_csv(os.path.join(OUT, "interval_scores_patient.csv"), index=False)
rng = np.random.RandomState(0)
out = []
for (arm, R, pat), g in pp.groupby(["arm", "R", "pattern"]):
    r = dict(arm=arm, R=R, pattern=pat, n=len(g))
    for col in ("cov90", "width90", "is90"):
        v = g[col].values
        bs = v[rng.randint(0, len(v), (10000, len(v)))].mean(1)
        r[col] = v.mean()
        r[col + "_lo"], r[col + "_hi"] = np.percentile(bs, [2.5, 97.5])
    out.append(r)
s = pd.DataFrame(out)
s.to_csv(os.path.join(OUT, "interval_scores_summary.csv"), index=False)
pd.set_option("display.width", 220)
print(s.round(4).to_string())
