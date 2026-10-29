"""Option 3 analysis: SAINT-PG (geometry-aware learned scale) vs SAINT-P learned and post-hoc, shared mean.

Calibration exactly as analyze.py (lambda on b_t, patient-averaged CAL uniform coverage = alpha, per R).
Outputs results/geo/{case_metrics,summary,primary}.csv
"""
import glob
import os

import numpy as np
import pandas as pd
import torch

from isbi import common as C
from isbi import uq as U

TAG = os.environ.get("GEO_TAG", "saint_geo")
EV, EG = os.path.join(C.ISBI, "runs", "eval"), os.path.join(C.ISBI, "runs", "eval_geo" if TAG == "saint_geo" else f"eval_{TAG}")
OUT = os.path.join(C.ISBI, "results", "geo" if TAG == "saint_geo" else TAG)
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(16)


def merged(role):
    for f in sorted(glob.glob(os.path.join(EG, role, "*.npz"))):
        c = dict(np.load(os.path.join(EV, role, "cases", os.path.basename(f))))
        c.update(dict(np.load(f)))
        yield c


def cover(c, m, a, lam):
    F = U.mix_cdf(torch.from_numpy(c["y"]), torch.from_numpy(c[f"{m}__mu_t"]), torch.from_numpy(c[f"{m}__b_t"]), lam).numpy()
    return (F >= (1 - a) / 2) & (F <= (1 + a) / 2)


def width(c, m, a, lam):
    mu, b = torch.from_numpy(c[f"{m}__mu_t"]), torch.from_numpy(c[f"{m}__b_t"])
    lo = U.mix_quantile(mu, b, (1 - a) / 2, lam).numpy()
    hi = U.mix_quantile(mu, b, (1 + a) / 2, lam).numpy()
    return np.clip(hi, 0, 1) - np.clip(lo, 0, 1)


cal = list(merged("CAL"))
lam = {}
for R in C.STRIDES:
    cs = [c for c in cal if int(c["R"]) == R]
    for a in U.LEVELS:
        lam[(R, a)] = U.calibrate_scalar(lambda c, l: cover(c, "SAINT_PG", a, l).mean(), cs, a, 1e-2, 1e2)
pd.Series({f"x{k[0]}_{k[1]}": v for k, v in lam.items()}).to_csv(os.path.join(OUT, "calibration.csv"))
rows = []
for c in merged("TEST"):
    R = int(c["R"])
    e = c["y"] - c["SAINT_P__x"]
    s = c["SAINT_PG__epi"] + c["SAINT_PG__alea"]
    r = dict(patient=c["patient"].item(), pattern=c["pattern"].item(), R=R, rep=int(c["rep"]),
             mean_check=float(c["mean_check_maxabs"]), spearman=U.spearman(np.abs(e), s),
             strat_spearman=U.strat_spearman(e, s, c["gap_mm"], c["near_mm"]), ause_norm=U.ause(e, s)[0] / np.abs(e).mean())
    for a in U.LEVELS:
        for kind, l_ in (("raw", 1.0), ("cal", lam[(R, a)])):
            r[f"cov{int(a * 100)}_{kind}"] = float(cover(c, "SAINT_PG", a, l_).mean())
            if a == 0.9:
                r[f"width90_{kind}"] = float(width(c, "SAINT_PG", a, l_).mean())
    rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "case_metrics.csv"), index=False)
cols = [c for c in d.columns if c.startswith(("cov", "width", "spearman", "strat", "ause"))]
pp = d.groupby(["R", "pattern", "patient"], as_index=False)[cols].mean()
pp.groupby(["R", "pattern"])[cols].mean().to_csv(os.path.join(OUT, "summary.csv"))
# comparison with the frozen SAINT-P arms (same patients, same mean)
ref = pd.read_csv(os.path.join(C.ISBI, "results", "patient_level_intervals.csv"))
ref = ref[ref.arm.isin(["SAINT_P|learned_cal", "SAINT_P|posthoc"])]
rng = np.random.RandomState(0)
prim = []
for R in C.STRIDES:
    g = pp[pp.R == R].pivot(index="patient", columns="pattern", values="cov90_cal")
    dPG = g["clustered"] - g["uniform"]
    out = dict(R=R, n=len(dPG), cov90_uniform=g["uniform"].mean(), cov90_clustered=g["clustered"].mean(),
               dcov90_PG=dPG.mean())
    for arm, k in (("SAINT_P|learned_cal", "L"), ("SAINT_P|posthoc", "PH")):
        h = ref[(ref.arm == arm) & (ref.R == R)].pivot(index="patient", columns="pattern", values="cov90")
        dd = (dPG - (h["clustered"] - h["uniform"])).dropna().values
        bs = dd[rng.randint(0, len(dd), (10000, len(dd)))].mean(1)
        out[f"PG_minus_{k}"] = dd.mean()
        out[f"PG_minus_{k}_ci"] = f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]"
    w = pp[pp.R == R].pivot(index="patient", columns="pattern", values="width90_cal")
    out["width_ratio_PG"] = (w["clustered"] / w["uniform"]).mean()
    prim.append(out)
prim = pd.DataFrame(prim)
prim.to_csv(os.path.join(OUT, "primary.csv"), index=False)
pd.set_option("display.width", 220)
print(prim.round(3).to_string())
print(pp.groupby(["R", "pattern"])[cols].mean().round(3).to_string())
print("max shared-mean deviation", d.mean_check.max())
