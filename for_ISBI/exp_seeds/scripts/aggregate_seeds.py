"""Aggregate exp_seeds/results/<dataset>_s<seed>_<method>/summary.csv over the three fine-tuning seeds (42, 43, 44).
For each (dataset, method, arm, R, pattern): mean and SD ACROSS SEEDS of the patient-averaged metric (each seed's value is the
mean over TEST patients). Linear interpolation has no training seed: its single (s42) value is reported with SD = 0.
Also: clustered-minus-uniform coverage change per seed, and gap-bin coverage per seed.
Writes exp_seeds/results/seeds_summary.csv, seeds_dcov.csv, seeds_gapbin.csv.
"""

from project_paths import root_path
import os

import numpy as np
import pandas as pd

R_ = root_path('for_ISBI/exp_seeds/results')
MET = ["cov90", "width90", "is90", "mae", "ause_norm", "uce_e3", "ence"]
rows, gb = [], []
for d in ("brats", "ixi"):
    for m in ("SAINT_P", "ArSSR", "linear", "SAINT_PG", "ArSSR_PG"):
        for s in ((42, 43, 44) if m != "linear" else (42,)):
            f = os.path.join(R_, f"{d}_s{s}_{m}", "summary.csv")
            t = pd.read_csv(f)
            t["seed"] = s
            rows.append(t)
            c = pd.read_csv(os.path.join(R_, f"{d}_s{s}_{m}", "case.csv"))
            pp = c.groupby(["arm", "R", "pattern", "patient"], as_index=False).cov90.mean()
            w = pp.pivot_table(index=["arm", "R", "patient"], columns="pattern", values="cov90").reset_index()
            w["dcov"] = w["clustered"] - w["uniform"]
            pw = c.groupby(["arm", "R", "pattern", "patient"], as_index=False).width90.mean()
            ww = pw.pivot_table(index=["arm", "R", "patient"], columns="pattern", values="width90").reset_index()
            ww["wr"] = ww["clustered"] / ww["uniform"]
            for (arm, R), g in w.groupby(["arm", "R"]):
                wr = ww[(ww.arm == arm) & (ww.R == R)].wr.mean()
                gb.append(dict(dataset=d, method=m, arm=arm, R=R, seed=s, dcov=g.dcov.mean(), width_ratio=wr))
S = pd.concat(rows)
agg = S.groupby(["dataset", "method", "arm", "R", "pattern"]).agg(
    n_seeds=("seed", "nunique"), **{f"{k}_mean": (k, "mean") for k in MET}, **{f"{k}_sd": (k, "std") for k in MET}).reset_index()
agg = agg.fillna(0.0)
agg.to_csv(os.path.join(R_, "seeds_summary.csv"), index=False)
D = pd.DataFrame(gb)
D.groupby(["dataset", "method", "arm", "R"]).agg(dcov_mean=("dcov", "mean"), dcov_sd=("dcov", "std"), wr_mean=("width_ratio", "mean"),
                                                 wr_sd=("width_ratio", "std")).fillna(0).reset_index().to_csv(
    os.path.join(R_, "seeds_dcov.csv"), index=False)
S.to_csv(os.path.join(R_, "seeds_per_seed.csv"), index=False)
pd.set_option("display.width", 250)
c = agg[agg.pattern.isin(["uniform", "clustered"])]
for k in ("cov90", "is90", "ause_norm", "ence", "uce_e3"):
    t = c.copy()
    t[k] = t[f"{k}_mean"].round(3).astype(str) + "±" + t[f"{k}_sd"].round(3).astype(str)
    print(k)
    print(t.pivot_table(index=["dataset", "method", "arm"], columns=["R", "pattern"], values=k, aggfunc="first").to_string())
