"""Consolidate every uncertainty arm (main study + all follow-ups) into one patient-level table for the deck/report.

Reads only saved result CSVs. Unit = patient (realisations averaged). Outputs results/geometry_story/:
  arms_patient.csv   (arm, R, pattern, patient, cov90, width90)
  arms_summary.csv   (arm metadata + cov90 per pattern, dcov90 clustered-uniform with 95 % bootstrap CI, width ratio)
"""
import os

import numpy as np
import pandas as pd

from isbi import common as C

RES = os.path.join(C.ISBI, "results")
OUT = os.path.join(RES, "geometry_story")
os.makedirs(OUT, exist_ok=True)
rows = []


def add(df, arm, cov="cov90", wid="width90"):
    d = df[["patient", "pattern", "R", cov, wid]].rename(columns={cov: "cov90", wid: "width90"})
    d["arm"] = arm
    rows.append(d)


m = pd.read_csv(os.path.join(RES, "test_case_metrics.csv"))
for arm, key in [("SAINT-P learned", "SAINT_P|learned_cal"), ("I3Net-P learned", "I3Net_P|learned_cal"),
                 ("SAINT-P post-hoc", "SAINT_P|posthoc"), ("I3Net-P post-hoc", "I3Net_P|posthoc"),
                 ("linear post-hoc", "linear|posthoc"), ("SAINT-P constant width", "SAINT_P|const")]:
    add(m[m.arm == key], arm)
ex = pd.read_csv(os.path.join(RES, "exposure", "case_metrics.csv"))
add(ex[(ex.model == "saint_expa") & (ex.kind == "learned_cal")], "SAINT exposure learned")
add(ex[(ex.model == "i3net_expa") & (ex.kind == "learned_cal")], "I3Net exposure learned")
g = pd.read_csv(os.path.join(RES, "geo", "case_metrics.csv"))
add(g, "SAINT-PG learned", "cov90_cal", "width90_cal")
gx = pd.read_csv(os.path.join(RES, "saint_geo_mixed", "case_metrics.csv"))
add(gx, "SAINT-PGX learned", "cov90_cal", "width90_cal")
na = pd.read_csv(os.path.join(RES, "native", "case_metrics.csv"))
add(na[(na.model == "arssr") & (na.kind == "learned_cal")], "ArSSR-P learned")
add(na[(na.model == "arssr") & (na.kind == "posthoc")], "ArSSR-P post-hoc")
ab = pd.read_csv(os.path.join(RES, "ablation", "case_metrics.csv"))
add(ab[(ab["mean"] == "SAINT_P") & (ab.variant == "no_geom")], "SAINT-P post-hoc, no geometry")
add(ab[(ab["mean"] == "I3Net_P") & (ab.variant == "no_geom")], "I3Net-P post-hoc, no geometry")
add(ab[(ab["mean"] == "linear") & (ab.variant == "no_geom")], "linear post-hoc, no geometry")
gc = pd.read_csv(os.path.join(RES, "gapcal", "case_metrics.csv"))
for mm, lab in (("ArSSR_P", "ArSSR-P"), ("linear", "linear"), ("SAINT_P", "SAINT-P"), ("I3Net_P", "I3Net-P")):
    add(gc[(gc["mean"] == mm) & (gc.kind == "gap_conditional")], f"{lab} post-hoc, gap-conditional")

META = {  # arm -> (uncertainty type, trained on irregular masks, uses gap geometry, reconstruction input)
    "SAINT-P learned": ("learned", "no", "no", "adapter"), "I3Net-P learned": ("learned", "no", "no", "adapter"),
    "SAINT exposure learned": ("learned", "yes", "no", "adapter"), "I3Net exposure learned": ("learned", "yes", "no", "adapter"),
    "SAINT-PG learned": ("learned", "no", "yes", "adapter"), "SAINT-PGX learned": ("learned", "yes", "yes", "adapter"),
    "ArSSR-P learned": ("learned", "yes", "partial", "native"),
    "SAINT-P post-hoc": ("post-hoc", "no", "yes", "adapter"), "I3Net-P post-hoc": ("post-hoc", "no", "yes", "adapter"),
    "ArSSR-P post-hoc": ("post-hoc", "-", "yes", "native"), "linear post-hoc": ("post-hoc", "-", "yes", "native"),
    "SAINT-P post-hoc, no geometry": ("post-hoc", "no", "no", "adapter"),
    "I3Net-P post-hoc, no geometry": ("post-hoc", "no", "no", "adapter"),
    "linear post-hoc, no geometry": ("post-hoc", "-", "no", "native"),
    "SAINT-P constant width": ("constant", "no", "no", "adapter"),
    "ArSSR-P post-hoc, gap-conditional": ("post-hoc (gap-cond.)", "-", "yes", "native"),
    "linear post-hoc, gap-conditional": ("post-hoc (gap-cond.)", "-", "yes", "native"),
    "SAINT-P post-hoc, gap-conditional": ("post-hoc (gap-cond.)", "no", "yes", "adapter"),
    "I3Net-P post-hoc, gap-conditional": ("post-hoc (gap-cond.)", "no", "yes", "adapter"),
}
d = pd.concat(rows)
pp = d.groupby(["arm", "R", "pattern", "patient"], as_index=False)[["cov90", "width90"]].mean()
pp.to_csv(os.path.join(OUT, "arms_patient.csv"), index=False)
rng = np.random.RandomState(0)
summ = []
for (arm, R), gg in pp.groupby(["arm", "R"]):
    w = gg.pivot(index="patient", columns="pattern", values="cov90")
    wd = gg.pivot(index="patient", columns="pattern", values="width90")
    dd = (w["clustered"] - w["uniform"]).values
    bs = dd[rng.randint(0, len(dd), (10000, len(dd)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    cb = w["clustered"].values[rng.randint(0, len(w), (10000, len(w)))].mean(1)
    t, irr, geo, inp = META[arm]
    summ.append(dict(arm=arm, R=R, uq=t, trained_irregular=irr, geometry=geo, input=inp, n=len(w),
                     cov90_uniform=w["uniform"].mean(), cov90_jittered=w["jittered"].mean(),
                     cov90_clustered=w["clustered"].mean(), clustered_lo=np.percentile(cb, 2.5),
                     clustered_hi=np.percentile(cb, 97.5), dcov90=dd.mean(), dcov90_lo=lo, dcov90_hi=hi,
                     width_ratio=(wd["clustered"] / wd["uniform"]).mean()))
s = pd.DataFrame(summ)
s.to_csv(os.path.join(OUT, "arms_summary.csv"), index=False)
pd.set_option("display.width", 220)
print(s[s.R == 4][["arm", "uq", "trained_irregular", "geometry", "input", "cov90_uniform", "cov90_clustered", "dcov90", "width_ratio"]].round(3).to_string())
