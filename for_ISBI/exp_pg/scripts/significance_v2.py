"""Significance for the current design (exp_pg, 2026-10-28): SAINT-P / ArSSR-P / linear on BraTS and IXI-tiny, arms
learned, learned + geometry (PG), post-hoc without / with geometry. CPU only, saved per-case metrics only.

Unit = TEST patient: mask realisations averaged within patient, then the three fine-tuning seeds averaged (linear has no
seed). Outcomes per patient: dcov = cov90(clustered) - cov90(uniform), is_C and ence_C (clustered interval score and ENCE).
(1) One-factor repeated-measures ANOVA across the uncertainty arms (all share the reconstruction mean), Greenhouse-Geisser
    corrected, Friedman sensitivity -- helpers copied from scripts/significance.py (balanced RM sums of squares).
(2) Planned paired contrasts: geometry effect post hoc (ph_geo - ph_nogeo), geometry effect learned (learned_geo - learned),
    learned_geo - ph_geo; paired t + Wilcoxon, t and bootstrap 95 % CIs, Holm within each (dataset, R, outcome) family.
Inference is across patients conditional on the fitted models (seed-averaged), not across training seeds.
Outputs results/significance_v2/{anova,contrasts}.csv and REPORT.md.
"""

from project_paths import root_path
import os

import numpy as np
import pandas as pd
from scipy import stats

R_ = root_path('for_ISBI/exp_seeds/results')
OUT = root_path('for_ISBI/results/significance_v2')
os.makedirs(OUT, exist_ok=True)


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return out


def paired(d, **labels):
    d = np.asarray(d, dtype=float)
    assert len(d) >= 3 and np.isfinite(d).all()
    n, sd = len(d), d.std(ddof=1)
    t = stats.ttest_1samp(d, 0)
    ci = stats.t.interval(.95, n - 1, loc=d.mean(), scale=sd / np.sqrt(n))
    bs = np.random.default_rng(20261023).choice(d, (10000, n), replace=True).mean(axis=1)
    return dict(**labels, n=n, difference=d.mean(), sd_difference=sd, dz=d.mean() / sd, t=t.statistic, df=n - 1,
                p_t=t.pvalue, p_wilcoxon=stats.wilcoxon(d, method='auto').pvalue, ci_t_lo=ci[0], ci_t_hi=ci[1],
                ci_boot_lo=np.quantile(bs, .025), ci_boot_hi=np.quantile(bs, .975))


def rm_anova(w, **labels):
    y = w.to_numpy(float)
    assert np.isfinite(y).all()
    n, k = y.shape
    effect = n * np.square(y.mean(0) - y.mean()).sum()
    error = np.square(y - y.mean(0) - y.mean(1)[:, None] + y.mean()).sum()
    df1, df2 = k - 1, (n - 1) * (k - 1)
    f = (effect / df1) / (error / df2)
    c = np.eye(k) - np.ones((k, k)) / k
    cov = c @ np.cov(y, rowvar=False) @ c
    eps = np.clip(np.trace(cov) ** 2 / ((k - 1) * np.square(cov).sum()), 1 / (k - 1), 1)
    fried = stats.friedmanchisquare(*y.T) if k >= 3 else None
    return dict(**labels, n=n, levels=k, F=f, df1=df1, df2=df2, epsilon_GG=eps, p_uncorrected=stats.f.sf(f, df1, df2),
                p_GG=stats.f.sf(f, eps * df1, eps * df2), partial_eta2=effect / (effect + error),
                friedman_chi2=fried.statistic if fried else np.nan, p_friedman=fried.pvalue if fried else np.nan)


# ------------------------------------------------------------------ per-patient outcomes
frames = []
for d in ("brats", "ixi"):
    for recon, methods, seeds in (("SAINT-P", ("SAINT_P", "SAINT_PG"), (42, 43, 44)), ("ArSSR-P", ("ArSSR", "ArSSR_PG"), (42, 43, 44)),
                                  ("Linear", ("linear",), (42,))):
        for m in methods:
            for s in seeds:
                c = pd.read_csv(os.path.join(R_, f"{d}_s{s}_{m}", "case.csv"))
                c = c[c.pattern.isin(["uniform", "clustered"])]
                pp = c.groupby(["arm", "R", "pattern", "patient"], as_index=False)[["cov90", "is90", "ence"]].mean()
                w = pp.pivot_table(index=["arm", "R", "patient"], columns="pattern", values=["cov90", "is90", "ence"])
                o = pd.DataFrame({"dcov": w[("cov90", "clustered")] - w[("cov90", "uniform")],
                                  "is_C": w[("is90", "clustered")], "ence_C": w[("ence", "clustered")]}).reset_index()
                o["dataset"], o["recon"], o["seed"] = d, recon, s
                frames.append(o)
P = pd.concat(frames)
P = P.groupby(["dataset", "recon", "arm", "R", "patient"], as_index=False)[["dcov", "is_C", "ence_C"]].mean()   # seed average
P.to_csv(os.path.join(OUT, "patient_outcomes.csv"), index=False)

ARMS = {"SAINT-P": ["learned", "learned_geo", "ph_nogeo", "ph_geo"], "ArSSR-P": ["learned", "learned_geo", "ph_nogeo", "ph_geo"],
        "Linear": ["ph_nogeo", "ph_geo"]}
CONTRASTS = [("geometry, post hoc", "ph_geo", "ph_nogeo"), ("geometry, learned", "learned_geo", "learned"),
             ("learned+geo vs post-hoc+geo", "learned_geo", "ph_geo")]
an, co = [], []
for (d, recon, R), g in P.groupby(["dataset", "recon", "R"]):
    for y in ("dcov", "is_C", "ence_C"):
        w = g.pivot(index="patient", columns="arm", values=y)[ARMS[recon]]
        if len(ARMS[recon]) >= 3:
            an.append(rm_anova(w, dataset=d, recon=recon, R=R, outcome=y))
        for name, a1, a0 in CONTRASTS:
            if a1 in w and a0 in w:
                co.append(paired(w[a1] - w[a0], dataset=d, recon=recon, R=R, outcome=y, contrast=name,
                                 mean_a=w[a1].mean(), mean_b=w[a0].mean()))
A = pd.DataFrame(an)
A["p_GG_holm"] = holm(A.p_GG)
A["p_friedman_holm"] = holm(A.p_friedman)
A.to_csv(os.path.join(OUT, "anova.csv"), index=False)
Cc = pd.DataFrame(co)
Cc["p_t_holm"] = np.nan
Cc["p_wilcoxon_holm"] = np.nan
for _, g in Cc.groupby(["dataset", "R", "outcome"]):
    Cc.loc[g.index, "p_t_holm"] = holm(g.p_t)
    Cc.loc[g.index, "p_wilcoxon_holm"] = holm(g.p_wilcoxon)
Cc.to_csv(os.path.join(OUT, "contrasts.csv"), index=False)

# ------------------------------------------------------------------ report
lines = ["# Significance analysis v2 (current design: SAINT-P / ArSSR-P / linear x learned / learned+geometry / post hoc +-geometry)", "",
         "CPU only, from saved per-case metrics (exp_seeds/results). Unit = TEST patient (BraTS 51, IXI-tiny 60): mask realisations "
         "averaged within patient, then three fine-tuning seeds averaged (linear: single run). Outcomes: dcov = clustered minus uniform "
         "90 % coverage; is_C and ence_C = clustered interval score and ENCE (lower is better). One-factor repeated-measures ANOVA across "
         "uncertainty arms sharing one reconstruction (GG-corrected, Friedman sensitivity; Holm over all ANOVA rows). Planned paired "
         "contrasts with Holm correction within each (dataset, stride, outcome) family. Retrospective, exploratory; inference is across "
         "patients conditional on the fitted (seed-averaged) models, not across training seeds.", "",
         "## ANOVA across uncertainty arms", "", "| Dataset | Recon | R | Outcome | F | partial eta2 | GG eps | Holm p (GG) | Holm p (Friedman) |",
         "|---|---|---:|---|---:|---:|---:|---:|---:|"]
for r in A.itertuples():
    lines.append(f"| {r.dataset} | {r.recon} | {r.R} | {r.outcome} | {r.F:.1f} | {r.partial_eta2:.2f} | {r.epsilon_GG:.2f} | "
                 f"{r.p_GG_holm:.1e} | {r.p_friedman_holm:.1e} |")
lines += ["", "## Planned contrasts (difference = first arm minus second; dcov in coverage units)", "",
          "| Dataset | Recon | R | Outcome | Contrast | Difference | Bootstrap 95% CI | dz | Holm p (t) | Holm p (Wilcoxon) |",
          "|---|---|---:|---|---|---:|---:|---:|---:|---:|"]
for r in Cc.itertuples():
    lines.append(f"| {r.dataset} | {r.recon} | {r.R} | {r.outcome} | {r.contrast} | {r.difference:+.3f} | "
                 f"[{r.ci_boot_lo:+.3f}, {r.ci_boot_hi:+.3f}] | {r.dz:+.2f} | {r.p_t_holm:.1e} | {r.p_wilcoxon_holm:.1e} |")
lines += ["", "Reproduce: `python for_ISBI/exp_pg/scripts/significance_v2.py` "
          "(after metrics_seed.py / metrics_pg.py)."]
open(os.path.join(OUT, "REPORT.md"), "w").write("\n".join(lines) + "\n")
print(f"ANOVA rows {len(A)}; significant (Holm GG<0.05): {(A.p_GG_holm < 0.05).sum()}")
print(Cc[["dataset", "recon", "R", "outcome", "contrast", "difference", "ci_boot_lo", "ci_boot_hi", "p_t_holm"]].round(4).to_string())
