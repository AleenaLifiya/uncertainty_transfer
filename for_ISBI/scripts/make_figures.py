"""Patient-level aggregation, primary endpoint, tables and quantitative figures (TEST).

    python scripts/make_figures.py   (reads results/test_*.csv; writes results/*.csv, results/tables/*.tex,
                                      figures/fig_*.{pdf,png})
Unit = patient: jittered / clustered realisations are averaged within patient first.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from isbi import common as C  # noqa: E402
from isbi import figstyle as FS  # noqa: E402

RES = os.environ.get("ISBI_RES", os.path.join(C.ISBI, "results"))
FIG = os.environ.get("ISBI_FIG", os.path.join(C.ISBI, "figures"))
TAB = os.path.join(RES, "tables")
for d in (FIG, TAB):
    os.makedirs(d, exist_ok=True)
FS.use()
B = 10000
rng = np.random.RandomState(0)
PATS = ["uniform", "jittered", "clustered"]
ARCH = {"SAINT": "SAINT", "I3Net": "I3Net"}
ARMLAB = {"learned_raw": "learned (raw)", "learned_cal": "learned (calibrated)", "posthoc": "post-hoc",
          "const": "constant width"}
SCORELAB = {"learned_total": "learned σ", "posthoc": "post-hoc b̂", "near_mm": "nearest-slice dist.",
            "gap_mm": "gap length", "random": "random", "learned_epi": "learned σ_ep", "learned_alea": "learned σ_al"}

cm = pd.read_csv(os.path.join(RES, "test_case_metrics.csv"))
rec = pd.read_csv(os.path.join(RES, "test_recon_metrics.csv"))


def boot_ci(x, b=B):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    m = rng.randint(0, len(x), (b, len(x)))
    s = x[m].mean(1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def per_patient(df, cols, keys):
    return df.groupby(keys + ["patient"], as_index=False)[cols].mean()


# ============================================================================ recon table
rp = per_patient(rec, ["psnr", "ssim", "mae", "nmse", "mae_fg_missing"], ["method", "pattern", "R"])
rows = []
for (m, pat, R), g in rp.groupby(["method", "pattern", "R"]):
    r = dict(method=m, pattern=pat, R=R, n_patients=len(g))
    for c in ("psnr", "ssim", "mae", "nmse"):
        r[c + "_mean"], r[c + "_sd"] = g[c].mean(), g[c].std()
    rows.append(r)
recon_tab = pd.DataFrame(rows)
recon_tab.to_csv(os.path.join(RES, "table_recon.csv"), index=False)

# ============================================================================ interval table
iv = cm[cm.kind.isin(["learned_raw", "learned_cal", "posthoc", "const"])].copy()
for a in (50, 80, 90, 95):
    iv[f"dev{a}"] = iv[f"cov{a}"] - a / 100
ivp = per_patient(iv, [c for c in iv.columns if c.startswith(("cov", "width", "dev"))] + ["mae"],
                  ["arm", "mean", "kind", "pattern", "R"])
rows = []
for (arm, mean, kind, pat, R), g in ivp.groupby(["arm", "mean", "kind", "pattern", "R"]):
    r = dict(arm=arm, mean=mean, kind=kind, pattern=pat, R=R, n_patients=len(g))
    for c in ("cov50", "cov80", "cov90", "cov95", "width90", "width95", "dev90"):
        r[c + "_mean"], r[c + "_sd"] = g[c].mean(), g[c].std()
        lo, hi = boot_ci(g[c])
        r[c + "_lo"], r[c + "_hi"] = lo, hi
    r["absdev90_mean"] = g.dev90.abs().mean()
    rows.append(r)
int_tab = pd.DataFrame(rows)
int_tab.to_csv(os.path.join(RES, "table_intervals.csv"), index=False)

# ============================================================================ ranking table
rk = cm[cm.kind == "rank"].copy()
rkp = per_patient(rk, ["spearman", "strat_spearman", "ause", "ause_norm", "auprc", "prevalence", "mae"],
                  ["mean", "score", "pattern", "R"])
rows = []
for (mean, score, pat, R), g in rkp.groupby(["mean", "score", "pattern", "R"]):
    r = dict(mean=mean, score=score, pattern=pat, R=R, n_patients=len(g))
    for c in ("spearman", "strat_spearman", "ause_norm", "auprc", "prevalence"):
        r[c + "_mean"], r[c + "_sd"] = g[c].mean(), g[c].std()
        r[c + "_lo"], r[c + "_hi"] = boot_ci(g[c])
    rows.append(r)
rank_tab = pd.DataFrame(rows)
rank_tab.to_csv(os.path.join(RES, "table_ranking.csv"), index=False)

# ============================================================================ primary endpoint
prim = []
for a in ARCH:
    for R in C.STRIDES:
        d = {}
        for kind in ("learned_cal", "posthoc", "learned_raw", "const"):
            g = ivp[(ivp.arm == f"{a}_P|{kind}") & (ivp.R == R)].pivot(index="patient", columns="pattern", values="cov90")
            w = ivp[(ivp.arm == f"{a}_P|{kind}") & (ivp.R == R)].pivot(index="patient", columns="pattern", values="width90")
            d[kind] = (g["clustered"] - g["uniform"], w["clustered"] / w["uniform"], g)
        D = (d["learned_cal"][0] - d["posthoc"][0]).dropna()
        for kind in d:
            x = d[kind][0].dropna()
            lo, hi = boot_ci(x)
            prim.append(dict(arch=a, R=R, contrast=f"dcov90[{kind}] clustered-uniform", n=len(x), mean=x.mean(),
                             sd=x.std(), ci_lo=lo, ci_hi=hi, width_ratio_clu_over_uni=d[kind][1].mean(),
                             frac_patients_negative=float((x < 0).mean())))
        lo, hi = boot_ci(D)
        prim.append(dict(arch=a, R=R, contrast="dcov90[learned_cal] - dcov90[posthoc] (PRIMARY at x4)", n=len(D),
                         mean=D.mean(), sd=D.std(), ci_lo=lo, ci_hi=hi,
                         frac_patients_negative=float((D < 0).mean())))
prim = pd.DataFrame(prim)
prim.to_csv(os.path.join(RES, "primary_endpoint.csv"), index=False)
per_pat = ivp[ivp.kind.isin(["learned_cal", "posthoc", "learned_raw", "const"])]
per_pat.to_csv(os.path.join(RES, "patient_level_intervals.csv"), index=False)
rkp.to_csv(os.path.join(RES, "patient_level_ranking.csv"), index=False)

# ============================================================================ figure 1: calibration transfer
SER = [("learned_raw", FS.SKY, "o", "--"), ("learned_cal", FS.BLUE, "o", "-"), ("posthoc", FS.VERM, "s", "-"),
       ("const", FS.MUTED, "^", ":")]


def fig_transfer(metric, ylabel, fname, nominal=None):
    fig, axs = plt.subplots(2, 2, figsize=(FS.DOUBLE * 0.82, 4.1), sharex=True, sharey=True)
    for j, a in enumerate(ARCH):
        for i, R in enumerate(C.STRIDES):
            ax = axs[i, j]
            for k, (kind, col, mk, ls) in enumerate(SER):
                t = int_tab[(int_tab.arm == f"{a}_P|{kind}") & (int_tab.R == R)].set_index("pattern").reindex(PATS)
                xs = np.arange(3) + (k - 1.5) * 0.07
                ax.errorbar(xs, t[f"{metric}_mean"], yerr=[t[f"{metric}_mean"] - t[f"{metric}_lo"],
                            t[f"{metric}_hi"] - t[f"{metric}_mean"]], color=col, marker=mk, ls=ls, lw=1.3,
                            ms=4, capsize=2, label=ARMLAB[kind])
            if nominal:
                ax.axhline(nominal, color=FS.INK2, lw=0.8, ls=(0, (4, 2)))
            ax.set_xticks(range(3))
            ax.set_xticklabels(PATS)
            ax.set_title(f"{a}-P  ×{R}" + ("  (primary)" if R == 4 else "  (confirmation)"), loc="left")
            if j == 0:
                ax.set_ylabel(ylabel)
    h, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{fname}.{ext}"))
    plt.close(fig)


fig_transfer("cov90", "empirical 90% coverage", "fig_coverage_transfer", 0.9)
fig_transfer("width90", "mean 90% interval width", "fig_width_transfer")

# ============================================================================ figure 2: primary endpoint forest
fig, ax = plt.subplots(figsize=(FS.SINGLE * 1.25, 2.6))
lab, y = [], 0
for a in ARCH:
    for R in C.STRIDES:
        for kind, col, mk in (("learned_cal", FS.BLUE, "o"), ("posthoc", FS.VERM, "s")):
            r = prim[(prim.arch == a) & (prim.R == R) & (prim.contrast == f"dcov90[{kind}] clustered-uniform")].iloc[0]
            g = ivp[(ivp.arm == f"{a}_P|{kind}") & (ivp.R == R)].pivot(index="patient", columns="pattern", values="cov90")
            dd = (g["clustered"] - g["uniform"]).dropna()
            ax.scatter(dd, y + rng.uniform(-0.12, 0.12, len(dd)), s=4, color=col, alpha=0.35, lw=0)
            ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r.ci_lo], [r.ci_hi - r["mean"]]], color=col, marker=mk,
                        ms=4.5, capsize=2.5, lw=1.4)
            lab.append(f"{a}-P ×{R}  {ARMLAB[kind]}")
            y += 1
        y += 0.5
ax.axvline(0, color=FS.INK2, lw=0.8)
ax.set_yticks([i + 0.5 * (i // 2) for i in range(len(lab))])
ax.set_yticklabels(lab, fontsize=6.5)
ax.invert_yaxis()
ax.grid(axis="x", color=FS.GRID)
ax.set_xlabel("change in 90% coverage, clustered − uniform")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig_primary_endpoint.{ext}"))
plt.close(fig)

# ============================================================================ figure 3: ranking beyond geometry
SC = [("learned_total", FS.BLUE, "o"), ("posthoc", FS.VERM, "s"), ("near_mm", FS.GREEN, "^"),
      ("gap_mm", FS.PURPLE, "D"), ("random", FS.MUTED, "v")]
fig, axs = plt.subplots(1, 4, figsize=(FS.DOUBLE, 2.3), sharey=True)
for j, (a, metric) in enumerate([("SAINT", "spearman"), ("SAINT", "strat_spearman"), ("I3Net", "spearman"),
                                 ("I3Net", "strat_spearman")]):
    ax = axs[j]
    for k, (s, col, mk) in enumerate(SC):
        t = rank_tab[(rank_tab["mean"] == f"{a}_P") & (rank_tab.score == s) & (rank_tab.R == 4)].set_index("pattern").reindex(PATS)
        if t[f"{metric}_mean"].isna().all():
            continue
        xs = np.arange(3) + (k - 2) * 0.09
        ax.errorbar(xs, t[f"{metric}_mean"], yerr=[t[f"{metric}_mean"] - t[f"{metric}_lo"],
                    t[f"{metric}_hi"] - t[f"{metric}_mean"]], color=col, marker=mk, ls="none", ms=4, capsize=2,
                    label=SCORELAB[s])
    ax.axhline(0, color=FS.INK2, lw=0.7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(PATS, rotation=20)
    ax.set_title(f"{a}-P ×4\n" + ("Spearman(|e|, score)" if metric == "spearman" else "within (gap, dist.) strata"),
                 loc="left", fontsize=7.5)
axs[0].set_ylabel("within-patient Spearman")
h, l_ = axs[0].get_legend_handles_labels()
fig.legend(h, l_, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig_ranking.{ext}"))
plt.close(fig)

# ============================================================================ figure 4: risk-coverage
cu = pd.read_csv(os.path.join(RES, "test_sparsification_curves.csv"))
fcols = [c for c in cu.columns if c.startswith("f")]
fr = np.array([float(c[1:]) for c in fcols])
fig, axs = plt.subplots(1, 4, figsize=(FS.DOUBLE, 2.2), sharey=False)
for j, (a, pat) in enumerate([(a, p) for a in ARCH for p in ("uniform", "clustered")]):
    ax = axs[j]
    for s, col, mk in SC + [("oracle", FS.INK, None)]:
        g = cu[(cu["mean"] == f"{a}_P") & (cu.score == s) & (cu.R == 4) & (cu.pattern == pat)]
        if not len(g):
            continue
        g = g.groupby("patient")[fcols].mean()
        ax.plot(1 - fr, g.mean(0).values, color=col, lw=1.2, ls="--" if s == "oracle" else "-",
                label=SCORELAB.get(s, s))
    ax.set_title(f"{a}-P ×4 {pat}", loc="left", fontsize=7.5)
    ax.set_xlabel("fraction of voxels retained")
    ax.invert_xaxis()
axs[0].set_ylabel("MAE of retained voxels")
h, l_ = axs[0].get_legend_handles_labels()
fig.legend(h, l_, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.05))
fig.tight_layout(rect=(0, 0.1, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig_risk_coverage.{ext}"))
plt.close(fig)

# ============================================================================ gap-length strata
rows = []
print("tables + figures written")
json.dump(dict(n_test_patients=int(cm.patient.nunique()), n_cases=int(cm.groupby(["patient", "pattern", "R", "rep"]).ngroups)),
          open(os.path.join(RES, "counts.json"), "w"), indent=1)

# ============================================================================ slide figure: x4 transfer (coverage + width)
fig, axs = plt.subplots(2, 2, figsize=(3.7, 3.0), sharex=True)
for j, a in enumerate(ARCH):
    for i, (metric, ylab) in enumerate((("cov90", "90% coverage"), ("width90", "90% width"))):
        ax = axs[i, j]
        for k, (kind, col, mk, ls) in enumerate(SER):
            t = int_tab[(int_tab.arm == f"{a}_P|{kind}") & (int_tab.R == 4)].set_index("pattern").reindex(PATS)
            xs = np.arange(3) + (k - 1.5) * 0.07
            ax.errorbar(xs, t[f"{metric}_mean"], yerr=[t[f"{metric}_mean"] - t[f"{metric}_lo"], t[f"{metric}_hi"] - t[f"{metric}_mean"]],
                        color=col, marker=mk, ls=ls, lw=1.2, ms=3.2, capsize=1.5, label=ARMLAB[kind])
        if metric == "cov90":
            ax.axhline(0.9, color=FS.INK2, lw=0.7, ls=(0, (4, 2)))
            ax.set_title(f"{a}-P ×4", loc="left", fontsize=8)
        ax.set_xticks(range(3)); ax.set_xticklabels(["uni.", "jit.", "clu."], fontsize=7)
        if j == 0:
            ax.set_ylabel(ylab, fontsize=7.5)
for i in range(2):
    lo = min(axs[i, 0].get_ylim()[0], axs[i, 1].get_ylim()[0]); hi = max(axs[i, 0].get_ylim()[1], axs[i, 1].get_ylim()[1])
    for j in range(2):
        axs[i, j].set_ylim(lo, hi)
h, l_ = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l_, loc="lower center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.12, 1, 1), h_pad=0.4, w_pad=0.6)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig_slide_transfer.{ext}"))
plt.close(fig)
