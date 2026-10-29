"""Manuscript tables, generated from saved results only (no hand-typed numbers).

tab_coverage.tex  90 % coverage (uniform / jittered / clustered), change clustered - uniform, width ratio and
                  clustered interval score per arm, x4 and x8.   Sources: results/geometry_story/arms_summary.csv
                  (coverage, change, width ratio) and data/interval_scores_summary.csv (IS, recomputed from saved
                  TEST voxels by compute_interval_scores.py; '--' where not recomputed).
tab_recon.tex     PSNR (dB) on GT-foreground voxels of missing slices.  Sources: results/table_recon.csv,
                  results/native/recon.csv.
"""

from project_paths import root_path
import os

import pandas as pd

ISBI = root_path('for_ISBI')
MS = os.path.join(ISBI, "manuscript_4page")
A = pd.read_csv(os.path.join(ISBI, "results", "geometry_story", "arms_summary.csv"))
IS = pd.read_csv(os.path.join(MS, "data", "interval_scores_summary.csv"))

# (arms_summary arm, interval-score arm or None, mean label, uncertainty label)
ROWS = [
    ("SAINT-P learned", "SAINT_P|learned_cal", "SAINT-P", "learned"),
    ("I3Net-P learned", "I3Net_P|learned_cal", "I3Net-P", "learned"),
    ("SAINT exposure learned", None, "SAINT-P$^{\\dagger}$", "learned, irreg.-trained"),
    ("I3Net exposure learned", None, "I3Net-P$^{\\dagger}$", "learned, irreg.-trained"),
    ("SAINT-PG learned", "SAINT_PG|learned_cal", "SAINT-P", "learned + geometry (PG)"),
    ("ArSSR-P learned", "ArSSR_P|learned_cal", "ArSSR-P", "learned (position in gap)"),
    None,
    ("SAINT-P post-hoc, no geometry", None, "SAINT-P", "post-hoc w/o geometry"),
    ("linear post-hoc, no geometry", None, "Linear", "post-hoc w/o geometry"),
    None,
    ("SAINT-P post-hoc", "SAINT_P|posthoc", "SAINT-P", "post-hoc"),
    ("I3Net-P post-hoc", "I3Net_P|posthoc", "I3Net-P", "post-hoc"),
    ("ArSSR-P post-hoc", "ArSSR_P|posthoc", "ArSSR-P", "post-hoc"),
    ("linear post-hoc", "linear|posthoc", "Linear", "post-hoc"),
    None,
    ("SAINT-P post-hoc, gap-conditional", "SAINT_P|posthoc_gapcond", "SAINT-P", "post-hoc, gap-cond.\\ $\\kappa_c(g)$"),
    ("I3Net-P post-hoc, gap-conditional", "I3Net_P|posthoc_gapcond", "I3Net-P", "post-hoc, gap-cond.\\ $\\kappa_c(g)$"),
    ("linear post-hoc, gap-conditional", "linear|posthoc_gapcond", "Linear", "post-hoc, gap-cond.\\ $\\kappa_c(g)$"),
]


def f2(v):
    return f"{v:.2f}"


def sgn(v):
    return ("$+$" if v >= 0 else "$-$") + f"{abs(v):.2f}"


lines = []
for r in ROWS:
    if r is None:
        lines.append("\\addlinespace[2pt]")
        continue
    arm, isarm, mean, uq = r
    cells = [mean, uq]
    for R in (4, 8):
        t = A[(A.arm == arm) & (A.R == R)].iloc[0]
        if isarm is not None:
            s = IS[(IS.arm == isarm) & (IS.R == R) & (IS.pattern == "clustered")].iloc[0]
            # consistency check: recomputed coverage must equal the consolidated table
            assert abs(s.cov90 - t.cov90_clustered) < 5e-4, (arm, R, s.cov90, t.cov90_clustered)
            isv = f"{s.is90:.2f}"
        else:
            isv = "--"
        cells += [f2(t.cov90_uniform), f2(t.cov90_jittered), f2(t.cov90_clustered), sgn(t.dcov90),
                  f"{t.width_ratio:.2f}", isv]
    lines.append(" & ".join(cells) + " \\\\")

tab = r"""\begin{table*}[t]
\centering
\caption{90\,\% voxel coverage on TEST (mean of 51 per-patient values). U/J/C: uniform/jittered/clustered;
$\Delta$: C$-$U; $W_{\mathrm C}/W_{\mathrm U}$: width ratio; IS: clustered interval score (lower is better).
$^{\dagger}$Own (jointly fine-tuned) mean.}
\label{tab:cov}
\setlength{\tabcolsep}{3.1pt}
\begin{tabular}{@{}ll cccccc c cccccc@{}}
\toprule
& & \multicolumn{6}{c}{$\times4$ ($K{=}39$)} && \multicolumn{6}{c}{$\times8$ ($K{=}20$)}\\
\cmidrule(lr){3-8}\cmidrule(lr){10-15}
Mean & Uncertainty & U & J & C & $\Delta$ & $W_{\mathrm C}/W_{\mathrm U}$ & IS && U & J & C & $\Delta$ & $W_{\mathrm C}/W_{\mathrm U}$ & IS\\
\midrule
""" + "\n".join(l.replace(" & ", " & ", 1) for l in lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
# insert the empty spacer column between the x4 and x8 blocks
out = []
for l in tab.split("\n"):
    if l.endswith("\\\\") and l.count("&") == 13:
        p = l[:-2].split(" & ")
        l = " & ".join(p[:8] + [""] + p[8:]) + " \\\\"
    out.append(l)
open(os.path.join(MS, "tab_coverage.tex"), "w").write("\n".join(out))

# ---------------------------------------------------------------- reconstruction table
T = pd.read_csv(os.path.join(ISBI, "results", "table_recon.csv"))
N = pd.read_csv(os.path.join(ISBI, "results", "native", "recon.csv"))
meth = [("linear", "Linear", "native"), ("bicubic", "Cubic", "native"), ("arssr_P", "ArSSR-P", "native"),
        ("SAINT_P", "SAINT-P", "adapter"), ("I3Net_P", "I3Net-P", "adapter")]
best = {}
vals = {}
for m, lab, _ in meth:
    for R in (4, 8):
        for pat in ("uniform", "jittered", "clustered"):
            if m == "arssr_P":
                v = N[(N.method == m) & (N.R == R) & (N.pattern == pat)].psnr.iloc[0]
            else:
                v = T[(T.method == m) & (T.R == R) & (T.pattern == pat)].psnr_mean.iloc[0]
            vals[(m, R, pat)] = v
for R in (4, 8):
    for pat in ("uniform", "jittered", "clustered"):
        best[(R, pat)] = max(vals[(m, R, pat)] for m, _, _ in meth)
rl = []
for m, lab, kind in meth:
    c = [lab]
    for R in (4, 8):
        for pat in ("uniform", "jittered", "clustered"):
            v = vals[(m, R, pat)]
            s = f"{v:.2f}"
            c.append(f"\\textbf{{{s}}}" if abs(v - best[(R, pat)]) < 1e-9 else s)
    rl.append(" & ".join(c) + " \\\\")
rt = r"""\begin{table}[t]
\centering
\caption{Reconstruction PSNR (dB) on GT-foreground voxels of missing slices, TEST (51 patients); best per column in bold.
SAINT-P/I3Net-P: MC mean, adapter for J/C.}
\label{tab:recon}
\setlength{\tabcolsep}{3.4pt}
\begin{tabular}{@{}l ccc ccc@{}}
\toprule
& \multicolumn{3}{c}{$\times4$} & \multicolumn{3}{c}{$\times8$}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Method & U & J & C & U & J & C\\
\midrule
""" + "\n".join(rl) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(MS, "tab_recon.tex"), "w").write(rt)
print(open(os.path.join(MS, "tab_coverage.tex")).read())
print(rt)

# ---------------------------------------------------------------- headline table (the only table in the paper)
HEAD = [("SAINT-P learned", "SAINT_P|learned_cal", "SAINT-P", "learned"),
        ("SAINT-P post-hoc", "SAINT_P|posthoc", "SAINT-P", "post-hoc"),
        ("I3Net-P learned", "I3Net_P|learned_cal", "I3Net-P", "learned"),
        ("I3Net-P post-hoc", "I3Net_P|posthoc", "I3Net-P", "post-hoc"),
        ("ArSSR-P post-hoc", "ArSSR_P|posthoc", "ArSSR-P", "post-hoc"),
        ("linear post-hoc", "linear|posthoc", "Linear", "post-hoc")]
hl = []
for i, (arm, isarm, mean, uq) in enumerate(HEAD):
    if i in (2, 4):
        hl.append("\\addlinespace[2pt]")
    c = [mean, uq]
    for R in (4, 8):
        t = A[(A.arm == arm) & (A.R == R)].iloc[0]
        s = IS[(IS.arm == isarm) & (IS.R == R) & (IS.pattern == "clustered")].iloc[0]
        assert abs(s.cov90 - t.cov90_clustered) < 5e-4
        c += [f2(t.cov90_uniform), f2(t.cov90_jittered), f2(t.cov90_clustered),
              f"{sgn(t.dcov90)} [{sgn(t.dcov90_lo)}, {sgn(t.dcov90_hi)}]", f"{t.width_ratio:.2f}", f"{s.is90:.2f}"]
    hl.append(" & ".join(c[:8] + [""] + c[8:]) + " \\\\")
ht = r"""\begin{table*}[t]
\centering
\caption{Headline result: calibrated 90\,\% voxel coverage on TEST (mean of 51 per-patient values) when calibration
(uniform CAL data only) is transferred to jittered (J) and clustered (C) sampling with the same $K$. Learned and post-hoc
rows of one network share the same mean $\bar y$. $\Delta$: C$-$U with 95\,\% patient-bootstrap CI;
$W_{\mathrm C}/W_{\mathrm U}$: width ratio; IS: clustered 90\,\% interval score (lower is better).}
\label{tab:main}
\setlength{\tabcolsep}{3.0pt}
\begin{tabular}{@{}ll cccccc c cccccc@{}}
\toprule
& & \multicolumn{6}{c}{$\times4$ ($K{=}39$)} && \multicolumn{6}{c}{$\times8$ ($K{=}20$)}\\
\cmidrule(lr){3-8}\cmidrule(lr){10-15}
Mean & Interval & U & J & C & $\Delta$ [95\,\% CI] & $W_{\mathrm C}/W_{\mathrm U}$ & IS && U & J & C & $\Delta$ [95\,\% CI] & $W_{\mathrm C}/W_{\mathrm U}$ & IS\\
\midrule
""" + "\n".join(hl) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
open(os.path.join(MS, "tab_headline.tex"), "w").write(ht)
print(ht)
