"""Headline table v4 (three fine-tuning seeds): from exp_seeds/results/seeds_summary.csv and seeds_dcov.csv
(exp_seeds/scripts/aggregate_seeds.py). Cells = mean over seeds 42/43/44 of the patient-averaged metric; clustered coverage
also shows the SD across seeds. Linear interpolation has no training seed (single value).
Columns per factor: U, J, C (+-SD), Delta = C - U, clustered IS, clustered ENCE.
"""

from project_paths import root_path
import os

import pandas as pd

R_ = root_path('for_ISBI/exp_seeds/results')
MS = root_path('for_ISBI/manuscript_4page')
A = pd.read_csv(os.path.join(R_, "seeds_summary.csv"))
D = pd.read_csv(os.path.join(R_, "seeds_dcov.csv"))


def g(d, m, arm, R, pat, k):
    t = A[(A.dataset == d) & (A.method == m) & (A.arm == arm) & (A.R == R) & (A.pattern == pat)].iloc[0]
    return t[f"{k}_mean"], t[f"{k}_sd"]


def f2(f):
    return ".2f" if f == ".2f" else ".1f"


def sg(v):
    return ("$+$" if v >= 0 else "$-$") + f"{abs(v):.2f}"


BLOCKS = [("brats", r"BraTS 2023 glioma FLAIR (51 test patients)"), ("ixi", r"IXI-tiny T1 (60 test patients)")]
ROWS = [("SAINT_P", "SAINT-P", "learned", "learned"), ("SAINT_PG", "SAINT-P", "learned_geo", "learned + geometry"),
        ("SAINT_P", "SAINT-P", "ph_nogeo", "post hoc"), ("SAINT_P", "SAINT-P", "ph_geo", "post hoc + geometry"), None,
        ("ArSSR", "ArSSR-P", "learned", "learned"), ("ArSSR_PG", "ArSSR-P", "learned_geo", "learned + geometry"),
        ("ArSSR", "ArSSR-P", "ph_nogeo", "post hoc"), ("ArSSR", "ArSSR-P", "ph_geo", "post hoc + geometry"), None,
        ("linear", "Linear", "ph_nogeo", "post hoc"), ("linear", "Linear", "ph_geo", "post hoc + geometry")]
lines = []
for d, title in BLOCKS:
    if lines:
        lines.append(r"\midrule")
    lines.append(r"\multicolumn{13}{@{}l}{\textbf{" + title + r"}}\\")
    for r in ROWS:
        if r is None:
            lines.append(r"\addlinespace[2pt]")
            continue
        m, lab, arm, ulab = r
        c = [lab, ulab]
        for R in (4, 8):
            lin = m == "linear"
            pm = lambda v, sd, f: (f"{v:{f}}" if lin else f"{v:{f}}{{\\tiny$\\pm${sd:{f2(f)}}}}")  # noqa: E731
            for pat in ("uniform", "jittered", "clustered"):
                v, sd = g(d, m, arm, R, pat, "cov90")
                c.append(pm(100 * v, 100 * sd, ".0f"))
            t = D[(D.dataset == d) & (D.method == m) & (D.arm == arm) & (D.R == R)].iloc[0]
            c.append(pm(100 * t.dcov_mean, 100 * t.dcov_sd, "+.0f").replace("-", "$-$").replace("+", "$+$"))
            v, sd = g(d, m, arm, R, "clustered", "ence")
            c.append(pm(v, sd, ".2f"))
        lines.append(" & ".join(c[:7] + [""] + c[7:]) + r" \\")
tab = r"""\begin{table*}[t]
\centering
\caption{Coverage of the 90\,\% intervals on test data (\%; best is 90, $\rightarrow$90) with mean\,$\pm$\,SD over three
fine-tuning seeds (linear interpolation has no seed). All intervals were calibrated on uniform sampling only. Change:
clustered minus uniform coverage in percentage points (best is 0, $\rightarrow$0). ENCE: expected normalized calibration
error on clustered sampling (lower is better, $\downarrow$).}
\label{tab:main}
\setlength{\tabcolsep}{1.5pt}\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll ccccc c ccccc@{}}
\toprule
& & \multicolumn{5}{c}{$\times4$} && \multicolumn{5}{c}{$\times8$}\\
\cmidrule(lr){3-7}\cmidrule(lr){9-13}
& & \multicolumn{3}{c}{Coverage (\%) $\rightarrow$90} & & && \multicolumn{3}{c}{Coverage (\%) $\rightarrow$90} & & \\
\cmidrule(lr){3-5}\cmidrule(lr){9-11}
Reconstruction & Uncertainty & Uniform & Jittered & Clustered & Change $\rightarrow$0 & ENCE $\downarrow$ && Uniform & Jittered & Clustered & Change $\rightarrow$0 & ENCE $\downarrow$\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
open(os.path.join(MS, "tab_headline.tex"), "w").write(tab)
open(root_path('for_ISBI/ISBI_paper/tab_headline.tex'), "w").write(tab)
print(tab)
