"""Headline table for the matched setup (v3): SAINT-P (adapter, uniform-trained), ArSSR-P (native, uniform-trained), linear.
Rows: learned (probabilistic models only), post-hoc WITHOUT geometry, post-hoc WITH geometry (main-effects designs that
differ only by the geometry features). Reads exp_uniform_arssr/results/{base,uniform}/summary.csv (written by
exp_uniform_arssr/scripts/analysis_cpu.py from saved outputs). Any cell without a result yet is printed as '--'.
Re-run after the uniform ArSSR analysis finishes; nothing else needs editing.
"""

from project_paths import root_path
import os

import pandas as pd

ISBI = root_path('for_ISBI')
RES = os.path.join(ISBI, "exp_uniform_arssr", "results")
MS = os.path.join(ISBI, "manuscript_4page")
frames = [pd.read_csv(os.path.join(RES, t, "summary.csv")) for t in ("base", "uniform")
          if os.path.exists(os.path.join(RES, t, "summary.csv"))]
S = pd.concat(frames)
for t_ in ("ixi", "ixi_arssr"):
    IXI = os.path.join(ISBI, "exp_ixi", "results", t_, "summary.csv")
    if os.path.exists(IXI):
        SX = pd.read_csv(IXI)
        SX["method"] = "IXI_" + SX["method"]
        S = pd.concat([S, SX])
ROWS = ["BraTS 2023 glioma FLAIR, 1\\,mm slices, 51 TEST patients ($K{=}39$ / 20)", ("SAINT_P", "SAINT-P", "learned", "learned"), ("SAINT_P", "SAINT-P", "ph_nogeo", "PH, no geometry"),
        ("SAINT_P", "SAINT-P", "ph_geo", "PH + geometry"), None,
        ("ArSSR_uniform", "ArSSR-P", "learned", "learned"), ("ArSSR_uniform", "ArSSR-P", "ph_nogeo", "PH, no geometry"),
        ("ArSSR_uniform", "ArSSR-P", "ph_geo", "PH + geometry"), None,
        ("linear", "Linear", "ph_nogeo", "PH, no geometry"), ("linear", "Linear", "ph_geo", "PH + geometry"),
        "IXI-tiny T1, 2.18\\,mm slices, 60 TEST patients ($K{=}21$ / 11)",
        ("IXI_SAINT_P", "SAINT-P", "learned", "learned"), ("IXI_SAINT_P", "SAINT-P", "ph_nogeo", "PH, no geometry"),
        ("IXI_SAINT_P", "SAINT-P", "ph_geo", "PH + geometry"), None,
        ("IXI_ArSSR_uniform", "ArSSR-P", "learned", "learned"), ("IXI_ArSSR_uniform", "ArSSR-P", "ph_nogeo", "PH, no geometry"),
        ("IXI_ArSSR_uniform", "ArSSR-P", "ph_geo", "PH + geometry"), None,
        ("IXI_linear", "Linear", "ph_nogeo", "PH, no geometry"), ("IXI_linear", "Linear", "ph_geo", "PH + geometry")]


def sg(v):
    return ("$+$" if v >= 0 else "$-$") + f"{abs(v):.2f}"


lines = []
for r in ROWS:
    if r is None:
        lines.append("\\addlinespace[2pt]")
        continue
    if isinstance(r, str):
        if lines:
            lines.append("\\midrule")
        lines.append("\\multicolumn{15}{@{}l}{\\emph{" + r + "}}\\\\")
        continue
    m, lab, arm, ulab = r
    c = [lab, ulab]
    for R in (4, 8):
        t = S[(S.method == m) & (S.arm == arm) & (S.R == R)]
        if len(t) == 0:
            c += ["--"] * 6
            continue
        t = t.iloc[0]
        c += [f"{t.cov_U:.2f}", f"{t.cov_J:.2f}", f"{t.cov_C:.2f}", f"{sg(t.dcov)} [{sg(t.dcov_lo)}, {sg(t.dcov_hi)}]",
              f"{t.width_ratio:.2f}", f"{t.is_C:.2f}"]
    lines.append(" & ".join(c[:8] + [""] + c[8:]) + " \\\\")
tab = r"""\begin{table*}[t]
\centering
\caption{Calibrated 90\,\% voxel coverage on TEST (mean of per-patient values), calibrated on uniform CAL data only and
transferred to jittered (J) and clustered (C) sampling with the same $K$. Rows of one reconstruction share the same mean
$\bar y$; the two post-hoc (PH) rows differ only by the geometry features. $\Delta$: C$-$U with 95\,\% patient-bootstrap CI;
$W_{\mathrm C}/W_{\mathrm U}$: width ratio; IS: clustered interval score (lower is better).}
\label{tab:main}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}ll cccccc c cccccc@{}}
\toprule
& & \multicolumn{6}{c}{$\times4$} && \multicolumn{6}{c}{$\times8$}\\
\cmidrule(lr){3-8}\cmidrule(lr){10-15}
Mean & Interval & U & J & C & $\Delta$ [95\,\% CI] & $W_{\mathrm C}/W_{\mathrm U}$ & IS && U & J & C & $\Delta$ [95\,\% CI] & $W_{\mathrm C}/W_{\mathrm U}$ & IS\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
open(os.path.join(MS, "tab_headline.tex"), "w").write(tab)
print(tab)
