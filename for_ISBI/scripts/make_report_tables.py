"""LaTeX tables for report/ (booktabs), generated only from saved result CSVs."""
import os

import pandas as pd

from isbi import common as C

RES = os.path.join(C.ISBI, "results")
OUT = os.path.join(C.ISBI, "report", "tables")
os.makedirs(OUT, exist_ok=True)
S = pd.read_csv(os.path.join(RES, "geometry_story", "arms_summary.csv"))

ORDER = ["SAINT-P learned", "I3Net-P learned", "SAINT exposure learned", "I3Net exposure learned", "SAINT-P constant width",
         "SAINT-P post-hoc, no geometry", "I3Net-P post-hoc, no geometry", "linear post-hoc, no geometry",
         "ArSSR-P learned", "SAINT-PG learned", "SAINT-PGX learned", "SAINT-P post-hoc", "I3Net-P post-hoc",
         "ArSSR-P post-hoc", "linear post-hoc", "SAINT-P post-hoc, gap-conditional", "I3Net-P post-hoc, gap-conditional",
         "ArSSR-P post-hoc, gap-conditional", "linear post-hoc, gap-conditional"]
lines = [r"\begin{tabular}{@{}lllcccccc@{}}", r"\toprule",
         r" & & & \multicolumn{3}{c}{$\times4$} & \multicolumn{3}{c}{$\times8$} \\ \cmidrule(lr){4-6}\cmidrule(l){7-9}",
         r"Arm & irregular & geometry & uniform & clustered & change [95\% CI] & uniform & clustered & change [95\% CI] \\",
         r" & training & & & & & & & \\ \midrule"]
for i, a in enumerate(ORDER):
    if i == 8:
        lines.append(r"\midrule")
    r4 = S[(S.arm == a) & (S.R == 4)].iloc[0]
    r8 = S[(S.arm == a) & (S.R == 8)].iloc[0]
    cell = lambda r: f"{r.cov90_uniform:.2f} & {r.cov90_clustered:.2f} & {r.dcov90:+.2f} [{r.dcov90_lo:+.2f}, {r.dcov90_hi:+.2f}]"  # noqa: E731
    lines.append(f"{a} & {r4.trained_irregular} & {r4.geometry} & {cell(r4)} & {cell(r8)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_arms.tex"), "w").write("\n".join(lines) + "\n")

# reconstruction accuracy (PSNR, missing slices, brain), patient means
rt = pd.read_csv(os.path.join(RES, "table_recon.csv"))
nat = pd.read_csv(os.path.join(RES, "native", "recon.csv"))
rows = []
for m, lab in (("linear", "linear"), ("bicubic", "bicubic"), ("SAINT_P", "SAINT-P (adapter)"), ("I3Net_P", "I3Net-P (adapter)")):
    for R in C.STRIDES:
        q = rt[(rt.method == m) & (rt.R == R)].set_index("pattern").psnr_mean
        rows.append((lab, R, q["uniform"], q["jittered"], q["clustered"]))
for R in C.STRIDES:
    q = nat[(nat.method == "arssr_P") & (nat.R == R)].set_index("pattern").psnr
    rows.append(("ArSSR-P (native)", R, q["uniform"], q["jittered"], q["clustered"]))
lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{$\times4$} & \multicolumn{3}{c}{$\times8$} \\ \cmidrule(lr){2-4}\cmidrule(l){5-7}",
         r"Method & uniform & jittered & clustered & uniform & jittered & clustered \\ \midrule"]
for lab in dict.fromkeys(r[0] for r in rows):
    a = [r for r in rows if r[0] == lab and r[1] == 4][0]
    b = [r for r in rows if r[0] == lab and r[1] == 8][0]
    lines.append(f"{lab} & {a[2]:.2f} & {a[3]:.2f} & {a[4]:.2f} & {b[2]:.2f} & {b[3]:.2f} & {b[4]:.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_recon.tex"), "w").write("\n".join(lines) + "\n")

# error ranking beyond geometry (within-stratum Spearman, x4)
rk = pd.read_csv(os.path.join(RES, "table_ranking.csv"))
nk = pd.read_csv(os.path.join(RES, "native", "summary.csv"))
lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Score ($\times4$) & uniform & jittered & clustered \\ \midrule"]
for m, s, lab in (("SAINT_P", "learned_total", "SAINT-P learned $\\sigma$"), ("SAINT_P", "posthoc", "SAINT-P post-hoc $\\hat b$"),
                  ("I3Net_P", "learned_total", "I3Net-P learned $\\sigma$"), ("I3Net_P", "posthoc", "I3Net-P post-hoc $\\hat b$"),
                  ("SAINT_P", "near_mm", "nearest-slice distance (geometry only)")):
    q = rk[(rk["mean"] == m) & (rk.score == s) & (rk.R == 4)].set_index("pattern").strat_spearman_mean
    lines.append(f"{lab} & {q['uniform']:.2f} & {q['jittered']:.2f} & {q['clustered']:.2f} \\\\")
for k, lab in (("learned_cal", "ArSSR-P learned $\\sigma$"), ("posthoc", "ArSSR-P post-hoc $\\hat b$")):
    q = nk[(nk.kind == k) & (nk.R == 4)].set_index("pattern").strat_spearman
    lines.append(f"{lab} & {q['uniform']:.2f} & {q['jittered']:.2f} & {q['clustered']:.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_ranking.tex"), "w").write("\n".join(lines) + "\n")
print("tables written", os.listdir(OUT))
