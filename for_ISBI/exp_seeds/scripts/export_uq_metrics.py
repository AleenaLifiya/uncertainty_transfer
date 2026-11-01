"""Export interval score, AUSE/MAE and UCE for every method, dataset, undersampling factor and sampling pattern.
Reads exp_seeds/results/seeds_summary.csv (written by aggregate_seeds.py) and writes
results/uq_metrics_all.csv and results/uq_metrics_all.md at the repository root (values: mean and SD over seeds).
Standard library only.
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(REPO, "for_ISBI", "exp_seeds", "results", "seeds_summary.csv")
OUT = os.path.join(REPO, "results")

DATASETS = [("brats", "BraTS 2023 glioma FLAIR (51 test patients)"), ("ixi", "IXI-tiny T1 (60 test patients)")]
RECON = [("SAINT", "SAINT-P"), ("ArSSR", "ArSSR-P"), ("linear", "Linear")]
ARMS = [("learned", "learned"), ("learned_geo", "learned + geometry"), ("ph_nogeo", "post hoc"),
        ("ph_geo", "post hoc + geometry")]
PATTERNS = ["uniform", "jittered", "clustered"]
METRICS = [("is90", "IS", 3), ("ause_norm", "AUSE/MAE", 3), ("uce_e3", "UCE x1e3", 2)]

rows = {}
with open(SRC) as f:
    for r in csv.DictReader(f):
        recon = "SAINT" if r["method"].startswith("SAINT") else r["method"].replace("_PG", "")
        rows[(r["dataset"], recon, r["arm"], int(r["R"]), r["pattern"])] = r

os.makedirs(OUT, exist_ok=True)
table = []
for d, _ in DATASETS:
    for R in (4, 8):
        for rk, rn in RECON:
            for ak, an in ARMS:
                for p in PATTERNS:
                    r = rows.get((d, rk, ak, R, p))
                    if r is None:
                        continue
                    table.append(dict(dataset=d, reconstruction=rn, uncertainty=an, R=R, pattern=p, n_seeds=int(r["n_seeds"]),
                                      **{f"{k}_{s}": float(r[f"{k}_{s}"]) for k, _, _ in METRICS for s in ("mean", "sd")}))
assert len(table) == 120, len(table)

with open(os.path.join(OUT, "uq_metrics_all.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0]))
    w.writeheader()
    for t in table:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in t.items()})


def cell(t, k, nd):
    m = f"{t[k + '_mean']:.{nd}f}"
    return m if t["n_seeds"] == 1 else f"{m} ± {t[k + '_sd']:.{nd}f}"


md = ["# Interval score, AUSE/MAE and UCE for all methods, datasets and undersampling factors", "",
      "Supplement to the ISBI paper *Different Gaps, Different Maps?* (Evaluation paragraph). Values are means over the TEST",
      "patients, then mean ± SD over the three fine-tuning seeds (42, 43, 44); linear interpolation has no seed, so a single",
      "value is given. Coverage and ENCE are in Table 1 of the paper. All intervals are nominal 90 % and were calibrated on",
      "uniform sampling only; jittered and clustered values are averaged over three random masks per patient.", "",
      "- **IS**: interval score of the 90 % interval (Gneiting & Raftery 2007), intensities scaled to [0, 1]. Lower is better.",
      "- **AUSE/MAE**: area under the sparsification-error curve divided by the mean absolute error. Lower is better.",
      "- **UCE ×10³**: uncertainty calibration error (Laves et al. 2020), 20 equal-count bins, squared-intensity units",
      "  multiplied by 10³. Lower is better.",
      "- Learned intervals use the predictive variance of the MC-dropout Laplace mixture; post-hoc intervals use the variance",
      "  of the Laplace distribution whose central 90 % interval is the post-hoc interval.", "",
      "Machine-readable version: [`uq_metrics_all.csv`](uq_metrics_all.csv). Regenerate with",
      "`python for_ISBI/exp_seeds/scripts/export_uq_metrics.py` (input: `for_ISBI/exp_seeds/results/seeds_summary.csv`).", ""]
for d, dn in DATASETS:
    for R in (4, 8):
        md += [f"## {dn}, ×{R}", "",
               "| Reconstruction | Uncertainty | Pattern | IS ↓ | AUSE/MAE ↓ | UCE ×10³ ↓ |", "|---|---|---|---|---|---|"]
        for t in table:
            if t["dataset"] == d and t["R"] == R:
                md.append(f"| {t['reconstruction']} | {t['uncertainty']} | {t['pattern']} | "
                          + " | ".join(cell(t, k, nd) for k, _, nd in METRICS) + " |")
        md.append("")
with open(os.path.join(OUT, "uq_metrics_all.md"), "w") as f:
    f.write("\n".join(md))
print(f"wrote {len(table)} rows to {OUT}/uq_metrics_all.csv and .md")
