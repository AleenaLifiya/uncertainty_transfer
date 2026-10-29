"""Step 3d -- native-architecture analysis (ArSSR-P, DCSRN-P), derived from analyze_exposure.py: learned (raw / calibrated), post-hoc and constant-width intervals around each
exposure-trained model, fitted (FIT) and calibrated (CAL, uniform) exactly as in analyze.py; compared with the
uniform-trained models (results/patient_level_intervals.csv) on the same 51 test patients.
Outputs results/exposure/{summary,dcov90,recon}.csv
"""
import glob
import os

import numpy as np
import pandas as pd
import torch

from isbi import common as C
from isbi import uq as U

EV = os.path.join(C.ISBI, "runs", "eval")
EX = os.path.join(C.ISBI, "runs", "eval_native")
OUT = os.path.join(C.ISBI, "results", "native")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(16)
TAGS = os.environ.get("EXP_TAGS", "").split(",") if os.environ.get("EXP_TAGS") else [t for t in ("arssr",) if os.path.isdir(os.path.join(EX, t, "TEST"))]


def cases(tag, role):
    for f in sorted(glob.glob(os.path.join(EX, tag, role, "cases", "*.npz"))):
        c = dict(np.load(os.path.join(EV, role, "cases", os.path.basename(f))))
        c.update(dict(np.load(f)))
        yield c


def cov_l(c, a, lam):
    F = U.mix_cdf(torch.from_numpy(c["y"]), torch.from_numpy(c["NAT__mu_t"]), torch.from_numpy(c["NAT__b_t"]), lam).numpy()
    return (F >= (1 - a) / 2) & (F <= (1 + a) / 2)


def wid_l(c, a, lam):
    mu, b = torch.from_numpy(c["NAT__mu_t"]), torch.from_numpy(c["NAT__b_t"])
    return (np.clip(U.mix_quantile(mu, b, (1 + a) / 2, lam).numpy(), 0, 1)
            - np.clip(U.mix_quantile(mu, b, (1 - a) / 2, lam).numpy(), 0, 1))


rows, recon = [], []
for tag in TAGS:
    fit, cal, test = list(cases(tag, "FIT")), list(cases(tag, "CAL")), list(cases(tag, "TEST"))
    glm = U.LaplaceGLM(1e-4, True).fit(np.concatenate([c["NAT__feat"] for c in fit]),
                                       np.concatenate([c["y"] - c["NAT__x"] for c in fit]))
    for R in C.STRIDES:
        cs = [c for c in cal if int(c["R"]) == R]
        lam_l = U.calibrate_scalar(lambda c, l: cov_l(c, 0.9, l).mean(), cs, 0.9, 1e-2, 1e2)
        pre = [(np.abs(c["y"] - c["NAT__x"]), glm.predict(c["NAT__feat"])) for c in cs]
        kap = U.calibrate_scalar(lambda q, l: (q[0] <= l * q[1]).mean(), pre, 0.9, 1e-2, 1e2)
        w_c = U.calibrate_scalar(lambda q, l: (q[0] <= l).mean(), pre, 0.9, 1e-5, 1.0)
        for c in [c for c in test if int(c["R"]) == R]:
            e = c["y"] - c["NAT__x"]
            x = c["NAT__x"]
            bh = glm.predict(c["NAT__feat"])
            sl = c["NAT__epi"] + c["NAT__alea"]
            base = dict(model=tag, patient=c["patient"].item(), pattern=c["pattern"].item(), R=R, rep=int(c["rep"]),
                        mae=float(np.abs(e).mean()))
            for kind, cov, w in (("learned_raw", cov_l(c, 0.9, 1.0), wid_l(c, 0.9, 1.0)),
                                 ("learned_cal", cov_l(c, 0.9, lam_l), wid_l(c, 0.9, lam_l)),
                                 ("posthoc", np.abs(e) <= kap * bh, np.clip(x + kap * bh, 0, 1) - np.clip(x - kap * bh, 0, 1)),
                                 ("const", np.abs(e) <= w_c, np.clip(x + w_c, 0, 1) - np.clip(x - w_c, 0, 1))):
                rows.append(dict(base, kind=kind, cov90=float(cov.mean()), width90=float(w.mean()),
                                 spearman=(U.spearman(np.abs(e), sl) if kind.startswith("learned")
                                           else U.spearman(np.abs(e), bh) if kind == "posthoc" else np.nan),
                                 strat_spearman=(U.strat_spearman(e, sl, c["gap_mm"], c["near_mm"]) if kind.startswith("learned")
                                                 else U.strat_spearman(e, bh, c["gap_mm"], c["near_mm"]) if kind == "posthoc" else np.nan)))
    recon.append(pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(EX, tag, "TEST", "rows", "*.csv"))]))
    print(tag, "done", flush=True)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "case_metrics.csv"), index=False)
pp = d.groupby(["model", "kind", "R", "pattern", "patient"], as_index=False)[["cov90", "width90", "spearman", "strat_spearman", "mae"]].mean()
summ = pp.groupby(["model", "kind", "R", "pattern"])[["cov90", "width90", "spearman", "strat_spearman", "mae"]].mean()
summ.to_csv(os.path.join(OUT, "summary.csv"))
# coverage change uniform -> clustered per model/arm (paired by patient), bootstrap CI
rng = np.random.RandomState(0)
out = []
for (tag, kind, R), g in pp.groupby(["model", "kind", "R"]):
    w = g.pivot(index="patient", columns="pattern", values="cov90")
    wr = g.pivot(index="patient", columns="pattern", values="width90")
    for pat in ("jittered", "clustered"):
        dd = (w[pat] - w["uniform"]).values
        bs = dd[rng.randint(0, len(dd), (10000, len(dd)))].mean(1)
        out.append(dict(model=tag, kind=kind, R=R, pattern=pat, n=len(dd), cov90_uniform=w["uniform"].mean(),
                        cov90=w[pat].mean(), dcov90=dd.mean(), ci=f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]",
                        width_ratio=(wr[pat] / wr["uniform"]).mean()))
out = pd.DataFrame(out)
out.to_csv(os.path.join(OUT, "dcov90.csv"), index=False)
rc = pd.concat(recon)
rc.groupby(["method", "R", "pattern"])[["psnr", "ssim", "mae", "bg_haze"]].mean().to_csv(os.path.join(OUT, "recon.csv"))
pd.set_option("display.width", 220)
print(out.round(3).to_string())
print(rc.groupby(["method", "R", "pattern"]).psnr.mean().round(2).unstack())
