"""v4 (2026-10-29): as qual_v3 but exports exact clipped 90 % interval ENDPOINTS (lo, hi) for every arm, incl.
learned + geometry (exp_pg seed-42 heads, lambda from exp_pg/logs/metrics_brats_*_s42.log), clustered pattern only.
Output planes4_clustered.npz.

Qualitative planes for the matched setup (SAINT-P, uniform-trained ArSSR-P, linear) on the protocol-selected median TEST
patient (runs/qual/selection.json), x4, uniform and clustered r0, coronal plane through the brain centre of mass (same
plane and crop rule as scripts/qualitative.py). Same MC seeds as the evaluation (hash(patient, condition)), T = 20.
Per model: mean, |error|, learned 90 % half-width (calibrated lambda), post-hoc 90 % half-width without and with geometry
(main-effects GLMs fitted on FIT, kappa on CAL uniform x4 -- exactly the 'ph_nogeo' / 'ph_geo' arms of analysis_cpu.py).
GPU: two forward passes per model and pattern (minutes). Output: exp_uniform_arssr/results/qual/planes_<pattern>.npz
"""
import glob
import json
import os

import numpy as np
import torch

from isbi import common as C
from isbi import evalcore as E
from isbi import native as N
from isbi import recon as RC
from isbi import uq as U
from comparisons import protocol as P
from IR3C_LPIPS import brats_methods as BM

R, VIEW_AX = 4, 1
EXP = os.path.join(C.ISBI, "exp_uniform_arssr")
OUT = os.path.join(EXP, "results", "qual")
import re, sys
sys.path.insert(0, os.path.join(C.ISBI, "exp_pg", "scripts"))
import pg as PGm  # noqa: E402
from isbi import models as Mm  # noqa: E402
os.makedirs(OUT, exist_ok=True)
EV = os.path.join(C.ISBI, "runs", "eval")
NATU = os.path.join(EXP, "runs", "eval_native", "arssr")
proto = json.load(open(os.path.join(C.ISBI, "configs", "protocol_frozen.json")))
roles = C.load_roles()
p = json.load(open(os.path.join(C.ISBI, "runs", "qual", "selection.json")))["picks"]["median"]
scan = roles["scan_of"][p]
EPS = U.EPS


# ------------------------------------------------------------------ post-hoc fits (FIT) and calibration (CAL), x4 uniform
def phi_me(F, geo):
    cols = [F[:, 0], np.log(F[:, 1] + EPS), np.log(F[:, 2] + EPS), np.log(F[:, 3] + EPS)]
    if geo:
        cols += [np.log(F[:, 4]), np.log(F[:, 5]), F[:, 6]]
    return np.stack(cols, 1).astype(np.float64)


def fit_glm(X, e):
    m, s = X.mean(0), X.std(0) + 1e-8
    Xt = torch.from_numpy((X - m) / s)
    aa = torch.from_numpy(np.abs(e).astype(np.float64))
    th = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    c0 = torch.tensor([float(np.log(np.mean(np.abs(e)) + 1e-6))], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([th, c0], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        lb = c0 + Xt @ th
        loss = (lb + aa * torch.exp(-lb)).mean() + 1e-4 * (th ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    th_, c_ = th.detach().numpy(), float(c0.detach())
    return lambda X_: np.exp(c_ + ((X_ - m) / s) @ th_).astype(np.float32)


def calib(vals, c=0.9):
    srt = [np.sort(v) for v in vals]
    f = lambda k: np.mean([np.searchsorted(v, k, side="right") / len(v) for v in srt])  # noqa: E731
    lo, hi = np.log(1e-4), np.log(1e4)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(np.exp(mid)) < c else (lo, mid)
    return float(np.exp(0.5 * (lo + hi)))


def cases(role, m):
    out = []
    for f in sorted(glob.glob(os.path.join(EV, role, "cases", "*.npz"))):
        z = np.load(f)
        if role == "CAL" and int(z["R"]) != R:
            continue
        if m == "ArSSR_P":
            n = np.load(os.path.join(NATU, role, "cases", os.path.basename(f)))
            out.append(dict(y=z["y"], x=n["NAT__x"], F=n["NAT__feat"], mu=n["NAT__mu_t"] if role == "CAL" else None,
                            b=n["NAT__b_t"] if role == "CAL" else None))
        else:
            out.append(dict(y=z["y"], x=z[f"{m}__x"], F=z[f"{m}__feat"],
                            mu=z[f"{m}__mu_t"] if role == "CAL" and m == "SAINT_P" else None,
                            b=z[f"{m}__b_t"] if role == "CAL" and m == "SAINT_P" else None))
    return out


def lam_thr(c):
    y, mu, b = (torch.from_numpy(np.asarray(v, np.float64)) for v in (c["y"], c["mu"], c["b"]))

    def cov(lam):
        z = (y[None] - mu) / (lam * b)
        F = torch.where(z < 0, 0.5 * torch.exp(z.clamp(max=0)), 1 - 0.5 * torch.exp(-z.clamp(min=0))).mean(0)
        return (F >= 0.05) & (F <= 0.95)
    lo = torch.full(y.shape, np.log(1e-3), dtype=torch.float64)
    hi = torch.full(y.shape, np.log(1e3), dtype=torch.float64)
    c_lo, c_hi = cov(torch.exp(lo)), cov(torch.exp(hi))
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        cm = cov(torch.exp(mid))
        hi, lo = torch.where(cm, mid, hi), torch.where(cm, lo, mid)
    t = torch.exp(hi).numpy()
    t[c_lo.numpy()] = 0
    t[~c_hi.numpy()] = np.inf
    return t


PH = {}
for m in ("SAINT_P", "ArSSR_P", "linear"):
    fit, cal = cases("FIT", m), cases("CAL", m)
    F = np.concatenate([c["F"] for c in fit])
    e = np.concatenate([c["y"] - c["x"] for c in fit])
    d = {}
    for kind, geo in (("ph_nogeo", False), ("ph_geo", True)):
        g = fit_glm(phi_me(F, geo), e)
        k = calib([np.abs(c["y"] - c["x"]) / g(phi_me(c["F"], geo)) for c in cal])
        d[kind] = (g, k, geo)
    if m != "linear":
        d["lambda"] = calib([lam_thr(c) for c in cal])
    PH[m] = d
    print(m, {k: (v[1] if isinstance(v, tuple) else v) for k, v in d.items()}, flush=True)

# ------------------------------------------------------------------ models
net, prfn, chunk, _ = RC.build_prob_saint(proto["selected_checkpoints"]["saint_prob"])
base, inf, _ = BM._load_arssr_slowlr()
pm = N.ProbArSSR(base).cuda()
pm.load_state_dict(torch.load(json.load(open(os.path.join(EXP, "runs", "train", "arssr_prob", "selected.json")))["selected_path"],
                              map_location="cpu", weights_only=False)["state_dict"])
pm.eval()
gt = P.prepare_one(C.scan_path(scan))["vol01"]
com = np.array(np.nonzero(gt > 0)).mean(1).astype(int)


def plane(v):
    return np.rot90(v[:, com[1]]) if v.ndim == 3 else np.rot90(v[:, :, com[1]] if False else v[:, com[1]])


def pg_lambda(arch):
    t = open(os.path.join(C.ISBI, "exp_pg", "logs", f"metrics_brats_{arch}_s42.log")).read()
    return float(re.search(r"lambda \{4: ([0-9.eE+-]+)", t).group(1))


pgsel = lambda arch: json.load(open(os.path.join(C.ISBI, "exp_pg", "runs", f"brats_{arch}_s42", "train", "selected.json")))["selected_path"]  # noqa: E731
pg_s = Mm.ProbRFNGeo(prfn).cuda()
pg_s.load_state_dict(torch.load(pgsel("saint"), map_location="cpu", weights_only=False)["state_dict"]); pg_s.eval()
pg_a = PGm.ArSSRGeo(pm).cuda()
pg_a.load_state_dict(torch.load(pgsel("arssr"), map_location="cpu", weights_only=False)["state_dict"]); pg_a.eval()
LAM_PG = {"SAINT_P": pg_lambda("saint"), "ArSSR_P": pg_lambda("arssr")}
print("lambda PG", LAM_PG, flush=True)


def mix_lohi(mu, b, lam):
    mus = mu[:, :, com[1], :].reshape(mu.shape[0], -1).cpu()
    bs = b[:, :, com[1], :].reshape(b.shape[0], -1).cpu()
    lo_ = U.mix_quantile(mus, bs, 0.05, lam).clamp(0, 1).numpy().reshape(gt.shape[0], gt.shape[2])
    hi_ = U.mix_quantile(mus, bs, 0.95, lam).clamp(0, 1).numpy().reshape(gt.shape[0], gt.shape[2])
    return np.rot90(lo_), np.rot90(hi_)


for pattern in ("clustered",):
    ret = C.make_mask(pattern, R, p, 0)
    geo = C.gap_geometry(ret)
    seed = C._h("isbi-mc", p, C.cond_key(pattern, R, 0))
    lin = RC.linear(gt, ret)
    vin, G = RC.adapter_input(gt, ret, R)
    sag, cor = RC.saint_marginals(net, vin, G, R, chunk)
    mu_s, b_s = RC.saint_mc(prfn, sag, cor, proto["T"], seed)
    mu_a, b_a = N.arssr_mc(pm, gt, ret, proto["T"], seed, **inf)
    mu_sg, b_sg = PGm.saint_geo_mc(pg_s, sag, cor, ret, proto["T"], seed, 1.0)
    mu_ag, b_ag = PGm.arssr_geo_mc(pg_a, gt, ret, proto["T"], seed, C.D, 1.0, **inf)
    print("PG mean check", float((mu_sg - mu_s).abs().max()), float((mu_ag - mu_a).abs().max()), flush=True)
    res = {}
    grid = np.zeros(gt.shape, bool)
    grid[:, com[1], :] = True
    grid[:, :, ret] = False
    sub = np.nonzero(grid)
    for m, mu, b in (("SAINT_P", mu_s, b_s), ("ArSSR_P", mu_a, b_a), ("linear", None, None)):
        if mu is None:
            x = RC.copy_back(np.clip(lin, 0, 1), gt, ret)
        else:
            x = RC.copy_back(np.clip(mu.mean(0).cpu().numpy(), 0, 1), gt, ret)
        Fp = E.features(x, lin, gt, ret, geo, sub)
        r = dict(mean=plane(x), err=plane(np.abs(gt - x)))
        for kind in ("ph_nogeo", "ph_geo"):
            g, k, geo_flag = PH[m][kind]
            hw = np.full(gt.shape, np.nan, np.float32)
            hw[sub] = k * g(phi_me(Fp, geo_flag))
            lo_, hi_ = np.clip(x[sub] - hw[sub], 0, 1), np.clip(x[sub] + hw[sub], 0, 1)
            L_ = np.full(gt.shape, np.nan, np.float32); H_ = np.full(gt.shape, np.nan, np.float32)
            L_[sub], H_[sub] = lo_, hi_
            r[kind + "_lo"], r[kind + "_hi"] = plane(L_), plane(H_)
        if mu is not None:
            r["learned_lo"], r["learned_hi"] = mix_lohi(mu, b, PH[m]["lambda"])
            mg, bg_ = (mu_sg, b_sg) if m == "SAINT_P" else (mu_ag, b_ag)
            r["learned_geo_lo"], r["learned_geo_hi"] = mix_lohi(mg, bg_, LAM_PG[m])
        res[m] = r
    acq = np.zeros(gt.shape, bool)
    acq[:, :, ret] = True
    fg = plane(gt) > 0
    ys, xs = np.nonzero(fg)
    box = (max(ys.min() - 4, 0), ys.max() + 5, max(xs.min() - 4, 0), xs.max() + 5)
    cr = lambda a_: a_[box[0]:box[1], box[2]:box[3]]  # noqa: E731
    out = dict(gt=cr(plane(gt)), fg=cr(fg), acq=cr(plane(acq)), sparse=cr(plane(np.where(acq, gt, 0))))
    for m, r in res.items():
        for k, v in r.items():
            out[f"{m}__{k}"] = cr(v)
        ev = P.evaluate_volume(np.flip(np.rot90(np.zeros(1)), 0) if False else
                               (RC.copy_back(np.clip(lin, 0, 1), gt, ret) if m == "linear" else
                                RC.copy_back(np.clip((mu_s if m == "SAINT_P" else mu_a).mean(0).cpu().numpy(), 0, 1), gt, ret)),
                               gt, ret, C.D, "missing")
        out[f"{m}__psnr"] = np.float32(ev["psnr_masked"])
    out["ret"] = np.asarray(ret)
    out["box"] = np.asarray(box)
    np.savez(os.path.join(OUT, f"planes4_{pattern}.npz"), **out)
    print(pattern, {m: float(out[f"{m}__psnr"]) for m in res}, flush=True)
json.dump(dict(patient=p, R=R, view="coronal", consts={m: {k: (v[1] if isinstance(v, tuple) else v) for k, v in d.items()}
                                                         for m, d in PH.items()}),
          open(os.path.join(OUT, "meta4.json"), "w"), indent=1)
print("done")
