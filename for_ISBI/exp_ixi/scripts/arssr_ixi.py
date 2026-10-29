"""Uniform-trained ArSSR(-P) on IXI-tiny (added 2026-10-23). Mirrors how the IXI SAINT was made (BraTS model fine-tuned
on IXI) but with UNIFORM masks only, so it is matched to SAINT-P.

  --stage det   init = BraTS uniform-trained ArSSR (exp_uniform_arssr/runs/train/arssr_ext/selected.json); fine-tune on the
                436 IXI TRAIN subjects, uniform masks R {2,4,8} (w .30/.35/.35), D = 81, L1, batch 15 x 8000 queries,
                Adam 1e-4, 218-step warm-up, cosine, 4360 steps (= the V2 IXI ArSSR budget); select best SEL brain-PSNR
                (uniform x4 + x8, 10 SEL subjects).
  --stage prob  ProbArSSR (2 MCDropout + log-scale head), recipe of exp_uniform_arssr/scripts/train_uarssr_prob.py
                (2000 head-only + 3000 joint, lr 2e-5 / 1e-3, cosine, region-balanced Laplace NLL with the BRAIN mask
                as foreground); selection: phase-2, dropout-off SEL PSNR >= start-0.2 dB and MC(T=5) >= start-0.5 dB,
                min SEL brain NLL.
  --stage eval --role FIT|CAL|TEST   NAT__* arrays for every saved IXI case (same masks, voxel subsets, MC seeds as
                SAINT-P) -> exp_ixi/runs/eval_native/arssr/<ROLE>/{cases,rows}.
Inference tiling = ir3c_v2.baselines_v2.ARSSR_INFERENCE['IXI'].
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import ixi_isbi as X
from isbi import common as C
from isbi import evalcore as E
from isbi import models as M
from isbi import native as N
from isbi import recon as RC
from comparisons import protocol as P
from comparisons import arssr_adapter as AR
from IR3C_LPIPS import brats_methods as BM
from ir3c_v2 import baselines_v2 as BV

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["det", "prob", "eval"])
ap.add_argument("--role", default=None)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
INF = BV.ARSSR_INFERENCE["IXI"]
TRD = os.path.join(X.EXP, "runs", "train" + ("_smoke" if a.smoke else ""))
SEED, DEV = 42, "cuda"
P.seed_everything(SEED)
roles = X.roles()


class UniformSlabSamplerIXI(BV.ArSSRSlabSamplerV2):
    def _mask(self):
        import ir2_components as IC
        return IC.sample_mask_weighted(self.D, self.rng, self.strides, list(self.w), uniform_prob=1.0)


def log(wd, m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    open(os.path.join(wd, "train.log"), "a").write(m + "\n")


def train_preps(n=None):
    subs = roles["TRAIN"] if not a.smoke else roles["TRAIN"][:6]
    out = []
    for s in subs[:n]:
        v, br = X.load(s)
        out.append(dict(vol01=v, brain=br))
    return out


def sel_val():
    out = []
    for s in roles["SEL"][: 2 if a.smoke else None]:
        gt, br = X.load(s)
        for R in (4, 8):
            out.append((s, R, gt, br, C.uniform_mask(R, X.D)))
    return out


def recon_det(net, gt, ret):
    return AR.reconstruct(net, gt, ret, X.D, DEV, INF["z_chunk"], INF["z_margin"], INF["slice_batch"])


# ================================================================== det
if a.stage == "det":
    WD = os.path.join(TRD, "arssr_ext")
    os.makedirs(WD, exist_ok=True)
    STEPS, VE, WU = (4360, 545, 218) if not a.smoke else (10, 5, 2)
    init = json.load(open(os.path.join(C.ISBI, "exp_uniform_arssr", "runs", "train", "arssr_ext", "selected.json")))["selected_path"]
    net, _, _ = BM._load_arssr_slowlr()
    net.load_state_dict(torch.load(init, map_location="cpu", weights_only=False)["state_dict"])
    net.to(DEV)
    preps = train_preps()
    sampler = UniformSlabSamplerIXI(preps, X.D, strides=[2, 4, 8], stride_weights=[0.30, 0.35, 0.35], n_lr=4, base=48,
                                    sample_size=8000, seed=SEED)
    for _ in range(200):
        assert sampler._mask()[1]["mask_family"] == "U0"
    val = sel_val()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    rng = np.random.RandomState(SEED + 1)
    lr_fn = lambda s: 1e-4 * min(1.0, (s + 1) / WU) * 0.5 * (1 + np.cos(np.pi * s / STEPS))  # noqa: E731

    def validate(step):
        net.eval()
        with torch.no_grad():
            return pd.DataFrame([dict(step=step, subject=s, R=R,
                                      psnr=X.psnr_brain(X.copy_back(np.clip(recon_det(net, gt, ret), 0, 1), gt, ret), gt, br, ret),
                                      linear=X.psnr_brain(X.linear(gt, ret), gt, br, ret)) for s, R, gt, br, ret in val])

    hist = validate(0)
    best, best_step = hist.psnr.mean(), 0
    torch.save(dict(state_dict=net.state_dict(), step=0), os.path.join(WD, "ckpt_step000000.pt"))
    log(WD, f"step 0 (BraTS uniform ArSSR {init}): {best:.3f} dB, linear {hist.linear.mean():.3f}")
    tt = time.time()
    for s in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_fn(s)
        net.train()
        lr_t, xyz, v, _ = sampler.batch(rng.randint(0, len(preps), 15), DEV)
        loss = torch.nn.functional.l1_loss(net(lr_t, xyz), v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (s + 1) % VE == 0 or s + 1 == STEPS:
            v_ = validate(s + 1)
            hist = pd.concat([hist, v_])
            hist.to_csv(os.path.join(WD, "val_history.csv"), index=False)
            torch.save(dict(state_dict=net.state_dict(), step=s + 1), os.path.join(WD, f"ckpt_step{s + 1:06d}.pt"))
            if v_.psnr.mean() > best:
                best, best_step = v_.psnr.mean(), s + 1
            log(WD, f"step {s + 1} {time.time() - tt:.0f}s loss {float(loss.detach()):.5f} val {v_.psnr.mean():.3f} best {best:.3f}@{best_step}")
    lin = float(hist[hist.step == 0].linear.mean())
    json.dump(dict(selected_step=int(best_step), selected_path=os.path.join(WD, f"ckpt_step{best_step:06d}.pt"), val_psnr=float(best),
                   val_linear=lin, beats_linear=bool(best > lin), init=init, train_hours=(time.time() - tt) / 3600,
                   rule="best SEL brain-PSNR, uniform x4+x8; training masks uniform only"),
              open(os.path.join(WD, "selected.json"), "w"), indent=1)
    log(WD, f"SELECTED {best_step}: {best:.3f} vs linear {lin:.3f}")

# ================================================================== prob
if a.stage == "prob":
    WD = os.path.join(TRD, "arssr_prob")
    os.makedirs(WD, exist_ok=True)
    H1, H2, VE = (2000, 3000, 1000) if not a.smoke else (4, 4, 4)
    srcj = json.load(open(os.path.join(TRD, "arssr_ext", "selected.json")))
    base, _, _ = BM._load_arssr_slowlr()
    base.load_state_dict(torch.load(srcj["selected_path"], map_location="cpu", weights_only=False)["state_dict"])
    model = N.ProbArSSR(base).to(DEV)
    preps = train_preps()
    sampler = UniformSlabSamplerIXI(preps, X.D, strides=[2, 4, 8], stride_weights=[0.30, 0.35, 0.35], n_lr=4, base=48,
                                    sample_size=8000, seed=SEED + 3)
    val = sel_val()
    rng = np.random.RandomState(SEED + 11)

    def predict(gt, ret, T=0):
        mu, b = N.arssr_mc(model, gt, ret, T, 0, D=X.D, **INF)
        return (mu[0], b[0]) if T == 0 else (mu.mean(0), None)

    def validate(step):
        rows = []
        for s, R, gt, br, ret in val:
            mu, b = predict(gt, ret)
            x = X.copy_back(np.clip(mu.cpu().numpy(), 0, 1), gt, ret)
            miss = np.setdiff1d(np.arange(X.D), ret)
            m = torch.from_numpy(br[:, :, miss]).to(DEV)
            e = (torch.from_numpy(gt[:, :, miss]).to(DEV) - mu[:, :, miss]).abs()[m]
            bb = b[:, :, miss][m]
            mmc, _ = predict(gt, ret, 5)
            xm = X.copy_back(np.clip(mmc.cpu().numpy(), 0, 1), gt, ret)
            rows.append(dict(step=step, subject=s, R=R, psnr=X.psnr_brain(x, gt, br, ret), psnr_mc5=X.psnr_brain(xm, gt, br, ret),
                             nll=float((torch.log(2 * bb) + e / bb).mean()),
                             spearman=float(spearmanr(e[::10].cpu(), bb[::10].cpu()).correlation)))
        return pd.DataFrame(rows)

    def batch():
        lr_t, xyz, v, _ = sampler.batch(rng.randint(0, len(preps), 15), DEV)
        return lr_t, xyz, v, N.arssr_batch_weights(xyz)

    head_p = [q for n_, q in model.named_parameters() if "scale_head" in n_]
    base_p = [q for n_, q in model.named_parameters() if "scale_head" not in n_]
    opt = torch.optim.Adam([dict(params=base_p, lr=2e-5, base=2e-5), dict(params=head_p, lr=1e-3, base=1e-3)])
    sched = lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + np.cos(np.pi * s / H2))  # noqa: E731
    hist = validate(0)
    log(WD, f"step 0: psnr {hist.psnr.mean():.3f} mc5 {hist.psnr_mc5.mean():.3f} nll {hist.nll.mean():.3f}")
    tt = time.time()
    for s in range(H1 + H2):
        phaseA = s < H1
        for i, g in enumerate(opt.param_groups):
            g["lr"] = 0.0 if (phaseA and i == 0) else g["base"] * (1.0 if phaseA else sched(s - H1))
        M.set_mc(model, not phaseA)
        for q in base_p:
            q.requires_grad_(not phaseA)
        x, xyz, y, w = batch()
        mu, b = model(x, xyz)
        fg = (y > 0).float() * w           # fg/bg by intensity > 0 as in the BraTS recipe (the sampler does not expose the brain mask)
        loss = 0.5 * M.laplace_nll(y, mu, b, fg) + 0.5 * M.laplace_nll(y, mu, b, (y <= 0).float() * w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_p + head_p, 1.0)
        opt.step()
        if (s + 1) % VE == 0:
            v = validate(s + 1)
            hist = pd.concat([hist, v])
            hist.to_csv(os.path.join(WD, "val_history.csv"), index=False)
            torch.save(dict(state_dict=model.state_dict(), step=s + 1), os.path.join(WD, f"ckpt_step{s + 1:06d}.pt"))
            log(WD, f"step {s + 1} {time.time() - tt:.0f}s psnr {v.psnr.mean():.3f} mc5 {v.psnr_mc5.mean():.3f} nll {v.nll.mean():.3f} "
                    f"spearman {v.spearman.mean():.3f}")
    h = hist.groupby("step")[["psnr", "psnr_mc5", "nll"]].mean().reset_index()
    sp = float(h[h.step == 0].psnr.iloc[0])
    ok = h[(h.step > H1) & (h.psnr >= sp - 0.2) & (h.psnr_mc5 >= sp - 0.5)]
    sel = ok.sort_values("nll").iloc[0] if len(ok) else h[h.step > H1].sort_values("psnr_mc5").iloc[-1]
    json.dump(dict(selected_step=int(sel.step), selected_path=os.path.join(WD, f"ckpt_step{int(sel.step):06d}.pt"), start_psnr=sp,
                   sel_psnr=float(sel.psnr), sel_psnr_mc5=float(sel.psnr_mc5), sel_nll=float(sel.nll), n_eligible=int(len(ok)),
                   train_hours=(time.time() - tt) / 3600), open(os.path.join(WD, "selected.json"), "w"), indent=1)
    log(WD, f"SELECTED {int(sel.step)}")

# ================================================================== eval
if a.stage == "eval":
    sel = json.load(open(os.path.join(TRD, "arssr_prob", "selected.json")))
    dsel = json.load(open(os.path.join(TRD, "arssr_ext", "selected.json")))
    b2, _, _ = BM._load_arssr_slowlr()
    pm = N.ProbArSSR(b2).to(DEV)
    pm.load_state_dict(torch.load(sel["selected_path"], map_location="cpu", weights_only=False)["state_dict"])
    pm.eval()
    det, _, _ = BM._load_arssr_slowlr()
    det.load_state_dict(torch.load(dsel["selected_path"], map_location="cpu", weights_only=False)["state_dict"])
    det.to(DEV).eval()
    OUT = os.path.join(X.EXP, "runs", "eval_native", "arssr", a.role)
    for d in ("cases", "rows"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    files = sorted(glob.glob(os.path.join(X.EXP, "runs", "eval", a.role, "cases", "*.npz")))
    t0 = time.time()
    loaded = {}
    for k, f in enumerate(files):
        key = os.path.basename(f)[:-4]
        if os.path.exists(os.path.join(OUT, "rows", key + ".csv")):
            continue
        c = np.load(f)
        s, pat, R, rep = c["patient"].item(), c["pattern"].item(), int(c["R"]), int(c["rep"])
        if s not in loaded:
            loaded = {s: X.load(s)}
        gt, brain = loaded[s]
        ret = [int(r) for r in c["ret"]]
        mu, b = N.arssr_mc(pm, gt, ret, 20, C._h("isbi-ixi-mc", s, C.cond_key(pat, R, rep)), D=X.D, **INF)
        mean, epi, alea = RC.moments(mu, b)
        sub = np.unravel_index(c["idx"], gt.shape)
        st = [torch.from_numpy(np.asarray(v)).to(mu.device) for v in sub]
        x = X.copy_back(np.clip(mean.cpu().numpy(), 0, 1), gt, ret)
        with torch.no_grad():
            xd = X.copy_back(np.clip(recon_det(det, gt, ret), 0, 1), gt, ret)
        lin = X.linear(gt, ret)
        out = {"NAT__mu_t": mu[:, st[0], st[1], st[2]].cpu().numpy(), "NAT__b_t": b[:, st[0], st[1], st[2]].cpu().numpy(),
               "NAT__epi": epi[st[0], st[1], st[2]].cpu().numpy(), "NAT__alea": alea[st[0], st[1], st[2]].cpu().numpy(),
               "NAT__x": x[sub].astype(np.float32), "NAT__feat": E.features(x, lin, gt, ret, X.geometry(ret), sub)}
        np.savez(os.path.join(OUT, "cases", key + ".tmp.npz"), **out)
        os.replace(os.path.join(OUT, "cases", key + ".tmp.npz"), os.path.join(OUT, "cases", key + ".npz"))
        pd.DataFrame([dict(patient=s, pattern=pat, R=R, rep=rep, method=m, psnr_brain=X.psnr_brain(v, gt, brain, ret))
                      for m, v in (("ArSSR_P", x), ("ArSSR_det", xd))]).to_csv(os.path.join(OUT, "rows", key + ".csv"), index=False)
        if k % 50 == 0:
            print(f"[{a.role}] {k + 1}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    print(f"[{a.role}] complete", flush=True)
