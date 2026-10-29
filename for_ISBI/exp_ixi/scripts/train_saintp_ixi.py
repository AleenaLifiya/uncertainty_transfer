"""SAINT-P on IXI-tiny: probabilistic fine-tuning of the IXI SAINT RFN (copy of scripts/train_prob.py, SAINT branch).

Recipe identical to BraTS SAINT-P: frozen MSR; RFN + 2 Dropout2d (p=0.1) + log-scale head; phase 1 scale head only
(2000 steps, mean frozen, dropout off), phase 2 joint (3000 steps, dropout on), lr 2e-5 source / 1e-3 head, cosine,
grad clip 1.0, batch 8 axial slices; region-balanced Laplace NLL (1/2 brain, 1/2 non-brain) on missing slices.
IXI differences (disclosed): training inputs = MSR marginals of UNIFORM R 2/4/8 masks on 120 seeded IXI TRAIN subjects
(computed here, D = 81); the foreground/background split uses the brain mask (IXI is not skull-stripped);
selection = phase-2 checkpoints with SEL (10 val subjects, uniform x4+x8, dropout off) brain-PSNR >= source - 0.2 dB,
lowest SEL brain NLL (the BraTS background-haze guard is not meaningful on unstripped IXI and is dropped).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import ixi_isbi as X
from isbi import common as C
from isbi import models as M
from isbi import recon as RC

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
H1, H2, VAL_EVERY, NTR = (2000, 3000, 500, 120) if not a.smoke else (4, 4, 4, 3)
WD = os.path.join(X.EXP, "runs", "train" + ("_smoke" if a.smoke else ""), "saint_prob")
CACHE = os.path.join(X.EXP, "cache", "train_marginals")
os.makedirs(WD, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    open(os.path.join(WD, "train.log"), "a").write(m + "\n")


roles = X.roles()
net = X.load_saint()
model = X.build_prob(net)
# ------------------------------------------------------------------ training cache (uniform masks only)
tr = sorted(np.random.RandomState(SEED).choice(roles["TRAIN"], NTR, replace=False).tolist())
files = []
t0 = time.time()
for s in tr:
    gt = brain = None
    for R in (2, 4, 8):
        f = os.path.join(CACHE, f"{s}_R{R}.npz")
        if not os.path.exists(f):
            if gt is None:
                gt, brain = X.load(s)
            ret = C.uniform_mask(R, X.D)
            sag, cor = X.saint_marginals(net, gt, ret, R)
            np.savez(f, sag=sag.astype(np.float16), cor=cor.astype(np.float16), gt=gt, brain=brain, ret=np.asarray(ret))
        files.append(f)
log(f"train cache: {len(files)} files ({len(tr)} subjects x R 2/4/8) in {time.time() - t0:.0f}s")
cache = {}


def get(f):
    if f not in cache:
        z = np.load(f)
        miss = np.setdiff1d(np.arange(X.D), z["ret"])
        cache[f] = (torch.from_numpy(np.stack([z["sag"], z["cor"]], 0).astype(np.float32)),
                    torch.from_numpy(z["gt"]), torch.from_numpy(z["brain"]), miss)
    return cache[f]


rng = np.random.RandomState(SEED + 2)


def batch(bs=8):
    sc, gt, br, miss = get(files[int(rng.randint(len(files)))])
    zs = rng.choice(miss, bs)
    x = sc[:, :, :, zs].permute(3, 0, 1, 2).to(X.DEV)
    y = gt[:, :, zs].permute(2, 0, 1)[:, None].to(X.DEV)
    b = br[:, :, zs].permute(2, 0, 1)[:, None].to(X.DEV).float()
    return x, y, b


# ------------------------------------------------------------------ validation (SEL, uniform x4/x8, dropout off)
val = []
for s in roles["SEL"]:
    gt, brain = X.load(s)
    for R in (4, 8):
        ret = C.uniform_mask(R, X.D)
        sag, cor = X.saint_marginals(net, gt, ret, R)
        val.append((s, R, gt, brain, ret, sag, cor))


def validate(step):
    rows = []
    for s, R, gt, brain, ret, sag, cor in val:
        mu, b = RC.saint_mc(model, sag, cor, 0, 0)
        mu, b = mu[0], b[0]
        x = X.copy_back(np.clip(mu.cpu().numpy(), 0, 1), gt, ret)
        miss = np.setdiff1d(np.arange(X.D), ret)
        y = torch.from_numpy(gt[:, :, miss]).to(X.DEV)
        m = torch.from_numpy(brain[:, :, miss]).to(X.DEV)
        e = (y - mu[:, :, miss]).abs()[m]
        bb = b[:, :, miss][m]
        rows.append(dict(step=step, subject=s, R=R, psnr=X.psnr_brain(x, gt, brain, ret),
                         nll=float((torch.log(2 * bb) + e / bb).mean()), e_over_b=float((e / bb).mean())))
    return pd.DataFrame(rows)


head_p = [q for n, q in model.named_parameters() if "scale_head" in n]
base_p = [q for n, q in model.named_parameters() if "scale_head" not in n]
opt = torch.optim.Adam([dict(params=base_p, lr=2e-5, base=2e-5), dict(params=head_p, lr=1e-3, base=1e-3)])
sched = lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + np.cos(np.pi * s / H2))  # noqa: E731
hist = validate(0)
hist.to_csv(os.path.join(WD, "val_history.csv"), index=False)
log(f"step 0 (= source SAINT): psnr {hist.psnr.mean():.3f} nll {hist.nll.mean():.3f}")
tt = time.time()
for s in range(H1 + H2):
    phaseA = s < H1
    for i, g in enumerate(opt.param_groups):
        g["lr"] = 0.0 if (phaseA and i == 0) else g["base"] * (1.0 if phaseA else sched(s - H1))
    M.set_mc(model, not phaseA)
    for q in base_p:
        q.requires_grad_(not phaseA)
    x, y, br = batch()
    mu, b = model(x)
    loss = 0.5 * M.laplace_nll(y, mu, b, br) + 0.5 * M.laplace_nll(y, mu, b, 1 - br)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(base_p + head_p, 1.0)
    opt.step()
    if (s + 1) % VAL_EVERY == 0:
        v = validate(s + 1)
        hist = pd.concat([hist, v])
        hist.to_csv(os.path.join(WD, "val_history.csv"), index=False)
        torch.save(dict(state_dict=model.state_dict(), step=s + 1), os.path.join(WD, f"ckpt_step{s + 1:06d}.pt"))
        log(f"step {s + 1} {time.time() - tt:.0f}s loss {float(loss):.3f} psnr {v.psnr.mean():.3f} nll {v.nll.mean():.3f} "
            f"e/b {v.e_over_b.mean():.3f}")
h = hist.groupby("step")[["psnr", "nll"]].mean().reset_index()
src = h[h.step == 0].iloc[0]
ok = h[(h.step > H1) & (h.psnr >= src.psnr - 0.2)]
sel = ok.sort_values("nll").iloc[0] if len(ok) else h[h.step > H1].sort_values("psnr").iloc[-1]
json.dump(dict(selected_step=int(sel.step), selected_path=os.path.join(WD, f"ckpt_step{int(sel.step):06d}.pt"),
               sel_psnr=float(sel.psnr), sel_nll=float(sel.nll), src_psnr=float(src.psnr), src_nll=float(src.nll),
               n_eligible=int(len(ok)), rule="phase-2; SEL brain-PSNR >= source-0.2 dB; min SEL brain NLL",
               train_hours=(time.time() - tt) / 3600), open(os.path.join(WD, "selected.json"), "w"), indent=1)
log(f"SELECTED step {int(sel.step)}")
