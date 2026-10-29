"""Fine-tune a probabilistic adaptation (Laplace NLL + trained MC dropout), resumable.

    python scripts/train_prob.py --arch saint            (one model, all strides, like the source)
    python scripts/train_prob.py --arch i3net --R 4|8    (one model per scale, like the source)

Training data: GLI TrainingData train split ONLY, uniform (U0) masks ONLY.
  SAINT: the read-only MSR cache of the source experiment (40 train subjects x R{2,4,8}, U0),
         i.e. exactly the RFN's original training inputs; NLL on MISSING slices of whole axial slices.
  I3Net: N_TRAIN_VOL seeded train volumes, upstream run sampler (gt[..., ::R]); NLL on the
         non-copied output slices of each run (all voxels, background included).
Validation / selection: SEL patients (uniform x4 and x8), dropout OFF.
  rule (prespecified): among checkpoints with val PSNR >= source PSNR - 0.2 dB and val background
  haze <= source + 0.002, take the lowest val NLL (missing foreground voxels); if none, highest PSNR.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from isbi import common as C
from isbi import models as M
from isbi import recon as RC

ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True, choices=["saint", "i3net"])
ap.add_argument("--R", type=int, default=None)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--val_every", type=int, default=500)
ap.add_argument("--lr", type=float, default=2e-5, help="pretrained (source) weights")
ap.add_argument("--lr_head", type=float, default=1e-3, help="new scale head (log-scale parametrisation)")
ap.add_argument("--head_warm", type=int, default=500, help="phase A: scale head only, mean frozen, dropout off")
ap.add_argument("--bs", type=int, default=None)
ap.add_argument("--n_train_vol", type=int, default=160)
ap.add_argument("--pilot", type=int, default=0, help="time N steps + 1 validation, write nothing durable")
args = ap.parse_args()
name = f"{args.arch}_prob" + (f"_x{args.R}" if args.arch == "i3net" else "")
WD = os.path.join(C.ISBI, "runs", "train" + ("_pilot" if args.pilot else ""), name)
os.makedirs(WD, exist_ok=True)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
from comparisons import protocol as P          # noqa: E402

roles = C.load_roles()
SEL = [roles["scan_of"][p] for p in roles["SEL"]]
VAL_R = (4, 8) if args.arch == "saint" else (args.R,)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    with open(os.path.join(WD, "train.log"), "a") as f:
        f.write(msg + "\n")


# ------------------------------------------------------------------ validation data
t0 = time.time()
val = []
vcache = os.path.join(C.ISBI, "cache", "data", "sel_val")
os.makedirs(vcache, exist_ok=True)
for s in SEL:
    gt = P.prepare_one(C.scan_path(s))["vol01"]
    val.append((s, gt))
log(f"loaded {len(val)} SEL volumes in {time.time() - t0:.0f}s")

# ------------------------------------------------------------------ model + data
if args.arch == "saint":
    net, model, chunk, src_sel = RC.build_prob_saint()
    params = list(model.parameters())
    bs = args.bs or 8
    idx = pd.read_csv(os.path.join(C.ROOT, "IR3C_more_comparisons/predictions/saint_msr_cache_index.csv"))
    tr = idx[idx.split == "train"].reset_index(drop=True)
    cache = {}

    def load_cache(f):
        if f not in cache:
            z = np.load(os.path.join(C.ROOT, "IR3C_more_comparisons", f))
            miss = np.setdiff1d(np.arange(z["gt"].shape[2]), z["ret"])
            cache[f] = (torch.from_numpy(np.stack([z["sag"], z["cor"]], 0)),       # float16 as stored
                        torch.from_numpy(z["gt"]), miss)
        return cache[f]

    rng = np.random.RandomState(SEED + 2)

    def batch():
        f = tr.file[int(rng.randint(len(tr)))]
        sc, gt, miss = load_cache(f)
        zs = rng.choice(miss, bs)
        x = sc[:, :, :, zs].permute(3, 0, 1, 2).to(RC.DEV).float()  # (bs,2,H,W)
        y = gt[:, :, zs].permute(2, 0, 1)[:, None].to(RC.DEV).float()
        return x, y, None

    def fwd(x):
        return model(x)

    # SEL marginals cache (frozen MSR, deterministic)
    vmarg = {}
    for s, gt in val:
        for R in VAL_R:
            p = os.path.join(vcache, f"saint_{s}_x{R}.npz")
            if not os.path.exists(p):
                sag, cor = RC.saint_marginals(net, gt, C.uniform_mask(R), R, chunk)
                np.savez(p + ".tmp.npz", sag=sag.astype(np.float16), cor=cor.astype(np.float16))
                os.replace(p + ".tmp.npz", p)
            z = np.load(p)
            vmarg[(s, R)] = (z["sag"].astype(np.float32), z["cor"].astype(np.float32))
    log(f"SEL marginals ready {time.time() - t0:.0f}s")

    def val_predict(s, gt, R):
        mu, b = RC.saint_mc(model, *vmarg[(s, R)], 0, 0)
        return mu[0], b[0]
else:
    R = args.R
    model, src_sel = RC.build_prob_i3net(R)
    params = list(model.parameters())
    bs = args.bs or 6
    from comparisons import i3net_adapter as I3
    from ldm.models.diffusion import fidelity_runtime as FR
    trs = P.load_splits()["train"]
    rs = np.random.RandomState(SEED)
    pick = sorted(rs.choice(len(trs), args.n_train_vol if not args.pilot else 12, replace=False).tolist())
    preps = []
    for i in pick:
        preps.append(dict(vol01=P.prepare_one(trs[i])["vol01"]))
    log(f"prepared {len(preps)} train volumes {time.time() - t0:.0f}s")
    sampler = I3.RunSampler(preps, R, 4, seed=SEED, cache_cap=len(preps))
    rng = np.random.RandomState(SEED + 1)
    S = sampler.hr
    wmask = torch.ones(1, 1, 1, S, device=RC.DEV)
    wmask[..., ::R] = 0                                  # copied (acquired) slices not supervised

    def batch():
        lr, gt = sampler.batch(rng.randint(0, len(preps), bs), RC.DEV)
        return lr, gt, wmask.expand_as(gt)

    def fwd(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, b = model(x)
        return mu.float(), b.float()

    def val_predict(s, gt, R):
        mu, b = RC.i3net_mc(model, gt, C.uniform_mask(R), R, 0, 0)
        return mu[0], b[0]


def validate(step):
    rows = []
    for s, gt in val:
        for R in VAL_R:
            ret = C.uniform_mask(R)
            mu, b = val_predict(s, gt, R)
            x = RC.copy_back(np.clip(mu.cpu().numpy(), 0, 1), gt, ret)
            ev = P.evaluate_volume(x, gt, ret, C.D, region="missing")
            miss = np.setdiff1d(np.arange(C.D), ret)
            y = torch.from_numpy(gt[:, :, miss]).to(RC.DEV)
            fg = y > 0
            nll = M.laplace_nll(y[fg], mu[:, :, miss][fg], b[:, :, miss][fg]).item()
            ee = (y - mu[:, :, miss]).abs()[fg][::10].cpu().numpy()
            bb = b[:, :, miss][fg][::10].cpu().numpy()
            from scipy.stats import spearmanr
            rows.append(dict(step=step, subject=s, R=R, psnr=ev["psnr_masked"], ssim=ev["ssim"],
                             spearman=float(spearmanr(ee, bb).correlation) if bb.std() > 0 else np.nan,
                             e_over_b=float((ee / bb).mean()),
                             mae=ev["mae"], bg_haze=ev["bg_haze"], nll=nll,
                             b_med=float(b[:, :, miss][fg].median()),
                             b_bg_med=float(b[:, :, miss][~fg].median())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ resume
head_p = [p for n, p in model.named_parameters() if "scale_head" in n]
base_p = [p for n, p in model.named_parameters() if "scale_head" not in n]
opt = torch.optim.Adam([dict(params=base_p, lr=args.lr, base=args.lr),
                        dict(params=head_p, lr=args.lr_head, base=args.lr_head)])
sched = lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + np.cos(np.pi * min(s, args.steps - args.head_warm) / (args.steps - args.head_warm)))  # noqa
step0 = 0
last = os.path.join(WD, "last.pt")
if os.path.exists(last) and not args.pilot:
    ck = torch.load(last, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"]); step0 = ck["step"]
    rng.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
    log(f"resumed at step {step0}")
hist_p = os.path.join(WD, "val_history.csv")
if step0 == 0:
    v = validate(0)
    v.to_csv(hist_p, index=False)
    log(f"step 0 (source-equivalent mean): {v.groupby('R')[['psnr','nll','bg_haze']].mean().to_dict()}")
    torch.save(dict(state_dict=model.state_dict(), step=0), os.path.join(WD, "ckpt_step000000.pt"))

# ------------------------------------------------------------------ train
tl, tt = [], time.time()
n_steps = args.pilot or args.steps
for step in range(step0, n_steps):
    phaseA = step < args.head_warm
    for i, g in enumerate(opt.param_groups):
        g["lr"] = (0.0 if (phaseA and i == 0) else g["base"] * (1.0 if phaseA else sched(step - args.head_warm)))
    M.set_mc(model, not phaseA)          # BN-free nets: train() only matters for dropout
    for p_ in base_p:
        p_.requires_grad_(not phaseA)
    x, y, w = batch()
    mu, b = fwd(x)
    fgm = (y > 0).float() if w is None else (y > 0).float() * w
    bgm = (y <= 0).float() if w is None else (y <= 0).float() * w
    # region-balanced NLL: foreground and background each weigh 1/2 (GT masks are training labels only)
    loss = 0.5 * M.laplace_nll(y, mu, b, fgm) + 0.5 * M.laplace_nll(y, mu, b, bgm)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    tl.append(float(loss.detach()))
    if not np.isfinite(tl[-1]):
        raise SystemExit(f"non-finite loss at step {step}")
    s1 = step + 1
    if s1 % 50 == 0:
        with open(os.path.join(WD, "train_loss.csv"), "a") as f:
            f.write(f"{s1},{np.mean(tl[-50:]):.6f},{opt.param_groups[0]['lr']:.3e},{time.time() - tt:.1f}\n")
    if args.pilot and s1 == args.pilot:
        dt = (time.time() - tt) / args.pilot
        tv = time.time(); v = validate(s1); dv = time.time() - tv
        json.dump(dict(sec_per_step=dt, sec_per_validation=dv, peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                       bs=bs, val=v.groupby("R")[["psnr", "nll", "spearman", "e_over_b", "b_med"]].mean().to_dict()),
                  open(os.path.join(WD, "pilot.json"), "w"), indent=1)
        torch.save(dict(state_dict=model.state_dict(), step=s1), os.path.join(WD, "pilot_last.pt"))
        log(f"PILOT {dt:.3f}s/step, validation {dv:.0f}s")
        break
    if s1 % args.val_every == 0 or s1 == args.steps:
        v = validate(s1)
        pd.concat([pd.read_csv(hist_p), v]).to_csv(hist_p, index=False)
        torch.save(dict(state_dict=model.state_dict(), step=s1),
                   os.path.join(WD, f"ckpt_step{s1:06d}.pt"))
        torch.save(dict(state_dict=model.state_dict(), opt=opt.state_dict(), step=s1,
                        np_rng=rng.get_state(), torch_rng=torch.get_rng_state()), last + ".tmp")
        os.replace(last + ".tmp", last)
        log(f"step {s1} loss {np.mean(tl[-args.val_every:]):.4f} "
            f"val {v.groupby('R')[['psnr','nll','bg_haze','b_med','spearman','e_over_b']].mean().round(4).to_dict()}")

# ------------------------------------------------------------------ select
if not args.pilot:
    h = pd.read_csv(hist_p).groupby("step")[["psnr", "nll", "bg_haze"]].mean().reset_index()
    src = h[h.step == 0].iloc[0]
    ok = h[(h.step > args.head_warm) & (h.psnr >= src.psnr - 0.2) & (h.bg_haze <= src.bg_haze + 0.002)]  # dropout must be trained
    sel = ok.sort_values("nll").iloc[0] if len(ok) else h[h.step > args.head_warm].sort_values("psnr").iloc[-1]
    out = dict(name=name, selected_step=int(sel.step), val=sel.to_dict(), source_val=src.to_dict(),
               n_eligible=int(len(ok)), rule="phase-2 checkpoints only; min val NLL s.t. PSNR >= src-0.2 dB and haze <= src+0.002", head_warm=args.head_warm, lr_head=args.lr_head,
               steps=args.steps, lr=args.lr, bs=bs, source=str(src_sel.get("selected_path")),
               selected_path=os.path.join(WD, f"ckpt_step{int(sel.step):06d}.pt"),
               train_hours=None)
    json.dump(out, open(os.path.join(WD, "selected.json"), "w"), indent=1, default=float)
    log(f"SELECTED {out}")
