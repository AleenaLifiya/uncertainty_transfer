"""Training-exposure arm: fine-tune the selected probabilistic models on uniform + irregular masks.

    python scripts/train_exposure.py --arch saint --variant a|b
    python scripts/train_exposure.py --arch i3net --R 4|8 --variant a|b
Start: the selected SAINT-P / I3Net-P checkpoint. Joint fine-tuning (mean + scale + trained dropout; SAINT MSR frozen),
region-balanced Laplace NLL on originally missing (non-copied) slices, 50 % uniform / 50 % irregular masks
(variant a: jittered + clustered; b: jittered only). Lr 2e-5 (source weights) / 1e-3 (scale head), cosine.
Selection (prespecified): lowest mean SEL foreground NLL over the variant's own mask families (uniform + its irregular
patterns, SEL patients only), subject to SEL uniform PSNR >= start checkpoint - 0.2 dB.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from isbi import common as C
from isbi import exposure as X
from isbi import models as M
from isbi import recon as RC
from comparisons import protocol as P

ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True, choices=["saint", "i3net"])
ap.add_argument("--R", type=int, default=None)
ap.add_argument("--variant", required=True, choices=["a", "b"])
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--val_every", type=int, default=500)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
name = f"{args.arch}_exp{args.variant}" + (f"_x{args.R}" if args.arch == "i3net" else "") + ("_smoke" if args.smoke else "")
if args.smoke:
    args.steps, args.val_every = 4, 4
WD = os.path.join(C.ISBI, "runs", "train", name)
os.makedirs(WD, exist_ok=True)
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
proto = json.load(open(os.path.join(C.ISBI, "configs", "protocol_frozen.json")))
roles = C.load_roles()
PATS = X.VARIANT_PATTERNS[args.variant]
ECACHE = os.path.join(C.ISBI, "cache", "data", "exposure_saint")
VCACHE = os.path.join(C.ISBI, "cache", "data", "sel_val")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    open(os.path.join(WD, "train.log"), "a").write(m + "\n")


# ------------------------------------------------------------------ SEL validation masks (SEL patients only)
VAL_R = (4, 8) if args.arch == "saint" else (args.R,)
val_masks = [("uniform", R, 0) for R in VAL_R] + [(p, R, 0) for R in VAL_R for p in PATS]
SELP = roles["SEL"][:2] if args.smoke else roles["SEL"]
val_gt = {p: P.prepare_one(C.scan_path(roles["scan_of"][p]))["vol01"] for p in SELP}

if args.arch == "saint":
    net, model, chunk, _ = RC.build_prob_saint(proto["selected_checkpoints"]["saint_prob"])
    for n_, p_ in net.named_parameters():
        p_.requires_grad = False
    idx = pd.read_csv(os.path.join(C.ROOT, "IR3C_more_comparisons/predictions/saint_msr_cache_index.csv"))
    uni = list(idx[idx.split == "train"].file)
    subj = sorted(idx[idx.split == "train"].subject.unique())
    if args.smoke:
        subj = subj[:2]
    # ---- irregular training cache (train subjects, frozen MSR on adapter inputs), built once, shared by a and b
    os.makedirs(ECACHE, exist_ok=True)
    gtf = {s: idx[(idx.split == "train") & (idx.subject == s) & (idx.R == 4)].file.iloc[0] for s in subj}
    irr = []
    t0 = time.time()
    for s in subj:
        gt = None
        for R in (4, 8):
            for pat in ("jittered", "clustered"):
                for r in range(2):
                    f = os.path.join(ECACHE, f"{s}_x{R}_{pat}_r{r}.npz")
                    if not os.path.exists(f):
                        if gt is None:
                            gt = np.load(os.path.join(C.ROOT, "IR3C_more_comparisons", gtf[s]))["gt"].astype(np.float32)
                        ret = X.train_mask(pat, R, np.random.RandomState(C._h("isbi-train-mask", s, R, pat, r)))
                        vin, G = RC.adapter_input(gt, ret, R)
                        sag, cor = RC.saint_marginals(net, vin, G, R, chunk)
                        np.savez(f + ".tmp.npz", sag=sag.astype(np.float16), cor=cor.astype(np.float16), ret=np.asarray(ret))
                        os.replace(f + ".tmp.npz", f)
                    if pat in PATS:
                        irr.append((f, gtf[s]))
    log(f"irregular cache ready ({len(irr)} entries for variant {args.variant}) {time.time() - t0:.0f}s")
    cache = {}

    def get(f, gfile=None):
        if f not in cache:
            z = np.load(f if gfile else os.path.join(C.ROOT, "IR3C_more_comparisons", f))
            gt = z["gt"] if not gfile else np.load(os.path.join(C.ROOT, "IR3C_more_comparisons", gfile))["gt"]
            miss = np.setdiff1d(np.arange(gt.shape[2]), z["ret"])
            cache[f] = (torch.from_numpy(np.stack([z["sag"], z["cor"]], 0)), torch.from_numpy(gt), miss)
        return cache[f]

    rng = np.random.RandomState(SEED + 5)

    def batch():
        if rng.rand() < 0.5:
            sc, gt, miss = get(uni[int(rng.randint(len(uni)))])
        else:
            f, g = irr[int(rng.randint(len(irr)))]
            sc, gt, miss = get(f, g)
        zs = rng.choice(miss, 8)
        x = sc[:, :, :, zs].permute(3, 0, 1, 2).to(RC.DEV).float()
        y = gt[:, :, zs].permute(2, 0, 1)[:, None].to(RC.DEV).float()
        return x, y, None

    fwd = model
    vmarg = {}

    def val_predict(p, pat, R, rep):
        key = (p, pat, R, rep)
        if key not in vmarg:
            f = os.path.join(VCACHE, f"saint_{roles['scan_of'][p]}_x{R}.npz" if pat == "uniform"
                             else f"saint_{roles['scan_of'][p]}_x{R}_{pat}_r{rep}.npz")
            if not os.path.exists(f):
                ret = C.make_mask(pat, R, p, rep)
                vin, G = RC.adapter_input(val_gt[p], ret, R)
                sag, cor = RC.saint_marginals(net, vin, G, R, chunk)
                np.savez(f + ".tmp.npz", sag=sag.astype(np.float16), cor=cor.astype(np.float16))
                os.replace(f + ".tmp.npz", f)
            z = np.load(f)
            vmarg[key] = (z["sag"].astype(np.float32), z["cor"].astype(np.float32))
        mu, b = RC.saint_mc(model, *vmarg[key], 0, 0)
        return mu[0], b[0]
else:
    model, _ = RC.build_prob_i3net(args.R, proto["selected_checkpoints"][f"i3net_prob_x{args.R}"])
    trs = P.load_splits()["train"]
    pick = sorted(np.random.RandomState(SEED).choice(len(trs), 160, replace=False).tolist())
    preps = [dict(vol01=P.prepare_one(trs[i])["vol01"]) for i in pick]
    samp = X.I3IrregularSampler(preps, args.R, PATS, 0.5, SEED + 7)
    log(f"prepared {len(preps)} train volumes")

    def batch():
        x, y, w = samp.batch(6, RC.DEV)
        return x, y, w

    def fwd(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, b = model(x)
        return mu.float(), b.float()

    def val_predict(p, pat, R, rep):
        ret = C.make_mask(pat, R, p, rep)
        vin, G = RC.adapter_input(val_gt[p], ret, R)
        mu, b = RC.i3net_mc(model, vin, G, R, 0, 0)
        return mu[0], b[0]


def validate(step):
    rows = []
    for p in SELP:
        gt = val_gt[p]
        for (pat, R, rep) in val_masks:
            ret = C.make_mask(pat, R, p, rep)
            mu, b = val_predict(p, pat, R, rep)
            x = RC.copy_back(np.clip(mu.cpu().numpy(), 0, 1), gt, ret)
            ev = P.evaluate_volume(x, gt, ret, C.D, region="missing")
            miss = np.setdiff1d(np.arange(C.D), ret)
            y = torch.from_numpy(gt[:, :, miss]).to(RC.DEV)
            fg = y > 0
            e = (y - mu[:, :, miss]).abs()[fg]
            bb = b[:, :, miss][fg]
            rows.append(dict(step=step, patient=p, pattern=pat, R=R, psnr=ev["psnr_masked"],
                             nll=float((torch.log(2 * bb) + e / bb).mean()),
                             spearman=float(spearmanr(e[::10].cpu(), bb[::10].cpu()).correlation),
                             e_over_b=float((e / bb).mean())))
    return pd.DataFrame(rows)


head_p = [p_ for n_, p_ in model.named_parameters() if "scale_head" in n_]
base_p = [p_ for n_, p_ in model.named_parameters() if "scale_head" not in n_ and p_.requires_grad]
opt = torch.optim.Adam([dict(params=base_p, lr=2e-5, base=2e-5), dict(params=head_p, lr=1e-3, base=1e-3)])
sched = lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + np.cos(np.pi * s / args.steps))  # noqa: E731
hist = validate(0)
hist.to_csv(os.path.join(WD, "val_history.csv"), index=False)
log(f"step 0: {hist.groupby(['pattern', 'R'])[['psnr', 'nll', 'spearman', 'e_over_b']].mean().round(4).to_dict()}")
t0 = time.time()
for step in range(args.steps):
    for g in opt.param_groups:
        g["lr"] = g["base"] * sched(step)
    M.set_mc(model, True)
    x, y, w = batch()
    mu, b = fwd(x)
    fgm = (y > 0).float() if w is None else (y > 0).float() * w
    bgm = (y <= 0).float() if w is None else (y <= 0).float() * w
    loss = 0.5 * M.laplace_nll(y, mu, b, fgm) + 0.5 * M.laplace_nll(y, mu, b, bgm)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(base_p + head_p, 1.0)
    opt.step()
    if (step + 1) % 50 == 0:
        open(os.path.join(WD, "train_loss.csv"), "a").write(f"{step + 1},{float(loss.detach()):.5f},{time.time() - t0:.1f}\n")
    if (step + 1) % args.val_every == 0:
        v = validate(step + 1)
        hist = pd.concat([hist, v])
        hist.to_csv(os.path.join(WD, "val_history.csv"), index=False)
        torch.save(dict(state_dict=model.state_dict(), step=step + 1), os.path.join(WD, f"ckpt_step{step + 1:06d}.pt"))
        log(f"step {step + 1} {time.time() - t0:.0f}s {v.groupby(['pattern', 'R'])[['psnr', 'nll', 'spearman', 'e_over_b']].mean().round(4).to_dict()}")
h = hist.groupby("step")
src_psnr = hist[(hist.step == 0) & (hist.pattern == "uniform")].psnr.mean()
cand = []
for s_, g in h:
    if s_ == 0:
        continue
    upsnr = g[g.pattern == "uniform"].psnr.mean()
    cand.append(dict(step=s_, nll=g.nll.mean(), uniform_psnr=upsnr, ok=upsnr >= src_psnr - 0.2))
cand = pd.DataFrame(cand)
ok = cand[cand.ok]
sel = (ok if len(ok) else cand).sort_values("nll").iloc[0]
json.dump(dict(name=name, selected_step=int(sel.step), selected_path=os.path.join(WD, f"ckpt_step{int(sel.step):06d}.pt"),
               start_uniform_psnr=float(src_psnr), sel=sel.to_dict(), n_eligible=int(len(ok)), patterns=PATS,
               rule="min SEL fg NLL over variant families; SEL uniform PSNR >= start - 0.2 dB", train_sec=time.time() - t0),
          open(os.path.join(WD, "selected.json"), "w"), indent=1, default=float)
log(f"SELECTED step {int(sel.step)}")
