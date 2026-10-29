"""Uniform-trained ArSSR, deterministic, FROM SCRATCH (added 2026-10-22 at the user's request).

Purpose: a training-matched counterpart to SAINT-P (uniform masks only). Adapted from scripts/extend_native.py
(read-only); the ONLY intended difference from the existing (mixed-mask) ArSSR is the training-mask distribution:
  * init      = the IR3C run's step-0 random initialisation (arssr_slowlr_step000000.pt), i.e. the same start point
                as the mixed-mask ArSSR -- no pretrained or mixed-trained weights;
  * masks     = UNIFORM ONLY, R in {2,4,8} with weights [0.30,0.35,0.35] (the original sampler draws 50 % uniform /
                50 % U1..U6); slab/crop sampling, batch 15 x 8000 queries, L1 loss, 400 train volumes (seed 42) unchanged;
  * budget    = 35,000 steps (~ the mixed model's 12,510 + 22,500), Adam, lr 1e-4, 1000-step warm-up, cosine to 0;
  * selection = best SEL validation PSNR on UNIFORM x4 and x8 masks only (no irregular data anywhere).
Outputs: <WD>/ckpt_step*.pt, val_history.csv, selected.json (name 'arssr_ext' so eval_native.py can read it via
ISBI_TRAIN_DIR without modification).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from isbi import common as C
from comparisons import protocol as P
from comparisons import arssr_adapter as AR
from IR3C_LPIPS import brats_methods as BM

ap = argparse.ArgumentParser()
ap.add_argument("--max_steps", type=int, default=35000)
ap.add_argument("--val_every", type=int, default=2500)
ap.add_argument("--n_vol", type=int, default=400)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
if a.smoke:
    a.max_steps, a.val_every, a.n_vol = 20, 10, 8
EXP = os.path.join(C.ISBI, "exp_uniform_arssr")
WD = os.path.join(EXP, "runs", "train" + ("_smoke" if a.smoke else ""), "arssr_ext")
os.makedirs(WD, exist_ok=True)
SEED = 42
P.seed_everything(SEED)
DEV = "cuda"
INIT = os.path.join(C.ROOT, "runs", "brats_ir3c_20260902", "checkpoints", "arssr_slowlr", "arssr_slowlr_step000000.pt")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    open(os.path.join(WD, "train.log"), "a").write(m + "\n")


class UniformSlabSampler(AR.SlabSampler):
    """AR.SlabSampler with the mask draw restricted to the uniform family (everything else inherited)."""

    def _mask(self):
        import ir2_components as IC
        return IC.sample_mask_weighted(P.DEPTH, self.rng, self.strides, list(self.w), uniform_prob=1.0)


roles = C.load_roles()
VALP = roles["SEL"][:2 if a.smoke else 6]
val_gt = {p: P.prepare_one(C.scan_path(roles["scan_of"][p]))["vol01"] for p in VALP}
cells = [(R, C.uniform_mask(R)) for R in (4, 8)]
trs = P.load_splits()["train"]
pick = sorted(np.random.RandomState(SEED).choice(len(trs), a.n_vol, replace=False).tolist())
t0 = time.time()
preps = [dict(vol01=P.prepare_one(trs[i])["vol01"]) for i in pick]
log(f"prepared {len(preps)} train volumes {time.time() - t0:.0f}s")

net, inf, _ = BM._load_arssr_slowlr()                      # architecture + inference settings only
net.load_state_dict(torch.load(INIT, map_location="cpu", weights_only=False)["state_dict"])  # random init
net.to(DEV)
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
sampler = UniformSlabSampler(preps, [2, 4, 8], [0.30, 0.35, 0.35], 4, 48, 8000, seed=SEED)
for _ in range(200):                                       # guard: the sampler must only ever produce uniform masks
    ret, meta = sampler._mask()
    assert meta["mask_family"] == "U0", meta
rng = np.random.RandomState(SEED + 1)
lr_fn = lambda s: 1e-4 * min(1.0, (s + 1) / 1000) * 0.5 * (1 + np.cos(np.pi * s / a.max_steps))  # noqa: E731


def validate(s):
    net.eval()
    rows = []
    with torch.no_grad():
        for p, gt in val_gt.items():
            for R, ret in cells:
                x = AR.reconstruct(net, gt, ret, C.D, DEV, **inf)
                rows.append(dict(step=s, patient=p, R=R,
                                 psnr=P.evaluate_volume(x, gt, ret, C.D, "missing")["psnr_masked"],
                                 linear=P.evaluate_volume(P.interpolate_baseline(gt, ret, "linear", C.D), gt, ret, C.D,
                                                          "missing")["psnr_masked"]))
    return pd.DataFrame(rows)


hist = validate(0)
best, best_step = -1e9, 0
log(f"step 0 (random init {INIT}): val {hist.psnr.mean():.3f} dB vs linear {hist.linear.mean():.3f} dB")
tt = time.time()
for s in range(a.max_steps):
    for g in opt.param_groups:
        g["lr"] = lr_fn(s)
    net.train()
    lr_t, xyz, v, _ = sampler.batch(rng.randint(0, len(preps), 15), DEV)
    loss = torch.nn.functional.l1_loss(net(lr_t, xyz), v)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if (s + 1) % 100 == 0:
        open(os.path.join(WD, "train_loss.csv"), "a").write(f"{s + 1},{float(loss.detach()):.6f},{time.time() - tt:.1f}\n")
    if (s + 1) % a.val_every == 0 or s + 1 == a.max_steps:
        v_ = validate(s + 1)
        hist = pd.concat([hist, v_])
        hist.to_csv(os.path.join(WD, "val_history.csv"), index=False)
        torch.save(dict(state_dict=net.state_dict(), step=s + 1), os.path.join(WD, f"ckpt_step{s + 1:06d}.pt"))
        m = v_.psnr.mean()
        if m > best:
            best, best_step = m, s + 1
        log(f"step {s + 1} loss {float(loss):.5f} {time.time() - tt:.0f}s val {m:.3f} (linear {v_.linear.mean():.3f}) "
            f"best {best:.3f}@{best_step}")
lin = hist[hist.step == 0].linear.mean()
json.dump(dict(name="arssr_ext (UNIFORM-trained, from scratch)", init=INIT, selected_step=int(best_step),
               selected_path=os.path.join(WD, f"ckpt_step{best_step:06d}.pt"), val_psnr=float(best), val_linear=float(lin),
               beats_linear=bool(best > lin), steps_run=int(a.max_steps), train_hours=(time.time() - tt) / 3600,
               rule="best SEL val PSNR, uniform x4+x8 masks only; training masks uniform only"),
          open(os.path.join(WD, "selected.json"), "w"), indent=1)
log(f"SELECTED step {best_step}: {best:.3f} dB vs linear {lin:.3f} dB")
