"""Step 3a: extend the deterministic training of ArSSR (slow-LR) and DCSRN-mixed beyond the old 12,510-step cap
(user, 2026-10-19: train to a better budget so they beat interpolation; separate venue from IR3C).

    python scripts/extend_native.py --arch arssr
    python scripts/extend_native.py --arch dcsrn --R 4|8

Start: the selected BraTS checkpoint of the source run (read-only). Recipe = the model's own: ArSSR L1, Adam,
batch 15 slabs x 8000 queries, SlabSampler (R 2/4/8, 50 % uniform / 50 % U1-U6) -- with a fresh cosine LR from 5e-5
(1000-step warm-up; the source schedule had decayed to 6.25e-6 while still improving); DCSRN L2, Adam lr 1e-4 constant (its own probed
value), batch 2 x 64^3, MixedMaskPatchSampler (uniform_prob 0.5). Training volumes: N_VOL seeded train-split volumes.
Validation (SEL patients 0-5, never test): own mask families U0/U2/U4/U6 (protocol masks) at x4 and x8 (ArSSR) or
the specialist's R (DCSRN), brain-masked PSNR on missing slices with the model's own inference; linear on the same
cells. Selection: best val PSNR; early stop after PATIENCE validations without improvement; hard cap MAX_STEPS.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from isbi import common as C
from comparisons import protocol as P
from comparisons import arssr_adapter as AR
from comparisons_more import dcsrn_adapter as DC
from IR3C_LPIPS import brats_methods as BM

ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True, choices=["arssr", "dcsrn"])
ap.add_argument("--R", type=int, default=None)
ap.add_argument("--max_steps", type=int, default=40000)
ap.add_argument("--val_every", type=int, default=2500)
ap.add_argument("--patience", type=int, default=4)
ap.add_argument("--n_vol", type=int, default=400)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
if a.smoke:
    a.max_steps, a.val_every, a.n_vol = 20, 10, 8
name = "arssr_ext" if a.arch == "arssr" else f"dcsrn_ext_x{a.R}"
WD = os.path.join(C.ISBI, "runs", "train" + ("_smoke" if a.smoke else ""), name)
os.makedirs(WD, exist_ok=True)
SEED = 42
P.seed_everything(SEED)
DEV = "cuda"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    open(os.path.join(WD, "train.log"), "a").write(m + "\n")


roles = C.load_roles()
VALP = roles["SEL"][:2 if a.smoke else 6]
val_gt = {p: P.prepare_one(C.scan_path(roles["scan_of"][p]))["vol01"] for p in VALP}
VAL_R = (4, 8) if a.arch == "arssr" else (a.R,)
cells = [(R, fam, P.build_mask(C.D, R, fam)) for R in VAL_R for fam in ("U0", "U2", "U4", "U6")]

trs = P.load_splits()["train"]
pick = sorted(np.random.RandomState(SEED).choice(len(trs), a.n_vol, replace=False).tolist())
t0 = time.time()
preps = [dict(vol01=P.prepare_one(trs[i])["vol01"]) for i in pick]
log(f"prepared {len(preps)} train volumes {time.time() - t0:.0f}s")

if a.arch == "arssr":
    net, inf, sel0 = BM._load_arssr_slowlr()
    opt = torch.optim.Adam(net.parameters(), lr=5e-5)
    sampler = AR.SlabSampler(preps, [2, 4, 8], [0.30, 0.35, 0.35], 4, 48, 8000, seed=SEED)
    rng = np.random.RandomState(SEED + 1)
    lr_fn = lambda s: 5e-5 * min(1.0, (s + 1) / 1000) * 0.5 * (1 + np.cos(np.pi * s / a.max_steps))  # noqa: E731

    def step():
        lr_t, xyz, v, _ = sampler.batch(rng.randint(0, len(preps), 15), DEV)
        return torch.nn.functional.l1_loss(net(lr_t, xyz), v)

    recon = lambda gt, ret, R: AR.reconstruct(net, gt, ret, C.D, DEV, **inf)  # noqa: E731
    src = sel0["selected_path"]
else:
    nets, sels, modes = BM._load_dcsrn(mixed=True)
    net, mode = nets[a.R], modes[a.R]
    for k in list(nets):
        if k != a.R:
            del nets[k]
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    sampler = DC.MixedMaskPatchSampler(preps, a.R, C.D, 64, seed=SEED, cache_cap=len(preps), uniform_prob=0.5)
    rng = np.random.RandomState(SEED + 1)
    lr_fn = lambda s: 1e-4 * min(1.0, (s + 1) / 1000)  # noqa: E731  (warm-up: fresh Adam state on a trained net)

    def step():
        x, y = sampler.batch(rng.randint(0, len(preps), 2), DEV)
        return DC.loss_fn(net(x), y)

    recon = lambda gt, ret, R: DC.reconstruct(net, gt, ret, R, C.D, DEV, mode=mode)  # noqa: E731
    src = sels[a.R]["selected_path"]


def validate(s):
    net.eval()
    rows = []
    for p, gt in val_gt.items():
        for R, fam, ret in cells:
            x = recon(gt, ret, R)
            rows.append(dict(step=s, patient=p, R=R, family=fam,
                             psnr=P.evaluate_volume(x, gt, ret, C.D, "missing")["psnr_masked"],
                             linear=P.evaluate_volume(C_lin(gt, ret), gt, ret, C.D, "missing")["psnr_masked"]))
    return pd.DataFrame(rows)


def C_lin(gt, ret):
    return P.interpolate_baseline(gt, ret, "linear", C.D)


hist = validate(0)
best, bad, best_step = hist.psnr.mean(), 0, 0
torch.save(dict(state_dict=net.state_dict(), step=0), os.path.join(WD, "ckpt_step000000.pt"))
log(f"step 0 (source {src}): val {hist.psnr.mean():.3f} dB vs linear {hist.linear.mean():.3f} dB")
tt = time.time()
for s in range(a.max_steps):
    for g in opt.param_groups:
        g["lr"] = lr_fn(s)
    net.train()
    loss = step()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if (s + 1) % 100 == 0:
        open(os.path.join(WD, "train_loss.csv"), "a").write(f"{s + 1},{float(loss.detach()):.6f},{time.time() - tt:.1f}\n")
    if (s + 1) % a.val_every == 0:
        v = validate(s + 1)
        hist = pd.concat([hist, v])
        hist.to_csv(os.path.join(WD, "val_history.csv"), index=False)
        torch.save(dict(state_dict=net.state_dict(), step=s + 1), os.path.join(WD, f"ckpt_step{s + 1:06d}.pt"))
        m = v.psnr.mean()
        if m > best + 0.01:
            best, bad, best_step = m, 0, s + 1
        else:
            bad += 1
        log(f"step {s + 1} loss {float(loss):.5f} {time.time() - tt:.0f}s val {m:.3f} (linear {v.linear.mean():.3f}) "
            f"best {best:.3f}@{best_step} bad {bad}")
        if bad >= a.patience:
            log("early stop")
            break
hist.to_csv(os.path.join(WD, "val_history.csv"), index=False)
lin = hist[hist.step == 0].linear.mean()
json.dump(dict(name=name, source=src, selected_step=int(best_step), selected_path=os.path.join(WD, f"ckpt_step{best_step:06d}.pt"),
               val_psnr=float(best), val_linear=float(lin), beats_linear=bool(best > lin), steps_run=int(s + 1),
               train_hours=(time.time() - tt) / 3600, rule="best SEL val PSNR on own families U0/U2/U4/U6"),
          open(os.path.join(WD, "selected.json"), "w"), indent=1)
log(f"SELECTED step {best_step}: {best:.3f} dB vs linear {lin:.3f} dB")
