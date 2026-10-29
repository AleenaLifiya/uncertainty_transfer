"""2x2 cell 4 (SAINT-PGX): geometry-aware scale head trained on UNIFORM + IRREGULAR masks (copy of train_geo.py;
only the training-mask distribution and the validation families differ). Original docstring follows.
Option 3: train SAINT-PG's geometry-aware scale head (mean path + dropout frozen = SAINT-P selected checkpoint).

Data: the same read-only MSR cache (40 train subjects, UNIFORM R 2/4/8) -> gap lengths 2/4/8 mm and all in-gap
positions are seen; irregular masks are never used. Loss: region-balanced Laplace NLL on missing slices, dropout ON
(the MC regime of the final SAINT-P phase). Selection (prespecified): lowest SEL foreground NLL on uniform x4+x8.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from isbi import common as C
from isbi import models as M
from isbi import recon as RC
from comparisons import protocol as P

STEPS, VAL_EVERY, LR, BS, SEED = 3000, 500, 1e-3, 8, 42
WD = os.path.join(C.ISBI, "runs", "train", "saint_geo_mixed")
os.makedirs(WD, exist_ok=True)
torch.manual_seed(SEED)
proto = json.load(open(os.path.join(C.ISBI, "configs", "protocol_frozen.json")))
net, pg, chunk = RC.build_prob_saint_geo(proto["selected_checkpoints"]["saint_prob"])
for n, p_ in pg.named_parameters():
    p_.requires_grad = n.startswith("scale_head")
params = [p_ for p_ in pg.parameters() if p_.requires_grad]
opt = torch.optim.Adam(params, lr=LR)
roles = C.load_roles()
idx = pd.read_csv(os.path.join(C.ROOT, "IR3C_more_comparisons/predictions/saint_msr_cache_index.csv"))
tr = idx[idx.split == "train"].reset_index(drop=True)
# irregular entries = the exposure-a cache (40 train subjects x R 4/8 x {jittered, clustered} x 2 masks; frozen MSR)
ECACHE = os.path.join(C.ISBI, "cache", "data", "exposure_saint")
gtf = {s: tr[(tr.subject == s) & (tr.R == 4)].file.iloc[0] for s in tr.subject.unique()}
irr = [(os.path.join(ECACHE, f"{s}_x{R}_{p}_r{r}.npz"), gtf[s]) for s in sorted(gtf) for R in (4, 8)
       for p in ("jittered", "clustered") for r in range(2)]
assert all(os.path.exists(f) for f, _ in irr), "exposure cache incomplete"
cache = {}
rng = np.random.RandomState(SEED + 3)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    open(os.path.join(WD, "train.log"), "a").write(m + "\n")


def load_cache(f, gfile=None):
    if f not in cache:
        z = np.load(f if gfile else os.path.join(C.ROOT, "IR3C_more_comparisons", f))
        zg = np.load(os.path.join(C.ROOT, "IR3C_more_comparisons", gfile)) if gfile else z
        ret = [int(r) for r in z["ret"]]
        g = C.gap_geometry(ret, zg["gt"].shape[2])
        geo = np.stack([np.log1p(g["near_mm"]), np.log(np.maximum(g["gap_mm"], 1.0)), 1 - np.abs(2 * g["t"] - 1)], 1)
        cache[f] = (torch.from_numpy(np.stack([z["sag"], z["cor"]], 0)), torch.from_numpy(zg["gt"]),
                    np.setdiff1d(np.arange(zg["gt"].shape[2]), ret), torch.from_numpy(geo.astype(np.float32)))
    return cache[f]


SEL = [roles["scan_of"][p] for p in roles["SEL"]]
vc = os.path.join(C.ISBI, "cache", "data", "sel_val")
val = [(s, P.prepare_one(C.scan_path(s))["vol01"]) for s in SEL]


def validate(step):
    rows = []
    for s, gt in val:
        pid = [p for p in roles["SEL"] if roles["scan_of"][p] == s][0]
        for R, pat in [(R, pat) for R in (4, 8) for pat in ("uniform", "jittered", "clustered")]:
            z = np.load(os.path.join(vc, f"saint_{s}_x{R}.npz" if pat == "uniform" else f"saint_{s}_x{R}_{pat}_r0.npz"))
            ret = C.make_mask(pat, R, pid, 0)
            mu, b = RC.saint_geo_mc(pg, z["sag"].astype(np.float32), z["cor"].astype(np.float32), ret, 0, 0)
            mu, b = mu[0], b[0]
            miss = np.setdiff1d(np.arange(C.D), ret)
            y = torch.from_numpy(gt[:, :, miss]).to(RC.DEV)
            fg = y > 0
            e = (y - mu[:, :, miss]).abs()[fg]
            bb = b[:, :, miss][fg]
            rows.append(dict(step=step, subject=s, R=R, pattern=pat, nll=float((torch.log(2 * bb) + e / bb).mean()),
                             spearman=float(spearmanr(e[::10].cpu(), bb[::10].cpu()).correlation),
                             e_over_b=float((e / bb).mean())))
    return pd.DataFrame(rows)


hist = validate(0)
log(f"step 0 (= SAINT-P): {hist.groupby('R')[['nll', 'spearman', 'e_over_b']].mean().round(4).to_dict()}")
torch.save(dict(state_dict=pg.state_dict(), step=0), os.path.join(WD, "ckpt_step000000.pt"))
t0 = time.time()
for step in range(STEPS):
    M.set_mc(pg, True)
    if rng.rand() < 0.5:
        sc, gt, miss, geo = load_cache(tr.file[int(rng.randint(len(tr)))])
    else:
        f, g_ = irr[int(rng.randint(len(irr)))]
        sc, gt, miss, geo = load_cache(f, g_)
    zs = rng.choice(miss, BS)
    x = sc[:, :, :, zs].permute(3, 0, 1, 2).to(RC.DEV).float()
    y = gt[:, :, zs].permute(2, 0, 1)[:, None].to(RC.DEV).float()
    gm = geo[zs][:, :, None, None].expand(BS, 3, x.shape[2], x.shape[3]).to(RC.DEV)
    with torch.no_grad():
        cat, f1 = pg.trunk(x)
    mu, b = pg.head(cat, f1, x, gm)
    mu = mu.detach()
    fg, bg = (y > 0).float(), (y <= 0).float()
    loss = 0.5 * M.laplace_nll(y, mu, b, fg) + 0.5 * M.laplace_nll(y, mu, b, bg)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if (step + 1) % VAL_EVERY == 0:
        v = validate(step + 1)
        hist = pd.concat([hist, v])
        torch.save(dict(state_dict=pg.state_dict(), step=step + 1), os.path.join(WD, f"ckpt_step{step + 1:06d}.pt"))
        log(f"step {step + 1} loss {float(loss):.3f} {time.time() - t0:.0f}s "
            f"{v.groupby('R')[['nll', 'spearman', 'e_over_b']].mean().round(4).to_dict()}")
hist.to_csv(os.path.join(WD, "val_history.csv"), index=False)
h = hist.groupby("step").nll.mean()
sel = int(h[h.index > 0].idxmin())
json.dump(dict(selected_step=sel, selected_path=os.path.join(WD, f"ckpt_step{sel:06d}.pt"),
               rule="min SEL fg NLL (uniform + jittered + clustered r0, x4+x8), step>0", sel_nll=float(h[sel]), step0_nll=float(h[0]),
               train_sec=time.time() - t0), open(os.path.join(WD, "selected.json"), "w"), indent=1)
log(f"SELECTED step {sel}")
