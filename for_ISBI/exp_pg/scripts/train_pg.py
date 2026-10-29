"""Train the geometry-aware scale head ('PG') on top of one seed's probabilistic model (exp_pg, 2026-10-26).

Recipe = SAINT-PG (scripts/train_geo.py): mean path + trained dropout frozen (= that seed's P model), only the scale head
is trained (first layer widened by 3 zero-init geometry inputs), dropout ON, Adam 1e-3, region-balanced Laplace NLL on
missing (non-acquired) positions, UNIFORM masks R 2/4/8 only, 3000 steps; selection = lowest SEL foreground NLL on uniform
x4 + x8 (step > 0). Foreground: y > 0 (BraTS, and IXI ArSSR as in its P recipe); brain mask for IXI SAINT (as its P recipe).
    python train_pg.py --arch saint|arssr --dataset brats|ixi --seed 42|43|44 [--smoke]
"""

from project_paths import root_path
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, root_path('for_ISBI/exp_ixi/scripts'))
import pg as PG  # noqa: E402
from isbi import common as C  # noqa: E402
from isbi import models as M  # noqa: E402
from isbi import native as N  # noqa: E402
from isbi import recon as RC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True, choices=["saint", "arssr"])
ap.add_argument("--dataset", required=True, choices=["brats", "ixi"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
I = C.ISBI
STEPS, VAL_EVERY = (3000, 500 if a.arch == "saint" else 1000) if not a.smoke else (6, 3)
SP = 1.0 if a.dataset == "brats" else 2.18
WD = os.path.join(I, "exp_pg", "runs" + ("_smoke" if a.smoke else ""), f"{a.dataset}_{a.arch}_s{a.seed}", "train")
os.makedirs(WD, exist_ok=True)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
rng = np.random.RandomState(a.seed + 3)
DEV = "cuda"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    open(os.path.join(WD, "train.log"), "a").write(m + "\n")


def base_prob_ckpt():
    S = os.path.join(I, "exp_seeds", "runs")
    if a.arch == "saint":
        if a.dataset == "brats":
            return (json.load(open(os.path.join(I, "configs", "protocol_frozen.json")))["selected_checkpoints"]["saint_prob"]
                    if a.seed == 42 else json.load(open(os.path.join(S, f"brats_s{a.seed}", "train", "saint_prob", "selected.json")))["selected_path"])
        d = os.path.join(I, "exp_ixi", "runs", "train", "saint_prob") if a.seed == 42 else os.path.join(S, f"ixi_s{a.seed}", "train", "saint_prob")
    else:
        d = ({"brats": os.path.join(I, "exp_uniform_arssr", "runs", "train", "arssr_prob"),
              "ixi": os.path.join(I, "exp_ixi", "runs", "train", "arssr_prob")}[a.dataset] if a.seed == 42 else
             os.path.join(S, f"{a.dataset}_s{a.seed}", "train", "arssr_prob"))
    return json.load(open(os.path.join(d, "selected.json")))["selected_path"]


CK = base_prob_ckpt()
log(f"base P checkpoint {CK}")

# ================================================================== models + data
if a.arch == "saint":
    if a.dataset == "brats":
        net, prfn, chunk, _ = RC.build_prob_saint(CK)
    else:
        import ixi_isbi as X
        net = X.load_saint()
        prfn = X.build_prob(net, CK)
    pg = M.ProbRFNGeo(prfn).to(DEV)
    if a.dataset == "brats":
        idx = pd.read_csv(os.path.join(C.ROOT, "IR3C_more_comparisons/predictions/saint_msr_cache_index.csv"))
        files = [os.path.join(C.ROOT, "IR3C_more_comparisons", f) for f in idx[idx.split == "train"].file]
    else:
        import glob
        files = sorted(glob.glob(os.path.join(I, "exp_ixi", "cache", "train_marginals", "*.npz")))
    if a.smoke:
        files = files[:3]
    cache = {}

    def get(f):
        if f not in cache:
            z = np.load(f)
            ret = [int(r) for r in z["ret"]]
            Dz = z["gt"].shape[2]
            fg = (z["brain"] if "brain" in z.files else (z["gt"] > 0)).astype(np.float32)
            cache[f] = (torch.from_numpy(np.stack([z["sag"], z["cor"]], 0).astype(np.float32)), torch.from_numpy(z["gt"]),
                        torch.from_numpy(fg), np.setdiff1d(np.arange(Dz), ret), torch.from_numpy(PG.geo_feats(ret, Dz, SP)))
        return cache[f]

    def step_loss():
        sc, gt, fg, miss, geo = get(files[int(rng.randint(len(files)))])
        zs = rng.choice(miss, 8)
        x = sc[:, :, :, zs].permute(3, 0, 1, 2).to(DEV)
        y = gt[:, :, zs].permute(2, 0, 1)[:, None].to(DEV).float()
        f_ = fg[:, :, zs].permute(2, 0, 1)[:, None].to(DEV)
        gm = geo[zs][:, :, None, None].expand(8, 3, x.shape[2], x.shape[3]).to(DEV)
        with torch.no_grad():
            cat, f1 = pg.trunk(x)
        mu, b = pg.head(cat, f1, x, gm)
        mu = mu.detach()
        return 0.5 * M.laplace_nll(y, mu, b, f_) + 0.5 * M.laplace_nll(y, mu, b, 1 - f_)

    # validation: SEL uniform x4/x8 marginals (MSR frozen and identical across seeds)
    roles = C.load_roles() if a.dataset == "brats" else None
    val = []
    if a.dataset == "brats":
        from comparisons import protocol as P
        for p in roles["SEL"][: 2 if a.smoke else None]:
            s = roles["scan_of"][p]
            gt = P.prepare_one(C.scan_path(s))["vol01"]
            for R in (4, 8):
                zc = np.load(os.path.join(I, "cache", "data", "sel_val", f"saint_{s}_x{R}.npz"))
                val.append((gt, gt > 0, C.uniform_mask(R), zc["sag"].astype(np.float32), zc["cor"].astype(np.float32)))
    else:
        for s in X.roles()["SEL"][: 2 if a.smoke else None]:
            gt, br = X.load(s)
            for R in (4, 8):
                ret = C.uniform_mask(R, X.D)
                sag, cor = X.saint_marginals(net, gt, ret, R)
                val.append((gt, br, ret, sag, cor))

    def predict(v):
        gt, fg, ret, sag, cor = v
        mu, b = PG.saint_geo_mc(pg, sag, cor, ret, 0, 0, SP)
        return mu[0], b[0]
else:
    from IR3C_LPIPS import brats_methods as BM
    from comparisons import protocol as P
    base, inf, _ = BM._load_arssr_slowlr()
    pm = N.ProbArSSR(base)
    pm.load_state_dict(torch.load(CK, map_location="cpu", weights_only=False)["state_dict"])
    pg = PG.ArSSRGeo(pm).to(DEV)
    if a.dataset == "brats":
        from comparisons import arssr_adapter as AR
        D_ = C.D

        class Smp(AR.SlabSampler):
            def _mask(self):
                import ir2_components as IC
                return IC.sample_mask_weighted(P.DEPTH, self.rng, self.strides, list(self.w), uniform_prob=1.0)
        trs = P.load_splits()["train"]
        pick = sorted(np.random.RandomState(42).choice(len(trs), 8 if a.smoke else 400, replace=False).tolist())
        preps = [dict(vol01=P.prepare_one(trs[i])["vol01"]) for i in pick]
        sampler = Smp(preps, [2, 4, 8], [0.30, 0.35, 0.35], 4, 48, 8000, seed=a.seed + 3)
        roles = C.load_roles()
        val = []
        for p in roles["SEL"][: 2 if a.smoke else 6]:
            gt = P.prepare_one(C.scan_path(roles["scan_of"][p]))["vol01"]
            for R in (4, 8):
                val.append((gt, gt > 0, C.uniform_mask(R)))
        INF = inf
    else:
        import ixi_isbi as X
        from ir3c_v2 import baselines_v2 as BV
        D_ = X.D

        class Smp(BV.ArSSRSlabSamplerV2):
            def _mask(self):
                import ir2_components as IC
                return IC.sample_mask_weighted(self.D, self.rng, self.strides, list(self.w), uniform_prob=1.0)
        subs = X.roles()["TRAIN"][: 6 if a.smoke else None]
        preps = [dict(vol01=X.load(s)[0]) for s in subs]
        sampler = Smp(preps, X.D, strides=[2, 4, 8], stride_weights=[0.30, 0.35, 0.35], n_lr=4, base=48, sample_size=8000,
                      seed=a.seed + 3)
        val = []
        for s in X.roles()["SEL"][: 2 if a.smoke else None]:
            gt, br = X.load(s)
            for R in (4, 8):
                val.append((gt, br, C.uniform_mask(R, X.D)))
        INF = BV.ARSSR_INFERENCE["IXI"]
    for _ in range(100):
        assert sampler._mask()[1]["mask_family"] == "U0"

    def step_loss():
        lr_t, xyz, y, metas = sampler.batch(rng.randint(0, len(preps), 15), DEV)
        geo = PG.arssr_train_geo(xyz, metas, SP)
        w = N.arssr_batch_weights(xyz)
        with torch.no_grad():
            fm = pg.base.encoder(lr_t)
        import torch.nn.functional as F_
        fv = F_.grid_sample(fm, xyz.flip(-1).unsqueeze(1).unsqueeze(1), mode="bilinear",
                            align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
        Nn, K = xyz.shape[:2]
        mu, b = pg.decode(torch.cat([fv, xyz], -1).view(Nn * K, -1), geo.reshape(Nn * K, 3))
        mu, b = mu.view(Nn, K, 1).detach(), b.view(Nn, K, 1)
        fg, bg = (y > 0).float() * w, (y <= 0).float() * w
        return 0.5 * M.laplace_nll(y, mu, b, fg) + 0.5 * M.laplace_nll(y, mu, b, bg)

    def predict(v):
        gt, fg, ret = v
        mu, b = PG.arssr_geo_mc(pg, gt, ret, 0, 0, D_, SP, **INF)
        return mu[0], b[0]

for n_, p_ in pg.named_parameters():
    p_.requires_grad = n_.startswith("scale_head")
params = [p_ for p_ in pg.parameters() if p_.requires_grad]
opt = torch.optim.Adam(params, lr=1e-3)


def validate(step):
    rows = []
    for v in val:
        gt, fg, ret = v[0], v[1], v[2]
        mu, b = predict(v)
        miss = np.setdiff1d(np.arange(gt.shape[2]), ret)
        y = torch.from_numpy(gt[:, :, miss]).to(DEV)
        m = torch.from_numpy(np.asarray(fg)[:, :, miss]).to(DEV).bool()
        e = (y - mu[:, :, miss]).abs()[m]
        bb = b[:, :, miss][m]
        rows.append(dict(step=step, nll=float((torch.log(2 * bb) + e / bb).mean())))
    return pd.DataFrame(rows)


hist = validate(0)
torch.save(dict(state_dict=pg.state_dict(), step=0), os.path.join(WD, "ckpt_step000000.pt"))
log(f"step 0 (= P): SEL NLL {hist.nll.mean():.4f}")
t0 = time.time()
for s in range(STEPS):
    M.set_mc(pg, True)
    loss = step_loss()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if (s + 1) % VAL_EVERY == 0:
        v = validate(s + 1)
        hist = pd.concat([hist, v])
        hist.to_csv(os.path.join(WD, "val_history.csv"), index=False)
        torch.save(dict(state_dict=pg.state_dict(), step=s + 1), os.path.join(WD, f"ckpt_step{s + 1:06d}.pt"))
        log(f"step {s + 1} {time.time() - t0:.0f}s loss {float(loss.detach()):.3f} SEL NLL {v.nll.mean():.4f}")
h = hist.groupby("step").nll.mean()
sel = int(h[h.index > 0].idxmin())
json.dump(dict(selected_step=sel, selected_path=os.path.join(WD, f"ckpt_step{sel:06d}.pt"), base_prob=CK, sel_nll=float(h[sel]),
               step0_nll=float(h[0]), train_sec=time.time() - t0,
               rule="head only, uniform R2/4/8, min SEL fg NLL (uniform x4+x8), step>0"),
          open(os.path.join(WD, "selected.json"), "w"), indent=1)
log(f"SELECTED step {sel}")
