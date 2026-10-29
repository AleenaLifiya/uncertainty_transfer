"""Probabilistic fine-tuning of the UNIFORM-trained ArSSR (ArSSR-P-U). Adapted from scripts/train_native_prob.py (read-only),
ArSSR branch only. The recipe is identical (ProbArSSR: 2 trained MCDropout + log-scale head; phase 1 head only 2000
steps, phase 2 joint 3000 steps; lr 2e-5 source / 1e-3 head, cosine; region-balanced Laplace NLL on non-acquired
queries; same prespecified selection rule), EXCEPT:
  * start  = exp_uniform_arssr/runs/train/arssr_ext/selected.json (uniform-trained deterministic ArSSR);
  * masks  = UNIFORM ONLY (training) -- the original used the mixed 50 % uniform / 50 % U1..U6 sampler;
  * validation = SEL patients 0-5 on uniform x4 and x8 only (the original used U0/U2/U4/U6).
Output name 'arssr_prob' so eval_native.py reads it through ISBI_TRAIN_DIR unmodified.
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
from isbi import models as M
from isbi import native as N
from comparisons import protocol as P
from comparisons import arssr_adapter as AR
from IR3C_LPIPS import brats_methods as BM

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
H1, H2, VAL_EVERY = 2000, 3000, 1000
if a.smoke:
    H1, H2, VAL_EVERY = 4, 4, 4
EXP = os.path.join(C.ISBI, "exp_uniform_arssr")
TRD = os.path.join(EXP, "runs", "train" + ("_smoke" if a.smoke else ""))
WD = os.path.join(TRD, "arssr_prob")
os.makedirs(WD, exist_ok=True)
SEED, DEV = 42, "cuda"
P.seed_everything(SEED)
srcj = json.load(open(os.path.join(TRD, "arssr_ext", "selected.json")))


class UniformSlabSampler(AR.SlabSampler):
    """AR.SlabSampler with the mask draw restricted to the uniform family (same as train_uarssr_det.py)."""

    def _mask(self):
        import ir2_components as IC
        return IC.sample_mask_weighted(P.DEPTH, self.rng, self.strides, list(self.w), uniform_prob=1.0)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    open(os.path.join(WD, "train.log"), "a").write(m + "\n")


roles = C.load_roles()
VALP = roles["SEL"][:2 if a.smoke else 6]
val_gt = {p: P.prepare_one(C.scan_path(roles["scan_of"][p]))["vol01"] for p in VALP}
cells = [(R, "U0", C.uniform_mask(R)) for R in (4, 8)]
trs = P.load_splits()["train"]
pick = sorted(np.random.RandomState(SEED).choice(len(trs), 8 if a.smoke else 400, replace=False).tolist())
preps = [dict(vol01=P.prepare_one(trs[i])["vol01"]) for i in pick]
log(f"prepared {len(preps)} volumes; start {srcj['selected_path']}")
rng = np.random.RandomState(SEED + 11)

base, inf, _ = BM._load_arssr_slowlr()
base.load_state_dict(torch.load(srcj["selected_path"], map_location="cpu", weights_only=False)["state_dict"])
model = N.ProbArSSR(base).to(DEV)
sampler = UniformSlabSampler(preps, [2, 4, 8], [0.30, 0.35, 0.35], 4, 48, 8000, seed=SEED + 3)
for _ in range(200):
    assert sampler._mask()[1]["mask_family"] == "U0"


def batch():
    lr_t, xyz, v, _ = sampler.batch(rng.randint(0, len(preps), 15), DEV)
    return lr_t, xyz, v, N.arssr_batch_weights(xyz)


def predict(gt, ret, R, T=0):
    mu, b = N.arssr_mc(model, gt, ret, T, 0, **inf)
    return (mu[0], b[0]) if T == 0 else (mu.mean(0), None)


def validate(step):
    rows = []
    for p, gt in val_gt.items():
        for R, fam, ret in cells:
            mu, b = predict(gt, ret, R)
            x = np.clip(mu.cpu().numpy(), 0, 1)
            x[:, :, ret] = gt[:, :, ret]
            ev = P.evaluate_volume(x, gt, ret, C.D, "missing")
            miss = np.setdiff1d(np.arange(C.D), ret)
            y = torch.from_numpy(gt[:, :, miss]).to(DEV)
            fg = y > 0
            e = (y - mu[:, :, miss]).abs()[fg]
            bb = b[:, :, miss][fg]
            mmc, _ = predict(gt, ret, R, 5)
            xm = np.clip(mmc.cpu().numpy(), 0, 1)
            xm[:, :, ret] = gt[:, :, ret]
            rows.append(dict(step=step, patient=p, R=R, family=fam, psnr=ev["psnr_masked"],
                             psnr_mc5=P.evaluate_volume(xm, gt, ret, C.D, "missing")["psnr_masked"],
                             nll=float((torch.log(2 * bb) + e / bb).mean()), e_over_b=float((e / bb).mean()),
                             spearman=float(spearmanr(e[::10].cpu(), bb[::10].cpu()).correlation) if bb.std() > 0 else np.nan))
    return pd.DataFrame(rows)


head_p = [q for n_, q in model.named_parameters() if "scale_head" in n_]
base_p = [q for n_, q in model.named_parameters() if "scale_head" not in n_]
opt = torch.optim.Adam([dict(params=base_p, lr=2e-5, base=2e-5), dict(params=head_p, lr=1e-3, base=1e-3)])
STEPS = H1 + H2
sched = lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + np.cos(np.pi * s / H2))  # noqa: E731
hist = validate(0)
log(f"step 0: psnr {hist.psnr.mean():.3f} mc5 {hist.psnr_mc5.mean():.3f} nll {hist.nll.mean():.3f} e/b {hist.e_over_b.mean():.3f}")
tt = time.time()
for s in range(STEPS):
    phaseA = s < H1
    for i, g in enumerate(opt.param_groups):
        g["lr"] = 0.0 if (phaseA and i == 0) else g["base"] * (1.0 if phaseA else sched(s - H1))
    M.set_mc(model, not phaseA)
    for q in base_p:
        q.requires_grad_(not phaseA)
    x, xyz, y, w = batch()
    mu, b = model(x, xyz)
    fgm, bgm = (y > 0).float() * w, (y <= 0).float() * w
    loss = 0.5 * M.laplace_nll(y, mu, b, fgm) + 0.5 * M.laplace_nll(y, mu, b, bgm)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(base_p + head_p, 1.0)
    opt.step()
    if (s + 1) % 50 == 0:
        open(os.path.join(WD, "train_loss.csv"), "a").write(f"{s + 1},{float(loss.detach()):.5f},{time.time() - tt:.1f}\n")
    if (s + 1) % VAL_EVERY == 0:
        v = validate(s + 1)
        hist = pd.concat([hist, v])
        hist.to_csv(os.path.join(WD, "val_history.csv"), index=False)
        torch.save(dict(state_dict=model.state_dict(), step=s + 1), os.path.join(WD, f"ckpt_step{s + 1:06d}.pt"))
        log(f"step {s + 1} {time.time() - tt:.0f}s psnr {v.psnr.mean():.3f} mc5 {v.psnr_mc5.mean():.3f} nll {v.nll.mean():.3f} "
            f"e/b {v.e_over_b.mean():.3f} spearman {v.spearman.mean():.3f}")
h = hist.groupby("step")[["psnr", "psnr_mc5", "nll"]].mean().reset_index()
src_psnr = float(h[h.step == 0].psnr.iloc[0])
ok = h[(h.step > H1) & (h.psnr >= src_psnr - 0.2) & (h.psnr_mc5 >= src_psnr - 0.5)]
sel = ok.sort_values("nll").iloc[0] if len(ok) else h[h.step > H1].sort_values("psnr_mc5").iloc[-1]
json.dump(dict(name="arssr_prob (UNIFORM-trained)", selected_step=int(sel.step),
               selected_path=os.path.join(WD, f"ckpt_step{int(sel.step):06d}.pt"),
               source=srcj["selected_path"], start_psnr=src_psnr, sel_psnr=float(sel.psnr), sel_nll=float(sel.nll),
               n_eligible=int(len(ok)), head_warm=H1,
               rule="phase-2; dropout-off val PSNR >= start-0.2 dB and MC(T=5)-mean val PSNR >= start-0.5 dB; min val fg NLL; "
                    "fallback best MC-mean PSNR (uniform val masks)", sel_psnr_mc5=float(sel.psnr_mc5),
               train_hours=(time.time() - tt) / 3600), open(os.path.join(WD, "selected.json"), "w"), indent=1)
log(f"SELECTED step {int(sel.step)}")
