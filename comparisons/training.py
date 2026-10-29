"""
comparisons/training.py
=======================
The SHARED training scaffold (ir3_comparisons.md §4, §13).

Budget, validation cadence, the validation metric, early stopping and checkpoint selection
are implemented ONCE here and called by every method, so they cannot differ between
methods. What each method keeps native: architecture, loss, optimiser, learning rate,
schedule shape, batch/patch strategy.

Validation metric (identical to IR3's):
    macro brain-masked PSNR over MISSING slices, averaged over the fixed validation matrix,
    with the guardrail  macro_SSIM >= linear_macro_SSIM - 0.01.
Test data is never read here.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import torch

from . import protocol as P


# --------------------------------------------------------------------------- #
#  budget
# --------------------------------------------------------------------------- #
class Budget:
    """A method-agnostic training budget expressed in OPTIMISATION STEPS.

    Steps -- not epochs -- because 'one epoch' means something different for a full-depth
    3D patch loop (IR3), a slice-run loop (I3Net) and a coordinate-sampling loop (ArSSR).
    ir3_comparisons.md §4 explicitly warns against equating completed epochs.
    """

    def __init__(self, total_steps, val_every, patience, warmup_frac=0.05):
        self.total_steps = int(total_steps)
        self.val_every = int(val_every)
        self.patience = int(patience)
        self.warmup_steps = int(round(float(warmup_frac) * self.total_steps))

    def as_dict(self):
        return dict(total_steps=self.total_steps, val_every=self.val_every,
                    patience=self.patience, warmup_steps=self.warmup_steps,
                    n_validations=self.total_steps // self.val_every)


# --------------------------------------------------------------------------- #
#  validation -- ONE implementation for every method
# --------------------------------------------------------------------------- #
def validate(recon_fn, val_preps, val_matrix, linear_cache, D=P.DEPTH):
    """Score `recon_fn` on the fixed validation matrix.

    recon_fn(gt_hw_d, ret, R, family) -> (240,240,153) float32 in [0,1].
    Returns (macro_psnr, macro_ssim, macro_ssim_bbox, linear_macro_psnr,
             linear_macro_ssim, per-cell DataFrame).
    """
    rows = []
    for prep in val_preps:
        gt = prep["vol01"]
        s = prep["subject"]
        for (R, fam, ret) in val_matrix:
            x = recon_fn(gt, ret, R, fam)
            m = P.evaluate_volume(x, gt, ret, D=D)
            key = (s, int(R), fam)
            if key not in linear_cache:
                lm = P.evaluate_volume(P.interpolate_baseline(gt, ret, "linear", D), gt, ret, D=D)
                linear_cache[key] = (lm["psnr_masked"], lm["ssim"], lm["ssim_B_brain_bbox"])
            lp, ls, lsb = linear_cache[key]
            rows.append(dict(subject=s, R=int(R), mask_family=fam,
                             psnr_masked=m["psnr_masked"], ssim=m["ssim"],
                             ssim_bbox=m["ssim_B_brain_bbox"], bg_haze=m["bg_haze"],
                             linear_psnr=lp, linear_ssim=ls, linear_ssim_bbox=lsb))
    df = pd.DataFrame(rows)
    ok = df[np.isfinite(df.psnr_masked)]
    return (float(ok.psnr_masked.mean()), float(ok.ssim.mean()), float(ok.ssim_bbox.mean()),
            float(ok.linear_psnr.mean()), float(ok.linear_ssim.mean()), df)


# --------------------------------------------------------------------------- #
#  learning-rate schedules -- each method keeps its OWN shape, mapped onto the
#  shared step budget (§4: "define an equivalent and defensible training budget")
# --------------------------------------------------------------------------- #
def lr_cosine_warmup(step, base_lr, budget: Budget):
    """I3Net native: linear warm-up over `warmup_epoch` fraction, then cosine to 0."""
    w = max(1, budget.warmup_steps)
    if step < w:
        return base_lr * step / w
    t = min(1.0, (step - w) / max(1, budget.total_steps - w))
    return 0.5 * base_lr * (1.0 + np.cos(np.pi * t))


def lr_step_halving(step, base_lr, budget: Budget, n_halvings=12):
    """ArSSR native: lr *= 0.5 every `lr_decay_epoch` epochs.

    Upstream runs 2500 epochs over 780 HCP volumes at batch 15 = 130,000 steps, halving
    every 200 epochs = every 10,400 steps, i.e. 12 halvings across the run. We preserve the
    SHAPE (12 halvings, geometrically spaced) across whatever total step budget is set.
    """
    period = max(1, budget.total_steps // int(n_halvings))
    return base_lr * (0.5 ** (step // period))


# --------------------------------------------------------------------------- #
#  the shared loop
# --------------------------------------------------------------------------- #
def train_method(*, name, model, optimizer, budget, lr_fn, step_fn, recon_fn,
                 val_preps, val_matrix, run_dir, device, extra_ckpt=None,
                 wallclock_limit_h=None, log_every=100):
    """Train one method under the shared protocol.

    step_fn(step) -> dict of scalar loss terms (must include 'loss'); it owns the forward,
                     backward and optimiser step, i.e. all method-specific logic.
    recon_fn(gt, ret, R, family) -> reconstruction, used ONLY for validation.
    """
    ckdir = os.path.join(run_dir, "checkpoints", name)
    os.makedirs(ckdir, exist_ok=True)
    hist, per_ckpt = [], []
    lincache = {}
    stopper = P.EarlyStopper(budget.patience)
    t0 = time.time()
    stop_reason = "budget_exhausted"

    def save(step, p, s, sb, lp, ls):
        path = os.path.join(ckdir, f"{name}_step{step:06d}.pt")
        torch.save(dict(state_dict=model.state_dict(), step=int(step),
                        macro_psnr=p, macro_ssim=s, macro_ssim_bbox=sb,
                        linear_macro_psnr=lp, linear_macro_ssim=ls,
                        budget=budget.as_dict(), method=name, **(extra_ckpt or {})), path)
        return os.path.relpath(path, run_dir)

    def do_val(step):
        model.eval()
        p, s, sb, lp, ls, df = validate(recon_fn, val_preps, val_matrix, lincache)
        model.train()
        rel = save(step, p, s, sb, lp, ls)
        per_ckpt.append(dict(step=int(step), epoch=int(step), path=rel, macro_psnr=p,
                             macro_ssim=s, macro_ssim_bbox=sb, linear_macro_psnr=lp,
                             linear_macro_ssim=ls,
                             gpu_peak_gb=(torch.cuda.max_memory_allocated() / 1e9
                                          if device == "cuda" else np.nan),
                             elapsed_h=(time.time() - t0) / 3600.0))
        df.assign(step=step).to_csv(
            os.path.join(run_dir, f"val_detail_{name}_step{step:06d}.csv"), index=False)
        print(f"[{name}] step {step}/{budget.total_steps}  val PSNR {p:.3f} (lin {lp:.3f})  "
              f"SSIM {s:.4f} (lin {ls:.4f})  bbox {sb:.4f}  "
              f"{(time.time()-t0)/3600:.2f} h", flush=True)
        return p

    print(f"[{name}] budget {json.dumps(budget.as_dict())}", flush=True)
    do_val(0)                                    # step-0 (untrained) reference point
    model.train()
    run_loss, n_acc = 0.0, 0
    for step in range(1, budget.total_steps + 1):
        lr = float(lr_fn(step))
        for g in optimizer.param_groups:
            g["lr"] = lr
        terms = step_fn(step)
        run_loss += float(terms["loss"]); n_acc += 1
        if step % log_every == 0:
            el = time.time() - t0
            eta = el / step * (budget.total_steps - step)
            print(f"[{name}] step {step}/{budget.total_steps} lr={lr:.2e} "
                  f"loss={run_loss/max(1,n_acc):.5f} "
                  f"({el/3600:.2f} h elapsed, ETA {eta/3600:.2f} h)", flush=True)
            hist.append(dict(step=step, lr=lr, train_loss=run_loss / max(1, n_acc),
                             elapsed_h=el / 3600.0, **{k: v for k, v in terms.items()
                                                       if k != "loss"}))
            run_loss, n_acc = 0.0, 0
        if step % budget.val_every == 0 or step == budget.total_steps:
            p = do_val(step)
            hist.append(dict(step=step, lr=lr, val_macro_psnr=p,
                             elapsed_h=(time.time() - t0) / 3600.0))
            if stopper.update(p, step):
                stop_reason = f"early_stop_patience_{budget.patience}"
                print(f"[{name}] EARLY STOP at step {step} "
                      f"(best {stopper.best:.3f} @ step {stopper.best_step})", flush=True)
                break
        if wallclock_limit_h and (time.time() - t0) / 3600.0 > float(wallclock_limit_h):
            stop_reason = "wallclock_limit"
            print(f"[{name}] WALL-CLOCK LIMIT hit at step {step}", flush=True)
            do_val(step)
            break

    pd.DataFrame(hist).to_csv(os.path.join(run_dir, f"train_history_{name}.csv"), index=False)
    pd.DataFrame(per_ckpt).to_csv(os.path.join(run_dir, f"val_per_checkpoint_{name}.csv"),
                                  index=False)
    seldf, sel = P.select_checkpoint(per_ckpt)
    seldf.to_csv(os.path.join(run_dir, f"checkpoint_selection_{name}.csv"), index=False)
    last_step = int(per_ckpt[-1]["step"])
    sel.update(method=name, stop_reason=stop_reason, stopped_at_step=last_step,
               total_steps_planned=budget.total_steps,
               training_hours=(time.time() - t0) / 3600.0,
               peak_gpu_gb=(torch.cuda.max_memory_allocated() / 1e9
                            if device == "cuda" else None),
               still_improving_at_stop=bool(sel["selected_step"] == last_step and
                                            last_step == budget.total_steps),
               budget=budget.as_dict())
    json.dump(sel, open(os.path.join(run_dir, f"selected_{name}.json"), "w"), indent=2)
    print(f"[{name}] SELECTED {json.dumps(sel, indent=2)}", flush=True)
    return sel
