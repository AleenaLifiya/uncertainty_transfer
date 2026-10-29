"""
fidelity_runtime.py
-------------------
Manual training / validation / checkpoint runtime for the fidelity ControlNet
experiment (Phases 5, 6, 8). Kept separate from the model so the notebook stays
short. Everything here is framework-light (no PyTorch-Lightning Trainer) which
gives explicit control over the three stages, differential LRs, AMP, gradient
clipping, the gradient audit, and — crucially — checkpoint selection by real
missing-slice reconstruction PSNR rather than diffusion loss.
"""

import copy
import hashlib
import itertools
import os
import random
import time

import numpy as np
import torch

from ldm.data.brats_preprocessing_fidelity import prepare_volume
from ldm.models.diffusion.fidelity_sampling import (
    reconstruct_volume, volume_metrics,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
#  VAE weight verification (frozen-VAE experiment: must be bitwise unchanged)
# --------------------------------------------------------------------------
@torch.no_grad()
def vae_weight_hash(model):
    """Deterministic SHA-256 over the frozen VAE (first_stage_model) parameters.

    Hashes params AND buffers in sorted-name order so it is sensitive to any
    weight change. Used before/after every training stage to prove the VAE never
    moved (requirement of xdiff_control_fullVAE.md).
    """
    h = hashlib.sha256()
    vae = model.first_stage_model
    items = list(vae.named_parameters()) + list(vae.named_buffers())
    for name, t in sorted(items, key=lambda x: x[0]):
        h.update(name.encode())
        h.update(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def vae_requires_grad_report(model):
    """Facts for the manifest: any VAE param trainable? total params?"""
    vae = model.first_stage_model
    n_total = sum(p.numel() for p in vae.parameters())
    n_trainable = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    return {"vae_params": int(n_total), "vae_trainable_params": int(n_trainable),
            "all_frozen": bool(n_trainable == 0)}


def rng_state_snapshot():
    """Capture python / numpy / torch (+cuda) RNG states for resumable runs."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }


def save_resumable(path, *, model, optimizer, scaler, ema, step, best, stage_name,
                   scheduler=None, extra=None):
    """Write a fully-resumable checkpoint (model+optim+scaler+ema+scheduler+RNG+step+best)."""
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema_shadow": (ema.shadow if ema is not None else None),
        "rng": rng_state_snapshot(),
        "step": int(step),
        "best": best,
        "stage": stage_name,
        "extra": (extra or {}),
    }, path)
    return path


def move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def disable_checkpointing(module):
    """Turn OFF gradient checkpointing everywhere under `module`.

    This repo's custom CheckpointFunction re-runs forward outside autocast, which
    is incompatible with AMP; disable it on any trainable U-Net (e.g. the Stage A
    backbone) before AMP training. Handles both attribute conventions.
    """
    for m in module.modules():
        if hasattr(m, "use_checkpoint") and isinstance(m.use_checkpoint, bool):
            m.use_checkpoint = False
        if hasattr(m, "checkpoint") and isinstance(m.checkpoint, bool):
            m.checkpoint = False


# --------------------------------------------------------------------------
#  checkpoint IO with explicit missing / unexpected reporting
# --------------------------------------------------------------------------
def load_filtered(model, ckpt_path, allow_missing_prefixes=(), verbose=True):
    """Load a state_dict, keeping only shape-matching keys.

    Returns dict(report) with n_loaded, skipped (shape/extra), missing (fresh).
    Raises if a MISSING key is not covered by allow_missing_prefixes (i.e. an
    unexplained incompatibility).
    """
    ck = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    msd = model.state_dict()
    load = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    skipped = [k for k in sd if k not in load]
    missing = [k for k in msd if k not in load]
    model.load_state_dict(load, strict=False)

    unexplained = [k for k in missing
                   if not any(k.startswith(p) for p in allow_missing_prefixes)]
    report = {
        "ckpt": os.path.basename(ckpt_path),
        "n_loaded": len(load),
        "n_skipped": len(skipped),
        "n_missing": len(missing),
        "skipped": skipped,
        "missing": missing,
        "unexplained_missing": unexplained,
    }
    if verbose:
        print(f"[load_filtered] {report['ckpt']}: loaded={report['n_loaded']} "
              f"skipped={report['n_skipped']} missing={report['n_missing']} "
              f"unexplained_missing={len(unexplained)}")
    if unexplained:
        raise RuntimeError(
            f"Unexplained checkpoint incompatibility: {len(unexplained)} missing "
            f"keys not covered by allow_missing_prefixes, e.g. {unexplained[:5]}")
    del ck, sd, load
    return report


# --------------------------------------------------------------------------
#  EMA over TRAINABLE params only (covers ControlNet + any unfrozen backbone)
# --------------------------------------------------------------------------
class ParamEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def copy_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in getattr(self, "backup", {}):
                p.data.copy_(self.backup[n])
        self.backup = {}

    def covers_controlnet(self, model):
        return any(n.startswith("control_model") for n in self.shadow)


# --------------------------------------------------------------------------
#  validation by reconstruction PSNR
# --------------------------------------------------------------------------
def prepare_volumes(files, p_lo, p_hi, depth):
    return [prepare_volume(f, p_lo=p_lo, p_hi=p_hi, depth=depth) for f in files]


@torch.no_grad()
def evaluate_reconstruction(model, preps, strides, sampler_cfg, method="interp",
                            seed=0, max_missing=None, batch_size=16):
    """Per-subject metrics averaged over the requested strides.

    Returns (rows, mean_masked_psnr). Each row: subject, stride, metrics.
    `max_missing` (per stride) enables fast validation via slice subsampling;
    metrics are computed ONLY on the generated slices.
    """
    model.eval()
    rows = []
    for si, prep in enumerate(preps):
        for R in strides:
            recon, gen_idx, _ = reconstruct_volume(
                model, prep, R, sampler_cfg, batch_size=batch_size,
                seed=seed + 1000 * si, method=method,
                max_missing=max_missing, subsample_seed=si)
            m = volume_metrics(recon, prep["vol01"], gen_idx)
            m.update({"subject": prep.get("subject", si), "stride": R})
            rows.append(m)
    mean_masked = float(np.mean([r["psnr_masked"] for r in rows]))
    return rows, mean_masked


# --------------------------------------------------------------------------
#  generic manual training stage
# --------------------------------------------------------------------------
def _infinite(loader):
    while True:
        for b in loader:
            yield b


def train_stage(model, train_loader, val_preps, *, stage_name, param_groups,
                max_steps, eval_every, ckpt_path, strides, sampler_cfg,
                device="cuda", grad_clip=1.0, use_amp=True, ema_decay=None,
                val_max_missing=6, val_method="interp", log_every=10,
                accumulate=1, seed=0, batch_size=1, train_dataset_size=None,
                resumable_path=None, verify_vae=True, val_batch_size=16,
                early_stop_patience=None):
    """Run one training stage; select the checkpoint by validation masked PSNR.

    Also (frozen-VAE experiment additions): logs learning rates, effective target
    slices processed and epoch-equivalent per validation; verifies the VAE weight
    hash is unchanged across the stage; and writes a fully-resumable checkpoint.

    Returns a history dict (losses, val curve, best metric/step, wall-clock, lrs,
    vae hashes, resumable path).
    """
    set_seed(seed)
    opt = torch.optim.AdamW(param_groups)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ParamEMA(model, ema_decay) if ema_decay else None
    if ema is not None:
        assert ema.covers_controlnet(model) or stage_name == "backbone", \
            "EMA must cover ControlNet params in ControlNet stages"

    vae_hash_before = vae_weight_hash(model) if verify_vae else None

    it = _infinite(train_loader)
    history = {"stage": stage_name, "step": [], "loss": [], "val_step": [],
               "val_psnr": [], "loss_components": [], "lr": [], "epoch_equiv": [],
               "eff_slices": [], "val_by_stride": []}
    best = {"psnr": -1e9, "step": -1}
    t0 = time.time()
    model.train()

    def _lrs():
        return {g.get("name", f"g{i}"): float(g["lr"]) for i, g in enumerate(opt.param_groups)}

    def _run_val(step):
        # evaluate (and, if improved, SAVE) with EMA weights applied, then restore.
        if ema is not None:
            ema.copy_to(model)
        rows, mpsnr = evaluate_reconstruction(
            model, val_preps, strides, sampler_cfg, method=val_method,
            seed=seed, max_missing=val_max_missing, batch_size=val_batch_size)
        by_stride = {}
        for R in strides:
            ps = [r["psnr_masked"] for r in rows if r["stride"] == R]
            if ps:
                by_stride[int(R)] = float(np.mean(ps))
        history["val_step"].append(step)
        history["val_psnr"].append(mpsnr)
        history["val_by_stride"].append(by_stride)
        eff = step * batch_size
        history["eff_slices"].append(eff)
        history["epoch_equiv"].append(
            (eff / train_dataset_size) if train_dataset_size else None)
        history["lr"].append(_lrs())
        improved = mpsnr > best["psnr"]
        if improved:
            best["psnr"] = mpsnr
            best["step"] = step
            torch.save({"state_dict": model.state_dict(),  # EMA weights currently applied
                        "val_psnr": mpsnr, "step": step, "stage": stage_name,
                        "val_by_stride": by_stride,
                        "ema_shadow": (ema.shadow if ema is not None else None)},
                       ckpt_path)
        if ema is not None:
            ema.restore(model)
        # resumable checkpoint (raw weights, optimizer/scaler/ema/rng/step/best)
        if resumable_path is not None:
            save_resumable(resumable_path, model=model, optimizer=opt, scaler=scaler,
                           ema=ema, step=step, best=dict(best), stage_name=stage_name,
                           extra={"lr": _lrs(), "eff_slices": eff})
        model.train()
        print(f"  [{stage_name}] step {step:5d}  val masked-PSNR {mpsnr:6.3f} dB"
              f"  by-stride {by_stride}  (best {best['psnr']:.3f} @ {best['step']})")
        return mpsnr, improved

    stale = 0
    opt.zero_grad(set_to_none=True)
    for step in range(1, max_steps + 1):
        batch = move_batch(next(it), device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, loss_dict = model.shared_step(batch)
            loss = loss / accumulate
        scaler.scale(loss).backward()
        if step % accumulate == 0:
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"]], grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
        if step % log_every == 0:
            history["step"].append(step)
            history["loss"].append(float(loss.detach()) * accumulate)
            comp = {k: float(v.detach()) for k, v in loss_dict.items()
                    if torch.is_tensor(v)}
            history["loss_components"].append(comp)
        if step % eval_every == 0 or step == max_steps:
            _, improved = _run_val(step)
            stale = 0 if improved else stale + 1
            if early_stop_patience is not None and stale >= early_stop_patience:
                print(f"  [{stage_name}] early stop at step {step} "
                      f"(no val improvement for {stale} evals)")
                break

    vae_hash_after = vae_weight_hash(model) if verify_vae else None
    if verify_vae:
        assert vae_hash_before == vae_hash_after, (
            f"VAE weights CHANGED during stage {stage_name}! "
            f"{vae_hash_before[:12]} -> {vae_hash_after[:12]}")

    history["best_psnr"] = best["psnr"]
    history["best_step"] = best["step"]
    history["wall_clock_s"] = time.time() - t0
    history["ckpt_path"] = ckpt_path
    history["resumable_path"] = resumable_path
    history["ema_covered_controlnet"] = bool(ema.covers_controlnet(model)) if ema else False
    history["vae_hash_before"] = vae_hash_before
    history["vae_hash_after"] = vae_hash_after
    history["vae_unchanged"] = bool(vae_hash_before == vae_hash_after) if verify_vae else None
    return history


