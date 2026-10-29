"""
image_residual/trainer.py
-------------------------
Patch-based smoke trainer for IR1 (§18-20). AMP, AdamW, cosine decay w/ warmup,
gradient clipping, gradient accumulation to an effective batch. Checkpoint selection
by macro brain-masked missing-slice PSNR on a fixed validation matrix (never training
loss, never SSA). Saves step-0 (zero-residual == linear), best and last checkpoints.
"""
from __future__ import annotations
import os, json, math, time
import numpy as np
import torch

from . import inputs as IN
from . import masks as M
from . import losses as LOSS
from . import inference as INF


# --------------------------------------------------------------------------- #
#  patch builder (positional channels come from the FULL mask; linear interp is
#  computed on the full volume so anchors outside the patch are respected)
# --------------------------------------------------------------------------- #
def _crop_indices(D, H, W, roi, rng, fg_slices):
    dp, hp, wp = roi
    dp = min(dp, D); hp = min(hp, H); wp = min(wp, W)
    if fg_slices:
        c = int(rng.choice(fg_slices))
        z0 = int(np.clip(c - dp // 2, 0, max(0, D - dp)))
    else:
        z0 = int(rng.randint(0, max(1, D - dp + 1)))
    y0 = int(rng.randint(0, max(1, H - hp + 1)))
    x0 = int(rng.randint(0, max(1, W - wp + 1)))
    return z0, y0, x0, dp, hp, wp


def build_patch(vol_hw_d, x_lin_hw_d, prof, D, crop):
    z0, y0, x0, dp, hp, wp = crop
    sl = (slice(y0, y0 + hp), slice(x0, x0 + wp), slice(z0, z0 + dp))
    gt = vol_hw_d[sl]                                        # (hp,wp,dp)
    lin = x_lin_hw_d[sl]
    known = prof["known"][z0:z0 + dp]
    alpha = prof["alpha"][z0:z0 + dp]
    gap_n = prof["local_gap"][z0:z0 + dp] / M.LOCAL_GAP_NORM
    near_n = prof["nearest"][z0:z0 + dp] / M.NEAREST_NORM
    sparse = np.where(known[None, None, :] > 0.5, gt, 0.0).astype(np.float32)

    def to_cdhw(hw_d): return np.moveaxis(hw_d, 2, 0).astype(np.float32)

    def bc(vec): return np.broadcast_to(vec[:, None, None], (dp, hp, wp)).astype(np.float32)

    x6 = np.stack([to_cdhw(sparse), to_cdhw(lin), bc(known), bc(alpha), bc(gap_n), bc(near_n)], 0)
    gt_c = to_cdhw(gt)[None]                                 # [1,dp,hp,wp]
    missing = bc(1.0 - known)[None]
    brain = (gt_c > 0.0).astype(np.float32)
    lin_c = to_cdhw(lin)[None]
    return x6, gt_c, lin_c, missing, brain


def cosine_warmup(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def validate(model, val_preps, val_matrix, D, device, roi):
    """Macro brain-masked missing-slice PSNR over the fixed validation matrix.
    Weighted equally across subject, R and mask family."""
    model.eval()
    psnrs = []
    per = []
    for prep in val_preps:
        gt = prep["vol01"]; s = prep["subject"]
        for (R, fam, ret) in val_matrix:
            x_hw_d, _, _ = INF.reconstruct_volume(model, gt, ret, D, device, roi=roi)
            miss = M.missing_of(ret, D)
            g = gt[:, :, miss]; p = x_hw_d[:, :, miss]
            fg = g > 0.0
            if fg.sum() < 1000:
                continue
            mse = float(((p - g)[fg] ** 2).mean())
            psnr = 10.0 * math.log10(1.0 / (mse + 1e-12))
            psnrs.append(psnr)
            per.append(dict(subject=s, R=R, mask_family=fam, psnr_masked=psnr))
    macro = float(np.mean(psnrs)) if psnrs else float("nan")
    return macro, per


def build_val_matrix(D, subjects_n, strides=(2, 4, 8, 16),
                     families=("U0", "U1", "U2", "U3", "U4", "U5", "U6"), seed=123):
    """Fixed validation masks (generated before training). U0 uniform; U1-U6 at a fixed
    per-family seed so the matrix is immutable."""
    rng = np.random.RandomState(seed)
    fam_seed = {f: int(rng.randint(1, 1_000_000)) for f in families}
    matrix = []
    for R in strides:
        for fam in families:
            if fam == "U0":
                ret = M.uniform_mask(D, R)
            else:
                ret = M.nonuniform_mask(D, R, fam, fam_seed[fam])
            matrix.append((int(R), fam, ret))
    return matrix, fam_seed


def train_smoke(model, train_preps, val_preps, val_matrix, cfg, run_dir, device, roi):
    """500-step (config) patch smoke. Returns history list + best macro PSNR."""
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    D = int(cfg["depth"])
    total = int(cfg["max_steps"]); warmup = int(cfg["warmup_steps"])
    base_lr = float(cfg["lr"]); accum = int(cfg["grad_accum"]); patch_bs = int(cfg["patch_batch"])
    val_every = int(cfg["val_every"]); seed = int(cfg["seed"])
    rng = np.random.RandomState(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=float(cfg["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["amp"]))
    grad_ok = True

    # cache full-volume linear interp lazily per (subject, mask) — masks vary per step
    prof_cache = {}

    def get_batch():
        xs, gts, lins, miss, brains = [], [], [], [], []
        for _ in range(patch_bs):
            prep = train_preps[int(rng.randint(len(train_preps)))]
            vol = prep["vol01"]; H, W, Dv = vol.shape
            ret, meta = M.sample_training_mask(D, rng, strides=tuple(cfg["strides"]))
            key = (id(prep), tuple(ret))
            if key not in prof_cache:
                prof_cache[key] = (IN.linear_interp_volume(vol, ret, D), M.positional_profiles(ret, D))
                if len(prof_cache) > 64:
                    prof_cache.pop(next(iter(prof_cache)))
            x_lin, prof = prof_cache[key]
            crop = _crop_indices(D, H, W, roi, rng, prep["fg_slices"])
            x6, gt_c, lin_c, m_c, b_c = build_patch(vol, x_lin, prof, D, crop)
            xs.append(x6); gts.append(gt_c); lins.append(lin_c); miss.append(m_c); brains.append(b_c)
            meta_rows.append(meta)
        t = lambda a: torch.from_numpy(np.stack(a)).to(device)
        return t(xs), t(gts), t(lins), t(miss), t(brains)

    history = []
    meta_rows = []
    best_macro = -1e9
    best_path = os.path.join(run_dir, "checkpoints", "best_ir1_macro_psnr.pt")
    last_path = os.path.join(run_dir, "checkpoints", "last_ir1.pt")
    step0_path = os.path.join(run_dir, "checkpoints", "step0_ir1_linear_identity.pt")

    def save_ckpt(path, step, macro):
        torch.save(dict(state_dict=model.state_dict(), optimizer=opt.state_dict(),
                        scaler=scaler.state_dict(), step=step, macro_psnr=macro,
                        residual_scale=float(model.residual_scale.item()),
                        rng_numpy=rng.get_state(), rng_torch=torch.get_rng_state(),
                        arch=model.arch, roi=list(roi), cfg=cfg), path)

    # ---- step-0 checkpoint (zero residual == linear); a valid selection candidate ----
    macro0, per0 = validate(model, val_preps, val_matrix, D, device, roi)
    save_ckpt(step0_path, 0, macro0)
    best_macro = macro0; save_ckpt(best_path, 0, macro0)
    history.append(dict(step=0, lr=0.0, **{k: float("nan") for k in ("total", "charbonnier", "grad_mae", "residual_magnitude")}, val_macro_psnr=macro0))
    print(f"[train] step 0 (linear identity) macro val PSNR = {macro0:.3f} dB")

    model.train()
    t0 = time.time()
    for step in range(1, total + 1):
        lr = cosine_warmup(step, total, warmup, base_lr)
        for pg in opt.param_groups: pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        acc_terms = None
        for _ in range(accum):
            x6, gt_c, lin_c, m_c, b_c = get_batch()
            with torch.cuda.amp.autocast(enabled=bool(cfg["amp"])):
                x_pred, delta = model(x6, lin_c, m_c, hardcopy=False, clip=True)
                loss, terms = LOSS.ir1_loss(x_pred, delta, gt_c, m_c, b_c, grad_ok=grad_ok)
                loss = loss / accum
            scaler.scale(loss).backward()
            acc_terms = terms if acc_terms is None else acc_terms
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
        assert torch.isfinite(gnorm), "non-finite grad norm"
        scaler.step(opt); scaler.update()
        if step % 25 == 0 or step == 1:
            print(f"[train] step {step}/{total} lr={lr:.2e} loss={acc_terms['total']:.4f} "
                  f"charb={acc_terms['charbonnier']:.4f} grad={acc_terms['grad_mae']:.4f}")
        if step % val_every == 0 or step == total:
            macro, per = validate(model, val_preps, val_matrix, D, device, roi)
            model.train()
            history.append(dict(step=step, lr=lr, **{k: acc_terms[k] for k in ("total", "charbonnier", "grad_mae", "residual_magnitude")}, val_macro_psnr=macro))
            print(f"[train] step {step} macro val PSNR = {macro:.3f} dB (best {best_macro:.3f})")
            if macro > best_macro:
                best_macro = macro; save_ckpt(best_path, step, macro)
                print(f"[train]   new best -> {best_path}")
    save_ckpt(last_path, total, history[-1]["val_macro_psnr"])
    dt = time.time() - t0
    print(f"[train] done in {dt:.1f}s; best macro val PSNR = {best_macro:.3f} dB")
    return history, best_macro, meta_rows, dict(step0=step0_path, best=best_path, last=last_path)
