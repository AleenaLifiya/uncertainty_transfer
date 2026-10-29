"""
bratsloader_controlnet_fidelity.py
----------------------------------
Fidelity dataloader for through-plane sparse-slice reconstruction (Phase 3 of
improve_xdiffcontrol.md). Replaces the random single-neighbour formulation.

For every training sample:
  * a stride R is sampled uniformly from {2, 4, 8};
  * two RETAINED real slices bracket the target: left_idx and right_idx=left_idx+R;
  * one missing target index d is strictly between them;
  * alpha = (d - left_idx) / (right_idx - left_idx);
  * linear = (1-alpha)*left + alpha*right  is the strong deterministic estimate.

X-Diffusion condition (unchanged plumbing):
  * image_cond   = linear   (repeated to 3 channels)  -> c_concat + CLIP/pose
  * image_target = target   (repeated to 3 channels)  -> reconstruct

ControlNet hint (single tensor, 5 channels, built directly in torch):
  0: left acquired slice
  1: right acquired slice
  2: linear interpolation
  3: constant alpha map
  4: constant normalised stride map (R / 8)

The target slice NEVER appears in any conditioning tensor.

`build_conditioning` is used by BOTH training and inference so the two are
guaranteed identical.
"""

import glob
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from ldm.data.brats_preprocessing_fidelity import (
    prepare_volume, pad_slice, TARGET_HW, DEPTH_CROP, PERCENTILE_LO, PERCENTILE_HI,
)

STRIDES = (2, 4, 8)
STRIDE_NORM = 8.0  # normalise stride map by max stride


def _to_3ch_hwc_pm1(slice01):
    """[0,1] HxW -> [-1,1] (H, W, 3) float32 tensor (model input space)."""
    t = torch.from_numpy(np.ascontiguousarray(slice01)).float()
    t = t * 2.0 - 1.0
    return t.unsqueeze(-1).repeat(1, 1, 3)


def build_conditioning(prep, left_idx, right_idx, d, target=TARGET_HW, debug=True):
    """Build all tensors for one (left_idx, right_idx, d) triple from a prepared
    volume dict (from prepare_volume). Shared by training and inference.

    Returns dict of torch tensors:
      image_target (H,W,3) [-1,1], image_cond (H,W,3) [-1,1],
      control (H,W,5) [0,1]-ish, T (4,), plus scalar metadata.
    """
    R = right_idx - left_idx
    alpha = (d - left_idx) / float(R)
    if debug:
        assert R in STRIDES, f"stride {R} not in {STRIDES}"
        assert left_idx < d < right_idx, f"target {d} not strictly between {left_idx},{right_idx}"
        assert abs(alpha - (d - left_idx) / (right_idx - left_idx)) < 1e-6

    left = pad_slice(prep["vol01"], left_idx, target)
    right = pad_slice(prep["vol01"], right_idx, target)
    tgt = pad_slice(prep["vol01"], d, target)
    linear = (1.0 - alpha) * left + alpha * right

    image_target = _to_3ch_hwc_pm1(tgt)
    image_cond = _to_3ch_hwc_pm1(linear)

    # 5-channel hint, built directly in torch, NOT via PIL, NOT repeated to RGB.
    hint = torch.stack([
        torch.from_numpy(np.ascontiguousarray(left)).float(),
        torch.from_numpy(np.ascontiguousarray(right)).float(),
        torch.from_numpy(np.ascontiguousarray(linear)).float(),
        torch.full((target, target), float(alpha)),
        torch.full((target, target), R / STRIDE_NORM),
    ], dim=-1)  # (H, W, 5)

    # Absolute target-depth conditioning kept for checkpoint compatibility (axial).
    depth_norm = 2.0 * (d / max(1, prep["depth"] - 1)) - 1.0
    T = torch.tensor([0.0, 0.0, 1.0, float(depth_norm)], dtype=torch.float32)

    if debug:
        assert image_target.shape == (target, target, 3)
        assert image_cond.shape == (target, target, 3)
        assert hint.shape == (target, target, 5)
        assert 0.0 - 1e-4 <= alpha <= 1.0 + 1e-4

    return {
        "image_target": image_target,
        "image_cond": image_cond,
        "control": hint,
        "T": T,
        "alpha": float(alpha),
        "stride": int(R),
        "left_idx": int(left_idx),
        "right_idx": int(right_idx),
        "d": int(d),
    }


class _VolumeCache:
    """Tiny per-worker LRU cache of prepared volumes (avoids re-reading NIfTI)."""

    def __init__(self, capacity=4, p_lo=PERCENTILE_LO, p_hi=PERCENTILE_HI, depth=DEPTH_CROP):
        self.capacity = capacity
        self.p_lo, self.p_hi, self.depth = p_lo, p_hi, depth
        self.store = OrderedDict()

    def get(self, path):
        if path in self.store:
            self.store.move_to_end(path)
            return self.store[path]
        prep = prepare_volume(path, p_lo=self.p_lo, p_hi=self.p_hi, depth=self.depth)
        self.store[path] = prep
        self.store.move_to_end(path)
        while len(self.store) > self.capacity:
            self.store.popitem(last=False)
        return prep


class BratsFidelityDataset(Dataset):
    """Axial through-plane fidelity dataset with dual-neighbour bracketing."""

    def __init__(self, files, samples_per_volume=8, strides=STRIDES,
                 p_foreground=0.9, target=TARGET_HW, depth=DEPTH_CROP,
                 p_lo=PERCENTILE_LO, p_hi=PERCENTILE_HI, debug_asserts=True,
                 cache_capacity=4, seed=0):
        self.files = list(files)
        if not self.files:
            raise ValueError("BratsFidelityDataset: file list is empty")
        self.spv = max(1, int(samples_per_volume))
        self.strides = tuple(strides)
        self.p_foreground = float(p_foreground)
        self.target = int(target)
        self.depth = int(depth)
        self.p_lo, self.p_hi = p_lo, p_hi
        self.debug_asserts = bool(debug_asserts)
        self.cache_capacity = int(cache_capacity)
        self.base_seed = int(seed)
        self._cache = None  # created lazily per worker

    def __len__(self):
        return len(self.files) * self.spv

    def _cache_get(self, path):
        if self._cache is None:
            self._cache = _VolumeCache(self.cache_capacity, self.p_lo, self.p_hi, self.depth)
        return self._cache.get(path)

    def _sample_triple(self, prep, rng):
        depth = prep["depth"]
        fg = prep["fg_slices"]
        R = rng.choice(self.strides)
        use_fg = (rng.random() < self.p_foreground) and len(fg) > (R + 1)
        for _ in range(20):
            if use_fg:
                lo_fg, hi_fg = min(fg), max(fg)
                left_idx = rng.randint(max(0, lo_fg - R), min(depth - 1 - R, hi_fg))
            else:
                left_idx = rng.randint(0, depth - 1 - R)
            right_idx = left_idx + R
            if right_idx > depth - 1:
                continue
            d = rng.randint(left_idx + 1, right_idx - 1)
            return int(left_idx), int(right_idx), int(d), int(R)
        # fallback: guaranteed-valid central triple
        left_idx = (depth - 1) // 2 - R
        return left_idx, left_idx + R, left_idx + R // 2, R

    def __getitem__(self, idx):
        path = self.files[idx // self.spv]
        # deterministic-but-varied per (sample, epoch) randomness via python rng
        rng = random.Random((self.base_seed * 1000003 + idx) & 0x7FFFFFFF)
        # decorrelate across epochs using torch's initial seed if present
        try:
            rng.seed((torch.initial_seed() + idx) & 0x7FFFFFFF)
        except Exception:
            pass
        prep = self._cache_get(path)
        left_idx, right_idx, d, R = self._sample_triple(prep, rng)
        cond = build_conditioning(prep, left_idx, right_idx, d,
                                  target=self.target, debug=self.debug_asserts)
        cond["filename"] = Path(path).parent.name
        return cond


def discover_files(root_dir, glob_pattern="*t2f.nii.gz"):
    """Recursively find volumes; verified against real files by the caller."""
    paths = sorted(glob.glob(os.path.join(root_dir, "**", glob_pattern), recursive=True))
    return paths


class BratsFidelityDataModule(pl.LightningDataModule):
    """Datamodule that takes EXPLICIT file lists (from immutable manifests)."""

    def __init__(self, train_files, val_files, batch_size=8, num_workers=4,
                 samples_per_volume=8, strides=STRIDES, p_foreground=0.9,
                 target=TARGET_HW, depth=DEPTH_CROP, p_lo=PERCENTILE_LO,
                 p_hi=PERCENTILE_HI, seed=42, debug_asserts=True):
        super().__init__()
        self.train_files = list(train_files)
        self.val_files = list(val_files)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.spv = samples_per_volume
        self.strides = strides
        self.p_foreground = p_foreground
        self.target = target
        self.depth = depth
        self.p_lo, self.p_hi = p_lo, p_hi
        self.seed = seed
        self.debug_asserts = debug_asserts

    def _loader(self, files, shuffle, spv):
        ds = BratsFidelityDataset(
            files, samples_per_volume=spv, strides=self.strides,
            p_foreground=self.p_foreground, target=self.target, depth=self.depth,
            p_lo=self.p_lo, p_hi=self.p_hi, debug_asserts=self.debug_asserts,
            seed=self.seed,
        )

        def _winit(worker_id):
            base = (self.seed + worker_id) & 0x7FFFFFFF
            np.random.seed(base)
            random.seed(base)

        return DataLoader(
            ds, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=shuffle, drop_last=shuffle, pin_memory=True,
            worker_init_fn=_winit,
            persistent_workers=(self.num_workers > 0),
        )

    def train_dataloader(self):
        return self._loader(self.train_files, shuffle=True, spv=self.spv)

    def val_dataloader(self):
        return self._loader(self.val_files, shuffle=False, spv=1)


