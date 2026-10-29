"""
brats_preprocessing_fidelity.py
--------------------------------
Volume-level preprocessing shared by the fidelity dataloader, the inference
code, the VAE round-trip ceiling test and the metric code, so that every stage
prepares slices IDENTICALLY.

Design (Phase 2 of improve_xdiffcontrol.md):
  * NO per-slice min-max normalisation.
  * One robust intensity window per volume, computed from the 1st and 99.5th
    percentiles INSIDE the nonzero foreground mask, then mapped once to [0,1].
  * Geometry preserved: 240x240 axial slices are SYMMETRICALLY PADDED to 256x256
    (never resized); depth is centrally cropped to D=153 so (D-1) is divisible
    by 2, 4 and 8.
  * The through-plane (axial) axis is axis 2 for BraTS 240x240x155 volumes.

All functions are pure/deterministic given their inputs.
"""

import hashlib
import numpy as np
import nibabel as nib

# Through-plane (axial) axis for BraTS GLI volumes (240, 240, 155).
THROUGH_PLANE_AXIS = 2
# In-plane target after symmetric padding.
TARGET_HW = 256
# Central depth crop so (D-1) is divisible by 2, 4, 8 (D-1 = 152 = 8*19).
DEPTH_CROP = 153
PERCENTILE_LO = 1.0
PERCENTILE_HI = 99.5


def load_raw_volume(path):
    """Load a NIfTI volume as float32 without any normalisation.

    Returns (vol, affine, header). Never writes to the source directory.
    """
    img = nib.load(str(path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    return vol, img.affine, img.header


def foreground_mask(vol):
    """Boolean nonzero-voxel foreground mask."""
    return vol > 0.0


def robust_window(vol, mask=None, p_lo=PERCENTILE_LO, p_hi=PERCENTILE_HI):
    """Robust intensity limits (lo, hi) from foreground percentiles.

    Computed once at volume level so every slice keeps a common scale.
    """
    if mask is None:
        mask = foreground_mask(vol)
    fg = vol[mask]
    if fg.size == 0:  # empty volume guard
        return 0.0, 1.0
    lo = float(np.percentile(fg, p_lo))
    hi = float(np.percentile(fg, p_hi))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def normalize_volume(vol, p_lo=PERCENTILE_LO, p_hi=PERCENTILE_HI):
    """Clip once at volume level and map once to [0,1] using the robust window.

    Returns (vol01, lo, hi). The SAME (lo, hi) scale applies to every slice of
    this volume, so target / left / right / linear all share one calibration.
    """
    mask = foreground_mask(vol)
    lo, hi = robust_window(vol, mask, p_lo, p_hi)
    vol01 = np.clip(vol, lo, hi)
    vol01 = (vol01 - lo) / (hi - lo)
    return vol01.astype(np.float32), lo, hi


def central_depth_crop(vol, axis=THROUGH_PLANE_AXIS, depth=DEPTH_CROP):
    """Central crop along the through-plane axis to `depth`.

    Returns (cropped_vol, start) where start is the first kept index in the
    original volume, so slice indices can be mapped back for metrics.
    """
    n = vol.shape[axis]
    if n <= depth:
        return vol, 0
    start = (n - depth) // 2
    sl = [slice(None)] * vol.ndim
    sl[axis] = slice(start, start + depth)
    return vol[tuple(sl)], start


def pad_hw_to(slice2d, target=TARGET_HW):
    """Symmetrically zero-pad a 2D slice to (target, target).

    Returns (padded, (top, bottom, left, right)). Assumes slice <= target in
    both dims; raises otherwise (we never resize).
    """
    h, w = slice2d.shape
    if h > target or w > target:
        raise ValueError(f"slice {slice2d.shape} larger than target {target}; refuse to resize")
    ph, pw = target - h, target - w
    top, left = ph // 2, pw // 2
    bottom, right = ph - top, pw - left
    padded = np.pad(slice2d, ((top, bottom), (left, right)), mode="constant", constant_values=0.0)
    return padded, (top, bottom, left, right)


def unpad_hw(slice2d, pad):
    """Undo pad_hw_to given (top, bottom, left, right)."""
    top, bottom, left, right = pad
    h, w = slice2d.shape
    return slice2d[top:h - bottom, left:w - right]


def prepare_volume(path, p_lo=PERCENTILE_LO, p_hi=PERCENTILE_HI,
                   axis=THROUGH_PLANE_AXIS, depth=DEPTH_CROP, target=TARGET_HW):
    """Full volume preprocessing used everywhere.

    Returns a dict with:
      vol01        : (H, W, D) normalised [0,1], depth-cropped (in-plane NOT padded)
      lo, hi       : intensity window used
      depth_start  : first kept index of the depth crop in the original volume
      fg_slices    : list of through-plane indices (into vol01) with foreground
      pad          : pad tuple that pad_slice() will apply per slice
      shape_raw    : original volume shape
      affine       : NIfTI affine
    Slices are padded lazily via pad_slice() to avoid storing padded volumes.
    """
    vol, affine, _ = load_raw_volume(path)
    vol01, lo, hi = normalize_volume(vol, p_lo, p_hi)
    cropped, depth_start = central_depth_crop(vol01, axis=axis, depth=depth)
    # move through-plane axis to last for uniform [H, W, D] indexing
    vol_hw_d = np.moveaxis(cropped, axis, 2)
    # foreground slice range (indices into the cropped volume)
    mask = cropped > 0.0
    mask_hw_d = np.moveaxis(mask, axis, 2)
    fg_per_slice = mask_hw_d.reshape(-1, mask_hw_d.shape[2]).sum(axis=0)
    fg_slices = np.where(fg_per_slice > 0)[0].tolist()
    # pad params are the same for every slice of this volume
    _, pad = pad_hw_to(vol_hw_d[:, :, 0], target)
    return {
        "vol01": vol_hw_d,
        "lo": lo, "hi": hi,
        "depth_start": int(depth_start),
        "fg_slices": fg_slices,
        "pad": pad,
        "shape_raw": tuple(int(s) for s in vol.shape),
        "affine": affine,
        "target": target,
        "axis": axis,
        "depth": int(vol_hw_d.shape[2]),
    }


def pad_slice(vol01_hw_d, idx, target=TARGET_HW):
    """Return the padded 2D slice at through-plane index `idx`, in [0,1]."""
    s = vol01_hw_d[:, :, idx]
    padded, _ = pad_hw_to(s, target)
    return padded.astype(np.float32)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


