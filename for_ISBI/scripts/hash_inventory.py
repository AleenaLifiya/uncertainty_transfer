"""Hash every reused (read-only) file. `--write` records, `--verify` compares.

    python scripts/hash_inventory.py --write  provenance/reused_sha256.json
    python scripts/hash_inventory.py --verify provenance/reused_sha256.json
"""

from project_paths import root_path
import glob
import hashlib
import json
import os
import sys

ROOT = root_path('')
PATTERNS = [
    "comparisons/*.py", "IR3C_LPIPS/brats_methods.py", "image_residual/*.py",
    "nonuniform_masks.py", "nonuniform_method_adapter.py", "ir1_audit_common.py", "ir2_components.py",
    "X-Diffusion/scripts/ldm/data/brats_preprocessing_fidelity.py",
    "X-Diffusion/scripts/ldm/models/diffusion/fidelity_runtime.py",
    "IR3C_more_comparisons/comparisons_more/*.py",
    "IR3C_more_comparisons/repositories/SAINT/model/*.py",
    "IR3C_more_comparisons/repositories/SAINT/data/srdata.py",
    "I3Net/model_zoo/i3net/*.py", "I3Net/select_model.py", "I3Net/opt/i3net.json",
    "configs/ir3c_comparisons.yaml", "IR3C_more_comparisons/configs/ir3c_more_comparisons.yaml",
    "runs/brats_frozen_vae_full_20260716_122401/splits/*.txt",
    "runs/brats_ir1_image_residual_smoke_20260728_121036/validation_matrix.json",
    "runs/brats_ir3c_20260902/selected_*.json", "runs/brats_ir3c_20260902/results_I3Net.csv",
    "runs/brats_ir3c_20260902/results_linear.csv", "runs/brats_ir3c_20260902/results_cubic.csv",
    "runs/brats_ir3c_20260902/results_ArSSR_slowlr.csv",
    "runs/brats_ir3c_20260902/checkpoints/i3net_x4/i3net_x4_step006255.pt",
    "runs/brats_ir3c_20260902/checkpoints/i3net_x8/i3net_x8_step003753.pt",
    "runs/brats_ir3c_20260902/checkpoints/arssr_slowlr/arssr_slowlr_step012510.pt",
    "IR3C_more_comparisons/metrics/selected_SAINT_*.json",
    "IR3C_more_comparisons/metrics/results_SAINT.csv",
    "IR3C_more_comparisons/checkpoints/SAINT/SAINT_rfn/SAINT_rfn_step011259.pt",
    "IR3C_more_comparisons/checkpoints/SAINT/SAINT_msr/SAINT_msr_step011259.pt",
    "IR3C_more_comparisons/predictions/saint_msr_cache_index.csv",
    "IR3C_more_comparisons/predictions/saint_msr_cache_manifest.json",
    "IR3C_more_comparisons/predictions/saint_msr_cache/*.npz",
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def collect():
    out = {}
    for pat in PATTERNS:
        hits = sorted(glob.glob(os.path.join(ROOT, pat)))
        if not hits:
            out[pat] = "MISSING"
        for p in hits:
            out[os.path.relpath(p, ROOT)] = sha(p)
    return out


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    cur = collect()
    if mode == "--write":
        json.dump(cur, open(path, "w"), indent=1)
        print(f"{len(cur)} entries; missing: {[k for k, v in cur.items() if v == 'MISSING']}")
    else:
        ref = json.load(open(path))
        bad = [k for k in ref if cur.get(k) != ref[k]]
        print(f"verified {len(ref)} entries; changed/missing: {bad}")
        sys.exit(1 if bad else 0)
