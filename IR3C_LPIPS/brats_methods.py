"""
IR3C_LPIPS/brats_methods.py
===========================
Rebuilds every BraTS reconstruction method of the IR3C comparison as a plain
callable `recon(gt01, ret, R, fam) -> (240,240,153) float32`.

Nothing here re-selects a checkpoint and nothing here is new science: each branch
is a verbatim copy of the call path that the ORIGINAL test stage used, so the
reconstruction it returns is the same array the stored PSNR/SSIM were computed
on. `IR3C_LPIPS/preflight.py` proves that by recomputing PSNR and comparing
against the stored CSV.

Two source experiments feed this:
  runs/brats_ir3c_20260902      nearest/linear/cubic/bicubic, IR3, ArSSR, I3Net
  IR3C_more_comparisons/        DCSRN, DCSRN_mixed, SAINT, SAINT_stage1

ArSSR: only the `ArSSR_slowlr` (slow LR-decay) variant is carried, on the user's
instruction -- it is the better of the two ArSSR runs (20.22 vs 19.99 dB mean
masked PSNR over the 2464-cell grid).
"""
from __future__ import annotations

from project_paths import root_path

import gc
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

PROJECT = root_path('')
EXP_MORE = os.path.join(PROJECT, "IR3C_more_comparisons")
for _d in (PROJECT, EXP_MORE, os.path.join(PROJECT, "X-Diffusion", "scripts")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from comparisons import protocol as P                  # noqa: E402
from comparisons import arssr_adapter as AR            # noqa: E402
from comparisons import i3net_adapter as I3            # noqa: E402
from comparisons_more import saint_adapter as SA       # noqa: E402
from comparisons_more import dcsrn_adapter as DC       # noqa: E402
import image_residual as IR                            # noqa: E402
import ir2_components as C2                            # noqa: E402

RUN_MAIN = os.path.join(PROJECT, "runs", "brats_ir3c_20260902")
CFG = OmegaConf.load(os.path.join(PROJECT, "configs", "ir3c_comparisons.yaml"))
CFG_MORE = OmegaConf.load(os.path.join(EXP_MORE, "configs", "ir3c_more_comparisons.yaml"))
D = 153
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#: method -> (source experiment, strides, families) exactly as each was tested
GRIDS = {
    "nearest":      ("main", [2, 4, 8, 16], list(CFG.families)),
    "linear":       ("main", [2, 4, 8, 16], list(CFG.families)),
    "cubic":        ("main", [2, 4, 8, 16], list(CFG.families)),
    "bicubic":      ("main", [2, 4, 8, 16], list(CFG.families)),
    "IR3_matched":  ("main", [2, 4, 8, 16], list(CFG.families)),
    "IR3_full":     ("main", [2, 4, 8, 16], list(CFG.families)),
    "ArSSR_slowlr": ("main", [2, 4, 8, 16], list(CFG.families)),
    "I3Net":        ("main", [2, 4, 8],     ["U0"]),
    "DCSRN":        ("more", [2, 4, 8],     ["U0"]),
    "DCSRN_mixed":  ("more", [2, 4, 8],     list(CFG.families)),
    "SAINT":        ("more", [2, 4, 8],     ["U0"]),
    "SAINT_stage1": ("more", [2, 4, 8],     ["U0"]),
}
ALL_METHODS = list(GRIDS)

#: where each method's stored reference CSV lives (preflight joins against these)
STORED = {m: (os.path.join(RUN_MAIN, f"results_{m}.csv") if src == "main"
              else os.path.join(EXP_MORE, "metrics", f"results_{m}.csv"))
          for m, (src, _, _) in GRIDS.items()}
# the main run wrote ArSSR_slowlr under its own name
STORED["ArSSR_slowlr"] = os.path.join(RUN_MAIN, "results_ArSSR_slowlr.csv")


def cells(method):
    """The frozen (R, family, retained-indices) grid this method was tested on."""
    _, strides, fams = GRIDS[method]
    grid, _ = P.eval_grid(strides, fams, D)
    return grid


def test_files():
    return P.load_splits()["test"]


def load_gt(path):
    prep = P.prepare_one(path)
    return prep["vol01"], prep["subject"]


# --------------------------------------------------------------------------- #
#  model loaders -- each mirrors the original test stage exactly
# --------------------------------------------------------------------------- #
def _sel_main(name):
    return json.load(open(os.path.join(RUN_MAIN, f"selected_{name}.json")))


def _sel_more(name):
    return json.load(open(os.path.join(EXP_MORE, "metrics", f"selected_{name}.json")))


def _load_ir3(tag):
    """IR3_matched / IR3_full, re-selected with the SAME rule the test stage used."""
    ir3_run = os.path.join(PROJECT, str(CFG.ir3_run))
    per = pd.read_csv(os.path.join(ir3_run, "validation_metrics_per_checkpoint.csv"))
    if tag == "IR3_matched":
        per = per[per.step <= int(CFG.budget.total_steps)]
    _, sel = P.select_checkpoint(per.to_dict("records"))
    ck = torch.load(os.path.join(ir3_run, sel["selected_path"]), map_location="cpu",
                    weights_only=False)
    m = IR.model.build_ir1(ck["residual_scale"], arch=ck.get("arch")).to(DEVICE)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, sel


def _load_arssr_slowlr():
    name = str(CFG.arssr_sensitivity.name)
    sel = json.load(open(os.path.join(RUN_MAIN, f"selected_{name}.json")))
    ck = torch.load(os.path.join(RUN_MAIN, sel["selected_path"]), map_location="cpu",
                    weights_only=False)
    c = CFG.arssr
    net = AR.build_model(DEVICE, str(c.encoder_name), int(c.feature_dim),
                         int(c.decoder_depth), int(c.decoder_width))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, OmegaConf.to_container(c.inference), sel


def _load_i3net():
    nets, sels = {}, {}
    for R in (2, 4, 8):
        sel = _sel_main(f"i3net_x{R}")
        ck = torch.load(os.path.join(RUN_MAIN, sel["selected_path"]), map_location="cpu",
                        weights_only=False)
        m, _ = I3.build_model(int(R), int(CFG.i3net.lr_slice_patch), DEVICE)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        nets[R], sels[R] = m, sel
    return nets, sels


def _load_dcsrn(mixed=False):
    if mixed:
        mc = CFG_MORE.dcsrn_mixed
        scales = [int(r) for r in mc.scales]
        names = {R: f"{mc.name_prefix}_x{R}" for R in scales}
    else:
        scales = [int(r) for r in CFG_MORE.dcsrn.scales]
        names = {R: f"DCSRN_x{R}" for R in scales}
    dec_path = os.path.join(EXP_MORE, "metrics", "dcsrn_inference_mode.json")
    dec = json.load(open(dec_path)) if os.path.exists(dec_path) else {}
    nets, sels, modes = {}, {}, {}
    for R in scales:
        sel = _sel_more(names[R])
        ck = torch.load(os.path.join(EXP_MORE, sel["selected_path"]), map_location="cpu",
                        weights_only=False)
        n = DC.build_model(DEVICE)
        n.load_state_dict(ck["state_dict"])
        n.eval()
        nets[R], sels[R] = n, sel
        modes[R] = (sel.get("inference_mode") or dec.get(f"DCSRN_x{R}", {}).get("chosen_mode")
                    or "tiled64")
    return nets, sels, modes


def _load_saint(stage):
    name = "SAINT_rfn" if stage == 2 else "SAINT_msr"
    sel = _sel_more(name)
    ck = torch.load(os.path.join(EXP_MORE, sel["selected_path"]), map_location="cpu",
                    weights_only=False)
    net, _ = SA.build_model(DEVICE, stage=2)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, int(CFG_MORE.saint.inference.chunk), sel


# --------------------------------------------------------------------------- #
#  public: build one method's recon callable
# --------------------------------------------------------------------------- #
def build(method):
    """Return (recon_fn, identity_dict, teardown_fn)."""
    if method in ("nearest", "linear", "cubic", "bicubic"):
        return (lambda gt, ret, R, fam, k=method: P.interpolate_baseline(gt, ret, k, D),
                dict(kind="interpolation"), lambda: None)

    if method in ("IR3_matched", "IR3_full"):
        m, sel = _load_ir3(method)
        fn = lambda gt, ret, R, fam: C2.reconstruct_ir2(m, gt, ret, D, DEVICE)[0]
        return fn, dict(kind="learned", selected_step=int(sel["selected_step"]),
                        selected_path=sel["selected_path"]), _teardown(m)

    if method == "ArSSR_slowlr":
        net, inf, sel = _load_arssr_slowlr()
        fn = lambda gt, ret, R, fam: AR.reconstruct(net, gt, ret, D, DEVICE, **inf)
        return fn, dict(kind="learned", selected_step=int(sel["selected_step"]),
                        selected_path=sel["selected_path"], inference=inf), _teardown(net)

    if method == "I3Net":
        nets, sels = _load_i3net()
        patch = int(CFG.i3net.lr_slice_patch)
        fn = lambda gt, ret, R, fam: I3.reconstruct(nets[int(R)], gt, ret, int(R), D,
                                                    DEVICE, patch)
        return fn, dict(kind="learned",
                        selected_step={str(k): int(v["selected_step"]) for k, v in sels.items()}
                        ), _teardown(nets)

    if method in ("DCSRN", "DCSRN_mixed"):
        nets, sels, modes = _load_dcsrn(mixed=(method == "DCSRN_mixed"))
        fn = lambda gt, ret, R, fam: DC.reconstruct(nets[int(R)], gt, ret, int(R), D,
                                                    DEVICE, mode=modes[int(R)])
        return fn, dict(kind="learned", inference_mode={str(k): v for k, v in modes.items()},
                        selected_step={str(k): int(v["selected_step"]) for k, v in sels.items()}
                        ), _teardown(nets)

    if method in ("SAINT", "SAINT_stage1"):
        stage = 2 if method == "SAINT" else 1
        net, chunk, sel = _load_saint(stage)
        fn = lambda gt, ret, R, fam: SA.reconstruct(net, gt, ret, int(R), D, DEVICE,
                                                    stage=stage, chunk=chunk)
        return fn, dict(kind="learned", stage=stage, chunk=chunk,
                        selected_step=int(sel["selected_step"])), _teardown(net)

    raise KeyError(method)


def _teardown(obj):
    ref = [obj]

    def _f():
        ref.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return _f
