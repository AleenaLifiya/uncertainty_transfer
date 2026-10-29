"""
ir3c_v2/v2paths.py
==================
Every path IR3C_experiments_V2 uses, in one place, plus the vendor-dir bootstrap.

Importing this module is the only supported way for V2 code to learn where anything is.
It never writes outside V2_DIR.
"""
from __future__ import annotations
import os
import sys

V2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.dirname(V2_DIR)
VENDOR = os.path.join(V2_DIR, "vendor")

# vendored FLOP libraries -- installed with `pip install --target`, NOT into the xdiff env
if os.path.isdir(VENDOR) and VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)
# the project itself
for p in (PROJECT, os.path.join(PROJECT, "X-Diffusion", "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

PY = os.environ.get("V2_PY", __import__('sys').executable)

# ---- V2 stage directories -------------------------------------------------
STAGE0 = os.path.join(V2_DIR, "stage0_audit")
STAGE1 = os.path.join(V2_DIR, "stage1_profiling")
STAGE2 = os.path.join(V2_DIR, "stage2_finetune")
STAGE3 = os.path.join(V2_DIR, "stage3_nmse")
STAGE4 = os.path.join(V2_DIR, "stage4_ablationA")
STAGE5 = os.path.join(V2_DIR, "stage5_validation")
STAGE6 = os.path.join(V2_DIR, "stage6_reports")
LOGS = os.path.join(V2_DIR, "logs")
CONFIGS = os.path.join(V2_DIR, "configs")

# ---- read-only source experiments ----------------------------------------
RUNS = os.path.join(PROJECT, "runs")
IR3_RUN = os.path.join(RUNS, "brats_ir3_standalone_20260830_182349")
IR3C_RUN = os.path.join(RUNS, "brats_ir3c_20260902")
IR1_RUN = os.path.join(RUNS, "brats_ir1_image_residual_smoke_20260728_121036")
ABL1_RUN = os.path.join(RUNS, "IR3C_brats_ablation1")
OAIZIB_RUN = os.path.join(RUNS, "IR3C_OAIZIB_09Sept_fulltrain")
IXI_RUN = os.path.join(RUNS, "IR3C_IXItiny_08Sept")
MORE_DIR = os.path.join(PROJECT, "IR3C_more_comparisons")

# ---- the checkpoints the BraTS comparison reports -------------------------
CKPT = dict(
    IR3C=os.path.join(IR3_RUN, "checkpoints", "ir2_epoch023.pt"),
    I3Net_x2=os.path.join(IR3C_RUN, "checkpoints", "i3net_x2", "i3net_x2_step005004.pt"),
    I3Net_x4=os.path.join(IR3C_RUN, "checkpoints", "i3net_x4", "i3net_x4_step006255.pt"),
    I3Net_x8=os.path.join(IR3C_RUN, "checkpoints", "i3net_x8", "i3net_x8_step003753.pt"),
    ArSSR=os.path.join(IR3C_RUN, "checkpoints", "arssr_slowlr", "arssr_slowlr_step012510.pt"),
    DCSRN_x2=os.path.join(MORE_DIR, "checkpoints", "DCSRN", "DCSRN_x2", "DCSRN_x2_step008757.pt"),
    DCSRN_x4=os.path.join(MORE_DIR, "checkpoints", "DCSRN", "DCSRN_x4", "DCSRN_x4_step011259.pt"),
    DCSRN_x8=os.path.join(MORE_DIR, "checkpoints", "DCSRN", "DCSRN_x8", "DCSRN_x8_step011259.pt"),
    SAINT_msr=os.path.join(MORE_DIR, "checkpoints", "SAINT", "SAINT_msr", "SAINT_msr_step011259.pt"),
    SAINT_rfn=os.path.join(MORE_DIR, "checkpoints", "SAINT", "SAINT_rfn", "SAINT_rfn_step011259.pt"),
    IR3C_OAIZIB=os.path.join(OAIZIB_RUN, "IR3C_OAIZIB_09Sept_fulltrain_best_checkpoint.pt"),
    IR3C_IXI=os.path.join(IXI_RUN, "IR3C_IXItiny_08Sept_best_checkpoint.pt"),
    ABL1_CONTROL=os.path.join(ABL1_RUN, "checkpoints", "linear", "epoch025.pt"),
    ABL1_SHARED_INIT=os.path.join(ABL1_RUN, "provenance", "initial_model.pt"),
)

# ---- datasets -------------------------------------------------------------
OAIZIB_ROOT = os.path.join(PROJECT, "OAIZIB_dataset")
IXI_ROOT = os.path.join(PROJECT, "IXI_tiny")

# ---- report sources (Stage 6) --------------------------------------------
REPORT_COMPLETE = os.path.join(PROJECT, "reports", "ir3_complete_report")
REPORT_JOURNAL = os.path.join(PROJECT, "IR3C_journal")


def ensure(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs[0] if len(dirs) == 1 else dirs
