"""No-change replay of deterministic inference + per-volume timing (GPU).

Reconstructs 2 subjects of the old IR3C test88 list (stored references exist only there; these
patients were examined before and are NOT used as fresh evidence) at U0 R4/R8 through the
proven call paths of IR3C_LPIPS/brats_methods.py, and compares masked PSNR with the stored CSVs.
"""

from project_paths import root_path
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, root_path(''))
from IR3C_LPIPS import brats_methods as BM          # noqa: E402

OUT = os.path.join(os.environ["ISBI"], "runs", "pilot")
os.makedirs(OUT, exist_ok=True)
P = BM.P
files = BM.test_files()[:2]
methods = sys.argv[1:] or ["linear", "cubic", "SAINT", "I3Net", "ArSSR_slowlr"]
rows = []
for m in methods:
    fn, ident, td = BM.build(m)
    st = pd.read_csv(BM.STORED[m], low_memory=False)
    st = st[(st.region == "missing") & (st.mask_family == "U0")]
    rcol = "nominal_R" if "nominal_R" in st.columns else "R"
    for f in files:
        gt, subj = BM.load_gt(f)
        for R in (4, 8):
            ret = P.build_mask(BM.D, R, "U0")
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
            x = fn(gt, ret, R, "U0")
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            dt = time.time() - t0
            ev = P.evaluate_volume(x, gt, ret, BM.D, region="missing")
            ref = st[(st.subject == subj) & (st[rcol] == R)].psnr_masked
            ref = float(ref.iloc[0]) if len(ref) else np.nan
            rows.append(dict(method=m, subject=subj, R=R, psnr=ev["psnr_masked"], stored=ref,
                             abs_diff=abs(ev["psnr_masked"] - ref), sec=dt,
                             peak_gb=torch.cuda.max_memory_allocated() / 1e9))
            print(rows[-1], flush=True)
    td()
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, f"replay_{'_'.join(methods)}.csv"), index=False)
print(df.to_string())
print("MAX |dPSNR| =", df.abs_diff.max())
