"""Resumable evaluation over one role.

    python scripts/run_eval.py --role FIT|CAL|TEST|MCCONV [--T 20] [--n N]

FIT / CAL : uniform x4, x8 only (post-hoc fitting and interval calibration are uniform-only).
TEST      : all frozen conditions (uniform; jittered r0-2; clustered r0-2) at x4 and x8.
MCCONV    : first 6 SEL patients, uniform + clustered r0 at x4, T=100 (MC convergence check).
Output: runs/eval/<ROLE>/cases/<patient>__<cond>.npz and rows/<patient>__<cond>.csv (atomic).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from isbi import common as C
from isbi import evalcore as E

ap = argparse.ArgumentParser()
ap.add_argument("--role", required=True)
ap.add_argument("--T", type=int, default=20)
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--dry", action="store_true", help="pipeline test on SEL patients only, source-init models")
args = ap.parse_args()
torch.backends.cudnn.deterministic = True

roles = C.load_roles()
if args.dry:
    proto = dict(selected_checkpoints={"saint_prob": None, "i3net_prob_x4": None, "i3net_prob_x8": None})
    dry = {"FIT": roles["SEL"][6:9], "CAL": roles["SEL"][9:12], "TEST": roles["SEL"][6:8]}
elif args.role == "TEST":
    proto = json.load(open(os.path.join(C.ISBI, "configs", "protocol_frozen.json")))   # TEST only after freeze
    args.T = proto["T"]
else:
    pf = os.path.join(C.ISBI, "configs", "protocol_frozen.json")
    proto = json.load(open(pf if os.path.exists(pf) else os.path.join(C.ISBI, "configs", "protocol_prefreeze.json")))
    if args.role != "MCCONV":
        args.T = proto["T"]
if args.dry:
    pats = dry[args.role]
    conds = C.conditions(test=(args.role == "TEST"))
elif args.role == "TEST":
    pats = roles["TEST_ORDER"][:proto["n_test"]]
    conds = C.conditions(test=True)
elif args.role == "MCCONV":
    pats = roles["SEL"][:6]
    conds = [("uniform", 4, 0), ("clustered", 4, 0)]
    args.T = 100
else:
    pats = roles[args.role]
    conds = C.conditions(test=False)
if args.n:
    pats = pats[:args.n]
OUT = os.path.join(C.ISBI, "runs", "eval_dry" if args.dry else "eval", args.role)
for d in ("cases", "rows"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
models = E.Models(proto["selected_checkpoints"])
todo = [(p, c) for p in pats for c in conds]
t0, n_done = time.time(), 0
for k, (p, (pat, R, rep)) in enumerate(todo):
    key = f"{p}__{C.cond_key(pat, R, rep)}"
    fr = os.path.join(OUT, "rows", key + ".csv")
    if os.path.exists(fr):
        continue
    ts = time.time()
    out, rows, _ = E.run_case(models, roles["scan_of"][p], p, pat, R, rep, args.T)
    np.savez(os.path.join(OUT, "cases", key + ".tmp.npz"), **out)
    os.replace(os.path.join(OUT, "cases", key + ".tmp.npz"), os.path.join(OUT, "cases", key + ".npz"))
    dfr = pd.DataFrame(rows)
    dfr["case_wall_sec"] = time.time() - ts
    dfr["T"] = args.T
    dfr.to_csv(fr + ".tmp", index=False)
    os.replace(fr + ".tmp", fr)
    n_done += 1
    el = time.time() - t0
    left = len(todo) - k - 1
    msg = (f"[{args.role}] {k + 1}/{len(todo)} {key} {time.time() - ts:.1f}s  ETA {el / n_done * left / 3600:.2f} h  "
           + " ".join(f"{r['method']}={r['psnr']:.2f}" for r in rows))
    print(msg, flush=True)
    with open(os.path.join(OUT, "progress.txt"), "w") as f:
        f.write(msg + "\n")
print(f"[{args.role}] complete", flush=True)
