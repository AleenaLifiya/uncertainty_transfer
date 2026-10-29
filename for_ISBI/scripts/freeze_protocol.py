"""Freeze checkpoints, T and n_test by the prespecified rules (PROTOCOL.md). Run before TEST."""
import glob, json, os
import numpy as np, pandas as pd
from isbi import common as C
sel = {n: json.load(open(os.path.join(C.ISBI, "runs", "train", n, "selected.json")))["selected_path"]
       for n in ("saint_prob", "i3net_prob_x4", "i3net_prob_x8")}
tf = []
for f in glob.glob(os.path.join(C.ISBI, "runs", "eval", "FIT", "rows", "*.csv")):
    d = pd.read_csv(f); tf.append(float(d.case_wall_sec.iloc[0]))
t_case = float(np.median(tf))
n_test = int(min(80, max(30, np.floor(12 * 3600 / (14 * t_case)))))
mc = pd.read_csv(os.path.join(C.ISBI, "results", "mc_convergence.csv"))
m20 = mc[mc["T"] == 20]
ok = bool((m20.groupby("method").rel_err_total_var.median() < 0.05).all() and
          ((m20.cov90 - m20.cov90_T100).abs().groupby(m20.method).mean() < 0.01).all())
out = dict(selected_checkpoints=sel, T=20 if ok else 50, mc_rule_passed=ok, n_test=n_test,
           median_fit_case_wall_sec=t_case, frozen_at=pd.Timestamp.now().isoformat(),
           note="checkpoints by prespecified SEL rule; T and n_test by PROTOCOL.md run-order rules")
json.dump(out, open(os.path.join(C.ISBI, "configs", "protocol_frozen.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
