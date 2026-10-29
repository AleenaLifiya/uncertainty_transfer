"""Pre-freeze config for FIT / CAL / MCCONV: selected checkpoints (SEL rule) and T = 20."""
import json, os
from isbi import common as C
sel = {n: json.load(open(os.path.join(C.ISBI, "runs", "train", n, "selected.json")))["selected_path"]
       for n in ("saint_prob", "i3net_prob_x4", "i3net_prob_x8")}
json.dump(dict(selected_checkpoints=sel, T=20), open(os.path.join(C.ISBI, "configs", "protocol_prefreeze.json"), "w"), indent=1)
print(sel)
