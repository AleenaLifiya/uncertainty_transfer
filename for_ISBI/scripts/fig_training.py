"""Learning curves + SEL validation diagnostics for the three probabilistic fine-tunes."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from isbi import common as C, figstyle as FS
FS.use()
runs = [("saint_prob", "SAINT-P"), ("i3net_prob_x4", "I3Net-P ×4"), ("i3net_prob_x8", "I3Net-P ×8")]
runs = [r for r in runs if os.path.exists(os.path.join(C.ISBI, "runs", "train", r[0], "val_history.csv"))]
fig, axs = plt.subplots(len(runs), 4, figsize=(FS.DOUBLE, 1.55 * len(runs) + 0.3), squeeze=False)
rows = []
for i, (n, lab) in enumerate(runs):
    d = os.path.join(C.ISBI, "runs", "train", n)
    tl = pd.read_csv(os.path.join(d, "train_loss.csv"), header=None, names=["step", "loss", "lr", "sec"])
    vh = pd.read_csv(os.path.join(d, "val_history.csv"))
    sel = json.load(open(os.path.join(d, "selected.json"))) if os.path.exists(os.path.join(d, "selected.json")) else {}
    hw = sel.get("head_warm", 2000 if n.startswith("saint") else 1500)
    for k, (col, ylab) in enumerate([("loss", "train NLL (balanced)"), ("psnr", "SEL PSNR (dB)"),
                                    ("nll", "SEL fg NLL"), ("spearman", "SEL Spearman(|e|, b)")]):
        ax = axs[i, k]
        if col == "loss":
            ax.plot(tl.step, tl.loss, color=FS.INK2, lw=0.8)
        else:
            for j, (R, g) in enumerate(vh.groupby("R")):
                g = g.groupby("step")[col].mean()
                ax.plot(g.index, g.values, marker=FS.MARKERS[j], color=FS.SERIES[j], ms=3, lw=1.1, label=f"×{R}")
        ax.axvline(hw, color=FS.MUTED, lw=0.7, ls=":")
        if "selected_step" in sel and col != "loss":
            ax.axvline(sel["selected_step"], color=FS.VERM, lw=0.8, ls="--")
        if i == 0:
            ax.set_title(ylab, loc="left", fontsize=7.5)
        if k == 0:
            ax.set_ylabel(lab)
        if i == len(runs) - 1:
            ax.set_xlabel("step")
    rows.append(dict(model=lab, head_warm=hw, total_steps=int(tl.step.max()), selected_step=sel.get("selected_step"),
                     sel_psnr=sel.get("val", {}).get("psnr"), src_psnr=sel.get("source_val", {}).get("psnr"),
                     sel_nll=sel.get("val", {}).get("nll"), src_nll=sel.get("source_val", {}).get("nll"),
                     train_hours=float(tl.sec.max()) / 3600))
axs[0, 1].legend(fontsize=6)
fig.text(0.01, 0.005, "dotted: end of phase 1 (scale head only); dashed red: selected checkpoint (prespecified SEL rule)", fontsize=6, color=FS.MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(C.ISBI, "figures", f"fig_training_curves.{ext}"))
pd.DataFrame(rows).to_csv(os.path.join(C.ISBI, "results", "training_summary.csv"), index=False)
print(pd.DataFrame(rows).to_string())
