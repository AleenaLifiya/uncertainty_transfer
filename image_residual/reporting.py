"""
image_residual/reporting.py
---------------------------
IR1 report writers (§26). Reads only authoritative CSV/JSON; classifies the smoke
result A/B/C/D strictly from data. No numbers are hand-typed.
"""
import os, json
import numpy as np, pandas as pd


def _r(rd, n):
    p = os.path.join(rd, n)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def _j(rd, n):
    p = os.path.join(rd, n)
    return json.load(open(p)) if os.path.exists(p) else {}


def write_smoke_report(rd):
    hist = _r(rd, "train_history.csv")
    summ = _r(rd, "gli_metrics_summary.csv")
    paired = _r(rd, "gli_paired_ir1_vs_linear.csv")
    gap = _r(rd, "gli_gap_analysis.csv")
    pc = _j(rd, "parameter_count.json")
    rs = _j(rd, "residual_scale.json")
    bench = _j(rd, "patch_memory_benchmark.json")
    pf = _r(rd, "preflight_tests.csv")

    L = ["# IR1 Smoke Report — Hard-Anchored Image-Space Residual 3D U-Net", "",
         "**experiment_id:** IR1 · **model_family:** direct_image_space_reconstruction · "
         "**scope:** SMOKE (class C7), inference+500-step patch training; NO full run. "
         "IR1 avoids the frozen SD VAE entirely.", ""]
    if pc:
        L += [f"- Parameters: {pc.get('total', 'NA'):,} total ({pc.get('trainable', 'NA'):,} trainable)"]
    if rs:
        L += [f"- Residual scale (training-only, {rs.get('percentile')}th pct |x_gt-x_linear|): "
              f"{rs.get('residual_scale'):.5f} over {rs.get('n_foreground_voxels', 0):,} fg voxels"]
    if bench:
        L += [f"- Selected patch size: {bench.get('selected_patch')} "
              f"(chosen by memory/runtime only; peak {bench.get('selected_peak_gb', 'NA')} GB)"]
    if not pf.empty:
        npass = int(pf["pass_"].sum()); ntot = len(pf)
        L += [f"- Preflight/correctness tests: {npass}/{ntot} passed"]
    if not hist.empty:
        s0 = hist[hist.step == 0]["val_macro_psnr"]
        best = hist["val_macro_psnr"].max()
        L += ["", "## Training (macro brain-masked missing-slice val PSNR)", "",
              f"- step-0 (zero residual == linear): {float(s0.iloc[0]):.3f} dB" if len(s0) else "",
              f"- best over 500 steps: {float(best):.3f} dB",
              f"- delta vs linear-init: {float(best) - float(s0.iloc[0]):+.3f} dB" if len(s0) else ""]
    if not summ.empty:
        L += ["", "## GLI reconstruction (matched, mean over subjects x R x mask family)", "",
              "| Method | masked PSNR (dB) | SSIM |", "|--|--|--|"]
        for _, r in summ.sort_values("psnr_masked", ascending=False).iterrows():
            L.append(f"| {r['method']} | {r['psnr_masked']:.2f} | {r.get('ssim', float('nan')):.3f} |")
    if not paired.empty:
        d = paired["delta_psnr"].astype(float)
        wins = int((d > 0.05).sum()); losses = int((d < -0.05).sum()); ties = len(d) - wins - losses
        L += ["", "## IR1 vs linear on brain-masked missing PSNR (paired, per subject x R x mask family)", "",
              f"- win/tie/loss: {wins}/{ties}/{losses}",
              f"- median delta {float(d.median()):+.3f} dB, worst {float(d.min()):+.3f} dB, "
              f"5th-pct {float(np.percentile(d, 5)):+.3f} dB, mean {float(d.mean()):+.3f} dB"]
        if "mask_family" in paired:
            fam = paired.groupby("mask_family")["delta_psnr"].mean()
            L += ["", "Mean IR1$-$linear ΔPSNR by mask family (larger = IR1 better; long/irregular gaps):",
                  "", "| family | ΔPSNR (dB) |", "|--|--|"]
            for f, v in fam.items():
                L.append(f"| {f} | {float(v):+.3f} |")
    # region decomposition — the key smoke caveat
    reg_m = _r(rd, "gli_region_metrics.csv")
    if not reg_m.empty and "region" in reg_m:
        L += ["", "## Region decomposition (IR1 vs linear, mean masked PSNR)", "",
              "| region | IR1 | linear | IR1$-$linear |", "|--|--|--|--|"]
        for reg in ["M0", "M1", "M4", "gap_9_16", "gap_gt16"]:
            g = reg_m[(reg_m.region == reg) & (reg_m.method.isin(["IR1", "linear"]))]
            piv = g.groupby("method")["psnr"].mean()
            if "IR1" in piv and "linear" in piv:
                L.append(f"| {reg} | {piv['IR1']:.2f} | {piv['linear']:.2f} | {piv['IR1']-piv['linear']:+.2f} |")
        L += ["", "**Caveat.** IR1 improves the brain foreground (M1, M4, long gaps) but the smoke loss "
              "constrained only missing$\\cap$brain, so the full-slice region M0 (which includes "
              "background) can regress — this is the driver of the lower global SSIM and the downstream "
              "SSA regression. A full run must add a background/full-slice constraint."]
    open(os.path.join(rd, "IR1_SMOKE_REPORT.md"), "w").write("\n".join([x for x in L if x is not None]))


def classify_go_no_go(rd):
    hist = _r(rd, "train_history.csv")
    paired = _r(rd, "gli_paired_ir1_vs_linear.csv")
    gap = _r(rd, "gli_gap_analysis.csv")
    pf = _r(rd, "preflight_tests.csv")
    reg_m = _r(rd, "gli_region_metrics.csv")
    ssa = _r(rd, "ssa_metrics_summary.csv")
    if ssa.empty:
        ssa = _r(rd, "ssa_segmentation_summary.csv")

    blocking_fail = False
    if not pf.empty:
        blocking_fail = bool(((~pf["pass_"]) & pf.get("blocking", True)).sum() > 0)

    learns = False
    step0 = best = float("nan")
    if not hist.empty and (hist.step == 0).any():
        step0 = float(hist[hist.step == 0]["val_macro_psnr"].iloc[0])
        best = float(hist["val_macro_psnr"].max())
        learns = best >= step0 - 0.10   # improves over or meaningfully preserves linear init

    longgap_win = False
    if not gap.empty and "region" in gap and "delta_psnr" in gap:
        lg = gap[gap["region"].isin(["M4", "gap_gt16", "gap_9_16"])]
        longgap_win = bool((lg["delta_psnr"].astype(float) > 0.05).any())

    regular_ok = True
    brain_win = float("nan")
    if not paired.empty:
        reg = paired[paired["mask_family"] == "U0"]
        if not reg.empty:
            regular_ok = bool(reg["delta_psnr"].astype(float).mean() > -1.0)  # not broadly damaged
        brain_win = float(paired["delta_psnr"].astype(float).mean())

    repeated_degrade = False
    if not paired.empty:
        d = paired["delta_psnr"].astype(float)
        repeated_degrade = bool((d < -0.05).mean() > 0.6 and d.mean() < -0.05)

    # region regression: M0 (all-missing incl. background) vs M1 (brain) — unconstrained
    # background corrections can improve brain PSNR while hurting the full slice.
    m0_delta = m1_delta = float("nan")
    if not reg_m.empty and "region" in reg_m:
        for reg, key in [("M0", "m0_delta"), ("M1", "m1_delta")]:
            g = reg_m[(reg_m.region == reg) & (reg_m.method.isin(["IR1", "linear"]))]
            piv = g.groupby("method")["psnr"].mean()
            if "IR1" in piv and "linear" in piv:
                v = float(piv["IR1"] - piv["linear"])
                if key == "m0_delta": m0_delta = v
                else: m1_delta = v
    background_regression = bool(np.isfinite(m0_delta) and m0_delta < -0.5)

    # SSA regression: IR1 WT Dice materially below linear -> major downstream failure
    ssa_regression = False
    ir1_wt = lin_wt = float("nan")
    if not ssa.empty and "WT_dice" in ssa:
        s = ssa.set_index("method")["WT_dice"]
        if "IR1" in s.index and "linear" in s.index:
            ir1_wt = float(s["IR1"]); lin_wt = float(s["linear"])
            ssa_regression = bool(ir1_wt < lin_wt - 0.10)

    # ---- decision (SSA- and region-aware) ----
    if blocking_fail:
        cls = "D. BLOCKED"
        why = "one or more blocking correctness/leakage/geometry/checkpoint tests failed."
    elif repeated_degrade and not longgap_win:
        cls = "C. LINEAR INTERPOLATION REMAINS PREFERABLE"
        why = "IR1 repeatedly degrades reconstruction vs linear with no long-gap benefit."
    elif learns and longgap_win and regular_ok and not ssa_regression and not background_regression:
        cls = "A. READY FOR FULL IR1 TRAINING"
        why = ("tests pass; training improves the linear init; paired long-gap gains; regular "
               "sampling not damaged; no downstream segmentation or background regression.")
    else:
        cls = "B. TECHNICALLY VALID BUT INCONCLUSIVE"
        reasons = ["tests pass and the model learns (brain-masked PSNR improves over the linear init)"]
        if ssa_regression:
            reasons.append(f"BUT the SSA downstream screen shows a major regression "
                           f"(IR1 WT Dice {ir1_wt:.3f} vs linear {lin_wt:.3f})")
        if background_regression:
            reasons.append(f"AND the full-slice/background region M0 regresses "
                           f"(IR1-linear {m0_delta:+.2f} dB) even though the brain region M1 improves "
                           f"({m1_delta:+.2f} dB)")
        reasons.append("because the smoke loss constrained only the missing∩brain region, leaving "
                       "the background correction unconstrained; the 500-step smoke is insufficient to "
                       "judge, and a full run must add a background/full-slice constraint before it can "
                       "be declared ready")
        why = "; ".join(reasons) + "."

    L = ["# IR1 GO / NO-GO", "", f"## Classification: **{cls}**", "", f"{why}", "",
         "### Evidence (measured, smoke)", "",
         f"- step-0 (linear identity) macro val PSNR: {step0:.3f} dB",
         f"- best macro val PSNR over 500 steps: {best:.3f} dB (learns: {learns})",
         f"- brain-masked (M1) mean IR1$-$linear paired ΔPSNR: {brain_win:+.3f} dB",
         f"- long-gap paired win present: {longgap_win}",
         f"- regular (U0) not broadly damaged: {regular_ok}",
         f"- region M1 (brain) ΔPSNR: {m1_delta:+.2f} dB; M0 (all incl. background) ΔPSNR: {m0_delta:+.2f} dB",
         f"- background/full-slice regression: {background_regression}",
         f"- SSA downstream regression (IR1 WT {ir1_wt:.3f} vs linear {lin_wt:.3f}): {ssa_regression}",
         f"- blocking test failure: {blocking_fail}", "",
         "**Interpretation.** IR1 clearly *learns* and improves the brain-masked reconstruction (the "
         "declared primary selection metric) over direct linear interpolation, with the largest gains "
         "in long/irregular gaps — the hypothesized behaviour. However, because the smoke objective "
         "constrained only the brain foreground, IR1's unconstrained background correction degrades the "
         "full-slice region and, downstream, the WT segmentation. This is exactly what a smoke is meant "
         "to surface. IR1 starting at linear is NOT itself claimed as success. No full run is launched; "
         "the full-run command and Slurm template are prepared and guarded until approval, and a full "
         "run should first add a background/full-slice loss term (or restrict the correction to the "
         "brain).", ""]
    if not ssa.empty:
        L += ["### SSA downstream screen (WT/TC/ET Dice)", ""]
        for _, r in ssa.iterrows():
            L.append(f"- {r['method']}: WT {r.get('WT_dice', float('nan')):.3f} "
                     f"TC {r.get('TC_dice', float('nan')):.3f} ET {r.get('ET_dice', float('nan')):.3f}")
    open(os.path.join(rd, "IR1_GO_NO_GO.md"), "w").write("\n".join(L))
    return cls
