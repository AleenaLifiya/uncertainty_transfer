# for_ISBI — frozen protocol (written before any TEST evaluation)

**Question.** Does uncertainty calibrated under uniform through-plane slice sampling stay reliable under
irregular sampling, and how does that depend on the reconstruction architecture and on the uncertainty
approach (learned probabilistic vs post-hoc)?

## Data (audited, `provenance/`, `configs/roles.json`)
- BraTS 2023 GLI, FLAIR (`t2f`), 240×240×155, 1 mm isotropic (header zooms), LPS; protocol crop to D = 153
  (central), foreground-percentile [1, 99.5] → [0, 1] (`comparisons/protocol.py`, unchanged). Through-plane axis = 2.
- Training split = GLI TrainingData (1251 scans, 1133 patients): used by the source checkpoints and by the
  probabilistic fine-tuning only.
- Held-out pool = GLI ValidationData (219 scans, 199 patients; patient = `BraTS-GLI-PPPPP`, longitudinal
  timepoints grouped). Train and pool patient IDs are disjoint. One scan per patient (lowest timepoint).
  All 199 patients were examined in earlier project experiments (val60 / IR3C test88) — none is "untouched".
- Roles (seed 20261016, patient-level, disjoint): SEL 12 (incl. the 4 patients of the source models' 6-scan
  selection set), FIT 30, CAL 30, TEST = first `n_test` of the remaining seeded order.

## Methods
| name | what | irregular masks |
|---|---|---|
| linear | linear interpolation in physical z | native |
| bicubic | project `bicubic`: Keys a = −0.5 cubic convolution over the acquired stack in LR-sample coordinates p(z) (piecewise-linear in physical z); chosen by the user | native |
| SAINT_det | SAINT (CVPR 2020) BraTS checkpoint, step 11,259 | + interpolation adapter |
| I3Net_det | I3Net (TMI 2024) BraTS checkpoints x4 step 6,255 / x8 step 3,753 | + interpolation adapter |
| SAINT_P | probabilistic SAINT: frozen MSR, RFN + 2 Dropout2d (p = 0.1) + log-scale head | + adapter |
| I3Net_P | probabilistic I3Net: tail Dropout2d ×2 (p = 0.1) + log-scale head, one model per R | + adapter |
ArSSR (slow-LR) was audited and NOT used: 1.1 dB below linear on validation and still improving at its budget end.
DCSRN: unstable validation curve. IR3C: excluded by instruction.

**Adapter** (SAINT, I3Net): slices on the uniform grid G = {0, R, …, 152} are linear interpolations (physical z)
of the originally acquired slices; identity for uniform masks. All originally missing slices are scored,
including adapter-filled G positions. **Copy-back** (all methods): acquired slices replaced by data.

## Sampling (`src/isbi/common.py`)
×4 (K = 39) primary, ×8 (K = 20) confirmation; anchors 0 and 152 always acquired; identical K across patterns.
- uniform: stride R. jittered: interior grid points ± U{−J..J}, J = R/2 − 1. clustered: interior slices in
  clusters of 2–3 adjacent slices, clusters spread evenly (±1). Max gap: ×4 4 / 6 / 9–12 mm; ×8 8 / 10–14 / 16–22 mm.
- Masks depend only on (patient id, pattern, R, realisation r ∈ {0,1,2}); uniform has one realisation.

## Probabilistic fine-tuning (`scripts/train_prob.py`)
Uniform masks, training split only; region-balanced Laplace NLL (fg / bg ½ each) on missing (non-copied) slices.
Phase 1: scale head only (mean frozen, dropout off); phase 2: joint, dropout on, lr 2e-5 (source weights),
1e-3 (head), cosine. SAINT 2000 + 3000 steps (MSR cache of the source run, 40 train subjects, R 2/4/8);
I3Net 1500 + 2500 steps per R (160 train volumes). Selection (prespecified): phase-2 checkpoints with SEL PSNR
≥ source − 0.2 dB and background haze ≤ source + 0.002; lowest SEL foreground NLL.
History of the scale-head design (all decided on SEL, before any FIT/CAL/TEST data): see logs/Executing_for_ISBI.md.

## Uncertainty
- Learned: T = 20 MC passes (only trained dropout active; seed = hash(patient, condition)); mean of μ_t;
  epistemic Var_t μ_t; aleatoric mean 2b_t²; intervals = central quantiles of the Laplace mixture
  (raw, and "cal" with b_t → λ b_t, λ from CAL). MC convergence vs T = 100 on 6 SEL patients.
- Post-hoc: one Laplace-scale GLM per mean method, common design picked by 5-fold patient CV on FIT (uniform);
  intervals x̂ ± κ_α b̂, κ_α from CAL (uniform). Applied around the SAME mean as the learned arm (shared mean),
  and around the deterministic sources and classical interpolants.
- Constant width x̂ ± w_α (per method, R, α) from CAL. Geometry scores: nearest-slice distance, gap length; random.
- Calibration target: mean over CAL patients of per-patient coverage = α. Empirical calibration only; no
  distribution-free guarantee, and none is claimed to survive the sampling change.

## Evaluation
- Region: originally missing slices; UQ metrics on GT-foreground voxels of those slices, fixed subset of
  40,000 voxels per (patient, condition), identical for every method. Recon metrics (PSNR-brain, SSIM, MAE,
  NMSE) on the whole missing region with the shared evaluator.
- Coverage/width at 50/80/90/95 %; patient deviation from nominal; within-patient Spearman(|e|, score);
  within-geometry-stratum Spearman (strata = exact (gap, nearest-distance) cells); AUSE (removal grid 0–0.98,
  step 0.02) and AUSE/MAE; AUPRC for |e| > τ (τ = CAL 90th percentile, per method and R) with prevalence.
- Unit = patient; jittered/clustered realisations averaged within patient. Paired bootstrap (10,000) CIs.

**Primary endpoint.** Per architecture: Δcov90 = cov90(clustered ×4) − cov90(uniform ×4), learned-cal vs
post-hoc (shared mean); paired patient-level difference with bootstrap 95 % CI. Interpreted with width.
**Key supporting.** Within-stratum Spearman of learned vs post-hoc vs geometry scores (×4).
**Confirmation.** The same at ×8. No other formal comparisons.

## Qualitative selection rule (fixed now)
Among TEST patients, rank by SAINT_P PSNR at ×4 uniform: the median patient and the 10th-percentile
(difficult) patient; show uniform vs clustered r0 for each. Same windows across methods; error and
uncertainty maps share a common scale per row except the epistemic map (own, labelled scale).

## Prespecified run-order rules (added 2026-10-16 23:20, before any FIT/CAL/TEST evaluation)
- Order: training → FIT → CAL → MCCONV → post-hoc fit + calibration + MC check (frozen) → freeze → TEST.
  Calibration constants are therefore fixed before any TEST case is reconstructed.
- `n_test` = clamp(floor(12 h / (14 × median FIT case time)), 30, 80) — a fixed 12 GPU-hour TEST budget.
- T: keep T = 20 if, on the MCCONV subset (×4), the median relative error of the total variance vs T = 100 is
  < 5 % AND |cov90(T=20) − cov90(T=100)| < 0.01 for both models; otherwise T = 50 (FIT/CAL then re-run).
