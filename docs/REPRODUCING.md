# Reproduction guide

## Scope and frozen design

This release copies the code used in the final ISBI manuscript as of 29 October 2026. Original workspace files were not moved or edited. Directory names are retained because the scripts share loaders and artifact conventions. Portability edits replace workstation-root literals with `project_paths.root_path`; JSON/text/YAML paths are materialized from `path_templates/`. Original source SHA-256 values are recorded in `source_manifest.json`.

The initial `for_ISBI/PROTOCOL.md` and frozen protocol JSON predate the final matched ArSSR, IXI, repeated-seed, and learned-geometry experiments. For the final paper:

| Setting | BraTS | IXI-tiny |
|---|---|---|
| Modality | 2023 GLI FLAIR (`t2f`) | T1 |
| Scored brain region | Non-zero voxels | Provided brain labels |
| Cropped depth / slice spacing | 153 / 1 mm | 81 / 2.18 mm |
| SEL / FIT / CAL / TEST patients | 12 / 30 / 30 / 51 | 10 / 30 / 30 / 60 |
| Undersampling | ×4 and ×8 | ×4 and ×8 |
| Patterns | Uniform; three jittered and three clustered draws | Same |
| Fine-tuning seeds / MC passes | 42, 43, 44 / 20 | Same |

SAINT uses the regular-grid interpolation adapter; ArSSR and linear interpolation use acquired slice positions. All methods copy acquired slices back. Evaluation uses missing brain voxels, with a deterministic subset of up to 40,000 voxels per case shared by methods. FIT and CAL use uniform masks only. The final post-hoc comparison uses main effects without/with geometry (`ph_nogeo`, `ph_geo`); the older `ph` interaction model is not the final headline comparison. Learned geometry heads freeze the reconstruction mean.

## Data and artifacts

Obtain BraTS 2023 GLI and IXI-tiny under their respective access terms. The release does not download or redistribute them. Preserve the frozen split files and the preprocessing recipe; replacing them changes the experiment.

From the clone root, place or symlink the data at:

```text
brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/<scan>/<scan>-t2f.nii.gz
brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/<scan>/<scan>-t2f.nii.gz
IXI_tiny/image/<subject>_image.nii.gz
IXI_tiny/label/<subject>_label.nii.gz
```

After `source env.sh`, run `python tools/configure_paths.py`, then `python tools/fetch_upstream.py`. The latter fetches the exact model revisions and applies recorded SAINT/DCSRN adaptations. It refuses to overwrite existing checkouts.

Full reproduction depends on artifacts from earlier model training, which are **not distributed in this release**:

- BraTS SAINT source weights selected by `IR3C_more_comparisons/metrics/selected_SAINT_{msr,rfn}.json`, with checkpoint paths relative to `IR3C_more_comparisons/`.
- BraTS SAINT training marginals: `IR3C_more_comparisons/predictions/saint_msr_cache_index.csv` and its referenced arrays, containing `sag`, `cor`, `gt`, and `ret`.
- IXI SAINT initialization: `IR3C_experiments_V2/stage2_finetune/IXI/checkpoints/SAINT_rfn/SAINT_rfn_step000654.pt`.
- Matched BraTS ArSSR random initialization: `runs/brats_ir3c_20260902/checkpoints/arssr_slowlr/arssr_slowlr_step000000.pt`.
- I3Net source and probabilistic ×4/×8 weights for the original `run_eval.py`, which computes these historical controls alongside SAINT and linear. The final paper does not report I3Net, but the archived joint evaluator still loads it.
- Frozen `selected.json` metadata identifies additional trained outputs needed for inference. These weights must be restored or produced by the training commands below. Metadata alone is not a checkpoint.

`python tools/check_inputs.py` reports missing core inputs and frozen absolute checkpoint references. Relative checkpoint references are specified by the source-selection JSON files above; the checker is an inventory aid, not an exhaustive validation of all experiment branches. Full training and inference were not rerun for this release, and there is currently no public download URL for the source weights/cache bundle. The saved-metric workflow in the README is executable without it.

## Environment

Use Python 3.10. `requirements-analysis.txt` is sufficient for saved-table aggregation and significance. `requirements.txt` records the installed scientific/GPU packages from the experiment environment; PyTorch was `2.12.1+cu130`. A CUDA-capable NVIDIA GPU is needed for the full pipeline (several IXI entry points explicitly select CUDA). Memory demand depends on the architecture and volume size. No fresh-environment install or hardware-independent runtime guarantee has been established.

```bash
source .venv/bin/activate
source env.sh
python tools/configure_paths.py
python for_ISBI/tests/test_uq.py
cd "$ISBI"
```

The remaining commands assume the working directory is `$ISBI`. Do not rerun `configure_paths.py` after training unless you intend to restore the frozen selection metadata; it overwrites generated configuration/selection files from the templates.

## Seed 42: reconstruction and learned uncertainty

Supply the source weights/caches first. The existing frozen selection files describe the original selections; training writes new selections in this clone.

```bash
# BraTS SAINT-P: 2,000 scale-head warm-up steps, then joint fine-tuning to 5,000.
python scripts/train_prob.py --arch saint --steps 5000 --head_warm 2000

# Required only when rebuilding the original joint evaluator's I3Net controls:
python scripts/train_prob.py --arch i3net --R 4 --steps 4000 --head_warm 1500 --bs 6
python scripts/train_prob.py --arch i3net --R 8 --steps 4000 --head_warm 1500 --bs 6

# BraTS uniformly trained ArSSR deterministic backbone, then probabilistic fine-tuning.
python exp_uniform_arssr/scripts/train_uarssr_det.py
python exp_uniform_arssr/scripts/train_uarssr_prob.py

# Build the base case files first; later seeds and native evaluations reuse their masks/voxel indices.
for role in FIT CAL TEST; do
  python scripts/run_eval.py --role "$role"
  ISBI_TRAIN_DIR="$ISBI/exp_uniform_arssr/runs/train" \
  ISBI_NATIVE_OUT="$ISBI/exp_uniform_arssr/runs/eval_native" \
    python scripts/eval_native.py --arch arssr --role "$role"
done

# IXI uses its own frozen split and brain masks.
python exp_ixi/scripts/train_saintp_ixi.py
python exp_ixi/scripts/arssr_ixi.py --stage det
python exp_ixi/scripts/arssr_ixi.py --stage prob
for role in FIT CAL TEST; do
  python exp_ixi/scripts/eval_ixi.py --role "$role"
  python exp_ixi/scripts/arssr_ixi.py --stage eval --role "$role"
done
```

The frozen protocol stores the original selected paths. If retraining selects different steps, update `configs/protocol_frozen.json` to those new selected paths before evaluation; do not silently mix old selection metadata with new weights. Exact numerical replication also requires the original source weights, training inputs, and software/hardware behavior.

## Seeds 43 and 44

The training-data subsets remain fixed; the optimization seed changes. These evaluations depend on seed-42 base case files.

```bash
for seed in 43 44; do
  python exp_seeds/scripts/train_saintp_brats_seed.py --arch saint --steps 5000 --head_warm 2000 --seed "$seed"
  python exp_seeds/scripts/train_arssrp_brats_seed.py --seed "$seed"
  python exp_seeds/scripts/train_saintp_ixi_seed.py --seed "$seed"
  python exp_seeds/scripts/arssr_ixi_seed.py --stage prob --seed "$seed"
  for role in FIT CAL TEST; do
    python exp_seeds/scripts/eval_saintp_brats_seed.py --seed "$seed" --role "$role"
    ISBI_TRAIN_DIR="$ISBI/exp_seeds/runs/brats_s${seed}/train" \
    ISBI_NATIVE_OUT="$ISBI/exp_seeds/runs/brats_s${seed}/eval_native" \
      python scripts/eval_native.py --arch arssr --role "$role"
    python exp_seeds/scripts/eval_saintp_ixi_seed.py --seed "$seed" --role "$role"
    python exp_seeds/scripts/arssr_ixi_seed.py --stage eval --seed "$seed" --role "$role"
  done
done
```

## Geometry-aware learned heads and final metrics

```bash
for dataset in brats ixi; do
  for arch in saint arssr; do
    for seed in 42 43 44; do
      python exp_pg/scripts/train_pg.py --dataset "$dataset" --arch "$arch" --seed "$seed"
    done
    for role in CAL TEST; do
      python exp_pg/scripts/eval_pg.py --dataset "$dataset" --arch "$arch" --role "$role"
    done
    for seed in 42 43 44; do
      python exp_pg/scripts/metrics_pg.py --dataset "$dataset" --arch "$arch" --seed "$seed"
    done
  done
  for seed in 42 43 44; do
    for method in SAINT_P ArSSR; do
      python exp_seeds/scripts/metrics_seed.py --dataset "$dataset" --seed "$seed" --method "$method"
    done
  done
  python exp_seeds/scripts/metrics_seed.py --dataset "$dataset" --seed 42 --method linear
done
python exp_seeds/scripts/aggregate_seeds.py
python exp_pg/scripts/significance_v2.py
```

`metrics_seed.py` fits post-hoc models on FIT, calibrates intervals on CAL, and evaluates TEST; `metrics_pg.py` calibrates/evaluates the geometry-aware learned head. Do not fit or calibrate on irregular TEST data. Linear interpolation is reported once, without an artificial seed SD.

## FIT/CAL reassignment sensitivity

These commands require the seed-42 voxel-level case arrays (not included); 100 reassignments refit the post-hoc model and calibration while preserving TEST.

```bash
python exp_uniform_arssr/scripts/analysis_cpu.py \
  --methods SAINT_P,ArSSR_uniform,linear --resplits 100 --rs_kind ph_geo --tag paper_brats
ISBI_EV="$ISBI/exp_ixi/runs/eval" \
ISBI_NAT_UNIFORM="$ISBI/exp_ixi/runs/eval_native/arssr" \
ISBI_OUT_BASE="$ISBI/exp_ixi/results" ISBI_BINS=ixi \
  python exp_uniform_arssr/scripts/analysis_cpu.py \
  --methods SAINT_P,ArSSR_uniform,linear --resplits 100 --rs_kind ph_geo --tag paper_ixi
```

## Figures and historical scripts

`for_ISBI/manuscript_4page/scripts/` retains the table/plot code used during manuscript preparation. Quantitative plots read generated metrics; qualitative/reliability plots additionally require saved voxel arrays and/or checkpoints. These scripts retain historical output-directory names and are not a manuscript build system. The final paper's LaTeX table is included as `docs/paper_table.tex` for reference; subsequent editorial formatting is not regenerated by the older table script.

Earlier geometry/exposure/native-control scripts are retained for traceability. `tests/test_preflight.py` is the original data/checkpoint-dependent integration test, not a data-free smoke test. Workstation process-control, watchdog, and launch scripts were intentionally excluded.
