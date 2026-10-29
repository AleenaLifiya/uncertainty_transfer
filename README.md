# Uncertainty Transfer in Through-Plane MRI Reconstruction

Code and numerical results for **“Different Gaps, Different Maps? Transfer of Calibrated Uncertainty from Regular to Irregular Slice Sampling in Through-Plane MRI Reconstruction”**, by Aleena Lifiya Jeffy, Mohammed Q. Alkhatib, Marouane Tliba, Aladine Chitouani, and Nour Aburaed (ISBI manuscript).

## Paper summary

Voxel-wise uncertainty in MRI reconstruction is commonly calibrated using regularly spaced slices. In practice, rejecting corrupted slices can leave irregular gaps, raising a question: do intervals calibrated on regular sampling remain reliable when the number of acquired slices stays fixed but their positions change? This study isolates that sampling-geometry shift using BraTS 2023 glioma FLAIR and IXI-tiny T1 volumes, with fourfold and eightfold through-plane undersampling and uniform, jittered, and clustered acquisition patterns.

We compare SAINT, which requires interpolation onto a regular input grid, ArSSR, which queries positions between acquired slices, and linear interpolation. Uncertainty is estimated using learned Laplace distributions with Monte Carlo dropout or a post-hoc error-scale model. Both approaches are tested with and without explicit gap-geometry features. Model training, post-hoc fitting, and calibration use regular sampling only; all calibration parameters are frozen before irregular-sampling evaluation. Patients are separated into model-selection, fitting, calibration, and test groups.

Under clustered sampling, nominal 90% intervals without geometry cover as little as 59% of evaluated voxels. Adding geometry to the post-hoc model improves coverage by 6–23 percentage points, reaching 78–92%, and also improves learned uncertainty. Native reconstructions transfer more reliably than the regridded SAINT pipeline, which retains under-coverage. Three fine-tuning seeds, patient-level statistical comparisons, and FIT/CAL reassignment analyses assess robustness. These experiments demonstrate the importance of acquisition geometry for uncertainty transfer under synthetic sampling shifts; they do not establish clinical reliability.

## Clone and install

```bash
git clone https://github.com/AleenaLifiya/uncertainty_transfer.git
cd uncertainty_transfer
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-analysis.txt
source env.sh
python tools/configure_paths.py
```

Use Bash on Linux. `env.sh` derives paths from your clone. Run it in each new terminal. Re-run `configure_paths.py` after relocating the clone; it materializes local configuration paths from tracked templates.

## Quick start: reproduce the numerical analysis

The repository includes per-case metrics and three-seed summaries. **No MRI data, checkpoints, or GPU are needed for these commands:**

```bash
python for_ISBI/exp_seeds/scripts/aggregate_seeds.py
python for_ISBI/exp_pg/scripts/significance_v2.py
```

Outputs:

- [`seeds_summary.csv`](for_ISBI/exp_seeds/results/seeds_summary.csv): coverage, interval score, AUSE/MAE, UCE, and ENCE, aggregated across seeds.
- [`seeds_dcov.csv`](for_ISBI/exp_seeds/results/seeds_dcov.csv): clustered-minus-uniform coverage change.
- [`REPORT.md`](for_ISBI/results/significance_v2/REPORT.md): patient-level repeated-measures ANOVA and paired comparisons for the final paper design.

The scripts overwrite their derived summaries in this clone. Original per-case metric CSVs remain the analysis inputs.

## Run reconstruction and uncertainty experiments

The full GPU pipeline additionally requires the licensed datasets, upstream model source, source checkpoints, and SAINT marginal caches. **Model weights and image/cache arrays are not included, and no public checkpoint download is currently supplied.** A fresh clone can reproduce the saved-metric analyses above, but cannot reproduce image-level inference or training until these artifacts are supplied.

```bash
# Install the recorded full environment; the extra index provides the recorded CUDA build.
pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
python tools/fetch_upstream.py
python tools/check_inputs.py
python for_ISBI/tests/test_uq.py
```

See **[Reproduction guide](docs/REPRODUCING.md)** for data layout, artifact requirements, training/evaluation commands, all three seeds, geometry heads, and analysis order. See [release validation](docs/VALIDATION.md) for what was actually tested.

## Repository map

| Location | Purpose |
|---|---|
| `for_ISBI/src/isbi/` | Sampling masks, geometry, probabilistic models, reconstruction, calibration, and metrics |
| `for_ISBI/scripts/` | Original BraTS training/evaluation and exploratory analyses |
| `for_ISBI/exp_uniform_arssr/scripts/` | Uniform-trained ArSSR and post-hoc FIT/CAL reassignment analysis |
| `for_ISBI/exp_ixi/scripts/` | IXI-tiny preprocessing, training, and evaluation |
| `for_ISBI/exp_seeds/scripts/` | Additional seeds, per-seed metrics, and aggregation |
| `for_ISBI/exp_pg/scripts/` | Learned geometry-aware scale heads and final significance analysis |
| `for_ISBI/exp_seeds/results/` | Saved numerical inputs and final three-seed summaries |
| `path_templates/` | Frozen split, selection, and configuration metadata with portable root placeholders |
| `comparisons/`, `image_residual/`, `IR3C_*`, `X-Diffusion/` | Shared code imported by the original experiments |
| `tools/` | Path setup, upstream source retrieval, and input checks |

The final paper uses **SAINT-P, ArSSR-P, and linear interpolation**, seeds **42, 43, 44**, and **20 MC passes**. Earlier I3Net/DCSRN and mixed-mask experiments remain in shared code for provenance; they are not additional methods in the final headline table. The original `for_ISBI/PROTOCOL.md` records the initial study; [the reproduction guide](docs/REPRODUCING.md) explains the later paper configuration.

## Attribution and citation

See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream sources and license boundaries. This release does not assign a new blanket license to the research code. Existing third-party licenses remain applicable.

```bibtex
@misc{jeffy_uncertainty_transfer_2026,
  title = {Different Gaps, Different Maps? Transfer of Calibrated Uncertainty
           from Regular to Irregular Slice Sampling in Through-Plane MRI Reconstruction},
  author = {Jeffy, Aleena Lifiya and Alkhatib, Mohammed Q. and Tliba, Marouane
            and Chitouani, Aladine and Aburaed, Nour},
  year = {2026},
  note = {ISBI manuscript; code repository},
  howpublished = {\url{https://github.com/AleenaLifiya/uncertainty_transfer}}
}
```
