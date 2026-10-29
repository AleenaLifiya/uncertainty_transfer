# Release validation — 29 October 2026

Validated using the existing Python 3.10 experiment environment:

- Parsed all 133 packaged Python files successfully.
- Ran `for_ISBI/tests/test_uq.py`: passed. Synthetic Laplace GLM log-scale correlation 0.9999; recovered 90% calibration multiplier 2.301; mixture coverage 0.90076.
- Regenerated the seed aggregation and final significance analysis. All six previously saved derived CSV files matched numerically with `rtol=atol=1e-12`.
- Copied the release to a temporary unrelated directory, regenerated its paths, and imported reconstruction/IXI modules. No modules were imported from the original experiment workspace.
- Checked retained-slice counts, ordering, endpoint anchors, and finite gap geometry on synthetic inputs for both depths, both undersampling factors, all three patterns, and three realizations.
- Verified that both recorded upstream adaptation patches apply to their pinned base revisions, using local Git archives.
- Verified SHA-256 hashes of all 301 extracted source/input files against the untouched original workspace.
- Checked the README paper-summary length: 228 words.

Not performed: a new full-environment installation, upstream network-clone execution, full training, image-level inference, or regeneration of medical-image figures. These depend on external code/data, source weights, caches, and GPU resources. This release does not claim fresh-clone end-to-end reproduction without those inputs.

The CPU summary/statistics workflow is the validated entry point. `tools/check_inputs.py` intentionally exits nonzero when full-pipeline artifacts are absent.
