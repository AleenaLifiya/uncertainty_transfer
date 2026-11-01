# Interval score, AUSE/MAE and UCE for all methods, datasets and undersampling factors

Supplement to the ISBI paper *Different Gaps, Different Maps?* (Evaluation paragraph). Values are means over the TEST
patients, then mean ± SD over the three fine-tuning seeds (42, 43, 44); linear interpolation has no seed, so a single
value is given. Coverage and ENCE are in Table 1 of the paper. All intervals are nominal 90 % and were calibrated on
uniform sampling only; jittered and clustered values are averaged over three random masks per patient.

- **IS**: interval score of the 90 % interval (Gneiting & Raftery 2007), intensities scaled to [0, 1]. Lower is better.
- **AUSE/MAE**: area under the sparsification-error curve divided by the mean absolute error. Lower is better.
- **UCE ×10³**: uncertainty calibration error (Laves et al. 2020), 20 equal-count bins, squared-intensity units
  multiplied by 10³. Lower is better.
- Learned intervals use the predictive variance of the MC-dropout Laplace mixture; post-hoc intervals use the variance
  of the Laplace distribution whose central 90 % interval is the post-hoc interval.

Machine-readable version: [`uq_metrics_all.csv`](uq_metrics_all.csv). Regenerate with
`python for_ISBI/exp_seeds/scripts/export_uq_metrics.py` (input: `for_ISBI/exp_seeds/results/seeds_summary.csv`).

## BraTS 2023 glioma FLAIR (51 test patients), ×4

| Reconstruction | Uncertainty | Pattern | IS ↓ | AUSE/MAE ↓ | UCE ×10³ ↓ |
|---|---|---|---|---|---|
| SAINT-P | learned | uniform | 0.233 ± 0.001 | 0.557 ± 0.011 | 1.56 ± 0.02 |
| SAINT-P | learned | jittered | 0.307 ± 0.002 | 0.586 ± 0.026 | 2.53 ± 0.02 |
| SAINT-P | learned | clustered | 0.631 ± 0.004 | 0.639 ± 0.032 | 7.91 ± 0.02 |
| SAINT-P | learned + geometry | uniform | 0.226 ± 0.002 | 0.421 ± 0.035 | 1.55 ± 0.04 |
| SAINT-P | learned + geometry | jittered | 0.292 ± 0.002 | 0.382 ± 0.028 | 2.47 ± 0.05 |
| SAINT-P | learned + geometry | clustered | 0.507 ± 0.009 | 0.340 ± 0.024 | 6.63 ± 0.12 |
| SAINT-P | post hoc | uniform | 0.203 ± 0.001 | 0.325 ± 0.003 | 1.30 ± 0.01 |
| SAINT-P | post hoc | jittered | 0.283 ± 0.000 | 0.334 ± 0.001 | 2.53 ± 0.00 |
| SAINT-P | post hoc | clustered | 0.624 ± 0.003 | 0.349 ± 0.001 | 8.08 ± 0.01 |
| SAINT-P | post hoc + geometry | uniform | 0.202 ± 0.001 | 0.312 ± 0.003 | 1.31 ± 0.01 |
| SAINT-P | post hoc + geometry | jittered | 0.263 ± 0.000 | 0.284 ± 0.001 | 2.28 ± 0.00 |
| SAINT-P | post hoc + geometry | clustered | 0.418 ± 0.000 | 0.268 ± 0.002 | 5.32 ± 0.01 |
| ArSSR-P | learned | uniform | 0.227 ± 0.005 | 0.334 ± 0.004 | 5.36 ± 0.98 |
| ArSSR-P | learned | jittered | 0.237 ± 0.004 | 0.343 ± 0.003 | 4.41 ± 0.85 |
| ArSSR-P | learned | clustered | 0.515 ± 0.009 | 0.418 ± 0.007 | 5.89 ± 0.12 |
| ArSSR-P | learned + geometry | uniform | 0.219 ± 0.003 | 0.333 ± 0.005 | 3.64 ± 0.44 |
| ArSSR-P | learned + geometry | jittered | 0.229 ± 0.003 | 0.332 ± 0.003 | 3.17 ± 0.43 |
| ArSSR-P | learned + geometry | clustered | 0.464 ± 0.013 | 0.361 ± 0.007 | 5.24 ± 0.06 |
| ArSSR-P | post hoc | uniform | 0.240 ± 0.003 | 0.352 ± 0.005 | 1.60 ± 0.05 |
| ArSSR-P | post hoc | jittered | 0.262 ± 0.002 | 0.370 ± 0.003 | 1.91 ± 0.03 |
| ArSSR-P | post hoc | clustered | 0.492 ± 0.005 | 0.429 ± 0.002 | 6.35 ± 0.12 |
| ArSSR-P | post hoc + geometry | uniform | 0.240 ± 0.003 | 0.345 ± 0.004 | 1.60 ± 0.05 |
| ArSSR-P | post hoc + geometry | jittered | 0.256 ± 0.002 | 0.345 ± 0.002 | 1.82 ± 0.03 |
| ArSSR-P | post hoc + geometry | clustered | 0.419 ± 0.006 | 0.373 ± 0.002 | 4.76 ± 0.19 |
| Linear | post hoc | uniform | 0.250 | 0.331 | 2.04 |
| Linear | post hoc | jittered | 0.275 | 0.335 | 2.34 |
| Linear | post hoc | clustered | 0.477 | 0.342 | 6.26 |
| Linear | post hoc + geometry | uniform | 0.248 | 0.317 | 1.98 |
| Linear | post hoc + geometry | jittered | 0.261 | 0.283 | 2.08 |
| Linear | post hoc + geometry | clustered | 0.373 | 0.270 | 3.80 |

## BraTS 2023 glioma FLAIR (51 test patients), ×8

| Reconstruction | Uncertainty | Pattern | IS ↓ | AUSE/MAE ↓ | UCE ×10³ ↓ |
|---|---|---|---|---|---|
| SAINT-P | learned | uniform | 0.384 ± 0.001 | 0.551 ± 0.006 | 3.50 ± 0.01 |
| SAINT-P | learned | jittered | 0.489 ± 0.003 | 0.606 ± 0.021 | 5.46 ± 0.05 |
| SAINT-P | learned | clustered | 0.830 ± 0.006 | 0.711 ± 0.023 | 13.81 ± 0.08 |
| SAINT-P | learned + geometry | uniform | 0.358 ± 0.003 | 0.335 ± 0.022 | 3.58 ± 0.24 |
| SAINT-P | learned + geometry | jittered | 0.438 ± 0.003 | 0.334 ± 0.015 | 4.88 ± 0.14 |
| SAINT-P | learned + geometry | clustered | 0.581 ± 0.007 | 0.329 ± 0.022 | 8.40 ± 1.07 |
| SAINT-P | post hoc | uniform | 0.346 ± 0.000 | 0.357 ± 0.001 | 2.89 ± 0.03 |
| SAINT-P | post hoc | jittered | 0.457 ± 0.001 | 0.361 ± 0.001 | 5.15 ± 0.01 |
| SAINT-P | post hoc | clustered | 0.834 ± 0.004 | 0.389 ± 0.000 | 14.50 ± 0.03 |
| SAINT-P | post hoc + geometry | uniform | 0.335 ± 0.000 | 0.299 ± 0.000 | 2.93 ± 0.04 |
| SAINT-P | post hoc + geometry | jittered | 0.405 ± 0.000 | 0.268 ± 0.000 | 4.24 ± 0.01 |
| SAINT-P | post hoc + geometry | clustered | 0.545 ± 0.001 | 0.278 ± 0.001 | 6.32 ± 0.04 |
| ArSSR-P | learned | uniform | 0.338 ± 0.000 | 0.356 ± 0.008 | 5.50 ± 1.66 |
| ArSSR-P | learned | jittered | 0.363 ± 0.002 | 0.357 ± 0.008 | 4.42 ± 1.25 |
| ArSSR-P | learned | clustered | 0.905 ± 0.021 | 0.438 ± 0.006 | 13.13 ± 0.59 |
| ArSSR-P | learned + geometry | uniform | 0.329 ± 0.002 | 0.347 ± 0.006 | 2.98 ± 0.59 |
| ArSSR-P | learned + geometry | jittered | 0.349 ± 0.001 | 0.334 ± 0.004 | 2.66 ± 0.50 |
| ArSSR-P | learned + geometry | clustered | 0.811 ± 0.022 | 0.368 ± 0.004 | 11.71 ± 0.31 |
| ArSSR-P | post hoc | uniform | 0.373 ± 0.003 | 0.384 ± 0.013 | 3.14 ± 0.01 |
| ArSSR-P | post hoc | jittered | 0.409 ± 0.001 | 0.402 ± 0.012 | 3.85 ± 0.04 |
| ArSSR-P | post hoc | clustered | 0.676 ± 0.008 | 0.474 ± 0.009 | 10.76 ± 0.27 |
| ArSSR-P | post hoc + geometry | uniform | 0.366 ± 0.003 | 0.351 ± 0.009 | 3.12 ± 0.00 |
| ArSSR-P | post hoc + geometry | jittered | 0.391 ± 0.001 | 0.342 ± 0.005 | 3.49 ± 0.05 |
| ArSSR-P | post hoc + geometry | clustered | 0.576 ± 0.007 | 0.387 ± 0.004 | 6.68 ± 0.37 |
| Linear | post hoc | uniform | 0.388 | 0.341 | 3.65 |
| Linear | post hoc | jittered | 0.423 | 0.346 | 4.10 |
| Linear | post hoc | clustered | 0.633 | 0.369 | 9.59 |
| Linear | post hoc + geometry | uniform | 0.370 | 0.275 | 3.52 |
| Linear | post hoc + geometry | jittered | 0.392 | 0.253 | 4.29 |
| Linear | post hoc + geometry | clustered | 0.537 | 0.270 | 13.87 |

## IXI-tiny T1 (60 test patients), ×4

| Reconstruction | Uncertainty | Pattern | IS ↓ | AUSE/MAE ↓ | UCE ×10³ ↓ |
|---|---|---|---|---|---|
| SAINT-P | learned | uniform | 0.205 ± 0.001 | 0.496 ± 0.052 | 0.52 ± 0.03 |
| SAINT-P | learned | jittered | 0.274 ± 0.001 | 0.512 ± 0.052 | 1.51 ± 0.02 |
| SAINT-P | learned | clustered | 0.576 ± 0.000 | 0.514 ± 0.048 | 6.74 ± 0.03 |
| SAINT-P | learned + geometry | uniform | 0.198 ± 0.000 | 0.370 ± 0.008 | 0.48 ± 0.01 |
| SAINT-P | learned + geometry | jittered | 0.252 ± 0.002 | 0.352 ± 0.003 | 1.33 ± 0.10 |
| SAINT-P | learned + geometry | clustered | 0.429 ± 0.013 | 0.340 ± 0.003 | 4.59 ± 0.44 |
| SAINT-P | post hoc | uniform | 0.202 ± 0.000 | 0.390 ± 0.000 | 0.69 ± 0.00 |
| SAINT-P | post hoc | jittered | 0.277 ± 0.000 | 0.396 ± 0.000 | 1.60 ± 0.00 |
| SAINT-P | post hoc | clustered | 0.611 ± 0.000 | 0.423 ± 0.000 | 7.15 ± 0.00 |
| SAINT-P | post hoc + geometry | uniform | 0.198 ± 0.000 | 0.365 ± 0.000 | 0.57 ± 0.00 |
| SAINT-P | post hoc + geometry | jittered | 0.256 ± 0.000 | 0.342 ± 0.000 | 1.32 ± 0.00 |
| SAINT-P | post hoc + geometry | clustered | 0.441 ± 0.000 | 0.322 ± 0.000 | 4.81 ± 0.00 |
| ArSSR-P | learned | uniform | 0.272 ± 0.004 | 0.495 ± 0.003 | 1.96 ± 0.07 |
| ArSSR-P | learned | jittered | 0.287 ± 0.003 | 0.487 ± 0.002 | 1.68 ± 0.06 |
| ArSSR-P | learned | clustered | 0.465 ± 0.007 | 0.466 ± 0.002 | 4.98 ± 0.15 |
| ArSSR-P | learned + geometry | uniform | 0.271 ± 0.003 | 0.502 ± 0.005 | 1.70 ± 0.06 |
| ArSSR-P | learned + geometry | jittered | 0.284 ± 0.002 | 0.487 ± 0.004 | 1.39 ± 0.05 |
| ArSSR-P | learned + geometry | clustered | 0.435 ± 0.007 | 0.446 ± 0.002 | 4.30 ± 0.21 |
| ArSSR-P | post hoc | uniform | 0.272 ± 0.003 | 0.403 ± 0.008 | 0.75 ± 0.02 |
| ArSSR-P | post hoc | jittered | 0.293 ± 0.003 | 0.422 ± 0.007 | 1.02 ± 0.02 |
| ArSSR-P | post hoc | clustered | 0.463 ± 0.001 | 0.544 ± 0.002 | 4.59 ± 0.06 |
| ArSSR-P | post hoc + geometry | uniform | 0.271 ± 0.003 | 0.396 ± 0.007 | 0.70 ± 0.01 |
| ArSSR-P | post hoc + geometry | jittered | 0.288 ± 0.003 | 0.396 ± 0.004 | 0.87 ± 0.01 |
| ArSSR-P | post hoc + geometry | clustered | 0.424 ± 0.003 | 0.503 ± 0.002 | 2.85 ± 0.10 |
| Linear | post hoc | uniform | 0.262 | 0.385 | 0.97 |
| Linear | post hoc | jittered | 0.289 | 0.394 | 1.15 |
| Linear | post hoc | clustered | 0.458 | 0.421 | 4.76 |
| Linear | post hoc + geometry | uniform | 0.257 | 0.359 | 0.91 |
| Linear | post hoc + geometry | jittered | 0.273 | 0.321 | 0.99 |
| Linear | post hoc + geometry | clustered | 0.393 | 0.315 | 2.81 |

## IXI-tiny T1 (60 test patients), ×8

| Reconstruction | Uncertainty | Pattern | IS ↓ | AUSE/MAE ↓ | UCE ×10³ ↓ |
|---|---|---|---|---|---|
| SAINT-P | learned | uniform | 0.320 ± 0.002 | 0.519 ± 0.050 | 1.26 ± 0.02 |
| SAINT-P | learned | jittered | 0.419 ± 0.001 | 0.511 ± 0.041 | 3.68 ± 0.03 |
| SAINT-P | learned | clustered | 0.745 ± 0.002 | 0.568 ± 0.039 | 12.10 ± 0.05 |
| SAINT-P | learned + geometry | uniform | 0.303 ± 0.003 | 0.350 ± 0.006 | 1.29 ± 0.15 |
| SAINT-P | learned + geometry | jittered | 0.400 ± 0.003 | 0.382 ± 0.004 | 3.26 ± 0.24 |
| SAINT-P | learned + geometry | clustered | 0.598 ± 0.012 | 0.402 ± 0.008 | 7.28 ± 0.86 |
| SAINT-P | post hoc | uniform | 0.316 ± 0.000 | 0.419 ± 0.000 | 1.69 ± 0.00 |
| SAINT-P | post hoc | jittered | 0.422 ± 0.000 | 0.426 ± 0.000 | 3.69 ± 0.00 |
| SAINT-P | post hoc | clustered | 0.784 ± 0.000 | 0.486 ± 0.000 | 12.76 ± 0.01 |
| SAINT-P | post hoc + geometry | uniform | 0.307 ± 0.000 | 0.356 ± 0.000 | 1.67 ± 0.00 |
| SAINT-P | post hoc + geometry | jittered | 0.401 ± 0.000 | 0.358 ± 0.000 | 3.40 ± 0.00 |
| SAINT-P | post hoc + geometry | clustered | 0.577 ± 0.000 | 0.346 ± 0.000 | 6.88 ± 0.00 |
| ArSSR-P | learned | uniform | 0.383 ± 0.004 | 0.448 ± 0.003 | 3.60 ± 0.26 |
| ArSSR-P | learned | jittered | 0.415 ± 0.004 | 0.451 ± 0.003 | 2.72 ± 0.19 |
| ArSSR-P | learned | clustered | 0.681 ± 0.007 | 0.533 ± 0.002 | 9.73 ± 0.17 |
| ArSSR-P | learned + geometry | uniform | 0.382 ± 0.004 | 0.454 ± 0.003 | 2.77 ± 0.18 |
| ArSSR-P | learned + geometry | jittered | 0.410 ± 0.004 | 0.446 ± 0.004 | 2.10 ± 0.16 |
| ArSSR-P | learned + geometry | clustered | 0.626 ± 0.005 | 0.498 ± 0.003 | 8.17 ± 0.22 |
| ArSSR-P | post hoc | uniform | 0.382 ± 0.003 | 0.424 ± 0.004 | 1.77 ± 0.02 |
| ArSSR-P | post hoc | jittered | 0.420 ± 0.003 | 0.449 ± 0.005 | 2.11 ± 0.00 |
| ArSSR-P | post hoc | clustered | 0.628 ± 0.001 | 0.540 ± 0.001 | 8.39 ± 0.08 |
| ArSSR-P | post hoc + geometry | uniform | 0.381 ± 0.003 | 0.419 ± 0.004 | 1.65 ± 0.03 |
| ArSSR-P | post hoc + geometry | jittered | 0.414 ± 0.003 | 0.424 ± 0.001 | 1.98 ± 0.04 |
| ArSSR-P | post hoc + geometry | clustered | 0.580 ± 0.002 | 0.516 ± 0.003 | 5.35 ± 0.12 |
| Linear | post hoc | uniform | 0.422 | 0.431 | 2.86 |
| Linear | post hoc | jittered | 0.453 | 0.439 | 3.03 |
| Linear | post hoc | clustered | 0.602 | 0.485 | 6.61 |
| Linear | post hoc + geometry | uniform | 0.398 | 0.326 | 2.78 |
| Linear | post hoc + geometry | jittered | 0.426 | 0.305 | 4.51 |
| Linear | post hoc + geometry | clustered | 0.573 | 0.322 | 13.88 |
