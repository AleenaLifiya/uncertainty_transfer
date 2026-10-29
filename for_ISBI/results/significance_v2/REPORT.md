# Significance analysis v2 (current design: SAINT-P / ArSSR-P / linear x learned / learned+geometry / post hoc +-geometry)

CPU only, from saved per-case metrics (exp_seeds/results). Unit = TEST patient (BraTS 51, IXI-tiny 60): mask realisations averaged within patient, then three fine-tuning seeds averaged (linear: single run). Outcomes: dcov = clustered minus uniform 90 % coverage; is_C and ence_C = clustered interval score and ENCE (lower is better). One-factor repeated-measures ANOVA across uncertainty arms sharing one reconstruction (GG-corrected, Friedman sensitivity; Holm over all ANOVA rows). Planned paired contrasts with Holm correction within each (dataset, stride, outcome) family. Retrospective, exploratory; inference is across patients conditional on the fitted (seed-averaged) models, not across training seeds.

## ANOVA across uncertainty arms

| Dataset | Recon | R | Outcome | F | partial eta2 | GG eps | Holm p (GG) | Holm p (Friedman) |
|---|---|---:|---|---:|---:|---:|---:|---:|
| brats | ArSSR-P | 4 | dcov | 229.2 | 0.82 | 0.37 | 1.1e-21 | 7.2e-29 |
| brats | ArSSR-P | 4 | is_C | 23.8 | 0.32 | 0.46 | 4.6e-07 | 2.2e-17 |
| brats | ArSSR-P | 4 | ence_C | 58.7 | 0.54 | 0.45 | 2.3e-12 | 2.3e-19 |
| brats | ArSSR-P | 8 | dcov | 793.0 | 0.94 | 0.38 | 2.6e-35 | 1.2e-31 |
| brats | ArSSR-P | 8 | is_C | 156.6 | 0.76 | 0.44 | 7.7e-21 | 4.4e-29 |
| brats | ArSSR-P | 8 | ence_C | 304.0 | 0.86 | 0.38 | 2.0e-24 | 2.8e-30 |
| brats | SAINT-P | 4 | dcov | 1181.3 | 0.96 | 0.57 | 3.8e-59 | 1.2e-31 |
| brats | SAINT-P | 4 | is_C | 172.2 | 0.77 | 0.36 | 1.7e-18 | 8.0e-29 |
| brats | SAINT-P | 4 | ence_C | 355.5 | 0.88 | 0.43 | 1.1e-29 | 1.6e-29 |
| brats | SAINT-P | 8 | dcov | 1896.2 | 0.97 | 0.54 | 3.1e-64 | 4.6e-30 |
| brats | SAINT-P | 8 | is_C | 215.8 | 0.81 | 0.36 | 2.7e-20 | 2.7e-27 |
| brats | SAINT-P | 8 | ence_C | 427.0 | 0.90 | 0.43 | 6.6e-32 | 5.7e-29 |
| ixi | ArSSR-P | 4 | dcov | 826.3 | 0.93 | 0.47 | 6.4e-49 | 1.0e-33 |
| ixi | ArSSR-P | 4 | is_C | 119.4 | 0.67 | 0.55 | 1.3e-23 | 2.3e-22 |
| ixi | ArSSR-P | 4 | ence_C | 166.5 | 0.74 | 0.57 | 4.2e-29 | 1.3e-24 |
| ixi | ArSSR-P | 8 | dcov | 758.5 | 0.93 | 0.46 | 2.5e-46 | 5.9e-34 |
| ixi | ArSSR-P | 8 | is_C | 156.7 | 0.73 | 0.47 | 1.4e-23 | 1.1e-23 |
| ixi | ArSSR-P | 8 | ence_C | 285.0 | 0.83 | 0.61 | 5.1e-41 | 8.6e-31 |
| ixi | SAINT-P | 4 | dcov | 4566.6 | 0.99 | 0.40 | 2.1e-67 | 2.1e-37 |
| ixi | SAINT-P | 4 | is_C | 302.7 | 0.84 | 0.34 | 2.2e-24 | 1.5e-31 |
| ixi | SAINT-P | 4 | ence_C | 456.0 | 0.89 | 0.35 | 1.4e-29 | 5.9e-31 |
| ixi | SAINT-P | 8 | dcov | 1704.5 | 0.97 | 0.36 | 1.3e-47 | 3.6e-37 |
| ixi | SAINT-P | 8 | is_C | 186.0 | 0.76 | 0.34 | 8.1e-20 | 1.5e-27 |
| ixi | SAINT-P | 8 | ence_C | 288.0 | 0.83 | 0.35 | 2.8e-24 | 4.3e-26 |

## Planned contrasts (difference = first arm minus second; dcov in coverage units)

| Dataset | Recon | R | Outcome | Contrast | Difference | Bootstrap 95% CI | dz | Holm p (t) | Holm p (Wilcoxon) |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| brats | ArSSR-P | 4 | dcov | geometry, post hoc | +0.104 | [+0.098, +0.109] | +4.79 | 1.4e-35 | 3.6e-09 |
| brats | ArSSR-P | 4 | dcov | geometry, learned | +0.053 | [+0.045, +0.060] | +2.01 | 2.5e-19 | 3.6e-09 |
| brats | ArSSR-P | 4 | dcov | learned+geo vs post-hoc+geo | -0.203 | [-0.228, -0.177] | -2.11 | 6.5e-20 | 3.6e-09 |
| brats | ArSSR-P | 4 | is_C | geometry, post hoc | -0.073 | [-0.088, -0.058] | -1.32 | 3.5e-12 | 8.9e-09 |
| brats | ArSSR-P | 4 | is_C | geometry, learned | -0.051 | [-0.059, -0.043] | -1.72 | 4.6e-16 | 3.6e-09 |
| brats | ArSSR-P | 4 | is_C | learned+geo vs post-hoc+geo | +0.045 | [+0.019, +0.070] | +0.48 | 1.2e-03 | 4.7e-04 |
| brats | ArSSR-P | 4 | ence_C | geometry, post hoc | -0.404 | [-0.467, -0.338] | -1.71 | 5.4e-16 | 1.1e-08 |
| brats | ArSSR-P | 4 | ence_C | geometry, learned | -0.257 | [-0.291, -0.222] | -2.02 | 8.7e-19 | 3.6e-09 |
| brats | ArSSR-P | 4 | ence_C | learned+geo vs post-hoc+geo | +0.467 | [+0.345, +0.583] | +1.05 | 8.9e-10 | 9.6e-08 |
| brats | ArSSR-P | 8 | dcov | geometry, post hoc | +0.098 | [+0.093, +0.103] | +5.03 | 1.4e-36 | 3.6e-09 |
| brats | ArSSR-P | 8 | dcov | geometry, learned | +0.058 | [+0.051, +0.066] | +2.07 | 1.5e-19 | 3.6e-09 |
| brats | ArSSR-P | 8 | dcov | learned+geo vs post-hoc+geo | -0.342 | [-0.367, -0.318] | -3.72 | 1.1e-30 | 3.6e-09 |
| brats | ArSSR-P | 8 | is_C | geometry, post hoc | -0.101 | [-0.120, -0.083] | -1.50 | 4.4e-14 | 3.6e-09 |
| brats | ArSSR-P | 8 | is_C | geometry, learned | -0.094 | [-0.105, -0.083] | -2.38 | 1.3e-21 | 3.6e-09 |
| brats | ArSSR-P | 8 | is_C | learned+geo vs post-hoc+geo | +0.236 | [+0.198, +0.277] | +1.64 | 2.6e-15 | 3.6e-09 |
| brats | ArSSR-P | 8 | ence_C | geometry, post hoc | -0.382 | [-0.430, -0.332] | -2.11 | 1.0e-19 | 3.6e-09 |
| brats | ArSSR-P | 8 | ence_C | geometry, learned | -0.501 | [-0.539, -0.463] | -3.56 | 2.0e-29 | 3.6e-09 |
| brats | ArSSR-P | 8 | ence_C | learned+geo vs post-hoc+geo | +1.477 | [+1.309, +1.651] | +2.33 | 2.1e-21 | 3.6e-09 |
| brats | Linear | 4 | dcov | geometry, post hoc | +0.161 | [+0.148, +0.173] | +3.50 | 2.4e-29 | 3.6e-09 |
| brats | Linear | 4 | is_C | geometry, post hoc | -0.104 | [-0.134, -0.076] | -0.98 | 1.2e-08 | 7.9e-08 |
| brats | Linear | 4 | ence_C | geometry, post hoc | -0.606 | [-0.733, -0.476] | -1.29 | 5.0e-12 | 3.2e-08 |
| brats | Linear | 8 | dcov | geometry, post hoc | +0.150 | [+0.139, +0.160] | +3.94 | 1.0e-31 | 3.6e-09 |
| brats | Linear | 8 | is_C | geometry, post hoc | -0.096 | [-0.126, -0.067] | -0.89 | 6.3e-08 | 3.6e-07 |
| brats | Linear | 8 | ence_C | geometry, post hoc | -0.343 | [-0.431, -0.255] | -1.06 | 1.6e-09 | 1.9e-07 |
| brats | SAINT-P | 4 | dcov | geometry, post hoc | +0.230 | [+0.220, +0.239] | +6.51 | 5.5e-42 | 3.6e-09 |
| brats | SAINT-P | 4 | dcov | geometry, learned | +0.113 | [+0.106, +0.121] | +4.08 | 2.4e-32 | 3.6e-09 |
| brats | SAINT-P | 4 | dcov | learned+geo vs post-hoc+geo | -0.063 | [-0.069, -0.056] | -2.66 | 4.7e-24 | 3.6e-09 |
| brats | SAINT-P | 4 | is_C | geometry, post hoc | -0.206 | [-0.239, -0.175] | -1.78 | 1.8e-16 | 3.6e-09 |
| brats | SAINT-P | 4 | is_C | geometry, learned | -0.125 | [-0.141, -0.109] | -2.13 | 1.6e-19 | 3.6e-09 |
| brats | SAINT-P | 4 | is_C | learned+geo vs post-hoc+geo | +0.089 | [+0.073, +0.105] | +1.55 | 1.9e-14 | 3.6e-09 |
| brats | SAINT-P | 4 | ence_C | geometry, post hoc | -1.439 | [-1.583, -1.293] | -2.71 | 4.4e-24 | 3.6e-09 |
| brats | SAINT-P | 4 | ence_C | geometry, learned | -0.907 | [-0.989, -0.823] | -2.94 | 1.3e-25 | 3.6e-09 |
| brats | SAINT-P | 4 | ence_C | learned+geo vs post-hoc+geo | +0.284 | [+0.236, +0.332] | +1.62 | 2.7e-15 | 7.9e-09 |
| brats | SAINT-P | 8 | dcov | geometry, post hoc | +0.239 | [+0.231, +0.248] | +7.65 | 2.1e-45 | 3.6e-09 |
| brats | SAINT-P | 8 | dcov | geometry, learned | +0.162 | [+0.153, +0.171] | +4.77 | 1.4e-35 | 3.6e-09 |
| brats | SAINT-P | 8 | dcov | learned+geo vs post-hoc+geo | -0.020 | [-0.024, -0.015] | -1.13 | 1.4e-10 | 1.3e-08 |
| brats | SAINT-P | 8 | is_C | geometry, post hoc | -0.289 | [-0.328, -0.251] | -2.03 | 7.9e-19 | 3.6e-09 |
| brats | SAINT-P | 8 | is_C | geometry, learned | -0.248 | [-0.280, -0.216] | -2.12 | 1.6e-19 | 3.6e-09 |
| brats | SAINT-P | 8 | is_C | learned+geo vs post-hoc+geo | +0.037 | [+0.028, +0.046] | +1.11 | 4.9e-10 | 3.6e-09 |
| brats | SAINT-P | 8 | ence_C | geometry, post hoc | -1.148 | [-1.241, -1.050] | -3.29 | 7.0e-28 | 3.6e-09 |
| brats | SAINT-P | 8 | ence_C | geometry, learned | -0.746 | [-0.825, -0.663] | -2.50 | 1.2e-22 | 3.6e-09 |
| brats | SAINT-P | 8 | ence_C | learned+geo vs post-hoc+geo | +0.043 | [+0.027, +0.059] | +0.72 | 4.5e-06 | 9.7e-06 |
| ixi | ArSSR-P | 4 | dcov | geometry, post hoc | +0.063 | [+0.060, +0.066] | +5.27 | 2.3e-44 | 1.1e-10 |
| ixi | ArSSR-P | 4 | dcov | geometry, learned | +0.038 | [+0.037, +0.039] | +11.59 | 5.2e-64 | 1.1e-10 |
| ixi | ArSSR-P | 4 | dcov | learned+geo vs post-hoc+geo | -0.069 | [-0.074, -0.064] | -3.65 | 9.9e-36 | 1.1e-10 |
| ixi | ArSSR-P | 4 | is_C | geometry, post hoc | -0.038 | [-0.045, -0.031] | -1.39 | 5.3e-15 | 1.6e-09 |
| ixi | ArSSR-P | 4 | is_C | geometry, learned | -0.029 | [-0.032, -0.027] | -3.00 | 1.4e-30 | 1.1e-10 |
| ixi | ArSSR-P | 4 | is_C | learned+geo vs post-hoc+geo | +0.011 | [+0.006, +0.016] | +0.57 | 4.2e-05 | 2.6e-04 |
| ixi | ArSSR-P | 4 | ence_C | geometry, post hoc | -0.210 | [-0.248, -0.170] | -1.34 | 2.0e-14 | 1.6e-09 |
| ixi | ArSSR-P | 4 | ence_C | geometry, learned | -0.141 | [-0.150, -0.132] | -3.97 | 3.7e-37 | 1.1e-10 |
| ixi | ArSSR-P | 4 | ence_C | learned+geo vs post-hoc+geo | +0.177 | [+0.145, +0.207] | +1.40 | 4.2e-15 | 1.6e-09 |
| ixi | ArSSR-P | 8 | dcov | geometry, post hoc | +0.065 | [+0.061, +0.068] | +4.64 | 3.9e-41 | 1.1e-10 |
| ixi | ArSSR-P | 8 | dcov | geometry, learned | +0.050 | [+0.048, +0.052] | +6.97 | 3.6e-51 | 1.1e-10 |
| ixi | ArSSR-P | 8 | dcov | learned+geo vs post-hoc+geo | -0.075 | [-0.080, -0.069] | -3.32 | 3.0e-33 | 1.1e-10 |
| ixi | ArSSR-P | 8 | is_C | geometry, post hoc | -0.049 | [-0.058, -0.039] | -1.23 | 6.1e-13 | 2.1e-09 |
| ixi | ArSSR-P | 8 | is_C | geometry, learned | -0.055 | [-0.059, -0.050] | -2.86 | 1.8e-29 | 1.1e-10 |
| ixi | ArSSR-P | 8 | is_C | learned+geo vs post-hoc+geo | +0.047 | [+0.037, +0.057] | +1.14 | 6.0e-12 | 7.8e-09 |
| ixi | ArSSR-P | 8 | ence_C | geometry, post hoc | -0.206 | [-0.244, -0.166] | -1.29 | 8.0e-14 | 1.6e-09 |
| ixi | ArSSR-P | 8 | ence_C | geometry, learned | -0.215 | [-0.229, -0.201] | -3.85 | 1.9e-36 | 1.1e-10 |
| ixi | ArSSR-P | 8 | ence_C | learned+geo vs post-hoc+geo | +0.273 | [+0.235, +0.310] | +1.80 | 1.4e-19 | 3.5e-10 |
| ixi | Linear | 4 | dcov | geometry, post hoc | +0.123 | [+0.114, +0.131] | +3.65 | 9.9e-36 | 1.1e-10 |
| ixi | Linear | 4 | is_C | geometry, post hoc | -0.065 | [-0.080, -0.050] | -1.08 | 3.7e-11 | 6.8e-09 |
| ixi | Linear | 4 | ence_C | geometry, post hoc | -0.350 | [-0.423, -0.273] | -1.16 | 2.5e-12 | 1.6e-09 |
| ixi | Linear | 8 | dcov | geometry, post hoc | +0.097 | [+0.089, +0.104] | +3.21 | 1.2e-32 | 1.1e-10 |
| ixi | Linear | 8 | is_C | geometry, post hoc | -0.029 | [-0.045, -0.012] | -0.44 | 1.1e-03 | 4.0e-03 |
| ixi | Linear | 8 | ence_C | geometry, post hoc | -0.103 | [-0.148, -0.056] | -0.56 | 1.2e-04 | 2.3e-03 |
| ixi | SAINT-P | 4 | dcov | geometry, post hoc | +0.172 | [+0.168, +0.175] | +12.35 | 1.5e-65 | 1.1e-10 |
| ixi | SAINT-P | 4 | dcov | geometry, learned | +0.142 | [+0.137, +0.147] | +6.63 | 5.7e-50 | 1.1e-10 |
| ixi | SAINT-P | 4 | dcov | learned+geo vs post-hoc+geo | +0.026 | [+0.025, +0.028] | +4.69 | 1.3e-41 | 1.1e-10 |
| ixi | SAINT-P | 4 | is_C | geometry, post hoc | -0.171 | [-0.187, -0.154] | -2.54 | 7.4e-27 | 1.1e-10 |
| ixi | SAINT-P | 4 | is_C | geometry, learned | -0.147 | [-0.165, -0.128] | -1.99 | 1.2e-21 | 3.6e-10 |
| ixi | SAINT-P | 4 | is_C | learned+geo vs post-hoc+geo | -0.012 | [-0.015, -0.008] | -0.80 | 1.1e-07 | 9.3e-06 |
| ixi | SAINT-P | 4 | ence_C | geometry, post hoc | -0.893 | [-0.951, -0.829] | -3.64 | 3.8e-35 | 1.1e-10 |
| ixi | SAINT-P | 4 | ence_C | geometry, learned | -0.568 | [-0.641, -0.489] | -1.86 | 2.5e-20 | 7.5e-10 |
| ixi | SAINT-P | 4 | ence_C | learned+geo vs post-hoc+geo | -0.076 | [-0.098, -0.052] | -0.82 | 3.0e-08 | 1.5e-05 |
| ixi | SAINT-P | 8 | dcov | geometry, post hoc | +0.174 | [+0.168, +0.180] | +7.16 | 9.0e-52 | 1.1e-10 |
| ixi | SAINT-P | 8 | dcov | geometry, learned | +0.146 | [+0.137, +0.154] | +4.11 | 3.2e-38 | 1.1e-10 |
| ixi | SAINT-P | 8 | dcov | learned+geo vs post-hoc+geo | +0.022 | [+0.019, +0.024] | +2.14 | 6.7e-24 | 1.1e-10 |
| ixi | SAINT-P | 8 | is_C | geometry, post hoc | -0.207 | [-0.230, -0.183] | -2.18 | 1.7e-23 | 1.1e-10 |
| ixi | SAINT-P | 8 | is_C | geometry, learned | -0.148 | [-0.173, -0.121] | -1.41 | 3.8e-15 | 2.1e-09 |
| ixi | SAINT-P | 8 | is_C | learned+geo vs post-hoc+geo | +0.021 | [+0.014, +0.028] | +0.77 | 3.1e-07 | 1.0e-07 |
| ixi | SAINT-P | 8 | ence_C | geometry, post hoc | -0.761 | [-0.825, -0.690] | -2.78 | 6.8e-29 | 1.1e-10 |
| ixi | SAINT-P | 8 | ence_C | geometry, learned | -0.401 | [-0.470, -0.327] | -1.40 | 4.8e-15 | 1.6e-09 |
| ixi | SAINT-P | 8 | ence_C | learned+geo vs post-hoc+geo | +0.015 | [-0.003, +0.034] | +0.20 | 1.2e-01 | 4.6e-01 |

Reproduce: `python for_ISBI/exp_pg/scripts/significance_v2.py` (after metrics_seed.py / metrics_pg.py).
