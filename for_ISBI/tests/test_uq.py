"""CPU unit tests for isbi.uq (run: $PY tests/test_uq.py)."""
import numpy as np, torch
from isbi import uq as U
rng = np.random.RandomState(0)
n = 60000
F = np.stack([rng.rand(n), rng.rand(n) * .2, rng.rand(n) * .1, rng.rand(n) * .3,
              rng.randint(1, 5, n).astype(float), rng.choice([4., 8., 16.], n), rng.rand(n)], 1).astype(np.float32)
b_true = np.exp(-4 + 0.5 * np.log(F[:, 5]) + 0.8 * F[:, 6])
e = rng.laplace(0, b_true)
g = U.LaplaceGLM(1e-6, False).fit(F, e)
r = np.corrcoef(np.log(g.predict(F)), np.log(b_true))[0, 1]
print("GLM log-scale corr", round(r, 4)); assert r > 0.97
# calibration recovers the Laplace central-90 multiplier -ln(0.1) = 2.3026
lam = U.calibrate_scalar(lambda p, l: (np.abs(e[p]) <= l * b_true[p]).mean(), [slice(0, n // 2), slice(n // 2, n)], 0.9)
print("kappa90", round(lam, 3)); assert abs(lam - np.log(10)) < 0.05
a, s, o = U.ause(e, np.abs(e)); assert abs(a) < 1e-12
ar, _, _ = U.ause(e, rng.rand(n)); ab, _, _ = U.ause(e, b_true); print("AUSE random/true-scale", ar, ab); assert ab < ar
# geometry-only score has no within-stratum ranking
print("strat spearman of gap score", U.strat_spearman(e, F[:, 5], F[:, 5], F[:, 4]))
# exact mixture coverage
mu = torch.zeros(1, 50000); b = torch.full((1, 50000), 0.1)
y = torch.from_numpy(rng.laplace(0, 0.1, 50000).astype(np.float32))
Fy = U.mix_cdf(y, mu, b).numpy(); print("mixture cov90", ((Fy >= .05) & (Fy <= .95)).mean())
q = U.mix_quantile(mu[:, :3], b[:, :3], 0.95); print("q95", q, 0.1 * np.log(10))
print("ALL OK")
