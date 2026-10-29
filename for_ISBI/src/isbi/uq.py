"""UQ analysis: post-hoc Laplace-scale GLM, patient-aware interval calibration, and metrics.

Conventions
-----------
* error e = y - x, x = the method's final reconstruction (clipped to [0,1], acquired slices copied
  back) at the fixed UQ voxel subset (missing slices, GT foreground).
* Learned intervals: central alpha-interval of the equal-weight Laplace mixture
  (1/T) sum_t Laplace(mu_t, lam * b_t). Coverage is evaluated exactly through the mixture CDF at y
  (y inside the central interval <=> (1-a)/2 <= F(y) <= (1+a)/2); widths by CDF bisection.
  "raw" = lam 1; "cal" = lam fitted on CAL (uniform) so the patient-averaged coverage is nominal.
* Post-hoc intervals: x +- kappa * b_hat; Constant-width: x +- w. kappa, w fitted the same way.
* Calibration target: mean over CAL patients of per-patient coverage == alpha (equal patient
  weight; empirical calibration -- no distribution-free guarantee is claimed).
* Widths are reported after intersecting the interval with the target support [0, 1].
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import stats

LEVELS = (0.5, 0.8, 0.9, 0.95)
EPS = 1e-3


# ============================================================================ mixture
def mix_cdf(y, mu, b, lam=1.0):
    """y: (n,), mu/b: (T,n) torch -> F(y) (n,)."""
    z = (y[None] - mu) / (lam * b)
    return torch.where(z < 0, 0.5 * torch.exp(z.clamp(max=0)), 1 - 0.5 * torch.exp(-z.clamp(min=0))).mean(0)


def mix_quantile(mu, b, q, lam=1.0, iters=36):
    bb = lam * b
    lo = (mu - 30 * bb).min(0).values
    hi = (mu + 30 * bb).max(0).values
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        F = mix_cdf(mid, mu, b, lam)
        go = F < q
        lo = torch.where(go, mid, lo)
        hi = torch.where(go, hi, mid)
    return 0.5 * (lo + hi)


# ============================================================================ post-hoc GLM
def design(F, interactions=True):
    """F columns: mu, grad, dlin, dacq, near_mm, gap_mm, s_mid (evalcore.FEATURES)."""
    mu, grad, dlin, dacq, near, gap, s = [F[:, i] for i in range(7)]
    cols = [mu, np.log(grad + EPS), np.log(dlin + EPS), np.log(dacq + EPS), np.log(near), np.log(gap), s]
    if interactions:
        cols += [np.log(gap) * s, np.log(dacq + EPS) * s, np.log(dacq + EPS) * np.log(gap)]
    return np.stack(cols, 1).astype(np.float64)


class LaplaceGLM:
    """log b = theta0 + theta . standardised(phi); fitted by minimising mean Laplace NLL
    (convex in theta) + ridge * |theta|^2 with L-BFGS."""

    def __init__(self, ridge=1e-3, interactions=True):
        self.ridge, self.inter = ridge, interactions

    def fit(self, F, e, w=None):
        X = design(F, self.inter)
        self.m, self.s = X.mean(0), X.std(0) + 1e-8
        X = torch.from_numpy((X - self.m) / self.s)
        a = torch.from_numpy(np.abs(e).astype(np.float64))
        w = torch.ones_like(a) if w is None else torch.from_numpy(w.astype(np.float64))
        w = w / w.sum()
        th = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([float(np.log(np.mean(np.abs(e)) + 1e-6))], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th, c], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            lb = c + X @ th
            loss = (w * (lb + a * torch.exp(-lb))).sum() + self.ridge * (th ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.th, self.c = th.detach().numpy(), float(c.detach())
        return self

    def predict(self, F):
        X = (design(F, self.inter) - self.m) / self.s
        return np.exp(self.c + X @ self.th).astype(np.float32)


def laplace_nll(e, b):
    return float(np.mean(np.log(2 * b) + np.abs(e) / b))


# ============================================================================ calibration
def calibrate_scalar(cover_fn, patients, alpha, lo=1e-3, hi=1e3, iters=60):
    """Find lam with mean_p cover_fn(p, lam) == alpha (cover_fn monotone increasing in lam)."""
    f = lambda lam: np.mean([cover_fn(p, lam) for p in patients])  # noqa: E731
    lo, hi = np.log(lo), np.log(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(np.exp(mid)) < alpha:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


# ============================================================================ ranking metrics
FR = np.round(np.arange(0.0, 0.99, 0.02), 2)   # fraction REMOVED (sparsification grid)


def sparsification(err, score):
    """MAE of the remaining voxels after removing the top fraction f by score, f in FR."""
    n = len(err)
    o = np.argsort(-score, kind="stable")
    e = np.abs(err)[o]
    cs = np.cumsum(e[::-1])[::-1]            # sum of e[k:] for each k
    out = []
    for f in FR:
        k = int(np.floor(f * n))
        out.append(cs[k] / (n - k))
    return np.asarray(out)


def ause(err, score):
    s = sparsification(err, score)
    o = sparsification(err, np.abs(err))
    return float(np.trapz(s - o, FR)), s, o


def auprc(pos, score):
    o = np.argsort(-score, kind="stable")
    p = pos[o].astype(float)
    tp = np.cumsum(p)
    prec = tp / np.arange(1, len(p) + 1)
    return float((prec * p).sum() / max(p.sum(), 1))


def spearman(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(stats.spearmanr(a, b).correlation)


def strat_spearman(err, score, gap, near, min_n=200):
    """Mean within-stratum Spearman, strata = exact (gap length, nearest distance) geometry cells
    (geometry-only scores are constant inside a stratum -> this is information BEYOND geometry)."""
    tot, acc = 0, 0.0
    key = gap * 1000 + near
    for k in np.unique(key):
        m = key == k
        if m.sum() < min_n:
            continue
        r = spearman(np.abs(err[m]), score[m])
        if np.isfinite(r):
            acc += r * m.sum()
            tot += m.sum()
    return acc / tot if tot else np.nan
