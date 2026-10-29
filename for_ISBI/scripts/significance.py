"""CPU-only inference from saved TEST metrics; run with the project's Python."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
OUT = RES / 'significance'
PATTERNS = ['uniform', 'jittered', 'clustered']


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return out


def paired(d, **labels):
    d = np.asarray(d, dtype=float)
    assert len(d) >= 3 and np.isfinite(d).all()
    n, sd = len(d), d.std(ddof=1)
    t = stats.ttest_1samp(d, 0)
    ci = stats.t.interval(.95, n - 1, loc=d.mean(), scale=sd / np.sqrt(n))
    bs = np.random.default_rng(20261023).choice(d, (10000, n), replace=True).mean(axis=1)
    return dict(**labels, n=n, difference=d.mean(), sd_difference=sd, dz=d.mean()/sd,
                t=t.statistic, df=n-1, p_t=t.pvalue,
                p_wilcoxon=stats.wilcoxon(d, method='auto').pvalue,
                ci_t_lo=ci[0], ci_t_hi=ci[1],
                ci_boot_lo=np.quantile(bs, .025), ci_boot_hi=np.quantile(bs, .975))


def rm_anova(w, **labels):
    """Balanced one-factor within-patient ANOVA, GG correction, Friedman check."""
    y = w.to_numpy(float)
    assert np.isfinite(y).all()
    n, k = y.shape
    effect = n * np.square(y.mean(0) - y.mean()).sum()
    error = np.square(y - y.mean(0) - y.mean(1)[:, None] + y.mean()).sum()
    df1, df2 = k-1, (n-1)*(k-1)
    f = (effect/df1)/(error/df2)
    c = np.eye(k) - np.ones((k,k))/k
    cov = c @ np.cov(y, rowvar=False) @ c
    eps = np.clip(np.trace(cov)**2 / ((k-1)*np.square(cov).sum()), 1/(k-1), 1)
    fried = stats.friedmanchisquare(*y.T) if k >= 3 else None
    return dict(**labels, n=n, levels=k, F=f, df1=df1, df2=df2,
                epsilon_GG=eps, p_uncorrected=stats.f.sf(f,df1,df2),
                p_GG=stats.f.sf(f,eps*df1,eps*df2),
                partial_eta2=effect/(effect+error), friedman_chi2=fried.statistic if fried else np.nan,
                p_friedman=fried.pvalue if fried else np.nan)


def save_tests(rows, name, pcols):
    df = pd.DataFrame(rows)
    for col in pcols:
        df[col+'_holm'] = df.groupby('family')[col].transform(lambda x: holm(x))
    df.to_csv(OUT/name, index=False)
    return df


def main():
    OUT.mkdir(exist_ok=True)
    inputs = ['patient_level_intervals.csv', 'test_recon_metrics.csv', 'exposure/case_metrics.csv']
    intervals = pd.read_csv(RES/inputs[0])
    assert not intervals.duplicated(['patient','arm','R','pattern']).any()
    expected = set(intervals.patient.unique())
    assert len(expected) == 51

    def wide(g, metric):
        w = g.pivot(index='patient', columns='pattern', values=metric).reindex(columns=PATTERNS)
        assert set(w.index) == expected and np.isfinite(w.to_numpy()).all()
        return w

    primary, pattern, anova = [], [], []
    for arch in ['SAINT', 'I3Net']:
        for r in [4,8]:
            group = intervals[(intervals['mean'] == arch+'_P') & (intervals.R == r)]
            a = wide(group[group.kind == 'learned_cal'], 'cov90')
            b = wide(group[group.kind == 'posthoc'], 'cov90')
            d = (a.clustered-a.uniform)-(b.clustered-b.uniform)
            primary.append(paired(d, arch=arch, R=r, family='primary_x4' if r==4 else 'confirmation_x8',
                                  contrast='(clustered-uniform): learned_cal minus posthoc'))
    for (arm,r), g in intervals.groupby(['arm','R']):
        for metric in ['cov90','width90']:
            w = wide(g, metric)
            anova.append(rm_anova(w, arm=arm,R=r,metric=metric,family='interval_pattern_ANOVA'))
            for pat in ['jittered','clustered']:
                pattern.append(paired(w[pat]-w.uniform, arm=arm,R=r,metric=metric,
                                      contrast=pat+' minus uniform',family='interval_pattern_pairs'))
    recon = pd.read_csv(RES/inputs[1])
    assert not recon.duplicated(['patient','method','R','pattern','rep']).any()
    recon = recon.groupby(['patient','method','R','pattern'],as_index=False)[['psnr','ssim','mae']].mean()
    for (method,r),g in recon.groupby(['method','R']):
        for metric in ['psnr','ssim','mae']:
            anova.append(rm_anova(wide(g,metric),arm=method,R=r,metric=metric,family='reconstruction_pattern_ANOVA'))
    exposure = pd.read_csv(RES/inputs[2])
    assert not exposure.duplicated(['patient','model','kind','R','pattern','rep']).any()
    exposure = exposure.groupby(['patient','model','kind','R','pattern'],as_index=False).cov90.mean()
    exp_tests = []
    for (model,kind,r),g in exposure.groupby(['model','kind','R']):
        arch = 'SAINT' if model.startswith('saint') else 'I3Net'
        ref = intervals[(intervals.arm == arch+'_P|'+kind) & (intervals.R == r)]
        a,b = wide(g,'cov90'),wide(ref,'cov90')
        exp_tests.append(paired((a.clustered-a.uniform)-(b.clustered-b.uniform),model=model,kind=kind,R=r,
                               family='exploratory_exposure',contrast='coverage change: exposure minus uniform training'))
    p = save_tests(primary,'primary_paired.csv',['p_t','p_wilcoxon'])
    save_tests(pattern,'pattern_paired.csv',['p_t','p_wilcoxon'])
    av = save_tests(anova,'repeated_measures_anova.csv',['p_GG','p_friedman'])
    save_tests(exp_tests,'exposure_paired.csv',['p_t','p_wilcoxon'])
    manifest = {f:hashlib.sha256((RES/f).read_bytes()).hexdigest() for f in inputs}
    (OUT/'inputs_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines = ['# Significance analysis of saved BraTS TEST results', '',
        'CPU only: no torch, inference, checkpoint loading, or training changes. 51 patients; mask repetitions averaged within patient. Complete paired data required; missing/nonfinite cells cause failure.', '',
        'Primary endpoint: (clustered − uniform coverage) for learned-calibrated minus post-hoc intervals around the same mean. Negative values mean the learned intervals lose more coverage. Two-sided paired t tests, standardized paired effect dz, t and percentile paired-bootstrap 95% CIs (10,000 draws, seed 20261023), and Wilcoxon sensitivity tests. CIs are pointwise, not multiplicity adjusted.', '',
        'Holm correction is separate for the two architecture tests at ×4 and the two confirmation tests at ×8. The endpoint was prespecified, but these added significance tests and multiplicity families are retrospective. Other analyses are exploratory. Each exploratory CSV family is corrected across all its rows, separately for each test type.', '',
        'ANOVA tests the three sampling patterns within patients, separately per arm/method, stride, and metric. Greenhouse–Geisser corrected p-values are used regardless of sphericity, with Friedman sensitivity tests. It does not establish pairwise differences between models. Exposure tests compare paired coverage changes against the corresponding uniform-trained model.', '',
        'Inference is across test patients conditional on these fitted models, not across training seeds. Coverage must be interpreted alongside interval width; a smaller coverage drop alone does not establish overall superiority. Normality of patient differences/residuals is an assumption of parametric tests; nonparametric sensitivity results are supplied.', '',
        '## Main endpoint results', '',
        '| Architecture | Stride | Difference (percentage points) | Bootstrap 95% CI (pp) | Holm p (t) | Holm p (Wilcoxon) |',
        '|---|---:|---:|---:|---:|---:|']
    for row in p.itertuples():
        lines.append(f'| {row.arch} | {row.R} | {100*row.difference:.2f} | [{100*row.ci_boot_lo:.2f}, {100*row.ci_boot_hi:.2f}] | {row.p_t_holm:.3g} | {row.p_wilcoxon_holm:.3g} |')
    lines += ['', f'ANOVA: {len(av)} tests; {int((av.p_GG_holm<.05).sum())} significant at family-adjusted 0.05. Full effect sizes, degrees of freedom and corrected p-values are in repeated_measures_anova.csv.', '',
        'Method reference: https://pingouin-stats.org/generated/pingouin.rm_anova.html (balanced repeated-measures sums of squares and GG correction).', '',
        'Reproduce: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python for_ISBI/scripts/significance.py`']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(p.to_string(index=False))
    print(f'Wrote {OUT}; {len(av)} ANOVAs')


if __name__ == '__main__':
    main()
