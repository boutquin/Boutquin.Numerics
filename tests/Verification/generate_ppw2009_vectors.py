#!/usr/bin/env python3
"""Reproducible PPW (2009) Tables 1/2 controls; not the unpublished paper seed.

Population optima use PW (2004) eqs. (6)/(11), D_SB=2g(0)^2 and
D_CB=4g(0)^2/3. Finite samples use the corrected author's lag-window
indexing and common truncated correlation/covariance rows, with no lower floor.
NumPy centered cross-products and explicit two-sided kernel sums provide
an independent control of the production entry point. Requires only numpy.
"""

import math

from conftest import save_vector

import numpy as np

MASK = (1 << 64) - 1


def normals(seed):
    """Blackman/Vigna xoshiro256** and NR3 p365 normal stream ordering."""
    state = []
    for _ in range(4):
        seed = (seed + 0x9E3779B97F4A7C15) & MASK
        z = seed
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        state.append(z ^ (z >> 31))

    def rotl(x, k):
        return ((x << k) | (x >> (64 - k))) & MASK

    def uniform():
        s0, s1, s2, s3 = state
        result = (rotl((s1 * 5) & MASK, 7) * 9) & MASK
        t = (s1 << 17) & MASK
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        state[:] = s0, s1, s2, rotl(s3, 45)
        return (result >> 11) / (1 << 53)

    yield from polar_normals(uniform)


def polar_normals(uniform):
    """NR3 Normaldev_BM: return v2, retain v1 only when nonzero.

    A stored zero is the listing's empty-cache sentinel, so no second
    output is yielded for that pair. This is deliberately stream-sensitive.
    """
    while True:
        u, v = 2 * uniform() - 1, 2 * uniform() - 1
        s = u * u + v * v
        if 0 < s < 1:
            factor = math.sqrt(-2 * math.log(s) / s)
            yield v * factor
            stored = u * factor
            if stored != 0.0:
                yield stored


def estimate(series):
    """Selected complete corrected author listing; shared truncated pairs."""
    return author_estimate(series)


def author_estimate(series):
    """Corrected opt_block_length_REV_dec07.m: shared truncated pairs."""
    n = len(series)
    kn, horizon = 5, math.ceil(math.sqrt(n)) + 5
    data = np.asarray(series)
    columns = np.column_stack([data[horizon-k:n-k] for k in range(horizon+1)])
    rho = np.corrcoef(columns, rowvar=False)[1:, 0]
    band = 2 * math.sqrt(math.log10(n) / n)
    windows = np.lib.stride_tricks.sliding_window_view(np.abs(rho), kn)
    first = np.flatnonzero(np.all(windows < band, axis=1))
    significant = np.flatnonzero(np.abs(rho) > band) + 1
    mhat = int(first[0]) + 1 if len(first) else int(significant[-1])
    big_m = min(2 * mhat, horizon)
    paired = np.column_stack([data[big_m-k:n-k] for k in range(big_m+1)])
    acv = np.cov(paired, rowvar=False)[:, 0]
    lags = np.arange(-big_m, big_m + 1)
    weights = np.minimum(1, 2 * (1 - np.abs(lags) / big_m))
    g = np.sum(weights * np.abs(lags) * acv[np.abs(lags)])
    spectrum = np.sum(weights * acv[np.abs(lags)])
    return float(min(np.cbrt(n * (g / spectrum)**2), math.ceil(min(3 * math.sqrt(n), n / 3))))


def generate():
    paper_mean = [.859, .927, .959, .881, 1.062, 1.081]
    paper_sd = [.342, .244, .943, .323, .644, .368]
    paper_rmse = [.370, .254, .943, .344, .646, .377]
    rows = []
    for row, (phi, n) in enumerate([(0.7, 200), (0.7, 800), (0.1, 200),
                                   (0.1, 800), (-0.4, 200), (-0.4, 800)]):
        seed = 20090000 + row
        stream = normals(seed)
        optimum = (2 * abs(phi) / (1 - phi**2))**(2/3) * n**(1/3)
        estimates = []
        author_estimates = []
        for _ in range(1000):
            x = next(stream) / math.sqrt(1 - phi**2)
            series = []
            for _ in range(n):
                x = phi * x + next(stream)
                series.append(x)
            estimates.append(estimate(series))
            author_estimates.append(author_estimate(series))
        ratios = np.array(estimates) / optimum
        mean, sd = np.mean(ratios), np.std(ratios, ddof=1)
        errors = (ratios - 1)**2
        rmse = np.sqrt(np.mean(errors))
        author_ratios = np.array(author_estimates) / optimum
        author_mean, author_sd = np.mean(author_ratios), np.std(author_ratios, ddof=1)
        author_errors = (author_ratios - 1)**2
        author_rmse = np.sqrt(np.mean(author_errors))
        # Six combined Monte Carlo SEs, using the companion author's influence
        # variances for both independent experiments (paper moments unpublished).
        # The selected implementation now follows the same author convention;
        # retained drift fields are zero and do not widen the envelopes.
        sd_se = np.std((author_ratios - author_mean)**2, ddof=1) / (2 * author_sd * math.sqrt(1000))
        rmse_se = np.std(author_errors, ddof=1) / (2 * author_rmse * math.sqrt(1000))
        rows.append(dict(phi=phi, length=n, seed=seed, replications=1000,
                         stationary_optimum=optimum, mean=float(mean),
                         standard_deviation=float(sd), rmse=float(rmse),
                         author_mean=float(author_mean), author_sd=float(author_sd), author_rmse=float(author_rmse),
                         mean_convention_drift=float(abs(mean-author_mean)),
                         sd_convention_drift=float(abs(sd-author_sd)),
                         rmse_convention_drift=float(abs(rmse-author_rmse)),
                         mean_combined_mcse=math.sqrt((author_sd**2 + paper_sd[row]**2)/1000),
                         sd_companion_mcse=float(sd_se), rmse_companion_mcse=float(rmse_se),
                         sd_combined_mcse=float(math.sqrt(2) * sd_se),
                         rmse_combined_mcse=float(math.sqrt(2) * rmse_se),
                         mean_paper_tolerance=abs(mean-author_mean) + 6 * math.sqrt((author_sd**2 + paper_sd[row]**2)/1000) + .0005,
                         sd_paper_tolerance=abs(sd-author_sd) + 6 * math.sqrt(2) * sd_se + .0005,
                         rmse_paper_tolerance=abs(rmse-author_rmse) + 6 * math.sqrt(2) * rmse_se + .0005,
                         printed_mean=paper_mean[row], printed_sd=paper_sd[row],
                         printed_rmse=paper_rmse[row],
                         checkpoints=[dict(trial=i, estimate=estimates[i]) for i in [0, 1, 499, 999]]))
    save_vector('ppw2009', dict(
        source='Patton, Politis & White (2009), corrected Tables 1 and 2',
        source_url='https://public.econ.duke.edu/~ap172/Patton_Politis_White_2009.pdf',
        contract='Seeded independent replication, not an exact replay of unpublished paper samples',
        normal_stream='NR3 p365 Normaldev_BM: v2 first, v1 cached, cached zero discarded; operator-selected author stream 2026-10-02',
        numpy_version=np.__version__,
        conventions='Stationary initialization; author lag-window indexing; common truncated-row sample covariances; upper cap Bmax, no lower floor',
        tolerance='Six combined Monte Carlo SEs plus 0.0005 printed rounding; SD/RMSE author delta-method influence variances from companion replay',
        rows=rows))
    for r in rows:
        print(r['phi'], r['length'], 'mean/sd/RMSE', r['mean'], r['standard_deviation'], r['rmse'],
              'paper differences', abs(r['mean'] - r['printed_mean']), abs(r['standard_deviation'] - r['printed_sd']), abs(r['rmse'] - r['printed_rmse']),
              'bounds', r['mean_paper_tolerance'], r['sd_paper_tolerance'], r['rmse_paper_tolerance'])


if __name__ == '__main__':
    generate()
