"""Covariance estimator reference vectors.

Covers the three baseline data regimes plus a small-scale OAS guard:

* ``small_well_conditioned`` — N=8, T=120, concentration c = N/T ≈ 0.067.
* ``moderate`` — N=30, T=60, c = 0.5.
* ``overconcentrated`` — N=50, T=40, c = 1.25 (more assets than
  observations; sample covariance is rank-deficient).
* ``small_scale_oas`` — N=8, T=120 with small return units, guarding the
  scale-relative OAS degeneracy branch.

For each regime we emit reference vectors for thirteen estimators:

* Baseline estimators:
  * ``sample`` — numpy.cov(ddof=1).
  * ``ledoit_wolf_scaled_identity`` — Ledoit-Wolf 2004 JMVA eq. 14 (sklearn oracle).
  * ``ewma_lambda_0_94`` — RiskMetrics exponentially-weighted cov.

* Phase 2 additions (numpy ports of the stated methods):
  * ``oas`` — Chen et al. 2010, Theorem 3 (OAS shrinkage).
  * ``lw_constant_correlation`` — Ledoit & Wolf 2004, constant-correlation target.
  * ``lw_single_factor`` — Ledoit & Wolf 2003, one-factor (equal-weight market) target.
  * ``denoised`` — López de Prado 2020, Marčenko-Pastur eigenvalue denoising (snippet 2.5).
  * ``detoned`` — López de Prado 2020, market-component removal and rescaling (section 2.6, α=1).
  * ``tracy_widom_denoised`` — Johnstone 2001 / Bun-Bouchaud-Potters 2017, TW finite-N 99% threshold.
  * ``qis`` — Ledoit & Wolf 2022, Quadratic Inverse Shrinkage.
  * ``nercome`` — Lam (2016), replicated matrix averaging and seven-candidate split selection.
  * ``poet`` — Fan-Liao-Mincheva 2013, principal-orthogonal-complement thresholding
    (derived from the paper's eq. 3.2 adaptive threshold, not ported from the C#).
  * ``doubly_sparse`` — Doubly Sparse Eigenvector Hard-Thresholding.

Library pins (documented so regression detection is decoupled from a
library update shifting the reference):

* numpy==2.x
* scipy==1.13+ (for numerically stable eigendecomposition routines)

Seed fixed per regime; regenerating produces bit-identical JSON.

The functions port the stated procedures to numpy. Each function is prefixed ``reference_`` so the port
is identifiable as the test's source of truth; source citations live above
each function.
"""

from __future__ import annotations

import math
import json
import sys
from pathlib import Path

import numpy as np

from conftest import save_vector


# =====================================================================
#  Generic helpers — shared baseline with the C# CovarianceHelpers class.
# =====================================================================


def sample_covariance_unbiased(returns: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance (T−1 divisor). Matches
    ``CovarianceHelpers.ComputeSampleCovariance``."""

    return np.cov(returns, rowvar=False, ddof=1)


def covariance_to_correlation(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (correlation, std-devs). Zero-variance assets get a 1 on
    the diagonal and 0 elsewhere, matching
    ``CovarianceHelpers.CovarianceToCorrelation``."""

    n = cov.shape[0]
    stds = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    corr = np.zeros_like(cov)
    for i in range(n):
        for j in range(n):
            if stds[i] == 0.0 or stds[j] == 0.0:
                corr[i, j] = 1.0 if i == j else 0.0
            else:
                corr[i, j] = cov[i, j] / (stds[i] * stds[j])
    return corr, stds


def correlation_to_covariance(corr: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Rebuild covariance from correlation + per-asset standard deviations."""

    return corr * np.outer(stds, stds)


def marcenko_pastur_upper(t: int, n: int) -> float:
    """λ₊ = (1 + √(N/T))² — the MP upper spectral edge for the empirical
    eigenvalue distribution of a sample correlation matrix under IID noise
    (Marčenko & Pastur 1967). ``q`` in the LdP book is T/N; equivalent."""

    if t <= 0:
        return float("inf")
    sqrt_inv_q = np.sqrt(n / t)
    return float((1.0 + sqrt_inv_q) ** 2)


def eigendecompose_sorted_ascending(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition with eigenvalues sorted ascending —
    C# Jacobi orders descending.
    Returns (values, vectors) where vectors[:, k] is the eigenvector for
    values[k].

    Consistent eigenpair permutation preserves reconstructed matrices; sign
    is not canonical but all downstream reconstructions ``V · diag(λ) · Vᵀ``
    and the eigenvector-magnitude operations used by the estimators that
    live here are sign-invariant."""

    values, vectors = np.linalg.eigh(matrix)
    # np.linalg.eigh returns ascending already — explicit reassertion for clarity.
    order = np.argsort(values)
    return values[order], vectors[:, order]


# =====================================================================
#  Baseline references for SampleCovariance, LedoitWolf, and EWMA.
#  EWMA uses the production contract's exponentially weighted mean.
# =====================================================================


def reference_ledoit_wolf_scaled_identity(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) linear shrinkage to the scaled identity.

    Reference: Ledoit & Wolf (2004), "A well-conditioned estimator for
    large-dimensional covariance matrices", JMVA 88(2), 365–411, eq. 14 and
    Lemma 3.4: the intensity is ``δ = min(b̄², d²)/d²`` — there is no ρ term.

    Independent oracle: ``sklearn.covariance.ledoit_wolf`` implements that
    intensity on the 1/T moment matrix of the demeaned data. The library
    returns the same estimator on its Bessel (T−1) scale, so the oracle output
    is multiplied by T/(T−1); the intensity itself is unchanged by that factor.
    """

    from sklearn.covariance import ledoit_wolf

    t = returns.shape[0]
    shrunk, _ = ledoit_wolf(returns)
    return shrunk * t / (t - 1)


def reference_ewma(returns: np.ndarray, lam: float) -> np.ndarray:
    """RiskMetrics EWMA (matches ``ExponentiallyWeightedCovarianceEstimator``)."""

    t, n = returns.shape
    weights = np.array([(1 - lam) * lam ** (t - 1 - k) for k in range(t)])
    weights = weights / weights.sum()
    means = np.sum(weights[:, np.newaxis] * returns, axis=0)
    centered = returns - means

    cov = np.zeros((n, n))
    for k in range(t):
        v = centered[k : k + 1].T
        cov += weights[k] * (v @ v.T)

    return cov


# =====================================================================
#  Phase 2 references — numpy ports of the stated C# methods. Detoned is
#  the repository variant, not the book's market-component removal recipe.
# =====================================================================


def reference_oas(returns: np.ndarray) -> np.ndarray:
    """Oracle Approximating Shrinkage — Chen, Wiesel, Eldar & Hero
    (2010), "Shrinkage Algorithms for MMSE Covariance Estimation",
    IEEE Trans. Signal Processing 58(10), 5016–5029, Theorem 3.

    Formula:
        ρ* = min(1, ((1 − 2/p)·tr(S²) + tr(S)²)
                    / ((n + 1 − 2/p)·(tr(S²) − tr(S)²/p)))
        Σ̂ = (1 − ρ*)·S + ρ*·(tr(S)/p)·I

    Known zero mean: S=X.T@X/n, without sample centering. The exact finite-p
    terms follow equation23; sklearn's simplified OAS omits the 2/p terms.
    This direct primary-equation oracle is separate from the NERCOME policy port.
    """

    t, p = returns.shape
    s = returns.T @ returns / t
    trace_s = float(np.trace(s))
    trace_s2 = float(np.sum(s ** 2))
    mu = trace_s / p
    numerator = (1.0 - 2.0 / p) * trace_s2 + trace_s ** 2
    dispersion = trace_s2 - trace_s ** 2 / p
    if dispersion <= 1e-12 * trace_s2:
        rho = 1.0
    else:
        denominator = (t + 1.0 - 2.0 / p) * dispersion
        rho = float(np.clip(numerator / denominator, 0.0, 1.0))
    target = mu * np.eye(p)
    return (1.0 - rho) * s + rho * target


def reference_lw_constant_correlation(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) shrinkage to the constant-correlation target.

    Reference: Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance
    Matrix", Journal of Portfolio Management 30(4), 110–119. Target from
    App. A eq. 3; intensity ``δ = max(0, min(κ̂/T, 1))`` with
    ``κ̂ = (π̂ − ρ̂)/γ̂`` and the App. B ``ρ̂`` (diagonal π̂_ii plus the
    r̄/2 · ϑ̂ cross terms). Written as a numpy port of the authors' published
    ``covCor.m`` matrix formulation (1/T moments of the demeaned data), not of
    the C# class. The library returns the estimator on its Bessel (T−1) scale,
    so the output is multiplied by T/(T−1).
    """

    t, n = returns.shape
    x = returns - returns.mean(axis=0)
    sample = x.T @ x / t
    var = np.diag(sample).copy()
    sqrtvar = np.sqrt(var)
    r_bar = (np.sum(sample / np.outer(sqrtvar, sqrtvar)) - n) / (n * (n - 1))
    prior = r_bar * np.outer(sqrtvar, sqrtvar)
    np.fill_diagonal(prior, var)

    # π̂: sum of asymptotic variances of the sample covariance entries.
    y = x ** 2
    phi_mat = y.T @ y / t - 2.0 * (x.T @ x) * sample / t + sample ** 2
    phi = float(phi_mat.sum())

    # ρ̂: App. B — ϑ̂_ii,ij = (1/T) Σ_t (x_it² − s_ii)(x_it x_jt − s_ij).
    term1 = (x ** 3).T @ x / t
    help_mat = x.T @ x / t
    help_diag = np.diag(help_mat)
    term2 = help_diag[:, None] * sample
    term3 = help_mat * var[:, None]
    term4 = var[:, None] * sample
    theta_mat = term1 - term2 - term3 + term4
    np.fill_diagonal(theta_mat, 0.0)
    rho = float(np.sum(np.diag(phi_mat))
                + r_bar * np.sum(np.outer(1.0 / sqrtvar, sqrtvar) * theta_mat))

    gamma = float(np.linalg.norm(sample - prior, "fro") ** 2)
    kappa = (phi - rho) / gamma
    shrinkage = max(0.0, min(1.0, kappa / t))
    sigma = shrinkage * prior + (1.0 - shrinkage) * sample
    return sigma * t / (t - 1)


def reference_lw_single_factor(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2003) shrinkage to the single-factor (market) target.

    Reference: Ledoit & Wolf (2003), "Improved Estimation of the Covariance
    Matrix of Stock Returns with an Application to Portfolio Selection",
    Journal of Empirical Finance 10(5), 603–621. Target ``f_ij = s_i0 s_j0/s_00``
    off the diagonal and ``f_ii = s_ii`` on it; intensity
    ``δ = max(0, min(1, (p − r)/(c·T)))`` [Theorem 2] with ``r`` from Lemma 2
    (``r_ii = p_ii``; off-diagonal eq. 8). Written as a numpy port of the
    authors' published ``covMarket.m`` matrix formulation (equal-weighted
    market of the demeaned returns, 1/T moments), not of the C# class. The
    library returns the estimator on its Bessel (T−1) scale, so the output is
    multiplied by T/(T−1).
    """

    t, n = returns.shape
    x = returns - returns.mean(axis=0)
    xmkt = x.mean(axis=1)

    full = np.cov(np.column_stack([x, xmkt]), rowvar=False) * (t - 1) / t
    covmkt = full[:n, n]
    varmkt = full[n, n]
    sample = full[:n, :n]
    prior = np.outer(covmkt, covmkt) / varmkt
    np.fill_diagonal(prior, np.diag(sample))

    c = float(np.linalg.norm(sample - prior, "fro") ** 2)
    y = x ** 2
    p = float(np.sum(y.T @ y) / t - np.sum(sample ** 2))
    rdiag = float(np.sum(y ** 2) / t - np.sum(np.diag(sample) ** 2))
    z = x * xmkt[:, None]
    v1 = y.T @ z / t - covmkt[:, None] * sample
    roff1 = float(np.sum(v1 * covmkt[None, :]) / varmkt
                  - np.sum(np.diag(v1) * covmkt) / varmkt)
    v3 = z.T @ z / t - varmkt * sample
    roff3 = float(np.sum(v3 * np.outer(covmkt, covmkt)) / varmkt ** 2
                  - np.sum(np.diag(v3) * covmkt ** 2) / varmkt ** 2)
    r = rdiag + 2.0 * roff1 - roff3
    k = (p - r) / c
    shrinkage = max(0.0, min(1.0, k / t))
    sigma = shrinkage * prior + (1.0 - shrinkage) * sample
    return sigma * t / (t - 1)


def cov2corr(cov: np.ndarray) -> np.ndarray:
    """Rescale to a unit diagonal and clip to [-1, 1] — López de Prado 2020,
    snippets 2.5/2.6 as implemented by emoen, *Machine-Learning-for-Asset-Managers*
    (``ch2_marcenko_pastur_pdf.py``, commit b9fd86b), ``cov2corr``."""

    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1
    return corr


def _denoise_core(
    returns: np.ndarray, threshold: float, detoning_alpha: float | None = None,
    clean_all_noise: bool = True
) -> np.ndarray:
    """Shared denoising kernel. ``threshold`` sets the noise/signal
    partition (MP edge, or the Tracy-Widom threshold). When ``detoning_alpha``
    is not None, additionally removes that fraction of the market component.

    Follows López de Prado 2020, snippet 2.5 (``denoisedCorr``) and section 2.6
    (``detoned_corr``) as implemented by emoen (commit b9fd86b): eigenvalues
    at or below the threshold are replaced by their mean, including all-noise
    spectra. The separate TW convention preserves its all-noise spectrum. The
    reconstruction is rescaled with ``cov2corr``, and detoning subtracts the
    first eigenpair of the *denoised* correlation before a second ``cov2corr``.
    ``detoning_alpha = 1`` is the book's recipe; removing a market component
    leaves a singular matrix when the residual diagonal remains positive. With
    no signal eigenvalue there is no market component and nothing is removed.
    """

    t, n = returns.shape
    s = sample_covariance_unbiased(returns)

    if n < 3:
        return s

    corr, stds = covariance_to_correlation(s)
    values, vectors = eigendecompose_sorted_ascending(corr)

    noise_mask = values <= threshold
    noise_count = int(np.sum(noise_mask))

    if noise_count > 0 and (clean_all_noise or noise_count < n):
        new_values = np.where(noise_mask, float(values[noise_mask].mean()), values)
    else:
        new_values = values.copy()

    cleaned_corr = cov2corr(vectors @ np.diag(new_values) @ vectors.T)

    if detoning_alpha is not None and noise_count < n:
        # Eigenvalues are ascending, so the market component is the last eigenpair.
        d_values, d_vectors = eigendecompose_sorted_ascending(cleaned_corr)
        v1 = d_vectors[:, -1:]
        market = v1 @ np.diag(d_values[-1:]) @ v1.T
        cleaned_corr = cov2corr(cleaned_corr - detoning_alpha * market)

    return correlation_to_covariance(cleaned_corr, stds)


def fitted_mp_upper(returns: np.ndarray) -> float:
    """Book snippets 2.1/2.4: 1000 MP ordinates vs Gaussian KDE, h=.01.

    SciPy differential evolution locates the global fit independently of the
    library's grid/golden-section implementation. Include endpoints explicitly.
    """
    from scipy.optimize import differential_evolution

    t, n = returns.shape
    q = t / n
    edge = (1 + np.sqrt(1 / q)) ** 2
    if q == 1:
        return edge
    eigenvalues = np.linalg.eigvalsh(covariance_to_correlation(sample_covariance_unbiased(returns))[0])
    z = np.linspace((1 - np.sqrt(1 / q)) ** 2, edge, 1000)

    def objective(parameter):
        variance = float(parameter[0])
        x = variance * z
        pdf = q / (2 * np.pi * variance * x) * np.sqrt(np.maximum(0, (x[-1] - x) * (x - x[0])))
        kde = np.exp(-0.5 * ((x[:, None] - eigenvalues) / .01) ** 2).mean(axis=1) / (.01 * np.sqrt(2 * np.pi))
        return float(np.sum((pdf - kde) ** 2))

    fit = differential_evolution(objective, [(1e-5, 1 - 1e-5)], seed=731, tol=1e-10, popsize=30)
    candidates = [1e-5, 1 - 1e-5, float(fit.x[0])]
    variance = min(candidates, key=lambda x: objective([x])) if fit.success else 1.0
    return variance * edge


def reference_denoised(returns: np.ndarray) -> np.ndarray:
    """López de Prado 2020, §§2.4–2.5 — fitted MP-edge denoising, no detoning.

    Reference: López de Prado, M. (2020), *Machine Learning for Asset
    Managers*, snippet 2.5.
    """

    t, n = returns.shape
    threshold = fitted_mp_upper(returns)
    return _denoise_core(returns, threshold, detoning_alpha=None)


def reference_detoned(returns: np.ndarray, detoning_alpha: float = 1.0) -> np.ndarray:
    """MP-denoised correlation with the market component removed (α = 1 is the book's recipe).

    Reference: López de Prado, M. (2020), *Machine Learning for Asset
    Managers*, section 2.6.
    """

    t, n = returns.shape
    threshold = fitted_mp_upper(returns)
    return _denoise_core(returns, threshold, detoning_alpha=detoning_alpha)


def reference_tracy_widom_denoised(returns: np.ndarray) -> np.ndarray:
    """Tracy-Widom finite-N corrected denoising at the 99% level.

    Reference: Johnstone (2001), "On the Distribution of the Largest
    Eigenvalue in Principal Components Analysis", Annals of Statistics
    29(2), 295–327, eqs. (1.3)–(1.4) and Theorem 1.1:

        mu_np    = (sqrt(n_obs - 1) + sqrt(p))**2
        sigma_np = (sqrt(n_obs - 1) + sqrt(p))
                   * (1/sqrt(n_obs - 1) + 1/sqrt(p))**(1/3)

    where n_obs is the observation count and p the variable count. For n_obs < p,
    reverse their roles in mu and sigma (p.300 after Theorem 1.1). Dividing
    by n_obs maps the relation onto the correlation-eigenvalue scale:

        threshold = mu_np / n_obs + (sigma_np / n_obs) * s_99

    with s_99 = F1^-1(0.99) = 2.0234 (Johnstone Table 1: 2.02). The TW mean
    is carried by mu_np, so no |E[TW1]| factor enters. This is an
    INDEPENDENT NumPy derivation from the paper formula, not a port of C# output.
    """

    t, n = returns.shape  # t = observations (Johnstone n_obs), n = variables (p)
    s_99 = 2.0234
    sqrt_nm1 = math.sqrt(max(t, n) - 1)
    sqrt_p = math.sqrt(min(t, n))
    mu_np = (sqrt_nm1 + sqrt_p) ** 2
    sigma_np = (sqrt_nm1 + sqrt_p) * (1.0 / sqrt_nm1 + 1.0 / sqrt_p) ** (1.0 / 3.0)
    tw = mu_np / t + (sigma_np / t) * s_99
    return _denoise_core(returns, tw, detoning_alpha=None, clean_all_noise=False)


def reference_qis(returns: np.ndarray) -> np.ndarray:
    """Quadratic-Inverse Shrinkage — Ledoit & Wolf (2022), "Quadratic
    shrinkage for large covariance matrices", Bernoulli 28(3), 1519–1547.

    Numpy port of the paper's own Appendix A code (``QIS(Y)`` with the default
    ``k = 1``), not of the C# class: demean, ``n = N − 1``, ``c = p/n``,
    ``sample = YᵀY/n``; ascending eigenvalues; bandwidth
    ``h = min(c², 1/c²)^0.35 / p^0.35``; smoothed Stein shrinker ``theta`` and
    its conjugate ``Htheta`` over the non-null inverse eigenvalues; null
    eigenvalues (p > n) receive ``δ₀ = 1/((c − 1)·mean(1/λ))``; trace rescale.
    """

    big_n, p = returns.shape
    y = returns - returns.mean(axis=0)
    n = big_n - 1
    c = p / n
    sample = y.T @ y / n
    sample = (sample + sample.T) / 2.0
    lam, u = np.linalg.eigh(sample)
    order = np.argsort(lam)
    lam = lam[order]
    u = u[:, order]

    h = min(c ** 2, 1.0 / c ** 2) ** 0.35 / p ** 0.35
    invlambda = 1.0 / lam[max(1, p - n + 1) - 1 : p]
    m = min(p, n)
    lj = np.tile(invlambda, (m, 1))        # lj[i, j] = 1/λ_j
    lj_i = lj - lj.T                       # 1/λ_j − 1/λ_i
    denom = lj_i ** 2 + h ** 2 * lj ** 2
    theta = np.mean(lj * lj_i / denom, axis=1)
    htheta = np.mean(lj * (h * lj) / denom, axis=1)
    atheta2 = theta ** 2 + htheta ** 2

    if p <= n:
        delta = 1.0 / ((1.0 - c) ** 2 * invlambda
                       + 2.0 * c * (1.0 - c) * invlambda * theta
                       + c ** 2 * invlambda * atheta2)
    else:
        delta0 = 1.0 / ((c - 1.0) * np.mean(invlambda))
        delta = np.concatenate([np.repeat(delta0, p - n), 1.0 / (invlambda * atheta2)])

    delta_qis = delta * (np.sum(lam) / np.sum(delta))
    return u @ np.diag(delta_qis) @ u.T


class _NercomePermutationStream:
    """Xoshiro256** with SplitMix64 expansion; reproducible library seed policy.

    The integer generator is a cross-language policy implementation, not a paper oracle.
    """

    _MASK = (1 << 64) - 1

    def __init__(self, seed: int):
        self.state = []
        for _ in range(4):
            seed = (seed + 0x9E3779B97F4A7C15) & self._MASK
            z = seed
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self._MASK
            self.state.append(z ^ (z >> 31))

    @classmethod
    def _rotate(cls, value: int, shift: int) -> int:
        return ((value << shift) | (value >> (64 - shift))) & cls._MASK

    def next_u64(self) -> int:
        a, b, c, d = self.state
        result = self._rotate((b * 5) & self._MASK, 7) * 9 & self._MASK
        temporary = (b << 17) & self._MASK
        c ^= a
        d ^= b
        b ^= c
        a ^= d
        c ^= temporary
        d = self._rotate(d, 45)
        self.state = [a, b, c, d]
        return result

    def next_int(self, upper: int) -> int:
        threshold = (1 << 32) % upper
        while True:
            product = (self.next_u64() >> 32) * upper
            if (product & 0xFFFFFFFF) >= threshold:
                return product >> 32


def reference_nercome(returns: np.ndarray, split_fraction: float | None = None,
                      replications: int = 50, seed: int = 0) -> np.ndarray:
    """Cross-language benchmark of Lam eq4.5–4.9 with declared finite policies.

    Zero-mean /n subset moments; distinct Fisher-Yates first subsets; full matrix
    averaging in each own basis; squared averaged residual; seven-candidate argmin.
    Floor/clamp/dedup and fixed Xoshiro seed are library policies. This numerical
    port is not independent paper-fidelity evidence: exact rational 2x2 projector
    tests separately establish the estimator and selector on exhaustive subsets.
    """
    count, dimension = returns.shape
    scale = float(np.max(np.abs(returns)))
    if scale == 0:
        return np.zeros((dimension, dimension))
    observations = returns / scale
    root = math.sqrt(count)
    locations = ([2 * root, .2 * count, .4 * count, .6 * count, .8 * count,
                  count - 2.5 * root, count - 1.5 * root]
                 if split_fraction is None else [split_fraction * count])
    candidates = list(dict.fromkeys(max(1, min(math.floor(x), count - 1)) for x in locations))
    stream = _NercomePermutationStream(seed)
    best = None
    best_score = math.inf
    for size in candidates:
        repetitions = min(replications, math.comb(count, size))
        seen = set()
        total = np.zeros((dimension, dimension))
        residual = np.zeros_like(total)
        while len(seen) < repetitions:
            order = list(range(count))
            for index in range(count - 1, 0, -1):
                other = stream.next_int(index + 1)
                order[index], order[other] = order[other], order[index]
            key = tuple(sorted(order[:size]))
            if key in seen:
                continue
            seen.add(key)
            first = observations[order[:size]]
            second = observations[order[size:]]
            a = first.T @ first / size
            b = second.T @ second / (count - size)
            _, vectors = np.linalg.eigh(a)
            diagonal = np.maximum(np.diag(vectors.T @ b @ vectors), 0)
            reconstructed = (vectors * diagonal) @ vectors.T
            total += reconstructed / repetitions
            residual += (reconstructed - b) / repetitions
        score = float(np.sum(residual * residual))
        if best is None or score < best_score:
            best, best_score = total, score
    return (best * scale) * scale


def reference_poet(returns: np.ndarray, num_factors: int = 1, threshold_multiplier: float = 0.5) -> np.ndarray:
    """Principal Orthogonal complEment Thresholding (POET), derived directly
    from the paper — not ported from the C#.

    Reference: Fan, Liao & Mincheva (2013), "Large Covariance Estimation
    by Thresholding Principal Orthogonal Complements", JRSS-B 75(4),
    603–680 (arXiv:1201.0175).

    * Factors by the paper's §2.3 PCA on the T×T matrix ``Y Yᵀ``: ``F̂`` is
      √T times its top-K eigenvectors, ``Λ̂ = Yᵀ F̂ / T`` and the residuals are
      ``û = Y − F̂ Λ̂ᵀ`` (this is a different computational route from
      eigendecomposing the N×N covariance, which the library does).
    * Adaptive entry-specific threshold, eq. (3.2):
      ``τ_ij = C · √θ̂_ij · ω_T`` with ``ω_T = 1/√p + √(log p / T)``,
      ``θ̂_ij = (1/T) Σ_t (û_it û_jt − σ̂_ij)²`` and
      ``σ̂_ij = (1/T) Σ_t û_it û_jt``; ``C = 0.5`` as in the paper's §5.
    * Soft thresholding of the residual off-diagonal (one of the paper's
      permitted generalized thresholding rules), diagonal kept (eq. 2.5).

    Library convention: covariances use the Bessel ``T − 1`` divisor, so the
    factor part is ``Yᵀ P_F Y / (T − 1)`` and the thresholded residual is
    ``û ᵀ û / (T − 1)``; ``θ̂`` keeps the paper's ``1/T`` averages,
    and the threshold is multiplied by T/(T−1) to use the same covariance scale.
    """

    t, n = returns.shape
    y = returns - returns.mean(axis=0)
    k = num_factors

    gram_values, gram_vectors = np.linalg.eigh(y @ y.T)
    order = np.argsort(gram_values)[::-1]
    f_hat = math.sqrt(t) * gram_vectors[:, order[:k]]  # T×K, F̂ᵀF̂ / T = I
    lam_hat = y.T @ f_hat / t  # N×K
    u_hat = y - f_hat @ lam_hat.T  # T×N residuals

    factor_cov = (lam_hat @ lam_hat.T) * t / (t - 1)
    residual = u_hat.T @ u_hat / (t - 1)

    omega = 1.0 / math.sqrt(n) + math.sqrt(math.log(n) / t)
    products = u_hat[:, :, None] * u_hat[:, None, :]  # T×N×N of û_it û_jt
    sigma_hat = products.mean(axis=0)
    theta_hat = ((products - sigma_hat) ** 2).mean(axis=0)
    tau = threshold_multiplier * np.sqrt(theta_hat) * omega * t / (t - 1)

    thresholded = np.sign(residual) * np.maximum(np.abs(residual) - tau, 0.0)
    np.fill_diagonal(thresholded, np.diag(residual))

    result = factor_cov + thresholded
    return 0.5 * (result + result.T)


def reference_doubly_sparse(returns: np.ndarray, eigenvector_threshold: float = 0.1) -> np.ndarray:
    """Doubly Sparse Covariance Estimator — hard-thresholds eigenvector
    entries on the signal subspace and replaces noise eigenvalues with
    their mean. Matches the C# ``DoublySparseEstimator``.

    Pipeline (verbatim from the C# implementation):
      1. sample cov → correlation
      2. eigen → partition by MP edge
      3. For noise eigenvalues: replace with noise mean (ALWAYS — no
         ``0 < |noise| < n`` guard, unlike the Denoised variant); keep
         the noise eigenvector unchanged.
      4. For signal eigenvalues: keep as-is; hard-threshold eigenvector
         entries with ``|entry| < ε`` and re-normalise the column.
      5. Reconstruct ``V · diag(λ) · Vᵀ`` — NO unit-diagonal forcing
         (C# goes directly from corrReconstructed to covariance).
      6. Covariance = diag(σ) · corrReconstructed · diag(σ).

    Sign-invariant: hard-thresholding uses ``|v_{ik}|``, and the final
    reconstruction is sign-invariant.
    """

    t, n = returns.shape
    s = sample_covariance_unbiased(returns)
    if n < 2:
        return s

    corr, stds = covariance_to_correlation(s)
    values, vectors = eigendecompose_sorted_ascending(corr)

    mp = marcenko_pastur_upper(t, n)
    noise_mask = values <= mp
    noise_count = int(np.sum(noise_mask))
    noise_avg = float(values[noise_mask].mean()) if noise_count > 0 else 0.0

    new_values = np.where(noise_mask, noise_avg, values)

    # Signal eigenvectors: hard-threshold + re-normalise. Noise eigenvectors:
    # unchanged (C# copies them over as-is in the ``else`` branch).
    new_vectors = vectors.copy()
    for k in range(n):
        if noise_mask[k]:
            # Noise: keep eigenvector unchanged.
            continue
        col = new_vectors[:, k]
        mask = np.abs(col) >= eigenvector_threshold
        thresholded = np.where(mask, col, 0.0)
        norm = np.linalg.norm(thresholded)
        if norm > 0.0:
            new_vectors[:, k] = thresholded / norm
        # Preserve the original signal direction if thresholding removes every entry.
        else:
            new_vectors[:, k] = col

    # Clip tiny-negative eigenvalues that can arise from rank-deficient samples.
    new_values = np.maximum(new_values, 0.0)

    cleaned_corr = new_vectors @ np.diag(new_values) @ new_vectors.T
    # NO unit-diagonal forcing — C# goes straight from cleanCorr to covariance.
    return correlation_to_covariance(cleaned_corr, stds)


# =====================================================================
#  Regime generation + orchestrator.
# =====================================================================


def _build_returns(seed: int, t: int, n: int) -> np.ndarray:
    """Synthetic Gaussian returns with unit variance per asset."""

    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.01, size=(t, n))


def _build_regime_from_returns(
    seed: int,
    returns: np.ndarray,
    extra: dict | None = None,
) -> dict:
    t, n = returns.shape
    sample = sample_covariance_unbiased(returns)
    lw = reference_ledoit_wolf_scaled_identity(returns)
    ewma = reference_ewma(returns, lam=0.94)

    block: dict = {
        "t": t,
        "n": n,
        "seed": seed,
        "returns": returns.tolist(),
        "sample_covariance": sample.tolist(),
        "ledoit_wolf_scaled_identity": lw.tolist(),
        "ewma_lambda_0_94": ewma.tolist(),
        "oas": reference_oas(returns).tolist(),
        "lw_constant_correlation": reference_lw_constant_correlation(returns).tolist(),
        "lw_single_factor": reference_lw_single_factor(returns).tolist(),
        "denoised": reference_denoised(returns).tolist(),
        "detoned": reference_detoned(returns, detoning_alpha=1.0).tolist(),
        "tracy_widom_denoised": reference_tracy_widom_denoised(returns).tolist(),
        "qis": reference_qis(returns).tolist(),
        "nercome": reference_nercome(returns).tolist(),
        "poet": reference_poet(returns, num_factors=1, threshold_multiplier=0.5).tolist(),
        "doubly_sparse": reference_doubly_sparse(returns, eigenvector_threshold=0.1).tolist(),
    }
    if extra:
        block.update(extra)
    return block


def _build_regime(name: str, seed: int, t: int, n: int, extra: dict | None = None) -> dict:
    return _build_regime_from_returns(seed, _build_returns(seed, t, n), extra)


def _build_small_scale_oas_regime() -> dict:
    base = _build_returns(seed=20260919, t=120, n=8)
    returns = base * 1e-7
    sample = sample_covariance_unbiased(returns)
    oas = reference_oas(returns)
    ratio = oas[0, 1] / sample[0, 0]
    if ratio == 0.0:
        raise AssertionError("small-scale OAS oracle collapsed a non-zero covariance ratio")

    return _build_regime_from_returns(
        seed=20260919,
        returns=returns,
        extra={"oas_scale_ratio_0_1": float(ratio)},
    )


def generate() -> None:
    regimes = {
        "small_well_conditioned": _build_regime("small_well_conditioned", seed=123, t=120, n=8),
        "small_scale_oas": _build_small_scale_oas_regime(),
        "moderate": _build_regime("moderate", seed=20260415, t=60, n=30),
        "overconcentrated": _build_regime("overconcentrated", seed=20260416, t=40, n=50),
    }

    # Keep the top-level compatibility keys that the existing three tests read
    # (SampleCovariance, LedoitWolf, EWMA), backed by the original (T=120, N=5)
    # input block. Sample/Ledoit-Wolf retain their values; EWMA is regenerated
    # with the corrected exponentially weighted mean.
    rng = np.random.default_rng(seed=123)
    legacy_returns = rng.normal(scale=0.01, size=(120, 5))
    legacy_sample = np.cov(legacy_returns, rowvar=False, ddof=1)
    legacy_lw = reference_ledoit_wolf_scaled_identity(legacy_returns)
    legacy_ewma = reference_ewma(legacy_returns, lam=0.94)

    save_vector(
        "covariance",
        {
            "library_pins": {"numpy": "2", "scipy": "1.13+"},
            # Compatibility top-level keys retained for the original tests.
            "returns": legacy_returns.tolist(),
            "sample_covariance": legacy_sample.tolist(),
            "ledoit_wolf_scaled_identity": legacy_lw.tolist(),
            "ewma_lambda_0_94": legacy_ewma.tolist(),
            # Phase 2 additions — per-regime blocks with all estimators.
            "regimes": regimes,
        },
    )


def refresh_selected_contract_vectors() -> None:
    """Refresh only OAS/NERCOME fields, preserving original panels and all other vectors."""
    path = Path(__file__).parent / "vectors" / "covariance.json"
    vectors = json.loads(path.read_text())
    for name, regime in vectors["regimes"].items():
        observations = np.asarray(regime["returns"], dtype=float)
        regime["oas"] = reference_oas(observations).tolist()
        regime["nercome"] = reference_nercome(observations).tolist()
        if "oas_scale_ratio_0_1" in regime:
            regime["oas_scale_ratio_0_1"] = regime["oas"][0][1] / regime["sample_covariance"][0][0]
    vectors["covariance_contract_provenance"] = {
        "numpy": np.__version__,
        "oas": "Chen2010 eq2 and23: known-zero-mean raw moment/n; exact2/p terms",
        "nercome": "Lam2016 eq4.5-4.9 cross-language policy benchmark; independent fidelity is tested separately by rational exhaustive projectors",
        "nercome_policy": "M50 capped at C(n,m); Xoshiro256** seed0 reset; distinct Fisher-Yates subsets; floor/clamp[1,n-1]/dedup in published order; first score tie",
        "unchanged_inputs": "Original checked-in returns panels and all unrelated estimator vectors retained"
    }
    save_vector("covariance", vectors)


if __name__ == "__main__":
    if "--refresh-selected-contracts" in sys.argv:
        refresh_selected_contract_vectors()
    else:
        generate()
    print("Wrote covariance.json")
