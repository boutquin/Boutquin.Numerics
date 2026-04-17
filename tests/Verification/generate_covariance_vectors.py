"""Covariance estimator reference vectors.

Covers three data regimes (spec §2.2):

* ``small_well_conditioned`` — N=8, T=120, concentration c = N/T ≈ 0.067.
* ``moderate`` — N=30, T=60, c = 0.5.
* ``overconcentrated`` — N=50, T=40, c = 1.25 (more assets than
  observations; sample covariance is rank-deficient).

For each regime we emit reference vectors for thirteen estimators:

* Legacy (unchanged from prior ship):
  * ``sample`` — numpy.cov(ddof=1).
  * ``ledoit_wolf_scaled_identity`` — own-formula Ledoit-Wolf-2004.
  * ``ewma_lambda_0_94`` — RiskMetrics exponentially-weighted cov.

* Phase 2 additions (paper-faithful numpy ports, spec §2.2):
  * ``oas`` — Chen et al. 2010, Theorem 3 (OAS shrinkage).
  * ``lw_constant_correlation`` — Ledoit & Wolf 2004, constant-correlation target.
  * ``lw_single_factor`` — Ledoit & Wolf 2003, one-factor (equal-weight market) target.
  * ``denoised`` — López de Prado 2018, Marčenko-Pastur eigenvalue denoising.
  * ``detoned`` — López de Prado 2020, MP-denoised + PC1 shrinkage (α=1).
  * ``tracy_widom_denoised`` — Johnstone 2001 / Bun-Bouchaud-Potters 2017, TW finite-N correction.
  * ``qis`` — Ledoit & Wolf 2022, Quadratic Inverse Shrinkage.
  * ``nercome`` — Abadir-Distaso-Žikeš 2014, leave-one-split rotated eigenvalues.
  * ``poet`` — Fan-Liao-Mincheva 2013, principal-orthogonal-complement thresholding.
  * ``doubly_sparse`` — Doubly Sparse Eigenvector Hard-Thresholding.

Library pins (documented so regression detection is decoupled from a
library update shifting the reference):

* numpy==2.x
* scipy==1.13+ (for numerically stable eigendecomposition routines)

Seed fixed per regime; regenerating produces bit-identical JSON.

All formulas are verbatim numpy ports of the published reference. Each
reference function is prefixed ``reference_`` so it is obvious the port
is the test's source of truth; paper citations live in the comment above
each function.
"""

from __future__ import annotations

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
    the convention the C# estimators' JacobiEigenDecomposition follows.
    Returns (values, vectors) where vectors[:, k] is the eigenvector for
    values[k].

    Eigenvalue order is identical between implementations (sorted); sign
    is not canonical but all downstream reconstructions ``V · diag(λ) · Vᵀ``
    and the eigenvector-magnitude operations used by the estimators that
    live here are sign-invariant."""

    values, vectors = np.linalg.eigh(matrix)
    # np.linalg.eigh returns ascending already — explicit reassertion for clarity.
    order = np.argsort(values)
    return values[order], vectors[:, order]


# =====================================================================
#  Legacy references — unchanged so that the existing three C# tests
#  (SampleCovariance, LedoitWolf, EWMA) keep passing.
# =====================================================================


def reference_ledoit_wolf_scaled_identity(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) with scaled-identity target, matching the C# one.

    Reference: Ledoit & Wolf (2004), "A well-conditioned estimator for
    large-dimensional covariance matrices", JMVA 88(2), 365–411.
    """

    t, n = returns.shape
    means = returns.mean(axis=0)
    centered = returns - means

    sample_cov = centered.T @ centered / (t - 1)
    mu = np.trace(sample_cov) / n
    target = mu * np.eye(n)

    gamma = float(np.sum((sample_cov - target) ** 2))

    pi_sum = 0.0
    for i in range(n):
        for j in range(n):
            sigma_ij = sample_cov[i, j]
            values = centered[:, i] * centered[:, j] - sigma_ij
            pi_sum += float(np.sum(values ** 2) / t)

    rho_sum = 0.0
    for i in range(n):
        sigma_ii = sample_cov[i, i]
        values = centered[:, i] ** 2 - sigma_ii
        rho_sum += float(np.sum(values ** 2) / t)

    if gamma == 0.0:
        delta = 1.0
    else:
        delta = (pi_sum - rho_sum) / (t * gamma)
    delta = float(np.clip(delta, 0.0, 1.0))

    return delta * target + (1 - delta) * sample_cov


def reference_ewma(returns: np.ndarray, lam: float) -> np.ndarray:
    """RiskMetrics EWMA (matches ``ExponentiallyWeightedCovarianceEstimator``)."""

    t, n = returns.shape
    means = returns.mean(axis=0)
    centered = returns - means

    weights = np.array([(1 - lam) * lam ** (t - 1 - k) for k in range(t)])
    weights = weights / weights.sum()

    cov = np.zeros((n, n))
    for k in range(t):
        v = centered[k : k + 1].T
        cov += weights[k] * (v @ v.T)

    return cov


# =====================================================================
#  Phase 2 references — each is a paper-faithful numpy port of the C#
#  implementation.
# =====================================================================


def reference_oas(returns: np.ndarray) -> np.ndarray:
    """Oracle Approximating Shrinkage — Chen, Wiesel, Eldar & Hero
    (2010), "Shrinkage Algorithms for MMSE Covariance Estimation",
    IEEE Trans. Signal Processing 58(10), 5016–5029, Theorem 3.

    Formula:
        ρ* = min(1, ((1 − 2/p)·tr(S²) + tr(S)²)
                    / ((n + 1 − 2/p)·(tr(S²) − tr(S)²/p)))
        Σ̂ = (1 − ρ*)·S + ρ*·(tr(S)/p)·I

    Note: sklearn's ``OAS`` differs because it uses the biased sample
    covariance (N divisor); Numerics uses the unbiased form (N-1), so a
    direct sklearn call will NOT match — this hand-port is the test's
    source of truth.
    """

    t, p = returns.shape
    s = sample_covariance_unbiased(returns)
    trace_s = float(np.trace(s))
    trace_s2 = float(np.sum(s ** 2))
    mu = trace_s / p
    numerator = (1.0 - 2.0 / p) * trace_s2 + trace_s ** 2
    denominator = (t + 1.0 - 2.0 / p) * (trace_s2 - trace_s ** 2 / p)
    if denominator <= 1e-28:
        rho = 1.0
    else:
        rho = float(np.clip(numerator / denominator, 0.0, 1.0))
    target = mu * np.eye(p)
    return (1.0 - rho) * s + rho * target


def reference_lw_constant_correlation(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) with constant-correlation target.

    Reference: Ledoit & Wolf (2004), "Honey, I Shrunk the Sample
    Covariance Matrix", Journal of Portfolio Management 30(4), 110–119.
    Schäfer-Strimmer analytical shrinkage form, as implemented in the C#
    class.
    """

    t, n = returns.shape
    means = returns.mean(axis=0)
    centered = returns - means
    s = centered.T @ centered / (t - 1)

    stds = np.sqrt(np.clip(np.diag(s), 0.0, np.inf))

    # Average pairwise sample correlation r̄ (zero-variance assets skipped).
    r_sum = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if stds[i] > 0.0 and stds[j] > 0.0:
                r_sum += s[i, j] / (stds[i] * stds[j])
                pair_count += 1
    r_bar = r_sum / pair_count if pair_count > 0 else 0.0

    # Target F.
    target = np.zeros((n, n))
    for i in range(n):
        target[i, i] = s[i, i]
        for j in range(i + 1, n):
            off = r_bar * stds[i] * stds[j]
            target[i, j] = off
            target[j, i] = off

    # π = Σᵢⱼ (1/T) · Σₖ (centered_ki · centered_kj − sᵢⱼ)².
    pi_sum = 0.0
    for i in range(n):
        for j in range(n):
            zij = centered[:, i] * centered[:, j]
            dev = zij - s[i, j]
            pi_sum += float(np.sum(dev ** 2) / t)

    gamma = float(np.sum((s - target) ** 2))
    delta = 1.0 if gamma == 0.0 else pi_sum / (t * gamma)
    delta = float(np.clip(delta, 0.0, 1.0))
    return delta * target + (1.0 - delta) * s


def reference_lw_single_factor(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2003) with one-factor (equal-weight market) target.

    Reference: Ledoit & Wolf (2003), "Improved Estimation of the
    Covariance Matrix of Stock Returns with an Application to Portfolio
    Selection", Journal of Empirical Finance 10(5), 603–621.
    """

    t, n = returns.shape
    means = returns.mean(axis=0)
    centered = returns - means
    s = centered.T @ centered / (t - 1)

    # Equal-weight market factor.
    market = returns.mean(axis=1)
    market_mean = market.mean()
    market_var = float(((market - market_mean) ** 2).sum() / (t - 1))

    # β_i = Cov(r_i, market) / Var(market) — sample cov with N-1 divisor.
    beta = np.zeros(n)
    for j in range(n):
        cov_j_m = float(((returns[:, j] - means[j]) * (market - market_mean)).sum() / (t - 1))
        beta[j] = cov_j_m / market_var if market_var != 0.0 else 0.0

    # Target F.
    target = np.zeros((n, n))
    for i in range(n):
        target[i, i] = s[i, i]
        for j in range(i + 1, n):
            off = beta[i] * beta[j] * market_var
            target[i, j] = off
            target[j, i] = off

    # Schäfer-Strimmer π and γ, same form as LW-CC.
    pi_sum = 0.0
    for i in range(n):
        for j in range(n):
            zij = centered[:, i] * centered[:, j]
            dev = zij - s[i, j]
            pi_sum += float(np.sum(dev ** 2) / t)

    gamma = float(np.sum((s - target) ** 2))
    delta = 1.0 if gamma == 0.0 else pi_sum / (t * gamma)
    delta = float(np.clip(delta, 0.0, 1.0))
    return delta * target + (1.0 - delta) * s


def _denoise_core(
    returns: np.ndarray, threshold: float, detoning_alpha: float | None = None
) -> np.ndarray:
    """Shared denoising kernel. ``threshold`` sets the noise/signal
    partition (MP edge, or MP + Tracy-Widom finite-N correction). When
    ``detoning_alpha`` is not None, additionally shrinks the PC1 (largest-
    eigenvalue) component toward the signal-eigenvalue mean.

    Mirrors the C# pipeline exactly: sample covariance → correlation →
    eigen → partition → replace noise with mean (iff 0 < |noise| < n)
    → detoning fallback logic (iff detoning requested) → reconstruct →
    force unit diagonal → covariance.

    Detoning semantics must match the C# ``DetonedCovarianceEstimator``:
    the detoning step runs whenever detoning is requested and there is at
    least one noise or signal eigenvalue to anchor the shrinkage — NOT
    guarded by the denoising guard. When all eigenvalues are noise, PC1
    is shrunk toward the noise mean; when at least one signal eigenvalue
    exists beyond PC1, PC1 is shrunk toward the mean of those signal
    eigenvalues.
    """

    t, n = returns.shape
    s = sample_covariance_unbiased(returns)

    if n < 3:
        return s

    corr, stds = covariance_to_correlation(s)
    values, vectors = eigendecompose_sorted_ascending(corr)

    noise_mask = values <= threshold
    noise_count = int(np.sum(noise_mask))
    noise_avg = float(values[noise_mask].mean()) if noise_count > 0 else 0.0

    # Denoising: replace noise eigenvalues with their mean, but only when
    # there is a mixed noise/signal split — if the spectrum is entirely
    # noise or entirely signal, the replacement would collapse information
    # with no statistical justification.
    if 0 < noise_count < n:
        new_values = np.where(noise_mask, noise_avg, values)
    else:
        new_values = values.copy()

    # Detoning: matches the C# Detoned pipeline's if / else-if branches.
    # Eigenvalues are ascending → PC1 (the largest) lives at index n − 1.
    if detoning_alpha is not None:
        pc1_idx = n - 1
        # Count signal eigenvalues EXCLUDING PC1, measured against the ORIGINAL
        # (not noise-replaced) eigenvalue array — equivalent to checking
        # values[i > 0 in descending] > threshold. Noise-replaced values are by
        # construction ≤ threshold, so the ``> threshold`` check on new_values
        # would also work; use original values for semantic clarity.
        signal_indices = [i for i in range(n) if i != pc1_idx and values[i] > threshold]
        if len(signal_indices) > 0:
            signal_mean = float(np.mean(new_values[signal_indices]))
            new_values[pc1_idx] = (
                (1.0 - detoning_alpha) * new_values[pc1_idx] + detoning_alpha * signal_mean
            )
        elif noise_count > 0:
            # Only PC1 qualifies as signal OR the spectrum is entirely noise —
            # shrink toward the noise mean.
            new_values[pc1_idx] = (
                (1.0 - detoning_alpha) * new_values[pc1_idx] + detoning_alpha * noise_avg
            )

    cleaned_corr = vectors @ np.diag(new_values) @ vectors.T
    np.fill_diagonal(cleaned_corr, 1.0)
    return correlation_to_covariance(cleaned_corr, stds)


def reference_denoised(returns: np.ndarray) -> np.ndarray:
    """López de Prado 2018, ch. 2 — MP-edge denoising, no detoning.

    Reference: López de Prado, M. (2018), *Advances in Financial Machine
    Learning*, chapter 2.
    """

    t, n = returns.shape
    threshold = marcenko_pastur_upper(t, n)
    return _denoise_core(returns, threshold, detoning_alpha=None)


def reference_detoned(returns: np.ndarray, detoning_alpha: float = 1.0) -> np.ndarray:
    """López de Prado 2020 — MP-denoised + PC1 shrinkage at intensity α.

    Reference: López de Prado, M. (2020), *Machine Learning for Asset
    Managers*, chapter 2.
    """

    t, n = returns.shape
    threshold = marcenko_pastur_upper(t, n)
    return _denoise_core(returns, threshold, detoning_alpha=detoning_alpha)


def reference_tracy_widom_denoised(returns: np.ndarray) -> np.ndarray:
    """Tracy-Widom finite-N corrected MP denoising.

    Reference: Johnstone (2001), "On the Distribution of the Largest
    Eigenvalue in Principal Components Analysis", Annals of Statistics
    29(2), 295–327. Bun-Bouchaud-Potters 2017 constants (c_α = 2.02,
    μ_TW = 1.21 for 95% threshold), matching the C# implementation.
    """

    t, n = returns.shape
    mp = marcenko_pastur_upper(t, n)
    c_alpha = 2.02
    mu_tw = 1.21
    tw = mp + (n ** (-2.0 / 3.0)) * c_alpha * mu_tw
    return _denoise_core(returns, tw, detoning_alpha=None)


def reference_qis(returns: np.ndarray) -> np.ndarray:
    """Quadratic Inverse Shrinkage — Ledoit & Wolf (2022), "Quadratic
    shrinkage for large covariance matrices", Bernoulli 28(3), 1519–1547
    (arXiv:1909.12522). Faithful port of the C# implementation.

    Pipeline (verbatim from the C# class):
      1. sample cov → correlation → eigen (ascending)
      2. effective_rank = min(n, t − 1) — the kernel is evaluated only
         over the top ``effective_rank`` eigenvalues; the bottom ``n −
         effective_rank`` are treated as noise and dropped, critical for
         the overconcentrated regime (t < n) where the bottom eigenvalues
         are numerically zero.
      3. For each eigenvalue λᵢ, density and Hilbert transform are
         estimated with bandwidth ``h·λᵢ`` (NOT ``h·λⱼ`` — the bandwidth
         is fixed per outer iteration).
      4. Shrunk eigenvalue: λᵢ / ((πc·λᵢ·f)² + (1 − c − πc·λᵢ·Hf)²).
      5. Reconstruct → force unit diagonal → covariance.
    """

    t, n = returns.shape
    s = sample_covariance_unbiased(returns)
    if n < 2:
        return s

    corr, stds = covariance_to_correlation(s)
    values, vectors = eigendecompose_sorted_ascending(corr)
    lam = np.asarray(values, dtype=float)

    c = n / t
    h = t ** (-0.35)
    effective_rank = min(n, t - 1)

    # Top-effective_rank eigenvalues (the trailing `effective_rank` in
    # ascending order). The bottom `n − effective_rank` noise eigenvalues
    # are excluded from both the density and Hilbert-transform sums.
    top_start = n - effective_rank

    shrunk = np.zeros(n, dtype=float)
    for i in range(n):
        ell = lam[i]
        density = 0.0
        hilbert = 0.0
        for j in range(effective_rank):
            ej = lam[top_start + j]
            if h * ell <= 0.0:
                continue
            u = (ell - ej) / (h * ell)
            # Quadratic-inverse kernel K(u) = √max(4 − u², 0) / (2π·u²).
            if abs(u) > 1e-12:
                disc = 4.0 - u * u
                if disc > 0.0:
                    density += np.sqrt(disc) / (2.0 * np.pi * u * u)
            # Hilbert transform (piecewise closed form, LW-2022 §3).
            if abs(u) >= 2.0:
                hilbert += (
                    np.sign(u) / (np.pi * u * u)
                    * (abs(u) * 0.5 * np.sqrt(u * u - 4.0) - 1.0)
                )
            elif abs(u) > 1e-12:
                hilbert += -1.0 / (np.pi * u * u)

        # Normalise: 1/(n · h · λᵢ). Note the divisor is ``n``, not
        # ``effective_rank``; matches the C# implementation.
        if h * ell > 0.0:
            density /= n * h * ell
            hilbert /= n * h * ell
        else:
            density = 0.0
            hilbert = 0.0

        a = np.pi * c * ell * density
        b = 1.0 - c - np.pi * c * ell * hilbert
        denom = a * a + b * b
        shrunk[i] = ell / denom if denom > 1e-18 else ell

    cleaned_corr = vectors @ np.diag(shrunk) @ vectors.T
    np.fill_diagonal(cleaned_corr, 1.0)
    return correlation_to_covariance(cleaned_corr, stds)


def reference_nercome(returns: np.ndarray, split_fraction: float = 0.5) -> np.ndarray:
    """Non-parametric eigenvalue regularized covariance matrix estimator
    (NERCOME) — Abadir, Distaso & Žikeš (2014), "Design-free estimation
    of variance matrices", Journal of Econometrics 181(2), 165–180.

    Pipeline: split rows into S₁ (first ⌊f·T⌋) and S₂ (remaining). Compute
    both sample covariances. Extract S₁'s eigenvectors V. Return
    V · diag(Vᵀ·S₂·V) · Vᵀ, flooring negatives at 0.
    """

    t, n = returns.shape
    # C# clamps split to [2, T-2] so both halves have ≥2 observations.
    split = int(np.floor(split_fraction * t))
    split = max(2, min(split, t - 2))

    s1 = sample_covariance_unbiased(returns[:split])
    s2 = sample_covariance_unbiased(returns[split:])

    _, v = eigendecompose_sorted_ascending(s1)
    # Rotated diag: d_k = v_k · S₂ · v_k (with v_k the k-th eigenvector column).
    d = np.diag(v.T @ s2 @ v)
    d = np.maximum(d, 0.0)
    return v @ np.diag(d) @ v.T


def reference_poet(returns: np.ndarray, num_factors: int = 1, threshold_multiplier: float = 0.5) -> np.ndarray:
    """Principal Orthogonal complEment Thresholding (POET).

    Reference: Fan, Liao & Mincheva (2013), "Large Covariance Estimation
    by Thresholding Principal Orthogonal Complements", JRSS-B 75(4),
    603–680 (arXiv:1201.0175).

    Pipeline: sample cov → eigen → keep top-K factors → soft-threshold
    off-diagonals of the residual at τ = c·√(log N / T).
    """

    t, n = returns.shape
    s = sample_covariance_unbiased(returns)
    values, vectors = eigendecompose_sorted_ascending(s)

    # Top-K eigenvalues are the LAST K in ascending order.
    k = num_factors
    top_values = values[n - k :]
    top_vectors = vectors[:, n - k :]

    factor_cov = top_vectors @ np.diag(top_values) @ top_vectors.T
    residual = s - factor_cov

    tau = threshold_multiplier * np.sqrt(np.log(n) / t)

    # Soft-threshold OFF-diagonals only (diagonal preserved).
    thresholded = residual.copy()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x = residual[i, j]
            thresholded[i, j] = np.sign(x) * max(0.0, abs(x) - tau)

    result = factor_cov + thresholded
    # Enforce symmetry (the C# implementation symmetrises at the end).
    result = 0.5 * (result + result.T)
    return result


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
        # else: C# leaves the column zero (sum of squares is 0; no renormalise).
        else:
            new_vectors[:, k] = thresholded

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


def _build_regime(name: str, seed: int, t: int, n: int, extra: dict | None = None) -> dict:
    returns = _build_returns(seed, t, n)
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
        "nercome": reference_nercome(returns, split_fraction=0.5).tolist(),
        "poet": reference_poet(returns, num_factors=1, threshold_multiplier=0.5).tolist(),
        "doubly_sparse": reference_doubly_sparse(returns, eigenvector_threshold=0.1).tolist(),
    }
    if extra:
        block.update(extra)
    return block


def generate() -> None:
    regimes = {
        "small_well_conditioned": _build_regime("small_well_conditioned", seed=123, t=120, n=8),
        "moderate": _build_regime("moderate", seed=20260415, t=60, n=30),
        "overconcentrated": _build_regime("overconcentrated", seed=20260416, t=40, n=50),
    }

    # Keep the top-level legacy keys that the existing three tests still read
    # (SampleCovariance, LedoitWolf, EWMA), backed by the original (T=120, N=5)
    # block for strict bit-for-bit continuity.
    rng = np.random.default_rng(seed=123)
    legacy_returns = rng.normal(scale=0.01, size=(120, 5))
    legacy_sample = np.cov(legacy_returns, rowvar=False, ddof=1)
    legacy_lw = reference_ledoit_wolf_scaled_identity(legacy_returns)
    legacy_ewma = reference_ewma(legacy_returns, lam=0.94)

    save_vector(
        "covariance",
        {
            "library_pins": {"numpy": "2", "scipy": "1.13+"},
            # Legacy top-level keys — unchanged from prior ship.
            "returns": legacy_returns.tolist(),
            "sample_covariance": legacy_sample.tolist(),
            "ledoit_wolf_scaled_identity": legacy_lw.tolist(),
            "ewma_lambda_0_94": legacy_ewma.tolist(),
            # Phase 2 additions — per-regime blocks with all estimators.
            "regimes": regimes,
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote covariance.json")
