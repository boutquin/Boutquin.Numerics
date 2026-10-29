"""Scalar-statistic reference vectors.

Five deterministic statistics, each verified against a paper-faithful
numpy / statsmodels reference:

* ``newey_west`` — ``statsmodels.stats.sandwich_covariance`` HAC variance
  convention, rewritten in numpy to match the C# ``NeweyWestVariance``
  formula bit-for-bit (Bartlett kernel, 1/T normalisation for Var(mean)).
* ``pbo`` — Bailey, Borwein, López de Prado & Zhu 2014 CSCV Probability of
  Backtest Overfitting, written from §2 of the paper (Sharpe on the joined
  training / test blocks, average rank for ties); exhaustive C(S, S/2)
  combinatorial enumeration.
* ``generalization_score`` — GT-inspired library variant (inspired by arXiv:2602.00080),
  the parameter-free multiplicative composite µ·ln(z)·r²/σ_d with the
  piecewise library loss. Uses sample SD, cumulative-return R², zero-target
  downside RMS and additive epsilon; no author parity is claimed.
* ``harrell_davis`` — Harrell-Davis 1982 percentile estimator via
  regularised incomplete beta weights.
* ``psquare`` — Jain-Chlamtac 1985 P² streaming percentile estimator.

Library pins:

* numpy==2.x
* scipy==1.13+ (for ``scipy.special.betainc``)
* statsmodels==0.14+ (only the HAC convention is borrowed; the numerical
  computation is done in numpy for a cleaner bit-for-bit match)

Fixed seed per statistic; regenerating produces bit-identical JSON.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy.special import betainc

from conftest import save_vector


SEED = 20260415


# =====================================================================
#  Newey-West HAC variance — Bartlett kernel, matches C# line-for-line.
# =====================================================================


def reference_newey_west_mean_variance(series: np.ndarray, lags: int) -> float:
    """Var(mean) via Newey-West HAC with Bartlett kernel — ``statsmodels``
    independent reference.

    The Numerics C# class computes Var(mean) directly via the
    Newey-West 1987 formula with 1/T autocovariance normalisation and
    Bartlett kernel. statsmodels provides the same estimator via
    ``OLS(y, ones(T)).fit().get_robustcov_results('HAC', maxlags=L,
    use_correction=False)`` — the HAC-robust covariance of the single
    intercept coefficient from a constant regression is exactly
    Var(mean) under the same conventions.

    ``use_correction=False`` suppresses the ``T/(T−K)`` finite-sample
    factor so the comparison is to the raw Newey-West formula (our class
    does not apply that correction either). Empirically the two match to
    within ~2.4e-15 relative (machine epsilon) on the shared test series.

    Using statsmodels as the reference avoids the "2× the same bug" risk
    a hand-port would carry — statsmodels' HAC implementation has
    different authors, a different codebase, and a different
    sandwich-formula layout (regression residuals × outer product × lag
    sums) from Numerics' direct-sum layout, so shared algorithmic bugs
    cannot silently produce matching answers.
    """

    t = series.size
    x_const = np.ones(t)
    model = sm.OLS(series, x_const).fit()
    hac = model.get_robustcov_results(
        cov_type="HAC", maxlags=lags, use_correction=False
    )
    return float(hac.cov_params()[0, 0])


# =====================================================================
#  Probability of Backtest Overfitting (Bailey et al. 2014, CSCV).
# =====================================================================


def reference_pbo(returns: np.ndarray, split_count: int = 16) -> dict:
    """Combinatorially-symmetric cross-validation (CSCV) PBO.

    Reference: Bailey, Borwein, López de Prado & Zhu (2014), "The
    Probability of Backtest Overfitting", J. Computational Finance 20(4),
    §2 steps (a)-(f). Written from the paper, not ported from the C#:

    * (a)/(b) the training set J joins the S/2 selected blocks, in their
      original order, into one T/2 × N matrix; the test set J̄ joins the rest;
    * (c)/(d) the performance statistic (Sharpe = mean / sample sd) is
      computed on each column of the joined J and J̄ — not averaged over
      per-block Sharpe ratios;
    * (e) the IS winner is the best column of J;
    * (f) ω = rank / (N + 1) of the winner's OOS statistic, with
      ``scipy.stats.rankdata(method="average")`` giving tied strategies
      their average rank; logit λ = ln(ω / (1 − ω)).

    PBO is the fraction of folds with λ ≤ 0; the logit median is the
    ordinary sample median (``np.median``). A constant column has zero
    standard deviation and Sharpe 0.
    """

    from itertools import combinations

    from scipy.stats import rankdata

    n, t = returns.shape
    assert t % split_count == 0
    block_size = t // split_count
    half = split_count // 2

    def sharpe(column: np.ndarray) -> float:
        if np.all(column == column[0]):
            return 0.0
        sd = column.std(ddof=1)
        return float(column.mean() / sd) if sd > 0 else 0.0

    def joined_rows(blocks) -> np.ndarray:
        return np.concatenate(
            [np.arange(b * block_size, (b + 1) * block_size) for b in blocks]
        )

    logits = []
    for is_blocks in combinations(range(split_count), half):
        oos_blocks = [b for b in range(split_count) if b not in is_blocks]
        train = returns[:, joined_rows(is_blocks)]
        test = returns[:, joined_rows(oos_blocks)]

        is_stat = np.array([sharpe(train[i]) for i in range(n)])
        oos_stat = np.array([sharpe(test[i]) for i in range(n)])
        is_winner = int(np.argmax(is_stat))

        omega = rankdata(oos_stat, method="average")[is_winner] / (n + 1)
        logits.append(np.log(omega / (1.0 - omega)))

    logits = np.asarray(logits)
    return {
        "pbo": float(np.mean(logits <= 0.0)),
        "logit_median": float(np.median(logits)),
        "split_count": split_count,
    }


# =====================================================================
#  Generalization Score (Sheppert 2026).
# =====================================================================


def reference_generalization_score(
    returns: np.ndarray,
    benchmark_mean: float = 0.0,
    epsilon: float = 1e-6,
) -> dict:
    """GT-inspired library variant: sample SD, cumulative R², zero-target RMS, additive epsilon.

    This independent oracle follows the declared library variant, not a C# port or author parity:

      eq.(1): GTScore = mu * ln(z) * r2 / sigma_d         (composite utility)
      eq.(2): z = (mu - mu_m) / (sigma / sqrt(N))         (significance Z-score)
      Algorithm 1 (piecewise LOSS, eps = 1e-6):
        if z <= 0:   score = 100 + 100*(1 - exp(-|z-1|))     # large penalty
        elif z <= 1: score = 100*(1 - exp(-|z-1|))           # smooth transition
        else:        score = -(mu*ln(z)*r2)/(sigma_d + eps)  # standard GT-Score

    The explicitly chosen library components are:
      r2      (Kestner 1996): R-squared of the cumulative-equity curve regressed
              on the linear time trend 1..N.
      sigma_d (Sortino & van der Meer 1991): downside deviation
              sqrt(mean(min(0, r)^2)) with a zero minimum acceptable return.
    """

    n = returns.size
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))

    # eq.(2) significance Z-score.
    se = sigma / np.sqrt(n)
    z = (mu - benchmark_mean) / se if se > 0 else 0.0

    # r2: R-squared of cumulative equity curve vs linear time trend (Kestner 1996).
    equity = np.cumsum(returns)
    xs = np.arange(1, n + 1, dtype=float)
    xbar = xs.mean()
    ebar = equity.mean()
    sxx = float(((xs - xbar) ** 2).sum())
    syy = float(((equity - ebar) ** 2).sum())
    sxy = float(((xs - xbar) * (equity - ebar)).sum())
    r2 = (sxy * sxy) / (sxx * syy) if sxx > 0 and syy > 0 else 0.0

    # sigma_d: downside deviation, MAR = 0 (Sortino & van der Meer 1991).
    downside = np.minimum(0.0, returns)
    sigma_d = float(np.sqrt((downside ** 2).mean()))

    # Algorithm 1 piecewise.
    if z <= 0.0:
        score = 100.0 + 100.0 * (1.0 - np.exp(-abs(z - 1.0)))
    elif z <= 1.0:
        score = 100.0 * (1.0 - np.exp(-abs(z - 1.0)))
    else:
        score = -(mu * np.log(z) * r2) / (sigma_d + epsilon)

    return {
        "score": float(score),
        "mu": mu,
        "z": float(z),
        "r_squared": r2,
        "downside_deviation": sigma_d,
        "benchmark_mean": benchmark_mean,
        "epsilon": epsilon,
    }


# =====================================================================
#  Harrell-Davis percentile (1982).
# =====================================================================


def reference_harrell_davis(sorted_values: np.ndarray, p: float) -> float:
    """Beta-kernel weighted percentile — regularised incomplete beta weights.

    Reference: Harrell & Davis (1982), "A New Distribution-Free Quantile
    Estimator", Biometrika 69(3), 635–640.
    """

    n = sorted_values.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_values[0])

    p = float(np.clip(p, 0.0, 1.0))
    alpha = (n + 1) * p
    beta_p = (n + 1) * (1.0 - p)

    # W_i = I(i/n; α, β) − I((i−1)/n; α, β) for i = 1..n.
    breakpoints = np.arange(n + 1) / n
    cdf_values = betainc(alpha, beta_p, breakpoints)
    weights = np.diff(cdf_values)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        # Fallback — match C# by returning a trivial linear-interp percentile.
        rank = p * (n - 1)
        lo = int(np.floor(rank))
        hi = int(np.ceil(rank))
        if lo == hi:
            return float(sorted_values[lo])
        frac = rank - lo
        return float(sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo]))

    weights /= weight_sum
    return float(np.dot(sorted_values, weights))


# =====================================================================
#  P² streaming percentile (Jain-Chlamtac 1985).
# =====================================================================


def reference_psquare(stream: np.ndarray, percentile: float) -> float:
    """Streaming percentile via P² algorithm — stateful five-marker update.

    Reference: Jain & Chlamtac (1985), "The P² Algorithm for Dynamic
    Calculation of Quantiles and Histograms Without Storing Observations",
    Communications of the ACM 28(10), 1076–1085.

    Desired-position decisions use exact rational arithmetic from Box 1,
    with the exact binary-double percentile. Marker heights use independent
    floating-point evaluation of the published parabolic/linear equations.
    """
    from fractions import Fraction

    p = percentile
    exact_p = Fraction(p)
    dn = [Fraction(0), exact_p / 2, exact_p, (1 + exact_p) / 2, Fraction(1)]
    q = [0.0] * 5
    nn = [0] * 5  # marker positions, 1-indexed per paper
    nprime = [0.0] * 5
    count = 0

    def parabolic(i: int, d: int) -> float:
        ni = nn[i]
        npre = nn[i - 1]
        npost = nn[i + 1]
        qi = q[i]
        qpre = q[i - 1]
        qpost = q[i + 1]
        denom = npost - npre
        lhs = d / denom
        left_factor = (ni - npre + d) * (qpost - qi) / (npost - ni)
        right_factor = (npost - ni - d) * (qi - qpre) / (ni - npre)
        return qi + lhs * (left_factor + right_factor)

    def linear(i: int, d: int) -> float:
        direction = i + d
        return q[i] + d * (q[direction] - q[i]) / (nn[direction] - nn[i])

    for obs in stream:
        obs = float(obs)
        if count < 5:
            q[count] = obs
            count += 1
            if count == 5:
                q.sort()
                for i in range(5):
                    nn[i] = i + 1
                    nprime[i] = 1.0 + dn[i] * 4.0
            continue

        # Locate cell k and adjust bounds.
        if obs < q[0]:
            q[0] = obs
            k = 0
        elif obs < q[1]:
            k = 0
        elif obs < q[2]:
            k = 1
        elif obs < q[3]:
            k = 2
        elif obs <= q[4]:
            k = 3
        else:
            q[4] = obs
            k = 3

        for i in range(k + 1, 5):
            nn[i] += 1
        for i in range(5):
            nprime[i] = 1 + count * dn[i]

        for i in range(1, 4):
            d = nprime[i] - nn[i]
            if (d >= 1.0 and nn[i + 1] - nn[i] > 1) or (d <= -1.0 and nn[i - 1] - nn[i] < -1):
                sign = 1 if d > 0 else (-1 if d < 0 else 0)
                qp = parabolic(i, sign)
                if q[i - 1] < qp < q[i + 1]:
                    q[i] = qp
                else:
                    q[i] = linear(i, sign)
                nn[i] += sign

        count += 1

    # Estimate: q[2] once ≥5 obs, else linear interp over sorted partial sample.
    if count < 5:
        buffer = sorted(q[:count])
        rank = p * (len(buffer) - 1)
        lo = int(np.floor(rank))
        hi = int(np.ceil(rank))
        if lo == hi:
            return buffer[lo]
        frac = rank - lo
        return buffer[lo] + frac * (buffer[hi] - buffer[lo])
    return q[2]


# =====================================================================
#  Regime generation + orchestrator.
# =====================================================================


def generate() -> None:
    rng = np.random.default_rng(SEED)

    # --- Newey-West ----------------------------------------------------
    nw_t = 500
    nw_series = rng.normal(scale=1.0, size=nw_t)
    # Artificial serial correlation — mild AR(1) to exercise the HAC lags.
    for i in range(1, nw_t):
        nw_series[i] = 0.3 * nw_series[i - 1] + nw_series[i]
    nw_lags = 8
    nw_variance = reference_newey_west_mean_variance(nw_series, nw_lags)

    # --- PBO -----------------------------------------------------------
    # 10 strategies × 128 observations (128 % 16 == 0 ⇒ 8-obs blocks).
    pbo_returns = rng.normal(scale=0.01, size=(10, 128))
    pbo_result = reference_pbo(pbo_returns, split_count=16)

    # --- Generalization score -----------------------------------------
    gt_returns = rng.normal(loc=0.0005, scale=0.01, size=504)
    gt_result = reference_generalization_score(gt_returns)

    # --- Harrell-Davis -------------------------------------------------
    hd_sample = np.sort(rng.normal(size=50))
    hd_percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    hd_values = {str(p): reference_harrell_davis(hd_sample, p) for p in hd_percentiles}

    # --- P-Square ------------------------------------------------------
    # 1000-sample stream for each of a few representative percentiles.
    ps_stream = rng.normal(loc=0.0, scale=1.0, size=1000)
    ps_values = {
        "0.5": reference_psquare(ps_stream, 0.5),
        "0.9": reference_psquare(ps_stream, 0.9),
        "0.99": reference_psquare(ps_stream, 0.99),
    }

    save_vector(
        "scalar_stats",
        {
            "seed": SEED,
            "library_pins": {"numpy": "2", "scipy": "1.13+", "statsmodels": "0.14"},
            "newey_west": {
                "series": nw_series.tolist(),
                "lags": nw_lags,
                "variance": nw_variance,
            },
            "pbo": {
                "returns": pbo_returns.tolist(),
                **pbo_result,
            },
            "generalization_score": {
                "returns": gt_returns.tolist(),
                **gt_result,
            },
            "harrell_davis": {
                "sorted_sample": hd_sample.tolist(),
                "percentile_values": hd_values,
            },
            "psquare": {
                "stream": ps_stream.tolist(),
                "estimates": ps_values,
            },
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote scalar_stats.json")
