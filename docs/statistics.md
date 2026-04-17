# Statistics, Distributions, and Monte Carlo

Boutquin.Numerics ships three complementary namespaces for quantitative work: `Statistics` (sample moments, covariance estimation, portfolio diagnostics), `Distributions` (normal CDF/PDF/quantile), and `MonteCarlo` (bootstrap resampling, quasi-random sequences, percentile estimation). All types are generic over `T` — the caller picks the precision regime. Tier constraints are documented in [`docs/tier-constraints.md`](tier-constraints.md).

## Statistics

### Sample moments

| Type | Purpose | Tier |
|------|---------|------|
| `WelfordMoments<T>` | Single-pass mean, variance, Pearson covariance via Welford + Chan-Golub-LeVeque parallel merge | A + √ |
| `PearsonCorrelation<T>` | Unbiased Pearson correlation coefficient from a returns matrix | A |
| `RankCorrelation<T>` | Spearman rank correlation | A |
| `FisherZTransform<T>` | Fisher Z-transformation and inverse for correlation inference | B |
| `DistanceCorrelation<T>` | Energy-based distance correlation (Székely-Rizzo 2007) | B |
| `SampleSkewness<T>` | Unbiased adjusted Fisher-Pearson skewness (G1) | A + √ |
| `SampleExcessKurtosis<T>` | Unbiased excess kurtosis (G2) | A + √ |
| `ReturnsMatrix<T>` | Immutable T × N matrix of return observations with named assets | A |

`WelfordMoments<T>` is the building block for all other estimators. Use it wherever streaming O(1) memory is preferred over a two-pass algorithm:

```csharp
using Boutquin.Numerics.Statistics;

var moments = new WelfordMoments<double>();
foreach (var r in returnSeries)
    moments.Add(r);

double mean   = moments.Mean;
double stdDev = moments.StandardDeviation;   // sample std dev, N-1 divisor
```

### Covariance estimation

The `ICovarianceEstimator<T>` interface abstracts all estimators:

```csharp
T[,] Estimate(T[,] returns);   // returns: T observations × N assets → N × N matrix
```

All implementations are Tier A (arithmetic-only, `IFloatingPoint<T>`) unless noted.

#### Baseline

| Type | Approach | When to use |
|------|----------|-------------|
| `SampleCovarianceEstimator<T>` | Standard Bessel-corrected (N−1) divisor | Baseline; accurate when T ≫ N |

The sample covariance is PSD by construction for T ≥ N + 1. It becomes singular (rank-deficient, still PSD) when T ≤ N and is known to have poorly-behaved extreme eigenvalues in the high-dimensional regime (c = N/T non-negligible).

#### Shrinkage estimators

Linear shrinkage blends the sample covariance S with a structured target F, yielding a well-conditioned matrix Σ̂ = δ·F + (1−δ)·S.

| Type | Target | Reference |
|------|--------|-----------|
| `LedoitWolfShrinkageEstimator<T>` | Scaled identity μ·I | Ledoit-Wolf (2004) — general-purpose default |
| `LedoitWolfConstantCorrelationEstimator<T>` | Constant-correlation (average ρ̄) | Ledoit-Wolf (2004) — when assets share a common correlation structure |
| `LedoitWolfSingleFactorEstimator<T>` | Single-index (market beta) model | Ledoit-Wolf (2003) — equity portfolios where a market factor dominates |
| `OracleApproximatingShrinkageEstimator<T>` | Non-linear shrinkage (OAS, direct inversion) | Chen-Wiesel-Eldar-Hero (2010) — improved bias correction for small T |

All shrinkage estimators are Tier A. They are PSD by construction — a convex combination of PSD matrices is PSD. For moderate-size portfolios (N ≤ 100, T ≥ 2N), `LedoitWolfShrinkageEstimator<T>` is the default choice.

#### Random Matrix Theory estimators

These estimators use eigenvalue analysis to separate signal from noise.

| Type | Method | Reference |
|------|--------|-----------|
| `DenoisedCovarianceEstimator<T>` | Clips eigenvalues below the Marchenko-Pastur bulk edge | Laloux et al. (1999), Plerou et al. (2002) |
| `DetonedCovarianceEstimator<T>` | Denoising + PC1 market-factor shrinkage | López de Prado (2020) — when the market factor dominates and hides diversification signal |
| `TracyWidomDenoisedCovarianceEstimator<T>` | Denoising with Tracy-Widom level-1 bulk threshold | Tracy-Widom (1994) — tighter noise boundary than Marchenko-Pastur |

RMT estimators require `JacobiEigenDecomposition<T>` internally (Tier A + √). They are the preferred choice in high-dimensional regimes where N > T/2.

#### Structured and non-parametric estimators

| Type | Method | Reference |
|------|--------|-----------|
| `ExponentiallyWeightedCovarianceEstimator<T>` | EWMA with decay factor λ | Standard; λ = 0.94 (RiskMetrics daily) |
| `PoetCovarianceEstimator<T>` | Principal Orthogonal complEment Thresholding | Fan-Liao-Mincheva (2013) — sparsity + factor structure |
| `NercomeCovarianceEstimator<T>` | Non-parametric Eigenvalue-Regularised Covariance Matrix Estimator | Lam-Yao (2012) — non-parametric, no distributional assumptions |
| `DoublySparseEstimator<T>` | Sparse factor loadings + sparse residual covariance | Structured sparsity across both dimensions |
| `QuadraticInverseShrinkageEstimator<T>` | Non-linear shrinkage via Quadratic Inverse Shrinkage (QIS) | Ledoit-Wolf (2022) — minimal Frobenius error in the precision matrix |

#### Choosing an estimator

```
T ≫ N           → SampleCovarianceEstimator<T>                    (baseline)
T moderate      → LedoitWolfShrinkageEstimator<T>                 (general-purpose)
equity universe → LedoitWolfConstantCorrelationEstimator<T>       (common ρ̄)
N > T/2         → DenoisedCovarianceEstimator<T>                   (RMT noise clipping)
market factor   → DetonedCovarianceEstimator<T>                    (shrink PC1)
regime-aware    → ExponentiallyWeightedCovarianceEstimator<T>     (EWMA)
precision focus → QuadraticInverseShrinkageEstimator<T>            (QIS)
```

`ICovarianceEstimator<T>` is also the shared interface consumed by `ActiveSetQpSolver<T>` in the Solvers namespace.

### Portfolio performance diagnostics

These types measure statistical significance, overfitting risk, and performance reliability. All are Tier A with transcendental tail (inner loop in `T`, final scalar via `double` for `Log`/inverse-normal).

| Type | Purpose | Reference |
|------|---------|-----------|
| `NeweyWestVariance<T>` | HAC variance of a return series (corrects for serial correlation) | Newey-West (1987) |
| `DeflatedSharpeRatio<T>` | Sharpe adjusted for skewness, kurtosis, and strategy selection bias | Bailey-López de Prado (2012) |
| `HaircutSharpe<T>` | Haircut Sharpe — required margin of safety for multi-tested strategies | Harvey-Liu (2015) |
| `MinimumTrackRecordLength<T>` | Minimum observation count for a Sharpe to be statistically significant | Bailey-López de Prado (2012) |
| `GeneralizationScore<T>` | Generalization score G — measures out-of-sample reliability | Tier B |
| `ProbabilityOfBacktestOverfitting<T>` | PBO via Combinatorially-Symmetric Cross-Validation (CSCV) | Bailey et al. (2014) |

`ProbabilityOfBacktestOverfitting<T>` exhausts all C(S, S/2) half-half splits of S balanced time blocks. PBO is the fraction of splits in which the in-sample-best strategy ranks below the out-of-sample median. It complements DSR: DSR adjusts a single observed Sharpe for the search process; PBO measures the fraction of selection decisions that are illusory.

```csharp
using Boutquin.Numerics.Statistics;

var pbo = new ProbabilityOfBacktestOverfitting<double>(numSubSeries: 16);
PboResult<double> result = pbo.Compute(strategySharpeMatrix);

Console.WriteLine($"PBO = {result.Pbo:P1}");            // e.g. "PBO = 37.5%"
Console.WriteLine($"Logit median = {result.LogitMedian:F3}");
```

---

## Distributions

| Type | Purpose | Tier |
|------|---------|------|
| `NormalDistribution<T>` | PDF, CDF, and quantile for N(μ, σ²) | C |
| `CumulativeNormal<T>` | Φ(x) — standard normal CDF via Hart (1968) rational approximation | C |
| `InverseNormal<T>` | Φ⁻¹(p) — probit function via Acklam's rational approximation | C |

All three are Tier C: their internal precision ceiling is Acklam's polynomial coefficients and the `erfc` continued-fraction table, both calibrated for `double` arithmetic. The public surface is generic over `T : IFloatingPoint<T>` so they compose into generic pipelines without an explicit cast at the boundary; internally they compute in `double` and return `T.CreateChecked(result)`.

```csharp
using Boutquin.Numerics.Distributions;

var nd = new NormalDistribution<double>(mu: 0.05, sigma: 0.15);

double pdf     = nd.Pdf(0.05);        // peak of the density
double cdf     = nd.Cdf(0.0);         // ≈ 0.3694 — probability of negative return
double var95   = nd.Quantile(0.05);   // 5% VaR threshold

// Standalone probit
double z = InverseNormal<double>.Compute(0.975);   // ≈ 1.96
```

---

## MonteCarlo

### Bootstrap resamplers

All resamplers implement a common interface: given a `T[]` sample, produce bootstrap replicates for a scalar statistic. The strategy pattern means they are interchangeable.

| Type | Method | Use when |
|------|--------|----------|
| `BootstrapResampler<T>` | IID resampling with replacement | i.i.d. observations |
| `MovingBlockBootstrapResampler<T>` | Non-overlapping blocks of fixed length | Weak serial dependence |
| `StationaryBootstrapResampler<T>` | Overlapping blocks with geometric random length (Politis-Romano 1994) | Stationary time series — preserves autocorrelation structure |
| `WildBootstrapResampler<T>` | Heteroskedasticity-robust (Mammen weights) | Returns with time-varying variance |

`PolitisWhiteBlockLength<T>` provides data-driven block-length selection for `StationaryBootstrapResampler<T>`.

### Bootstrap Monte Carlo engine

`BootstrapMonteCarloEngine<T>` wraps any resampler and a user-supplied statistic function, runs B iterations, and returns a `BootstrapMonteCarloResult<T>` with a sorted distribution plus summary statistics:

```csharp
using Boutquin.Numerics.MonteCarlo;
using Boutquin.Numerics.Random;

var rng       = new Pcg64RandomSource<double>(seed: 42);
var resampler = new StationaryBootstrapResampler<double>(blockLength: 12, rng);
var engine    = new BootstrapMonteCarloEngine<double>(resampler, simulations: 10_000);

BootstrapMonteCarloResult<double> result =
    engine.Run(returns, static r => r.Average());   // bootstrap distribution of the mean

Console.WriteLine($"Median = {result.Median:F4}");
Console.WriteLine($"90% CI = [{result.Percentile5:F4}, {result.Percentile95:F4}]");
```

### Quasi-random sequences

| Type | Method | Tier | Use when |
|------|--------|------|----------|
| `SobolSequence<T>` | Sobol low-discrepancy sequence (Joe-Kuo direction numbers) | C | Low-variance numerical integration in ≤ 1,111 dimensions |
| `HaltonSequence<T>` | Halton sequence (prime-base Van der Corput) | C | Simpler QMC, moderate dimensions |

Both are Tier C: direction numbers and scrambling constants are `uint32` bit patterns calibrated for the `double` lattice [0, 1). They accept generic `T` at the surface for pipeline composition; internally they generate in `double` and return `T.CreateChecked`.

`FastDoubleBootstrap<T>` uses `SobolSequence<T>` to accelerate the inner resampling loop of bootstrap engines — same distributional properties as the IID resampler, significantly fewer iterations needed for a given CI width.

### Percentile estimation

| Type | Purpose |
|------|---------|
| `Percentile<T>` | Linear interpolation percentile (Type 7 — R/numpy default) |
| `HarrellDavisPercentile<T>` | Smooth nonparametric quantile estimator (Harrell-Davis 1982) |
| `PSquareEstimator<T>` | Online streaming P² algorithm — approximate quantile in O(1) memory |
| `Subsampler<T>` | Subsample confidence intervals for heavy-tailed distributions |

`HarrellDavisPercentile<T>` uses Beta CDF weights for smooth quantile estimates — preferred for small samples where the standard interpolation percentile is noisy.

### Random number generation

| Type | Method | Tier |
|------|--------|------|
| `Pcg64RandomSource<T>` | PCG-64 permuted congruential generator with inverse-normal transform | C |

`Pcg64RandomSource<T>` generates normally-distributed `T` samples using the PCG-64 engine and Acklam's inverse-normal. Tier C: inverse-normal coefficients are `double`-precision; the generator is suitable wherever `double`-precision variates are adequate.

---

## Precision regime summary

| Namespace | Default `T` | `decimal` supported? | Note |
|-----------|-------------|----------------------|------|
| Statistics — moments, covariance, diagnostics | `decimal` or `double` | Yes (Tier A / A+√) | Inner loops in full working-type precision |
| Distributions | `double` | Cosmetic only | Acklam/Hart coefficients are `double`-ceiling |
| MonteCarlo — bootstrap engines, percentiles | `decimal` or `double` | Yes (Tier A) | Order statistics and arithmetic resampling |
| MonteCarlo — Sobol, Halton, fast bootstrap | `double` | Cosmetic only | Direction numbers calibrated for `double` lattice |
| MonteCarlo — PCG-64 | `double` | Cosmetic only | Inverse-normal is `double`-ceiling |

For portfolio moment calculations (covariance estimation, Sharpe diagnostics) where 28-digit precision matters, use `T = decimal` throughout. For Monte Carlo path simulation and distribution computations, `T = double` is the correct choice.
