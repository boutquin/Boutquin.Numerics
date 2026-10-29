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
| `ReturnsMatrix<T>` | T × N matrix wrapper for return observations with defensive array copies | A |

`WelfordMoments<T>` is the building block for all other estimators. Use it wherever streaming O(1) memory is preferred over a two-pass algorithm:

```csharp
using Boutquin.Numerics.Statistics;

var moments = new WelfordMoments<double>();
foreach (var r in returnSeries)
    moments.Add(r);

double mean   = moments.Mean;
double stdDev = moments.StdDev;   // sample std dev, N-1 divisor
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
| `SampleCovarianceEstimator<T>` | Standard Bessel-corrected (T−1) divisor | Baseline; accurate when T ≫ N |

The sample covariance is PSD by construction for T ≥ N + 1. It becomes singular (rank-deficient, still PSD) when T ≤ N and is known to have poorly-behaved extreme eigenvalues in the high-dimensional regime (c = N/T non-negligible).

#### Shrinkage estimators

Linear shrinkage blends the sample covariance S with a structured target F, yielding a well-conditioned matrix Σ̂ = δ·F + (1−δ)·S.

| Type | Target | Reference |
|------|--------|-----------|
| `LedoitWolfShrinkageEstimator<T>` | Scaled identity μ·I | Ledoit-Wolf (2004) — general-purpose default |
| `LedoitWolfConstantCorrelationEstimator<T>` | Constant-correlation (average ρ̄) | Ledoit-Wolf (2004) — when assets share a common correlation structure |
| `LedoitWolfSingleFactorEstimator<T>` | Single-index (market beta) model | Ledoit-Wolf (2003) — equity portfolios where a market factor dominates |
| `OracleApproximatingShrinkageEstimator<T>` | Known-zero-mean raw second moment / T, shrunk toward scaled identity (OAS) | Chen-Wiesel-Eldar-Hero (2010) — improved bias correction for small T |

The constant-correlation estimator is Tier A+√ because it normalizes by standard deviations; the other linear shrinkage estimators listed here are Tier A. They are PSD by construction — a convex combination of PSD matrices is PSD. For moderate-size portfolios (N ≤ 100, T ≥ 2N), `LedoitWolfShrinkageEstimator<T>` is the default choice.

#### Random Matrix Theory estimators

These estimators use eigenvalue analysis to separate signal from noise.

| Type | Method | Reference |
|------|--------|-----------|
| `DenoisedCovarianceEstimator<T>` | Replaces residual eigenvalues below the Marchenko-Pastur bulk edge (noise variance fitted to the eigenvalue density) with their average, then rescales the correlation to a unit diagonal | López de Prado (2020) *Machine Learning for Asset Managers* §2.5; Laloux et al. (1999) |
| `DetonedCovarianceEstimator<T>` | Denoising, then removal of the market component (first principal component of the denoised correlation) and a unit-diagonal rescale; full market-component removal with a positive residual diagonal is singular. The small-asset and all-noise paths, and zero-residual identity fallback, need not be singular | López de Prado (2020) §2.6 — when the market factor dominates and hides diversification signal |
| `TracyWidomDenoisedCovarianceEstimator<T>` | Denoising with the 99% Tracy-Widom threshold `mu_np/T + (sigma_np/T)·2.0234` | Johnstone (2001); Tracy-Widom (1994) — finite-N noise boundary |

Denoised and Detoned fit the MP noise variance over `[1e-5, 1-1e-5]` against a
Gaussian KDE (bandwidth 0.01; 1000 density points). The deterministic grid scan and
refinement selects the lowest objective among candidate minima; it can differ
from a local scipy L-BFGS-B search started at 0.5. Failed non-finite fits use 1.
Model selection runs in `double`, while covariance arithmetic remains in `T`.
For `decimal`, only the selected structure has double resolution.

RMT estimators require `JacobiEigenDecomposition<T>` internally (Tier A + √). They are the preferred choice in high-dimensional regimes where N > T/2.

#### Structured and non-parametric estimators

| Type | Method | Reference |
|------|--------|-----------|
| `ExponentiallyWeightedCovarianceEstimator<T>` | EWMA with decay factor λ | Standard; λ = 0.94 (RiskMetrics daily) |
| `PoetCovarianceEstimator<T>` | Principal Orthogonal complEment Thresholding | Fan-Liao-Mincheva (2013) — sparsity + factor structure |
| `NercomeCovarianceEstimator<T>` | Non-parametric Eigenvalue-Regularised Covariance Matrix Estimator | Lam (2016) — replicated full-matrix averaging and candidate-grid selection; Abadir-Distaso-Zikes (2014) antecedent |
| `DoublySparseEstimator<T>` | Hard-thresholded signal eigenvectors plus averaged noise eigenvalues | Declared repository variant; see distinction below |
| `QuadraticInverseShrinkageEstimator<T>` | Non-linear shrinkage via Quadratic Inverse Shrinkage (QIS) | Ledoit-Wolf (2022) — minimal Frobenius error in the precision matrix |

`NercomeCovarianceEstimator<T>` defaults to the seven-candidate grid of Lam (2016), Equation (4.9), with 50 distinct first subsets per candidate (capped by the available subsets) and deterministic seed 0. Candidate sizes are floored, clamped to nonempty subsets and deduplicated; the first candidate wins a score tie. Each split uses raw known-zero-mean second moments. Full reconstructed matrices are averaged, and selection minimizes the squared norm of the averaged residual. Explicit split fractions retain the replicated estimator at a fixed size.

`DoublySparseEstimator<T>` uses MP-edge signal selection, noise-eigenvalue averaging,
and a one-shot absolute threshold on signal-eigenvector entries with renormalisation.
It is PSD by construction. It does not implement the Kutateladze–Seregina estimator:
there is no truncated power iteration, rank or support-size input, least-squares
signal-eigenvalue re-estimation, or adaptive thresholding of the residual covariance.

#### Choosing an estimator

```
T ≫ N           → SampleCovarianceEstimator<T>                    (baseline)
T moderate      → LedoitWolfShrinkageEstimator<T>                 (general-purpose)
equity universe → LedoitWolfConstantCorrelationEstimator<T>       (common ρ̄)
N > T/2         → DenoisedCovarianceEstimator<T>                   (RMT noise clipping)
market factor   → DetonedCovarianceEstimator<T>                    (remove PC1)
regime-aware    → ExponentiallyWeightedCovarianceEstimator<T>     (EWMA)
precision focus → QuadraticInverseShrinkageEstimator<T>            (QIS)
```

`ActiveSetQpSolver<T>` in the Solvers namespace takes a plain `T[,]` covariance matrix, not an estimator; pass it the matrix returned by any `ICovarianceEstimator<T>.Estimate(...)` call.

### Portfolio performance diagnostics

These types measure statistical significance, overfitting risk, and performance reliability. The arithmetic-first diagnostics are Tier A with a transcendental tail (inner loop in `T`, final scalar via `double` for `Log`/inverse-normal); `MinimumTrackRecordLength<T>` and `GtInspiredScore<T>` are Tier B because transcendental operations are integral to their formulas.

| Type | Purpose | Reference |
|------|---------|-----------|
| `NeweyWestVariance<T>` | HAC variance of a return series (corrects for serial correlation) | Newey-West (1987) |
| `DeflatedSharpeRatio<T>` | Sharpe adjusted for skewness, kurtosis, and strategy selection bias | Bailey-López de Prado (2014) |
| `HaircutSharpe<T>` | Haircut Sharpe — required margin of safety for multi-tested strategies | Harvey-Liu (2015) |
| `MinimumTrackRecordLength<T>` | Minimum observation count for a Sharpe to be statistically significant | Bailey-López de Prado (2012) |
| `GtInspiredScore<T>` | GT-inspired library search loss with the explicit component definitions below | Tier B |

`GtInspiredScore<T>` uses sample SD (N − 1), cumulative arithmetic-return R² against time 1..N,
zero-target downside RMS over all N observations, and additive epsilon in the utility denominator.
Its inclusive minimum-count safeguard returns `999 − 899·N/minimumObservations` at or below the
threshold (default 50). Above it, z ≤ 0 returns `100 + 100·(1 − exp(−|z−1|))`,
0 < z ≤ 1 returns `100·(1 − exp(−|z−1|))`, and z > 1 returns `−µ·ln(z)·r²/(σ_d + ε)`.
Constant returns define z = 0; constant cumulative returns define R² = 0. At least two finite
observations, a finite benchmark, positive finite epsilon and a positive threshold are required.
This is a library variant inspired by Sheppert (2026), arXiv:2602.00080. No author parity,
empirical performance or out-of-sample generalization guarantee is claimed. Authenticated author
code instead uses population SD, regression of individual returns and dispersion of negative
returns, with a different zero-dispersion smoothing rule.
| `ProbabilityOfBacktestOverfitting<T>` | PBO via Combinatorially-Symmetric Cross-Validation (CSCV) | Bailey et al. (2014) |

`DeflatedSharpeRatio<T>` and `MinimumTrackRecordLength<T>` take annualized Sharpe ratios and divide them by √`tradingDaysPerYear` before applying the papers' formulas, which require Sharpe ratios and sample length at the same, non-annualized frequency. `DeflatedSharpeRatio<T>.ComputeWithTrialVariance` takes the variance of the trials' annualized Sharpe ratios, V[{SR̂ₙ}]; `Compute` and `ComputeFromReturns` use its null value, 1/(T − 1) per period. `DsrResult<T>.PValue` is 1 − DSR. `MinimumTrackRecordLength<T>` returns a number of per-period observations; divide by `tradingDaysPerYear` for years.

`ProbabilityOfBacktestOverfitting<T>` exhausts all C(S, S/2) half-half splits of S balanced time blocks, computing each strategy's Sharpe ratio on the joined in-sample and joined out-of-sample halves. PBO is the fraction of splits in which the in-sample-best strategy ranks at or below the out-of-sample median. It complements DSR: DSR adjusts a single observed Sharpe for the search process; PBO measures the fraction of selection decisions that are illusory.

```csharp
using Boutquin.Numerics.Statistics;

PboResult<double> result = ProbabilityOfBacktestOverfitting<double>.Compute(
    strategySharpeMatrix,
    splitCount: 16);

Console.WriteLine($"PBO = {result.Pbo:P1}");            // e.g. "PBO = 37.5%"
Console.WriteLine($"Logit median = {result.LogitMedian:F3}");
```

---

### Volatility, sparse precision, and tail dependence

`GarchOneOneEstimator<T>` and `GraphicalLassoSolver<T>` are Tier B and therefore require an
IEEE-754 type such as `double`; `LowerTailDependence<T>` is Tier A and also accepts `decimal`.

```csharp
using Boutquin.Numerics.Statistics;

GarchFitResult<double> garch = GarchOneOneEstimator<double>.Fit(returns);
double oneStepVariance = garch.ForecastVariance(horizon: 1);

double[,] sampleCovariance =
{
    { 1.00, 0.35, 0.10 },
    { 0.35, 1.00, 0.25 },
    { 0.10, 0.25, 1.00 },
};
GraphicalLassoResult<double> sparse = GraphicalLassoSolver<double>.Solve(
    sampleCovariance,
    lambda: 0.10);

TailResult<double> tail = LowerTailDependence<double>.Estimate(
    assetReturnsA,
    assetReturnsB,
    new TailOptions<double>(Quantiles: [0.05, 0.10]));
```

Use GARCH for conditional variance forecasting, graphical lasso for a sparse inverse-covariance
(precision) estimate, and lower-tail dependence for co-crash dependence across quantile levels.
`TailOptions<T>.UseGarchFilter` is currently unsupported and throws `NotSupportedException`; keep
its default `false`.
`LowerTailDependence<T>.EstimateMatrix` uses rows as the conditioning series for the default
conditional estimator. Its off-diagonal entries can differ by direction when tied ranks give
different realized marginal counts. The empirical-copula option has a shared `n·q` denominator
and produces a symmetric matrix.

## Distributions

| Type | Purpose | Tier |
|------|---------|------|
| `NormalDistribution<T>` | PDF, CDF, and quantile for N(μ, σ²) | C |
| `CumulativeNormal<T>` | Φ(x) — standard normal CDF via Hart (1968) rational approximation | C |
| `InverseNormal<T>` | Φ⁻¹(p) — probit function via Acklam's rational approximation | C |

All three are Tier C: their internal precision ceiling is Acklam's polynomial coefficients and the `erfc` continued-fraction table, both calibrated for `double` arithmetic. The public surface is generic over `T : IFloatingPoint<T>` so they compose into generic pipelines without an explicit cast at the boundary; internally they compute in `double` and return `T.CreateChecked(result)`.

```csharp
using Boutquin.Numerics.Distributions;

double pdf   = NormalDistribution<double>.Pdf(0.0);
double cdf   = NormalDistribution<double>.Cdf(0.0);       // 0.5
double z95   = NormalDistribution<double>.InverseCdf(0.95);

// Standalone probit
double z = InverseNormal<double>.Evaluate(0.975);   // ≈ 1.96
```

These are standard-normal APIs. For a normal distribution with mean `mu` and standard deviation
`sigma`, transform at the call site: `mu + sigma * z`.

---

## MonteCarlo

### Bootstrap resamplers

Each resampler is a separate concrete API that accepts a `ReadOnlySpan<T>` sample. They share a similar `Resample` shape, but do not implement a common resampler interface and are not injected into `BootstrapMonteCarloEngine<T>`. Callers can pass a `T[]` directly — it implicitly converts to `ReadOnlySpan<T>` at the call site.

| Type | Method | Use when |
|------|--------|----------|
| `BootstrapResampler<T>` | Circular fixed-length block resampling | Weakly dependent series; preserves local order and wraps at the end |
| `MovingBlockBootstrapResampler<T>` | Overlapping fixed-length blocks, no wrap-around | Weak serial dependence |
| `StationaryBootstrapResampler<T>` | Overlapping blocks with geometric random length (Politis-Romano 1994) | Stationary time series — preserves autocorrelation structure |
| `WildBootstrapResampler<T>` | Heteroskedasticity-robust (Mammen weights) | Returns with time-varying variance |

For IID resampling with replacement, use `BootstrapMonteCarloEngine<T>` or configure
`BootstrapResampler<T>` with `blockSize: 1`.

`PolitisWhiteBlockLength.EstimateDouble(ReadOnlySpan<double> series)` provides data-driven block-length selection for `StationaryBootstrapResampler<T>`.

It applies the corrected stationary constant `D_SB = 2g(0)²` and the corrected companion selector's stability window beginning at lag `m`. The search starts at lag 1. Correlations use the author's common horizon-truncated rows; covariances use common bandwidth-truncated rows, separate column means and the sample denominator. There is an upper cap and no lower floor: estimates may be below one. Callers must respect the input range of their chosen resampler. These finite-sample conventions are explicit in the [paper replication catalogue](paper-replication-tests.md#paper-ppw), which checks the population optima and six seeded simulations against independent NumPy controls.

### Bootstrap Monte Carlo engine

`BootstrapMonteCarloEngine<T>` performs IID sampling with replacement internally. Its constructor accepts the simulation count and an `IRandomSource`; `Run` accepts the observations and statistic function and returns a `BootstrapMonteCarloResult<T>` with a sorted distribution plus summary statistics:

```csharp
using Boutquin.Numerics.MonteCarlo;
using Boutquin.Numerics.Random;

double[] returns = [0.012, -0.004, 0.009, 0.015, -0.007, 0.003];
var rng = new Pcg64RandomSource(seed: 42UL);
var engine = new BootstrapMonteCarloEngine<double>(simulationCount: 10_000, rng);

BootstrapMonteCarloResult<double> result =
    engine.Run(returns, static r =>                 // bootstrap distribution of the mean
    {
        double sum = 0;
        foreach (var v in r) sum += v;
        return sum / r.Length;
    });

Console.WriteLine($"Median = {result.Median:F4}");
Console.WriteLine($"90% CI = [{result.Percentile5:F4}, {result.Percentile95:F4}]");
```

### Quasi-random sequences

| Type | Method | Tier | Use when |
|------|--------|------|----------|
| `SobolSequence<T>` | Sobol low-discrepancy sequence (Joe-Kuo direction numbers) | C | Low-variance numerical integration in ≤ 32 dimensions |
| `HaltonSequence<T>` | Halton sequence (prime-base Van der Corput) | C | Simpler QMC, moderate dimensions |

Both are Tier C: Sobol's direction numbers are `uint32` bit patterns calibrated for the `double` lattice [0, 1), and Halton evaluates its radical inverse in `double` (it is unscrambled and has no constants table). They accept generic `T` at the surface for pipeline composition; internally they generate in `double` and return `T.CreateChecked`.

`FastDoubleBootstrap` is a non-generic, decimal-working implementation that draws IID outer resamples and one IID inner resample per outer sample through its injected `IRandomSource`, then applies the Davidson–MacKinnon fast-double-bootstrap p-value correction. The FDB draws both levels from a DGP that respects the null hypothesis: `PValueImposingNull` takes a transform (for a location test, recentre the sample to the hypothesised mean) and applies it to the data and to every outer sample before its inner draw. `PValue` resamples raw data at both levels and is appropriate only for a statistic whose null distribution does not depend on the parameters the null restricts.

FDB excludes equality at both the first-level p-value and the final outer rank against the inner quantile: right tails use `>`, left tails use `<`. This follows Davidson (2023), §4, algorithm steps 4/6 and the right-rejecting remark. Earlier versions included ties, so tied-data results change. Inner quantiles still use Type-7 linear interpolation; the two-sided option folds the two strict one-sided FDB values.

### Percentile estimation

| Type | Purpose |
|------|---------|
| `Percentile` | `Compute<T>`: linear interpolation percentile (Type 7 — R/numpy default) |
| `HarrellDavisPercentile` | `Compute<T>`: smooth nonparametric quantile estimator (Harrell-Davis 1982) |
| `PSquareEstimator<T>` | Online streaming P² algorithm — approximate quantile in O(1) memory |
| `Subsampler` | `Run<T>`: subsample confidence intervals for heavy-tailed distributions |

`HarrellDavisPercentile` uses double-precision Beta CDF weights for smooth quantile estimates (Tier C) — preferred for small samples where the standard interpolation percentile is noisy. Its final accumulation cannot exceed the precision of those weights.

### Random number generation

| Type | Method | Tier |
|------|--------|------|
| `Pcg64RandomSource<T>` | PCG-64 uniform random source | C |
| `GaussianSampler<T>` | Normal sampler over an `IRandomSource` using the Marsaglia polar method | C |

`Pcg64RandomSource<T>` supplies uniform values; `GaussianSampler<T>` uses those values to draw normal samples via the Marsaglia polar method. Tier C: the generic adapters cast from `double`, so their sampled values have a `double`-precision ceiling. Acklam's inverse-normal belongs to `InverseNormal<T>`, not to the PCG source or the Gaussian sampler.

---

## Precision regime summary

| Namespace | Default `T` | `decimal` supported? | Note |
|-----------|-------------|----------------------|------|
| Statistics — moments, covariance, diagnostics | `decimal` or `double` | Yes (Tier A / A+√) | Inner loops in full working-type precision |
| Distributions | `double` | Cosmetic only | Acklam/Hart coefficients are `double`-ceiling |
| MonteCarlo — bootstrap engines, linear percentiles | `decimal` or `double` | Yes (Tier A) | Order statistics and arithmetic resampling |
| MonteCarlo — Harrell–Davis percentile | `decimal` or `double` | Yes (Tier C) | Beta weights have a `double`-precision ceiling |
| MonteCarlo — Sobol, Halton | `double` | Cosmetic only | Sobol direction numbers calibrated for the `double` lattice; Halton radical inverse evaluated in `double` |
| MonteCarlo — `FastDoubleBootstrap` | `decimal` | Decimal-only | IID resampling and statistic evaluation in `decimal`; no direction-number table |
| Random — PCG-64 | `double` | Cosmetic only | Uniform output follows the `double` bit lattice |

For portfolio moment calculations (covariance estimation, Sharpe diagnostics) where 28-digit precision matters, use `T = decimal` throughout. For Monte Carlo path simulation and distribution computations, `T = double` is the correct choice.

`GraphicalLassoSolver<T>` penalizes all precision entries, including the diagonal, by default. Its covariance iterate starts at S + λI and retains that diagonal. The default stopping threshold is 0.001 times the mean absolute off-diagonal sample covariance; a zero off-diagonal scale requires an exactly stationary iterate. Explicit zero diagonal penalty weights select an unpenalized diagonal.

At zero penalty the solver returns the sample covariance and its Cholesky inverse directly. Independent covariance blocks are solved in their own units. After convergence, the final lasso coefficients determine the sparse precision; its Cholesky inverse supplies the returned covariance. A sweep that loses positive definiteness returns the previous positive-definite iterate with `Converged=false`. BIC compares converged candidates and treats absolute partial correlations at or below 1e-10 as numerical zero; if no candidate converges with a finite BIC, it throws `InvalidOperationException`.
