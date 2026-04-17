# Changelog

All notable changes to `Boutquin.Numerics` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Versions are produced by [MinVer](https://github.com/adamralph/minver) from git tags on the public release repository.

## [Unreleased]

_No unreleased changes yet. See 1.0.0 below for the initial public release._

## [1.0.0] — 2026-04-16

First public release. Boutquin.Numerics ships a generic-math numerical methods library for .NET 10: the caller picks the working type (`double`, `float`, `Half`, `decimal`), and every algorithm is parameterised over `T` at the tier its critical path actually needs. Legacy concrete-typed public APIs are kept live as forwarding facades for pre-generic-math consumers.

### Highlights

- **Generic-math surface across the library.** `BrentSolver<T>`, `CholeskyDecomposition<T>`, `OrdinaryLeastSquares<T>`, and every other algorithm accept the caller's choice of `T`. The tier model — Tier A (arithmetic), Tier A + √ (plus `Sqrt` via the internal `NumericPrecision<T>` dispatcher), Tier B (transcendentals, IEEE only), Tier C (polynomial-approximation-bound) — is documented in [`docs/tier-constraints.md`](docs/tier-constraints.md). Tier B on `decimal` is a future enhancement; all other tiers accept `decimal`.
- **Decimal-precision OLS at `T = decimal`.** `OrdinaryLeastSquares<decimal>` runs the full QR factorisation, `Qᵀ·y`, upper-triangular solve, and `(XᵀX)⁻¹` in 28-digit `decimal` on every one of the 11 NIST StRD linear-regression problems. The 7 well-conditioned designs (`Norris`, `NoInt1/2`, `Longley`, `Wampler1/2/3`) clear the 1e-9 coefficient bar; the 4 ill-conditioned designs (`Pontius` cond ≈ 10⁷, `Filip` cond ≈ 10¹⁰, `Wampler4` cond ≈ 5·10¹⁰, `Wampler5` cond ≈ 5·10¹³) hold mathematically principled higher bars (5e-2, 5e-4, 1e-8, 1e-6 respectively), because single-precision decimal QR does not layer extended-precision residual refinement on top and its achievable accuracy is bounded by cond(X)·u. `T = double` remains available with mixed-precision decimal-residual iterative refinement — that path wins on the worst-conditioned designs by running the residual in extended precision.
- **Zero-allocation Levenberg-Marquardt.** `LevenbergMarquardtSolver<T>` with optional `LevenbergMarquardtBuffers<T>` pool. The pooled overload produces bit-identical results to the pool-free overload and allocates exactly zero bytes across 1000 warmup-stabilised inner iterations.
- **92 cross-language verification tests** against independent Python references (NIST StRD, `statsmodels`, `scipy`, `scipy.stats`, `scipy.optimize.minimize('SLSQP')`, `numpy`, `scikit-learn`, `arch`, `mpmath`, plus real Fama-French factor data). Every load-bearing public type ships with at least one external cross-check.
- **Zero `Boutquin.*` domain dependencies**, enforced at build time by architecture tests.

### Added

#### Solvers

- **Root finders** (Tier B at `T : IFloatingPointIeee754<T>`): `BrentSolver<T>`, `BisectionSolver<T>`, `NewtonRaphsonSolver<T>`, `MullerSolver<T>`, `SecantSolver<T>`. Generic interfaces `IBracketedRootSolver<T>`, `IUnbracketedRootSolver<T>`. Result record `RootSolverResult<T>`.
- **Levenberg-Marquardt** (Tier B): `LevenbergMarquardtSolver<T>` with Marquardt scaled-diagonal damping, Nielsen 1999 gain-ratio update, central finite-difference Jacobian, optional analytic Jacobian, optional box bounds via projection, relative function tolerance per MINPACK lmder. `LevenbergMarquardtBuffers<T>` caller-owned scratch pool for zero-allocation iteration in hot paths. Result record `MultivariateSolverResult<T>` reporting parameters, half-SSE cost, residuals, parameter covariance, termination reason, and a `BoundsActive` flag. Interface `IMultivariateLeastSquaresSolver<T>`.
- **Ordinary least squares** (Tier A + √): `OrdinaryLeastSquares<T>.Fit(x, y, includeIntercept)` backed by Householder QR with column pivoting. At `T = double`, mixed-precision iterative refinement uses `decimal` residuals for stability on ill-conditioned problems; at `T = decimal`, the full factorisation runs in 28-digit precision. Result record `OlsResult<T>` reporting coefficients, standard errors, residuals, residual sum of squares, residual standard deviation, degrees of freedom, R², and the covariance matrix.
- **Portfolio optimization** (Tier A + √): `ActiveSetQpSolver<T>` implements two standard long-only portfolio QPs — `SolveMinVariance(covariance, minWeight, maxWeight)` (minimises `w′Σw` subject to `1′w = 1` and `lb ≤ w ≤ ub`) and `SolveMeanVariance(covariance, means, riskAversion, minWeight, maxWeight)` (maximises `w′μ − (λ/2)·w′Σw` under the same constraints). Cholesky-based active-set method delegating to `CholeskyDecomposition<T>`; the inner solve uses the identity `Σ_sub⁻¹ · 1` together with the cross-covariance term `Σ_sub⁻¹ · (Σ_cross^T · w_fixed)` so that fixed-bound variables' linear contributions are correctly applied when the problem is reduced to the free block. Converges in at most 2N iterations (MinVar) or 3N + 3 iterations (MeanVar); `riskAversion = 0` reduces MeanVar to a pure linear programme. `ActiveSetQpSolver` decimal facade preserves pre-generic signatures.

#### Distributions

- **Normal distribution** (Tier C): `NormalDistribution<T>` with `Pdf`, `Cdf`, `InverseCdf`. `CumulativeNormal<T>` uses the Laikov (2025) exponential-free `erf` approximation. `InverseNormal<T>` uses Acklam (2000) polished with two Newton steps against the Laikov CDF.

#### Interpolation

- `IInterpolator<T>` stateless interface. `LinearInterpolator<T>`, `TwoPointLinearInterpolator<T>`, `FlatForwardInterpolator<T>` (Tier A). `LogLinearInterpolator<T>` (Tier B — IEEE only). `MonotoneCubicInterpolator<T>` with arXiv:2402.01324 endpoint improvements, `MonotoneCubicSpline<T>` for O(log n) repeated evaluation, `CubicSplineInterpolator<T>` with five `SplineBoundary<T>` modes (Natural, Clamped, NotAKnot, ModifiedNotAKnot per Jarre 2025, QSpline) — all Tier A + √. `InterpolatorFactory<T>` and `InterpolatorKind` for runtime selection.
- **`MonotoneConvexInterpolator<T>`** (Tier A) — Hagan-West (2006) monotone-convex interpolation (Applied Mathematical Finance 13(2), pp. 89–129) for discount-factor curves. Guarantees **non-negative instantaneous forward rates** across the full maturity range when applied to NCR (normalized cumulative return = `−ln(DF)`) node arrays. The no-arbitrage constraint for yield curves, distinct from `MonotoneCubicInterpolator` which preserves monotonicity of the y-values themselves. `InterpolatorKind.MonotoneConvex = 5` added to the factory.

#### Linear algebra

- All Tier A + √: `CholeskyDecomposition<T>` with unpivoted `Decompose(A)` and pivoted `DecomposePivoted(A, tolerance, maxRank)` for PSD matrices and low-rank approximations (arXiv:2507.20678); `Solve(L, b)` triangular back-substitution. `GaussianElimination<T>` with GERCP randomized partial pivoting (arXiv:2505.02023), deterministic under seed. `JacobiEigenDecomposition<T>` with cyclic Givens rotations, eigenvalues descending with eigenvectors as columns (`EigenResult<T>`). `NearestPsdProjection<T>` with `EigenClip` (one-shot, Higham 1988 closest-in-Frobenius), `Higham` (alternating projection for the nearest correlation matrix), and `IsPsd` check.
- **Principal component analysis** (Tier A + √): `PrincipalComponentAnalysis<T>` built on `JacobiEigenDecomposition<T>`. `Decompose(covarianceMatrix)` for pre-estimated covariance input; `FromReturns(ReturnsMatrix, standardize)` for raw-returns input. `PcaResult<T>` exposes descending eigenvalues, sign-flipped eigenvectors (deterministic across runs), explained-variance ratios, cumulative variance, and mean; `Project(data, k)` and `NumComponentsForExplainedVariance(threshold)` helpers.

#### Statistics

- **Online moments** (Tier A): `WelfordMoments<T>` single-pass mean/variance/stddev with parallel-merge (Chan-Golub-LeVeque 1979). `WelfordMoments<T>.Pearson` online bivariate correlation/covariance.
- **Sample skewness and kurtosis** (Tier A + √): `SampleSkewness<T>.Compute(values)` returns the adjusted Fisher-Pearson standardized third moment with bias-correction factor `n / ((n − 1)(n − 2))`, matching Excel `SKEW` and `scipy.stats.skew(bias=False)`. `SampleExcessKurtosis<T>.Compute(values)` returns the Fisher G₂ excess kurtosis with bias-correction matching Excel `KURT` and `scipy.stats.kurtosis(fisher=True, bias=False)`; uses `long` arithmetic for the integer correction-factor products to prevent `int32` overflow past `n = 1290`. Two-pass algorithm on a `ReadOnlySpan<T>`: Welford mean/variance in pass 1, cubed/fourth standardized deviations in pass 2. `SampleSkewness` and `SampleExcessKurtosis` `decimal`-facade classes preserve pre-generic call sites.
- **Covariance estimators** (Tier A or Tier A + √), all exposing `ICovarianceEstimator<T>.Estimate(ReturnsMatrix<T>)`: `SampleCovarianceEstimator<T>`, `ExponentiallyWeightedCovarianceEstimator<T>`, three Ledoit-Wolf variants (`LedoitWolfShrinkageEstimator<T>`, `LedoitWolfConstantCorrelationEstimator<T>`, `LedoitWolfSingleFactorEstimator<T>`), `OracleApproximatingShrinkageEstimator<T>` (OAS), `QuadraticInverseShrinkageEstimator<T>` (QIS), `DenoisedCovarianceEstimator<T>` (Marchenko-Pastur), `DetonedCovarianceEstimator<T>`, `TracyWidomDenoisedCovarianceEstimator<T>`, `DoublySparseEstimator<T>`, `NercomeCovarianceEstimator<T>`, `PoetCovarianceEstimator<T>`.
- **Correlation primitives**: `PearsonCorrelation<T>` (two-pass, Rolling window variant), `RankCorrelation<T>.Spearman`, `RankCorrelation<T>.KendallTauB` (Tier A). `DistanceCorrelation<T>` (Székely-Rizzo-Bakirov 2007), `FisherZTransform<T>` (Tier B).
- **Returns matrix**: `ReturnsMatrix<T>` value type accepting either T×N (`T[,]`) or asset-major (`T[][]`) layouts with implicit conversions; default method on `ICovarianceEstimator<T>` resolves the layout at the boundary.
- **Backtest statistics** (Tier A with transcendental tail, or Tier B): `DeflatedSharpeRatio<T>` with `DsrResult<T>`. `HaircutSharpe<T>` with Bonferroni/Holm/BHY corrections and `HaircutSharpeResult<T>`. `MinimumTrackRecordLength<T>`. `ProbabilityOfBacktestOverfitting<T>` via CSCV with `PboResult<T>`. `NeweyWestVariance<T>` HAC variance with Bartlett kernel and automatic lag selection. `GeneralizationScore<T>` (Sheppert 2026) four-component composite with `GtScoreResult<T>`.

#### Monte Carlo

- **Bootstrap resamplers**: `BootstrapResampler<T>` (circular block), `MovingBlockBootstrapResampler<T>` (Künsch / Liu-Singh), `StationaryBootstrapResampler<T>` (Politis-Romano geometric blocks), `WildBootstrapResampler<T>` with `WildBootstrapWeights` (Mammen / Rademacher / Gaussian), `Subsampler.Run<T>` (Politis-Romano-Wolf). `FastDoubleBootstrap` (double-only) with `PValueTail` parameter for right-tail / left-tail / two-sided conventions. `PolitisWhiteBlockLength.Estimate<T>` for data-driven optimal block length.
- **Monte Carlo engine**: `BootstrapMonteCarloEngine<T>` (Tier A + √) with single-statistic `Run` and multi-statistic `RunMulti` overload (computes K statistics in one pass). Results: `BootstrapMonteCarloResult<T>`, `MultiStatisticBootstrapResult<T>`.
- **Percentile**: `Percentile.Compute<T>` (NumPy `linear` / R `type=7`), `HarrellDavisPercentile.Compute<T>` with embedded incomplete-beta evaluation, `PSquareEstimator<T>` streaming five-marker estimator (Jain-Chlamtac 1985).
- **Quasi-Monte Carlo**: `HaltonSequence<T>` (up to 50 dimensions), `SobolSequence<T>` (Joe-Kuo 2008, up to 32 dimensions). Both Tier C — direction numbers and scrambling constants calibrated for `double`.

#### Random

- `IRandomSource<T>` with `NextULong`, `NextUInt`, `NextDouble`, `NextInt(upperExclusive)` (Lemire arXiv:1805.10941 nearly-divisionless rejection sampling). `Pcg64RandomSource<T>` (PCG-XSL-RR 128/64, stream-independent via `streamId`). `Xoshiro256StarStarRandomSource<T>` (Blackman-Vigna 2018, `Jump()` advances by 2^128). `GaussianSampler<T>` Marsaglia polar method with cached pair. All Tier C — rejection sampling and bit-tricks require IEEE types.

#### Collections

- **`RollingWindow<T>`** — fixed-capacity circular buffer implementing `IEnumerable<T>`. `new(int capacity)`, `Capacity` / `Count` / `IsFull`, indexer, `Add(item)` (overwrites the oldest element when full), `ToArray()`, `Clear()`, and enumerator. Unconstrained `T` — appropriate for any element type, including non-floating-point payloads. Zero domain dependencies: input validation uses BCL `ArgumentOutOfRangeException.ThrowIfNegativeOrZero` rather than a domain guard helper. Lives in `Boutquin.Numerics.Collections`.

#### Internal helpers

- `NumericPrecision<T>` — generic `Sqrt` dispatcher for `T : IFloatingPoint<T>` via compile-time `typeof(T)` branching (dotnet/runtime#74055 precedent). Delegates to `Math.Sqrt` / `MathF.Sqrt` / `Half.Sqrt` / Newton-Raphson for `decimal`.
- `CovarianceHelpers<T>` shared validation and reconstruction primitives used across every estimator.
- `InterpolationHelper.FindInterval<T>` bracket-and-validate logic shared across interpolators.

#### Validation

- **NIST Statistical Reference Datasets** — full coverage across three suites: 9 univariate-summary problems for `WelfordMoments` at 14-digit precision; all 26 nonlinear-regression problems for `LevenbergMarquardtSolver` converging from NIST Start 2 with finite-difference Jacobian; 11 linear-regression problems for `OrdinaryLeastSquares` at `T = double` (per-problem bars tied to each design's condition number, citing NIST's own c-Wampler5.shtml methodology note for the stability floor) and the same 11 at `T = decimal` (1e-9 on the 7 well-conditioned problems, principled higher bars on the 4 ill-conditioned ones).
- **Python cross-checks** — 92 tests under `Boutquin.Numerics.Tests.Verification` cross-check against independent references. Covariance estimators vs paper-faithful numpy ports across three regimes (small well-conditioned, moderate, overconcentrated). OLS vs `statsmodels.api.OLS` at 1e-10 relative. Levenberg-Marquardt vs `scipy.optimize.least_squares(method='lm')` at 1e-5 relative. `SampleSkewness` and `SampleExcessKurtosis` vs `scipy.stats.skew(bias=False)` and `scipy.stats.kurtosis(fisher=True, bias=False)` at 1e-10 absolute across 6 synthetic cases (normal, gamma, negated-gamma, uniform, t(3), small-n). `ActiveSetQpSolver.SolveMinVariance` / `SolveMeanVariance` vs `scipy.optimize.minimize(method='SLSQP')` at 1e-6 per-weight across 5 portfolio cases (including a boxed 3-asset MinVar case that exercises the cross-covariance sub-problem path). Bootstrap resamplers vs `arch.bootstrap.StationaryBootstrap` / `MovingBlockBootstrap` / `optimal_block_length`. Muller's reference validated against `mpmath.findroot(..., solver='muller')` — independent implementation. `NeweyWestVariance` vs `statsmodels.api.OLS(y, ones).fit().get_robustcov_results('HAC', ...)` at 2.4e-15 relative. `MonotoneConvexInterpolator` vs a paper-faithful hand-port of Hagan-West §2.3–§2.6 at 1e-10 absolute (no scipy equivalent exists for the algorithm). `PrincipalComponentAnalysis` with `standardize: true` verified on 60 months of real Fama-French 5-factor returns.

### Documentation

- `docs/tier-constraints.md` — canonical reference for the generic-math tier model, per-folder tier assignments, the scalar-cast sub-rule for `decimal → double`, the forbidden patterns, and when to reconsider a tier.
- `docs/solvers.md` — user-facing guide for root finders, `LevenbergMarquardtSolver<T>`, and `OrdinaryLeastSquares<T>` with worked examples at both `T = double` (Longley) and `T = decimal` (Wampler5), damping rule, termination semantics, parameter covariance, and the **Hot-path usage — pooled buffers** section.
- `docs/linear-algebra.md` — user-facing guide for the factorizations and `PrincipalComponentAnalysis<T>` with a **caller-chosen precision** worked example running Cholesky at `T = double` and `T = decimal` side-by-side.
- `README.md` — Features with tier labels on every type; Quick Start with generic-math worked examples; Architecture diagram showing the generic surface; Complexity Reference with a Tier column; Design Decisions including *Why generic math*.
- `CLAUDE.md` — Floating-Point Type Selection section organised around the tier model.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` — repository scaffolding.
