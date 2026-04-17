# Boutquin.Numerics

[![NuGet](https://img.shields.io/nuget/v/Boutquin.Numerics.svg)](https://www.nuget.org/packages/Boutquin.Numerics)
[![License](https://img.shields.io/github/license/boutquin/Boutquin.Numerics)](https://github.com/boutquin/Boutquin.Numerics/blob/main/LICENSE.txt)
[![Build](https://github.com/boutquin/Boutquin.Numerics/actions/workflows/pr-verify.yml/badge.svg)](https://github.com/boutquin/Boutquin.Numerics/actions/workflows/pr-verify.yml)

A foundational numerical methods library for the Boutquin financial-software ecosystem. Pure math — zero domain dependencies — implementing root finders, distributions, interpolators, linear algebra, covariance estimators, statistics primitives, Monte Carlo bootstrap variants, quasi-random sequences, and deterministic random number generators. Built with .NET 10 and strict `TreatWarningsAsErrors`.

Every algorithm exposes a generic-math surface (`INumber<T>` / `IFloatingPoint<T>`). The caller — not the library — picks the working type. `BrentSolver<double>`, `CholeskyDecomposition<decimal>`, and `OrdinaryLeastSquares<decimal>` all instantiate the same shared implementation at the precision regime that fits the pipeline. Tier constraints are documented on every type and summarised in `CLAUDE.md`. Legacy concrete-typed facades (`BrentSolver`, `CholeskyDecomposition`, …) stay live — existing callers need no change.

The library sits below MarketData, Curves, Trading, OptionPricing in the dependency graph — algorithms here can be reused without dragging in domain types.

## Solution Structure

| Project | NuGet Package | Description |
|---------|---------------|-------------|
| **Boutquin.Numerics** | `Boutquin.Numerics` | All public API: solvers, distributions, interpolation, linear algebra, statistics, Monte Carlo, random |
| **Boutquin.Numerics.Tests.Unit** | — | 540 xUnit + FluentAssertions tests (411 pre-migration baseline + generic parity, cross-type regression, `NistLinearRegressionDecimalTests`, `MonotoneConvexInterpolatorTests`, `SampleMomentTests`, `RollingWindowTests`, `ActiveSetQpSolverTests`) |
| **Boutquin.Numerics.ArchitectureTests** | — | 5 NetArchTest tests enforcing zero `Boutquin.*` dependencies and namespace purity |
| **Boutquin.Numerics.Tests.Verification** | — | 92 cross-language tests asserting C# kernels match Python reference vectors generated from numpy / scipy / scipy.stats / scipy.optimize / statsmodels / scikit-learn / arch / mpmath |
| **Boutquin.Numerics.BenchMark** | — | BenchmarkDotNet harness covering solvers, interpolation, linear algebra, covariance, bootstrap, RNG, and QMC |

## Features

Every generic public type carries a **tier label** in its XML `<remarks>` summarising the minimum constraint stack it requires and the numeric types it accepts. The tier vocabulary — A, A+√, B, C, and A with transcendental tail — is defined in `CLAUDE.md` → Floating-Point Type Selection.

### Solvers

#### Root Finding

Two interface families based on the bracket requirement, with no shared marker interface — concrete callers pick the appropriate variant. All root finders are **Tier B** (`T : IFloatingPointIeee754<T>`, accepting `double`, `float`, `Half`):

- **`IBracketedRootSolver<T>`** — requires a sign-changing bracket `[lo, hi]`
  - **`BisectionSolver<T>`** — guaranteed linear convergence; the safety net
  - **`BrentSolver<T>`** — superlinear (golden + inverse-quadratic) with **halving-interval guarantee** (Oliveira-Della-Pasqua-Steffen 2024) — worst-case convergence in `⌈log₂((b−a)/tol)⌉` iterations
  - **`NewtonRaphsonSolver<T>`** (safeguarded) — bracketed variant falls back to bisection when Newton diverges
- **`IUnbracketedRootSolver<T>`** — requires only an initial guess
  - **`SecantSolver<T>`** — superlinear without derivative, with `Solve(f, p0, p1)` two-point overload
  - **`MullerSolver<T>`** — handles complex/quadratic-shaped roots, with `Solve(f, x0, x1, x2)` three-point overload
  - **`NewtonRaphsonSolver<T>`** (pure) — quadratic convergence with optional analytical derivative

`RootSolverResult<T>` reports `Converged`, `Iterations`, `FinalResidual`, and `EstimatedError` (bracket half-width or step size).

Legacy concrete-typed types (`BrentSolver`, `NewtonRaphsonSolver`, …) remain live as forwarding facades over the `T = double` instantiation — existing callers need no change. Facade obsoletion is a future enhancement contingent on every downstream consumer migrating to the generic surface; no shipped date. Tier B on `T = decimal` (so `BrentSolver<decimal>`, `LevenbergMarquardtSolver<decimal>`, etc. become available) is also a future enhancement — it requires implementing 28-digit Remez / Chebyshev / CORDIC approximations for `Log`, `Exp`, `Sin`, `Cos`, `Atan`, `Tanh`, `Pow` inside the `NumericPrecision<T>` dispatcher; no shipped date, triggered by a consumer that surfaces the need. See [`docs/tier-constraints.md`](docs/tier-constraints.md) for the tier model that drives both.

**Validation:** every solver (Bisection, Brent, Newton-Raphson bracketed + unbracketed, Secant, Muller) converges to the certified mathematical root (`√2`, `e`, Dottie number, etc.) on the standard test-function pool at 1e-6 absolute. Muller's reference roots are independently validated at generation time against `mpmath.findroot(..., solver='muller')` — arbitrary-precision Muller from a different codebase, catching shared-implementation-bug risk that a numpy hand-port of Faires-Burden §2.6 would not. Generic parity (`BrentSolver<double>` vs legacy `BrentSolver`) is gated by `Solvers_ParityTests.cs`.

#### Multivariate Least Squares

For `r: ℝⁿ → ℝᵐ`, minimizing `½ Σᵢ rᵢ(θ)²`. **Tier B** (`T : IFloatingPointIeee754<T>`):

- **`IMultivariateLeastSquaresSolver<T>`** — contract for nonlinear least-squares solvers using the half-SSE convention (gradient is `Jᵀr`, Gauss-Newton Hessian is `JᵀJ`)
- **`LevenbergMarquardtSolver<T>`** — Marquardt scaled-diagonal damping `(JᵀJ + λ·diag(JᵀJ))·δ = −Jᵀr` (invariant under per-parameter rescaling) with Nielsen 1999 gain-ratio update — `λ ← λ·max(1/3, 1−(2ρ−1)³)` on acceptance, `λ ← λ·ν; ν ← 2ν` on rejection. Finite-difference Jacobian when no analytic form is provided. Optional box bounds via projection. Relative function tolerance per MINPACK lmder for scale-invariance as cost → 0. **Validation:** all 26 NIST StRD nonlinear-regression reference problems converge from NIST Start 2 with finite-difference Jacobian under per-tier tolerances; `scipy.optimize.least_squares(method='lm')` parity at 1e-5 relative on four smooth regimes disjoint from NIST (exponential decay, sum-of-sinusoids, Gompertz, logistic).
- **`LevenbergMarquardtBuffers<T>`** — caller-owned scratch pool enabling zero-allocation iteration. Pool-free and pool-accepting `Solve` overloads produce bit-identical results; the pooled form is appropriate for hot-path consumers (Monte Carlo calibration, bootstrap loops, real-time refits). Grow-only `Reset(parameterCount, residualCount)` supports fitting models of varying sizes against a single pool. Not thread-safe — one pool per thread. See the `Hot-path usage — pooled buffers` section in [`docs/solvers.md`](docs/solvers.md).

`MultivariateSolverResult<T>` reports `Parameters`, `FinalCost = ½ Σᵢ rᵢ²`, `FinalResiduals`, `ParameterCovariance` (`σ̂²·(JᵀJ)⁻¹` at convergence), `Iterations` (accepted-step count), `Converged`, `TerminationReason` (one of `FunctionToleranceReached`, `ParameterToleranceReached`, `GradientToleranceReached`, `MaxIterationsReached`, `DampingOverflow`), and `BoundsActive` (orthogonal — true when any parameter sits on a user-supplied bound at termination).

#### Linear Least Squares

For `y = X·β + ε`, minimizing `‖y − X·β‖₂²`. **Tier A + √** (`T : IFloatingPoint<T>` with `Sqrt` via `NumericPrecision<T>`; accepts `double`, `float`, `Half`, `decimal`):

- **`OrdinaryLeastSquares<T>.Fit(x, y, includeIntercept)`** — Householder QR with column pivoting (LAPACK DGEQP3-style). At `T = double`, mixed-precision iterative refinement computes the residual in `decimal` and feeds it back into the `double` back-substitution (equivalent to double-double refinement but using infrastructure Boutquin.Numerics already has). At `T = decimal`, the full factorisation — QR, `Qᵀ·y`, upper-triangular solve, and `(XᵀX)⁻¹` — runs in 28-digit `decimal`. Preserves the conditioning of `X` end-to-end, unlike normal-equation approaches that form `XᵀX` and double the condition number. Handles the `Longley` (cond ≈ 5·10⁷), `Filip` (cond ≈ 10¹⁰), `Wampler4` (cond ≈ 5·10¹⁰), and `Wampler5` (cond ≈ 5·10¹³) ill-conditioning where a normal-equation solver returns zero correct digits. **Validation:** all 11 NIST StRD linear-regression reference problems — at `T = double`, per-problem accuracy bars from 1e-9 on well-conditioned designs to 1e-6 on Wampler5 (NIST's own methodology note at [`c-Wampler5.shtml`](https://www.itl.nist.gov/div898/strd/lls/data/LINKS/c-Wampler5.shtml) documents this as the double-precision stability floor); at `T = decimal`, 1e-9 on the 7 well-conditioned problems and per-problem bars on the 4 ill-conditioned ones (Pontius 5e-2 at cond ≈ 10⁷, Filip 5e-4 at cond ≈ 10¹⁰, Wampler4 1e-8, Wampler5 1e-6) — the single-precision decimal QR lacks the extended-precision residual refinement that the double-precision path uses, so it caps higher than the mixed-precision double path on the most ill-conditioned designs. `statsmodels.api.OLS` parity at 1e-10 coefficient / 1e-8 derived-quantity relative tolerance across three regimes (well-conditioned random, polynomial degree 5, no-intercept).

`OlsResult<T>` reports `Coefficients` (β̂, length `p + 1` with intercept), `StandardErrors` (√diag(σ̂²·(XᵀX)⁻¹)), `Residuals` (y − X·β̂), `ResidualSumOfSquares`, `ResidualStandardDeviation` (σ̂), `DegreesOfFreedom` (n − p), `RSquared` (centred TSS when `includeIntercept=true`, uncentred otherwise — matches NIST StRD's convention for the `NoInt1`/`NoInt2` problems), and `CovarianceMatrix` (`Cov(β̂) = σ̂²·(XᵀX)⁻¹`, symmetric positive semi-definite at full column rank).

#### Portfolio Optimization (Quadratic Programming)

Two standard long-only portfolio problems under the sum-to-one constraint and per-asset box bounds. **Tier A + √** (`T : IFloatingPoint<T>` via `CholeskyDecomposition<T>`; accepts `double`, `float`, `Half`, `decimal`):

- **`ActiveSetQpSolver<T>.SolveMinVariance(covariance, minWeight, maxWeight)`** — minimum-variance portfolio: `min w′Σw  s.t.  1′w = 1,  lb ≤ w ≤ ub`. Active-set method: solve the Lagrangian `Σ·w = μ·1` with the unbound subset on each iteration via `CholeskyDecomposition<T>`, fix the most-violated bound, release variables whose KKT multipliers change sign. Terminates in at most **2N iterations**. The reduced free-block solve correctly applies the cross-covariance linear term `Σ_sub⁻¹ · (Σ_cross^T · w_fixed)` from the fixed variables, so results remain correct when any asset binds at an interior iteration.
- **`ActiveSetQpSolver<T>.SolveMeanVariance(covariance, means, riskAversion, minWeight, maxWeight)`** — mean-variance portfolio: `max w′μ − (λ/2)·w′Σw  s.t.  1′w = 1,  lb ≤ w ≤ ub`. Same active-set scaffold; the free-block solve uses two Cholesky back-substitutions (one for `Σ⁻¹·1`, one for `Σ⁻¹·μ_free_with_cross_terms`) to assemble the Lagrangian step. Terminates in at most **3N + 3 iterations**. `riskAversion = 0` reduces to a pure linear programme (max expected return under the constraints). **Validation:** 5 portfolio cases (including a boxed 3-asset MinVar that exercises the cross-covariance path) matched against `scipy.optimize.minimize(method='SLSQP')` at 1e-6 per-weight.

### Distributions

**Tier C** (`T : IFloatingPoint<T>`; internal polynomial-approximation tables are calibrated for `double` — `T` at the public surface is cosmetic, used so consumers can compose into generic pipelines without a type-swap):

- **`NormalDistribution<T>`** — `Pdf(x)` with pre-computed `1/√(2π)` constant; `Cdf(x)` and `InverseCdf(p)` convenience methods
- **`CumulativeNormal<T>`** — Laikov (2025, arXiv:2504.05068) exponential-free `erf` approximation, ~14-digit precision, branchless
- **`InverseNormal<T>`** — Acklam (2000) rational approximation polished with two Newton steps using the Laikov CDF, full double precision

Legacy static `NormalDistribution` / `CumulativeNormal` / `InverseNormal` classes remain live as forwarding facades over the `T = double` generic form.

### Interpolation

- **`IInterpolator<T>`** stateless interface with concrete implementations:
  - **`LinearInterpolator<T>`**, **`FlatForwardInterpolator<T>`**, **`TwoPointLinearInterpolator<T>`** — **Tier A** (arithmetic-only); accept `double`, `float`, `Half`, `decimal`
  - **`LogLinearInterpolator<T>`** — **Tier B** (requires `Log` / `Exp` on `T`; IEEE only)
  - **`MonotoneCubicInterpolator<T>`** — **Tier A + √** (Fritsch-Carlson tangents preserving monotonicity, with arXiv:2402.01324 endpoint improvements reducing boundary error from O(h) to O(h²))
  - **`MonotoneConvexInterpolator<T>`** — **Tier A** (Hagan-West 2006, Applied Mathematical Finance 13(2) pp. 89–129). Preserves **non-negative instantaneous forward rates** across the full maturity range when applied to NCR (normalized cumulative return, `−ln(DF)`) node arrays. The relevant no-arbitrage constraint for discount-factor curves — unlike `MonotoneCubicInterpolator`, which preserves monotonicity of the y-values themselves.
- **`MonotoneCubicSpline<T>`** — pre-computed tangents for O(log n) repeated evaluation (Tier A + √)
- **`CubicSplineInterpolator<T>`** (Tier A + √) with five `SplineBoundary<T>` modes:
  - `Natural`, `Clamped(leftSlope, rightSlope)`, `NotAKnot` — classical
  - `ModifiedNotAKnot` — spacing-ratio-weighted not-a-knot for non-uniform grids (Jarre 2025, arXiv:2507.05083)
  - `QSpline` — 4th-order optimal error using only function values
- **`InterpolatorFactory<T>`** + **`InterpolatorKind`** enum for runtime kind selection. The factory throws `NotSupportedException` at runtime when `LogLinear` / `FlatForward` (Tier B) is requested at a non-IEEE `T` such as `decimal` — the tier mismatch is surfaced at construction rather than at use site.

**Validation:** Linear / MonotoneCubic (PCHIP) / natural-cubic-spline vs `scipy.interpolate` at 1e-6 relative; LogLinear / FlatForward / TwoPointLinear vs numpy ports of the textbook formulas at 1e-10 absolute; MonotoneCubicSpline N=2 degenerate-case vs `scipy.interpolate.PchipInterpolator` at 1e-10 absolute; MonotoneConvex vs a paper-faithful hand-port of Hagan-West (2006) §2.3–§2.6 at 1e-10 absolute (no library equivalent exists for this finance-specific algorithm, so the reference is hand-ported from the paper with explicit equation citations). Generic parity (`LinearInterpolator<double>` vs legacy `LinearInterpolator`, all eight interpolators) is gated by `Interpolation_ParityTests.cs`.

### Linear Algebra

All matrix routines are **Tier A + √** (`T : IFloatingPoint<T>` with `Sqrt` via `NumericPrecision<T>`; accept `double`, `float`, `Half`, `decimal`). The legacy concrete-typed facades (`CholeskyDecomposition`, `GaussianElimination`, `JacobiEigenDecomposition`, `NearestPsdProjection`, `PrincipalComponentAnalysis`) forward to the `T = decimal` instantiation, so existing `decimal[,]` callers need no change.

- **`CholeskyDecomposition<T>`** — `Decompose(A)` for symmetric positive-definite matrices; **`DecomposePivoted(A, tolerance, maxRank)`** with diagonal pivoting for positive semi-definite matrices and early termination for low-rank approximations (arXiv:2507.20678); `Solve(L, b)` for triangular back-substitution
- **`JacobiEigenDecomposition<T>.Decompose(A)`** — cyclic Givens rotations, eigenvalues in descending order with eigenvectors as columns. Result record is `EigenResult<T>`.
- **`GaussianElimination<T>.Solve(A, b, seed?)`** — **GERCP randomized partial pivoting** (arXiv:2505.02023) achieving complete-pivoting growth bounds in expectation; deterministic when seeded
- **`NearestPsdProjection<T>`** — Higham (1988) projection onto the PSD cone:
  - **`EigenClip(A)`** — one-shot symmetrize → eigendecompose → clip negative eigenvalues → reconstruct (closest PSD in Frobenius norm)
  - **`Higham(A)`** — alternating projection between PSD cone and unit-diagonal subspace for the **nearest correlation matrix**
  - **`IsPsd(A, tolerance)`** — eigenvalue-based PSD check with configurable absorption of FP drift
- **`PrincipalComponentAnalysis<T>`** — factor-model eigendecomposition built on `JacobiEigenDecomposition<T>`:
  - **`Decompose(covarianceMatrix)`** — takes a pre-computed covariance (useful for running PCA on top of Ledoit-Wolf shrinkage, POET, Tracy-Widom, etc.); `PcaResult<T>.Mean` is empty and the caller pre-centers inputs to `Project`.
  - **`FromReturns(ReturnsMatrix, standardize)`** — takes raw returns, computes sample covariance (or correlation when `standardize: true`) internally, populates `PcaResult<T>.Mean` for downstream projection/reconstruction.
  - **`PcaResult<T>`** exposes `Eigenvalues` (descending — PC1 = largest, matching the Hull Ch 33.6 yield-curve level/slope/curvature convention), `Eigenvectors` (column-stored loadings, sign-flipped so each column's largest-magnitude entry is non-negative for cross-run determinism), `ExplainedVarianceRatio`, `CumulativeExplainedVariance`, `Mean`, plus `Project(data, k)` and `NumComponentsForExplainedVariance(threshold)`.

**Validation:** `CholeskyDecomposition<T>` / `GaussianElimination<T>` / `JacobiEigenDecomposition<T>` / `NearestPsdProjection<T>` vs `scipy.linalg` ground-truth at 1e-8 relative; `PrincipalComponentAnalysis<T>` additionally verified on 60 months of Fama-French 5-factor returns (trace identity under standardization, cumulative variance reaches 1, eigenvector orthonormality at 1e-8). **Cross-type regression** — [`CholeskyCrossTypeTests.cs`](tests/Boutquin.Numerics.Tests.Unit/CrossType/CholeskyCrossTypeTests.cs) runs `CholeskyDecomposition<double>` and `CholeskyDecomposition<decimal>` on a curated 10×10 SPD and asserts their lower factors agree to 12 significant digits — the single test that demonstrates the "caller chooses the precision" proposition is live.

### Statistics

All arithmetic-tier statistics are **Tier A** (`T : IFloatingPoint<T>`; accept `double`, `float`, `Half`, `decimal`). A `ReturnsMatrix<T>` value type accepts both layouts and resolves them at the interface boundary, so callers don't transpose:

- **`ReturnsMatrix<T>`** — readonly struct accepting either T×N (`T[,]`) or asset-major (`T[][]`); implicit conversions from both
- **`ICovarianceEstimator<T>`** — interface with explicit PSD-guarantee documentation and a default-method overload `Estimate(ReturnsMatrix<T>)`

Legacy non-generic `ReturnsMatrix` / `ICovarianceEstimator` / `SampleCovarianceEstimator` / `LedoitWolfShrinkageEstimator` / …  stay live as forwarding facades over the `T = decimal` instantiation.

#### Covariance Estimators

| Estimator | Reference | Use When |
|-----------|-----------|----------|
| **Sample** (N-1 divisor) | classical | baseline; T ≫ N |
| **Exponentially-Weighted Moving Average** | RiskMetrics | non-stationary returns; recent observations should dominate |
| **Ledoit-Wolf** (scaled identity target) | LW 2004, JMA | small T relative to N; minimal structure |
| **Ledoit-Wolf Constant Correlation** | LW 2003, JPM | equity universes with similar pairwise correlations |
| **Ledoit-Wolf Single Factor** | LW 2003, JEF | factor-structured equity data with a market beta |
| **Quadratic Inverse Shrinkage (QIS)** | LW 2022, arXiv:1909.12522 | concentration ratio c = p/n ≳ 0.1; nonlinear shrinkage strictly dominates linear |
| **Oracle Approximating Shrinkage (OAS)** | Chen-Wiesel-Eldar-Hero 2010, arXiv:0907.4698 | Gaussian data; closed-form alternative to LW |
| **Denoised** (Marčenko-Pastur) | López de Prado 2018, Ch. 2 | high-dim portfolios; replace noise eigenvalues with their mean |
| **Detoned** | López de Prado 2020, Ch. 2 | denoising + PC1 (market factor) shrinkage |
| **Tracy-Widom Denoised** | Johnstone 2001; Bun-Bouchaud-Potters 2017, arXiv:1610.08104 | finite-N denoising with sharper signal/noise threshold than MP edge |
| **Doubly Sparse (DSCE)** | Econometrics & Statistics 2024 | denoise eigenvalues AND threshold eigenvector entries for sparsity |
| **NERCOME** | Abadir-Distaso-Žikeš 2014, JoE | design-free split-sample estimator; breaks the in-sample correlation between eigenvectors and eigenvalues |
| **POET** | Fan-Liao-Mincheva 2013, JRSS-B, arXiv:1201.0175 | factor-model data; K leading PCs plus soft-thresholded residual covariance |

Every shrinkage/structured estimator is generic over `T : IFloatingPoint<T>` (Tier A for arithmetic shrinkage; Tier A + √ for estimators that normalise by standard deviations). `LedoitWolfShrinkageEstimator<T>`, `OracleApproximatingShrinkageEstimator<T>`, `PoetCovarianceEstimator<T>`, etc. all accept `T ∈ {double, float, Half, decimal}`. Legacy facades (`LedoitWolfShrinkageEstimator`, …) forward to the `T = decimal` instantiation.

**Validation:** every concrete estimator is cross-checked against a Python reference in `tests/Verification/vectors/covariance.json` across three regimes (small well-conditioned N=8/T=120, moderate N=30/T=60, overconcentrated N=50/T=40). Tiered tolerances reflect each estimator's algorithmic structure — shrinkage-only estimators (Sample, LW-family, OAS) agree at 1e-10 relative (no eigendecomposition to propagate LAPACK-vs-Jacobi noise); eigendecomposition-based estimators (Denoised, Detoned, Tracy-Widom, QIS, DoublySparse) at 1e-5 relative; POET at 1e-4 (threshold-nonlinearity amplification); NERCOME at 1e-1 on the well-conditioned regime only (rank-near-deficient splits in the other regimes make the reconstruction legitimately non-unique in the null-space basis). Python references: `sklearn.covariance.OAS` convention-adjusted for the N−1 divisor, paper-faithful numpy ports for LW-CC / LW-SF / QIS / Denoised / Detoned / TW / NERCOME / POET / DSCE with published equation references. **Cross-type regression** — [`LedoitWolfCrossTypeTests.cs`](tests/Boutquin.Numerics.Tests.Unit/CrossType/LedoitWolfCrossTypeTests.cs) runs `LedoitWolfShrinkageEstimator<double>` and `LedoitWolfShrinkageEstimator<decimal>` on a 20-asset 500-observation returns matrix and asserts shrinkage intensities agree to 10 digits — the arithmetic-only shrinkage formula is type-invariant by construction, and the test proves it on the shipped code.

#### Correlation Primitives

- **`PearsonCorrelation<T>.Compute`** (Tier A) — two-pass algorithm avoiding catastrophic cancellation; clamps to [−1, 1]; `Rolling` window variant
- **`RankCorrelation<T>.Spearman`** (Tier A) — Pearson on average ranks; detects monotonic non-linear dependence
- **`RankCorrelation<T>.KendallTauB`** (Tier A) — concordant-minus-discordant pairs with tie correction
- **`DistanceCorrelation<T>`** (Tier B — uses `Sqrt`/`Log`) — Székely-Rizzo-Bakirov 2007 (arXiv:0803.4101); detects arbitrary nonlinear dependence; dCor = 0 ⟺ independence
- **`FisherZTransform<T>`** (Tier B — uses `Atanh`) — `Forward` / `Inverse` / `ConfidenceInterval` for Pearson r

#### Online Moments

- **`WelfordMoments<T>`** (Tier A) — single-pass numerically-stable mean/variance (Welford 1962); `Compute(span)` static helper for batch use; `Merge(other)` and static `Combine(a, b)` parallel-merge (Chan-Golub-LeVeque 1979) for map-reduce aggregation
- **`WelfordMoments<T>.Pearson`** (Tier A) — single-pass online correlation/covariance (Chan-Golub-LeVeque 1979 bivariate form)

#### Higher Sample Moments

- **`SampleSkewness<T>.Compute(values)`** (Tier A + √) — adjusted Fisher-Pearson standardized third moment with bias-correction factor `n / ((n − 1)(n − 2))`. Matches Excel `SKEW` and `scipy.stats.skew(bias=False)`. Two-pass algorithm on a `ReadOnlySpan<T>`: Welford mean/variance in pass 1, cubed standardized deviations in pass 2.
- **`SampleExcessKurtosis<T>.Compute(values)`** (Tier A + √) — Fisher G₂ excess kurtosis with bias correction, matching Excel `KURT` and `scipy.stats.kurtosis(fisher=True, bias=False)`. Uses `long` arithmetic for the integer correction-factor products to prevent `int32` overflow past `n = 1290`.

**Validation:** both estimators agree with `scipy.stats.skew` / `scipy.stats.kurtosis` at 1e-10 absolute across 6 synthetic cases (normal, gamma, negated-gamma, uniform, t(3), small-n) via [`SampleMomentsVerificationTests.cs`](tests/Boutquin.Numerics.Tests.Verification/SampleMomentsVerificationTests.cs).

#### Backtest Statistics

Hybrid-tier statistics are **Tier A with transcendental tail** — the inner loop is arithmetic in `T`, and a final scalar `Log` / `Sqrt` / inverse-normal applies once. At `T = decimal`, the tail casts to `double`, runs the BCL function, and casts back (a single scalar, not an inner loop — the rule is documented in `CLAUDE.md` → Floating-Point Type Selection). At `T = double`, the tail is native.

- **`DeflatedSharpeRatio<T>`** (Tier A w/ transcendental tail) — Bailey-LdP 2014 multiple-testing correction; `Compute(...)` from pre-computed inputs or `ComputeFromReturns(...)` from a return series. Result: `DsrResult<T>`.
- **`HaircutSharpe<T>`** (Tier A w/ transcendental tail) — Harvey-Liu-Zhu 2016 with Bonferroni / Holm / BHY corrections; reports adjusted p-values plus the Sharpe haircut amount. Result: `HaircutSharpeResult<T>`.
- **`MinimumTrackRecordLength<T>`** (Tier B) — Bailey-LdP 2012 (arXiv:1205.1480); inverse of DSR — minimum sample length needed for an observed Sharpe to be statistically distinguishable from a benchmark
- **`ProbabilityOfBacktestOverfitting<T>`** (Tier A w/ transcendental tail) — Bailey-Borwein-LdP-Zhu 2014 (arXiv:1109.0776); CSCV (Combinatorially-Symmetric Cross-Validation) over a strategy panel, reporting PBO + logit distribution. Result: `PboResult<T>`.
- **`NeweyWestVariance<T>`** (Tier A w/ transcendental tail) — HAC variance with Bartlett kernel (Newey-West 1987); `AutomaticLags(T)` helper per Newey-West 1994
- **`GeneralizationScore<T>`** (Tier B) — GT-Score (Sheppert 2026, arXiv:2602.00080) composite anti-overfitting metric. Result: `GtScoreResult<T>`.

**Validation:** `DeflatedSharpeRatio` / `HaircutSharpe` / `MinimumTrackRecordLength` vs the embedded `dsr.json` vectors at 1e-8 relative; `ProbabilityOfBacktestOverfitting` vs a numpy port of Bailey et al. 2014 CSCV at 1e-8 relative with stable-sort tie-breaking matching the C# `OrderBy`; `NeweyWestVariance` vs `statsmodels.api.OLS(y, ones).fit().get_robustcov_results('HAC', ...)` at 1e-10 absolute (two genuinely independent implementations — sandwich-formula in statsmodels, direct-sum in Numerics — cross-check catches shared-implementation-bug risk); `GeneralizationScore` vs a numpy port of the Sheppert 2026 four-component formula at 1e-10 absolute per-component; `FisherZTransform` and `DistanceCorrelation` vs the Fisher 1915 / Székely-Rizzo-Bakirov 2007 analytic closed forms.

### Monte Carlo

#### Bootstrap Variants

Bootstrap resamplers are **Tier A** (arithmetic resampling / order statistics). `BootstrapMonteCarloEngine<T>` is Tier A + √ (standard-error computation uses `Sqrt`). `FastDoubleBootstrap`, `SobolSequence<T>`, and `HaltonSequence<T>` are Tier C (inverse-normal / direction numbers calibrated for `double`).

| Variant | Generic | Tier | Block size | Stationarity | Reference |
|---------|---------|------|------------|--------------|-----------|
| **`BootstrapMonteCarloEngine<T>`** (IID) | ✓ | A + √ | n/a | strictly stationary on IID | classical |
| **`BootstrapResampler<T>`** (circular block) | ✓ | A | fixed | approximately stationary | Politis-Romano 1992 |
| **`MovingBlockBootstrapResampler<T>`** | ✓ | A | fixed, no wrap | edge-effect biased | Künsch 1989; Liu-Singh 1992 |
| **`StationaryBootstrapResampler<T>`** | ✓ | A | geometric | strictly stationary | Politis-Romano 1994 |
| **`WildBootstrapResampler<T>`** | ✓ | A | n/a | preserves residual structure | Mammen 1993; Davidson-Flachaire 2008 |
| **`Subsampler.Run<T>`** | ✓ | A | b ≪ T overlapping | consistent under unit roots / extremes | Politis-Romano-Wolf 1999 |
| **`FastDoubleBootstrap`** | double-only | C | nested 1×1 | bias-corrected p-values | Davidson-MacKinnon 2007 |

- **`PolitisWhiteBlockLength.Estimate<T>(series)`** (Tier A) — data-driven optimal block length for stationary bootstrap (Politis-White 2004)
- **`BootstrapMonteCarloEngine<T>.RunMulti(...)`** — multi-statistic single-pass overload; computes K statistics on each resampled path simultaneously, avoiding K separate passes for bundles like `(Sharpe, Sortino, MaxDD, IR)`
- **`WildBootstrapWeights`** enum: `Mammen` (two-point, third-moment matching), `Rademacher` (±1 symmetric — Davidson-Flachaire default), `Gaussian`

Legacy concrete-typed facades (`BootstrapResampler`, `StationaryBootstrapResampler`, …) remain live as forwarding shims over the `T = decimal` generic form.

**Validation:** `StationaryBootstrapResampler` and `MovingBlockBootstrapResampler` resampled-mean distributions fall within the 95% band produced by `arch.bootstrap.StationaryBootstrap` / `MovingBlockBootstrap` references at 3000 trials; `WildBootstrapResampler` Mammen and Rademacher weight distributions match the analytic moments within 4·SE at 20 000 trials; `FastDoubleBootstrap.PValue` matches a numpy port of Davidson-MacKinnon 2007 within 8% absolute across all three `PValueTail` conventions (right, left, two-sided); `Subsampler` is deterministic and compared bit-identically to a numpy port; `PolitisWhiteBlockLength` agrees with `arch.bootstrap.optimal_block_length` within ±2 per spec §2.3.

#### Percentile

- **`Percentile.Compute<T>(sorted, p)`** (Tier A) — linear interpolation, NumPy `linear` / R `type=7` convention
- **`HarrellDavisPercentile.Compute<T>(sorted, p)`** (Tier A) — Harrell-Davis 1982 with embedded incomplete-beta evaluation; lower variance than linear for small samples (the bootstrap-CI sweet spot)
- **`PSquareEstimator<T>(p)`** (Tier A) — Jain-Chlamtac 1985 online percentile; five-marker parabolic-interpolation estimator with O(1) memory for unbounded streams (streaming bootstrap / very-large MC)

**Validation:** `HarrellDavisPercentile.Compute` vs a `scipy.special.betainc`-driven reference at five quantiles (10/25/50/75/90) at 1e-8 relative — the C# class uses a Numerical Recipes continued-fraction embed, scipy uses Lanczos coefficients, so agreement at this tolerance catches implementation drift in either library's incomplete-beta approximation. `PSquareEstimator.Estimate` vs a verbatim numpy port of the Jain-Chlamtac 1985 five-marker state transitions at 1e-10 absolute across three percentiles (0.5/0.9/0.99).

#### Quasi-Monte Carlo

- **`HaltonSequence<T>`** (Tier C) — radical-inverse in distinct primes, up to 50 dimensions; supports `skip` for low-index correlation avoidance
- **`SobolSequence<T>`** (Tier C) — Joe-Kuo (2008) direction numbers, up to 32 dimensions; `Jump` not exposed (use `skip`)

### Random Number Generation

Deterministic, cross-runtime-stable PRNGs replacing `System.Random` for reproducibility. All **Tier C** (`T : IFloatingPoint<T>` at the public surface; rejection sampling and the Gaussian polar method use IEEE bit tricks internally, bounding `T` to `double` / `float` / `Half` for the sampled-value path):

- **`IRandomSource<T>`** — interface with `NextULong`, `NextUInt`, `NextDouble`, `NextInt(upperExclusive)` (Lemire arXiv:1805.10941 nearly-divisionless rejection sampling)
- **`Pcg64RandomSource<T>`** — PCG-XSL-RR 128/64 (O'Neill 2014); period 2^128, k-equidistributed to k = 2, BigCrush-passing, **stream-independent** via the `streamId` constructor parameter — independent streams from the same seed for parallel MC
- **`Xoshiro256StarStarRandomSource<T>`** — Blackman-Vigna 2018 (arXiv:1805.01407); period 2^256 − 1; `Jump()` advances by 2^128 calls for partitioned parallel streams
- **`GaussianSampler<T>`** — Marsaglia polar method with cached pair; `Next()`, `Next(mean, stdDev)`, `NextBatch(count)`

### Collections

- **`RollingWindow<T>`** — fixed-capacity circular buffer implementing `IEnumerable<T>`. `new(int capacity)`, `Capacity` / `Count` / `IsFull`, indexer (oldest-to-newest), `Add(item)` (overwrites the oldest element when full), `ToArray()`, `Clear()`. Unconstrained `T` — suitable for any payload, not just floating-point types. Zero domain dependencies (input validation uses BCL `ArgumentOutOfRangeException.ThrowIfNegativeOrZero`). Lives in `Boutquin.Numerics.Collections`; intended for streaming-window algorithms (rolling moments, online filters, regime detectors) and hot-path MC bookkeeping.

### Internal Helpers (not part of public API)

- **`NumericPrecision<T>`** — compile-time-dispatched generic `Sqrt` for `T : IFloatingPoint<T>` — delegates to `Math.Sqrt` / `MathF.Sqrt` / `Half.Sqrt` / Newton-Raphson for `decimal`. Powers every Tier A + √ algorithm. Design follows the `System.Numerics.Tensors` precedent for typeof-based dispatch ([dotnet/runtime#74055](https://github.com/dotnet/runtime/pull/74055)).
- **`CovarianceHelpers<T>`** — shared `ValidateReturns`, `ComputeMeans`, `ComputeSampleCovariance`, `CovarianceToCorrelation`, `CorrelationToCovariance`, `MarcenkoPasturUpperBound`, `ReconstructFromEigen` — used by every estimator
- **`InterpolationHelper.FindInterval<T>`** — bracket-and-validate logic shared across interpolators

## Quick Start

### Installation

```sh
dotnet add package Boutquin.Numerics
```

### Root Finding

```csharp
using Boutquin.Numerics.Solvers;

// Brent — bracketed, superlinear, halving-interval-guaranteed.
var brent = new BrentSolver(tolerance: 1e-12, maxIterations: 100);
var result = brent.Solve(x => x * x - 2.0, lowerBound: 0.0, upperBound: 2.0);
// result.Root ≈ 1.41421356...
// result.Converged == true
// result.Iterations < 50

// Newton-Raphson with analytical derivative — quadratic convergence.
var newton = new NewtonRaphsonSolver(derivative: x => 2.0 * x);
var nr = newton.Solve(x => x * x - 2.0, initialGuess: 1.0);

// Secant — derivative-free, superlinear; two-point overload.
var secant = new SecantSolver();
var sr = secant.Solve(f: x => x * x - 2.0, p0: 0.5, p1: 2.0);
```

### Distributions

```csharp
using Boutquin.Numerics.Distributions;

double pdf  = NormalDistribution.Pdf(0.0);          // 0.39894228...
double cdf  = NormalDistribution.Cdf(1.96);         // 0.97500...
double quantile = NormalDistribution.InverseCdf(0.975); // 1.95996...

// Direct access to the precision-tuned underlying primitives.
double cdfDirect = CumulativeNormal.Evaluate(1.96);
double inverseDirect = InverseNormal.Evaluate(0.975);
```

### Interpolation

```csharp
using Boutquin.Numerics.Interpolation;

double[] xs = [1.0, 2.0, 3.0, 4.0, 5.0];
double[] ys = [1.0, 4.0, 9.0, 16.0, 25.0];

// Stateless interpolators — pass xs/ys at every call.
double linear = LinearInterpolator.Instance.Interpolate(2.5, xs, ys);   // 6.5
double mono   = MonotoneCubicInterpolator.Instance.Interpolate(2.5, xs, ys);

// Pre-computed cubic spline — O(log n) repeated evaluation.
var spline = new CubicSplineInterpolator(xs, ys, SplineBoundary.NotAKnot);
double y25 = spline.Interpolate(2.5);

// Modified not-a-knot for non-uniform grids.
var modSpline = new CubicSplineInterpolator(xs, ys, SplineBoundary.ModifiedNotAKnot);

// Q-Spline — 4th-order optimal error from values alone.
var qSpline = new CubicSplineInterpolator(xs, ys, SplineBoundary.QSpline);

// Runtime kind selection.
var dynamic = InterpolatorFactory.Create(InterpolatorKind.MonotoneCubic);

// Hagan-West monotone-convex for yield-curve NCR nodes. Input convention:
// xs[0]=0, ys[0]=0 is the virtual origin; ys[i] = -ln(P(xs[i])).
// Convert the result back to a discount factor via P(x) = exp(-ncr).
double[] curveXs = [0.0, 0.25, 1.0, 5.0, 10.0, 30.0];
double[] curveYs = [0.0, -Math.Log(0.995), -Math.Log(0.97),
                         -Math.Log(0.83),  -Math.Log(0.68), -Math.Log(0.25)];
double ncr2y = MonotoneConvexInterpolator.Instance.Interpolate(2.0, curveXs, curveYs);
double df2y  = Math.Exp(-ncr2y);  // implied 2y discount factor, non-negative forward rates guaranteed
```

### Linear Algebra

```csharp
using Boutquin.Numerics.LinearAlgebra;

// Cholesky for symmetric positive-definite matrices.
decimal[,] spd =
{
    { 4m, 12m, -16m },
    { 12m, 37m, -43m },
    { -16m, -43m, 98m },
};
decimal[,] L = CholeskyDecomposition.Decompose(spd);
decimal[] solution = CholeskyDecomposition.Solve(L, [1m, 2m, 3m]);

// Pivoted Cholesky for PSD matrices and low-rank approximations.
var pivoted = CholeskyDecomposition.DecomposePivoted(
    spd, tolerance: 1e-10m, maxRank: 2);
// pivoted.Lower (N×Rank), pivoted.Permutation, pivoted.Rank

// Eigendecomposition (descending eigenvalues).
var eigen = JacobiEigenDecomposition.Decompose(spd);
// eigen.Values, eigen.Vectors (column-stored)

// Gaussian elimination with randomized pivoting.
decimal[] x = GaussianElimination.Solve(spd, [1m, 2m, 3m], seed: 42);

// Project an indefinite matrix onto the PSD cone.
decimal[,] psd = NearestPsdProjection.EigenClip(maybeIndefinite);

// Find the nearest valid correlation matrix.
decimal[,] cleanCorr = NearestPsdProjection.Higham(messyCorr);

// Verify PSD-ness with FP-drift tolerance.
bool ok = NearestPsdProjection.IsPsd(myMatrix);
```

### Nonlinear Least Squares

```csharp
using Boutquin.Numerics.Solvers;

// Fit y = a·exp(b·x) to (xs, ys) — classic exponential decay calibration.
double[] xs = LoadXs();
double[] ys = LoadYs();

double[] Residuals(double[] theta)
{
    var a = theta[0];
    var b = theta[1];
    var r = new double[xs.Length];
    for (var i = 0; i < xs.Length; i++)
    {
        r[i] = ys[i] - a * Math.Exp(b * xs[i]);
    }
    return r;
}

var solver = new LevenbergMarquardtSolver();
var result = solver.Solve(
    residuals: Residuals,
    initialGuess: [1.0, 0.1]);

if (result.Converged)
{
    double a = result.Parameters[0];
    double b = result.Parameters[1];
    // Standard errors from the Gauss-Newton asymptotic approximation.
    double sigmaA = Math.Sqrt(result.ParameterCovariance![0, 0]);
    double sigmaB = Math.Sqrt(result.ParameterCovariance[1, 1]);
}

// Analytic Jacobian — faster and more accurate than central finite differences.
double[,] Jacobian(double[] theta) => /* ∂rᵢ/∂θⱼ, shape [m, n] */;
var analytic = solver.Solve(Residuals, [1.0, 0.1], Jacobian);

// Box-bounded calibration — parameters clipped to [lower, upper] via projection.
var bounded = solver.Solve(
    Residuals, [1.0, 0.1],
    jacobian: null,
    lowerBounds: [0.0, -1.0],
    upperBounds: [10.0, 1.0]);
// bounded.BoundsActive == true iff any parameter sits on a supplied bound at termination.

// Hot-path consumers (bootstrap, Monte Carlo, real-time refit) pool scratch buffers
// once and reuse them across solves — bit-identical results with zero per-iteration
// managed-heap allocation.
var buffers = new LevenbergMarquardtBuffers(parameterCount: 2, residualCount: xs.Length);
foreach (var resample in bootstrap.Resamples)
{
    var pooled = solver.Solve(resample.Residuals, [1.0, 0.1], buffers);
    // ...
}
```

### Ordinary Least Squares

```csharp
using Boutquin.Numerics.Solvers;

// NIST StRD "Longley" — cond ≈ 5·10⁷, the canonical ill-conditioned OLS problem.
// Normal-equation OLS returns garbage here; QR with pivoting recovers every digit.
var x = new double[16, 6]
{
    {  83.0, 234289, 2356, 1590, 107608, 1947 },
    {  88.5, 259426, 2325, 1456, 108632, 1948 },
    //  … 14 more rows …
};
var y = new[] { 60323.0, 61122, /* … */ };

var fit = OrdinaryLeastSquares.Fit(x, y);

// Coefficients are returned in [intercept, b1, …, bp] order.
double intercept  = fit.Coefficients[0];
double slopeB1    = fit.Coefficients[1];
double seB1       = fit.StandardErrors[1];
double sigmaHat   = fit.ResidualStandardDeviation;
int    dof        = fit.DegreesOfFreedom;            // n − p
double rSquared   = fit.RSquared;

// Fit through origin (no intercept) — NoInt1/NoInt2 pattern.
var noIntercept = OrdinaryLeastSquares.Fit(x, y, includeIntercept: false);
```

### Portfolio Optimization (Quadratic Programming)

```csharp
using Boutquin.Numerics.Solvers;

// Minimum-variance portfolio — min w'Σw s.t. 1'w = 1, 0.0 ≤ w_i ≤ 0.4.
decimal[,] covariance = LoadAssetCovariance();  // N×N SPD
decimal[] minVarWeights = ActiveSetQpSolver.SolveMinVariance(
    covariance,
    minWeight: 0.0m,
    maxWeight: 0.4m);

// Mean-variance portfolio with λ = 3 — max w'μ − (λ/2)·w'Σw under the same bounds.
decimal[] expectedReturns = LoadExpectedReturns();
decimal[] mvWeights = ActiveSetQpSolver.SolveMeanVariance(
    covariance,
    means: expectedReturns,
    riskAversion: 3.0m,
    minWeight: 0.0m,
    maxWeight: 0.4m);

// Generic form — instantiate at T = double for hot-path Monte Carlo calibration.
var mvDouble = ActiveSetQpSolver<double>.SolveMeanVariance(
    covarianceD, meansD, riskAversion: 3.0, minWeight: 0.0, maxWeight: 0.4);
```

### Principal Component Analysis

```csharp
using Boutquin.Numerics.LinearAlgebra;
using Boutquin.Numerics.Statistics;

// From raw returns — the most common path.
decimal[,] returns = LoadReturns();                    // rows = periods, cols = variables
var pca = PrincipalComponentAnalysis.FromReturns(returns);

// Explained variance and factor count for a target threshold.
decimal level     = pca.ExplainedVarianceRatio[0];     // PC1 share
decimal cumThree  = pca.CumulativeExplainedVariance[2]; // top-3 share
int     k95       = pca.NumComponentsForExplainedVariance(0.95m);

// Project observations onto the top-k principal components.
decimal[,] scores = pca.Project(returns, numComponents: k95);

// Standardize — divides each column by its sample std before covariance estimation
// (equivalent to PCA on the correlation matrix). Useful when variables span
// heterogeneous scales.
var std = PrincipalComponentAnalysis.FromReturns(returns, standardize: true);

// From a pre-estimated covariance — typical for curve-risk workflows that already
// shrunk with Ledoit-Wolf, Tracy-Widom, POET, or DSCE before running PCA.
decimal[,] shrunk = new LedoitWolfShrinkageEstimator().Estimate(returns);
var pcaShrunk = PrincipalComponentAnalysis.Decompose(shrunk);
// pcaShrunk.Mean is empty — caller pre-centers inputs to Project when using this path.
```

### Covariance Estimation

```csharp
using Boutquin.Numerics.Statistics;

// T×N return matrix (rows = time, columns = assets).
decimal[,] returns = LoadReturns();

// Pick the appropriate estimator for your sample regime.
var sample   = new SampleCovarianceEstimator().Estimate(returns);
var ewma     = new ExponentiallyWeightedCovarianceEstimator(lambda: 0.94m).Estimate(returns);
var lwLin    = new LedoitWolfShrinkageEstimator().Estimate(returns);
var lwCC     = new LedoitWolfConstantCorrelationEstimator().Estimate(returns);
var lwFM     = new LedoitWolfSingleFactorEstimator().Estimate(returns);
var qis      = new QuadraticInverseShrinkageEstimator().Estimate(returns);
var oas      = new OracleApproximatingShrinkageEstimator().Estimate(returns);
var denoised = new DenoisedCovarianceEstimator(applyLedoitWolfShrinkage: false).Estimate(returns);
var detoned  = new DetonedCovarianceEstimator(detoningAlpha: 1m).Estimate(returns);
var tw       = new TracyWidomDenoisedCovarianceEstimator().Estimate(returns);
var dsce     = new DoublySparseEstimator(eigenvectorThreshold: 0.1m).Estimate(returns);

// Asset-major (Trading-shape) input via ReturnsMatrix wrapper.
decimal[][] assetMajor = LoadAssetMajorReturns();
ICovarianceEstimator estimator = new SampleCovarianceEstimator();
decimal[,] cov = estimator.Estimate(new ReturnsMatrix(assetMajor));

// Enforce PSD before passing to a Cholesky-based portfolio optimizer.
decimal[,] safeCov = NearestPsdProjection.EigenClip(detoned);
```

### Correlation

```csharp
using Boutquin.Numerics.Statistics;

decimal[] x = LoadSeriesA();
decimal[] y = LoadSeriesB();

// Pearson — linear dependence, two-pass algorithm.
decimal pearson = PearsonCorrelation.Compute(x, y);
decimal[] rolling = PearsonCorrelation.Rolling(x, y, windowSize: 21);

// Rank correlations — robust to outliers, detect monotonic dependence.
decimal spearman = RankCorrelation.Spearman(x, y);
decimal kendall  = RankCorrelation.KendallTauB(x, y);

// Distance correlation — detects arbitrary nonlinear dependence.
double[] xd = ToDoubles(x);
double[] yd = ToDoubles(y);
double dcor = DistanceCorrelation.Compute(xd, yd);

// Fisher z-transform CI on Pearson r.
var (lo, hi) = FisherZTransform.ConfidenceInterval(r: 0.6, n: 100, confidenceLevel: 0.95);
```

### Online Moments

```csharp
using Boutquin.Numerics.Statistics;

var w = new WelfordMoments();
foreach (var x in stream)
{
    w.Add(x);
}
decimal mean = w.Mean;
decimal var_ = w.Variance;
decimal sd   = w.StdDev;

// Single-pass batch convenience.
var (m, v) = WelfordMoments.Compute(span);

// Bivariate (online correlation).
var pearson = new WelfordMoments.Pearson();
foreach (var (x, y) in pairedStream)
{
    pearson.Add(x, y);
}
decimal r   = pearson.Correlation;
decimal cov = pearson.Covariance;
```

### Higher Sample Moments

```csharp
using Boutquin.Numerics.Statistics;

// Fisher-Pearson bias-adjusted G1 / G2 — matches Excel SKEW / KURT and
// scipy.stats.skew(bias=False) / scipy.stats.kurtosis(fisher=True, bias=False).
decimal[] returns = LoadReturns();
decimal skew = SampleSkewness.Compute(returns);          // third standardized moment
decimal exKurt = SampleExcessKurtosis.Compute(returns);  // G2 excess kurtosis (normal -> 0)

// Generic form — run the same moments at double speed for a downstream
// double-denominated pipeline.
double[] returnsD = LoadReturnsDouble();
double skewD  = SampleSkewness<double>.Compute(returnsD);
double kurtD  = SampleExcessKurtosis<double>.Compute(returnsD);
```

### Backtest Statistics

```csharp
using Boutquin.Numerics.Statistics;

// Deflated Sharpe Ratio — adjust observed SR for multiple-testing bias.
var dsr = DeflatedSharpeRatio.Compute(
    observedSharpe: 1.5m,
    numTrials: 1000,
    backTestYears: 5m,
    skewness: -0.5m,
    kurtosis: 4.0m);
// dsr.DeflatedSharpe, dsr.PValue, dsr.ExpectedMaxSharpe

// Or directly from a return series.
decimal[] dailyReturns = LoadReturns();
var dsrFromReturns = DeflatedSharpeRatio.ComputeFromReturns(dailyReturns, numTrials: 100);

// Harvey-Liu-Zhu Sharpe haircut with Bonferroni / Holm / BHY corrections.
var hlz = HaircutSharpe.Compute(observedSharpe: 1.5m, numTrials: 1000, backTestYears: 5m);
// hlz.HaircutSharpe, hlz.HaircutAmount, hlz.BonferroniPValue, hlz.HolmPValue, hlz.BhyPValue

// Minimum Track Record Length — inverse of DSR.
double minYears = MinimumTrackRecordLength.Compute(observedSharpe: 1.5);

// Probability of Backtest Overfitting via CSCV.
decimal[,] strategyPanel = LoadStrategyReturns(); // [strategy, time]
var pbo = ProbabilityOfBacktestOverfitting.Compute(strategyPanel, splitCount: 16);
// pbo.Pbo, pbo.LogitMedian, pbo.LogitValues

// HAC variance for autocorrelated return series.
int lags = NeweyWestVariance.AutomaticLags(returns.Length);
decimal hacVar = NeweyWestVariance.MeanVariance(returns, lags);
```

### Monte Carlo Bootstrap

```csharp
using Boutquin.Numerics.MonteCarlo;
using Boutquin.Numerics.Random;

decimal[] returns = LoadReturns();

// IID bootstrap with deterministic seeded PCG64.
var iid = BootstrapMonteCarloEngine.FromSeed(simulationCount: 5000, seed: 42);
var iidResult = iid.Run(returns, paths => paths.Average());
// iidResult.Mean, .Median, .Percentile5, .Percentile95, .Statistics

// Stationary bootstrap with Politis-White auto block length.
double blockLength = PolitisWhiteBlockLength.Estimate(returns);
var stationary = StationaryBootstrapResampler.FromSeed(blockLength, seed: 42);
decimal[] resampled = stationary.Resample(returns);

// Multi-statistic bootstrap (Sharpe + Sortino + MaxDD in one pass).
var multi = iid.RunMulti(
    returns,
    statisticCount: 3,
    statisticWriter: (path, output) =>
    {
        output[0] = ComputeSharpe(path);
        output[1] = ComputeSortino(path);
        output[2] = ComputeMaxDrawdown(path);
    },
    names: ["sharpe", "sortino", "maxdd"]);

// Wild bootstrap for heteroskedastic residuals.
var wild = WildBootstrapResampler.FromSeed(WildBootstrapWeights.Rademacher, seed: 42);
decimal[] wildResampled = wild.Resample(residuals);

// Subsampling for non-stationary or extreme-value statistics.
int b = Subsampler.SuggestSubsampleLength(returns.Length); // T^(2/3)
decimal[] subsamples = Subsampler.Run(returns, b, span =>
{
    decimal sum = 0m;
    for (var i = 0; i < span.Length; i++) sum += span[i];
    return sum / span.Length;
});

// Fast Double Bootstrap for bias-corrected p-values.
var fdb = FastDoubleBootstrap.FromSeed(outerCount: 500, seed: 42);
decimal pValue = fdb.PValue(returns, observed: 0.005m, paths => paths.Average());

// Harrell-Davis percentile — lower-variance than linear for small samples.
decimal[] sorted = (decimal[])iidResult.Statistics;
Array.Sort(sorted);
decimal hd05 = HarrellDavisPercentile.Compute(sorted, 0.05);
```

### Quasi-Monte Carlo

```csharp
using Boutquin.Numerics.MonteCarlo;

// Halton — good for d ≲ 8.
var halton = new HaltonSequence(dimension: 4, skip: 64);
for (var i = 0; i < 10_000; i++)
{
    double[] point = halton.Next(); // length 4, each coord in [0, 1)
}

// Sobol — better for d > 8 (up to 32 supported).
var sobol = new SobolSequence(dimension: 16, skip: 1024);
for (var i = 0; i < 10_000; i++)
{
    double[] point = sobol.Next();
}
```

### Random Number Generation

```csharp
using Boutquin.Numerics.Random;

// PCG-64 — independent streams via streamId.
var rng = new Pcg64RandomSource(seed: 42UL, streamId: 0UL);
ulong u = rng.NextULong();
double d = ((IRandomSource)rng).NextDouble();
int i = ((IRandomSource)rng).NextInt(upperExclusive: 100);

// xoshiro256** — slightly faster, with Jump for parallel partitioning.
var xrng = new Xoshiro256StarStarRandomSource(seed: 42UL);
xrng.Jump(); // advance by 2^128 calls — independent stream for worker thread

// Marsaglia polar Gaussian sampler.
var gaussian = new GaussianSampler(rng);
double normal = gaussian.Next();           // N(0, 1)
double scaled = gaussian.Next(mean: 0.05, stdDev: 0.20);
double[] batch = gaussian.NextBatch(10_000);
```

### Rolling Window

```csharp
using Boutquin.Numerics.Collections;

// Fixed-capacity circular buffer — streaming rolling statistics without
// per-observation allocation. Enumeration walks oldest -> newest.
var window = new RollingWindow<double>(capacity: 21);
foreach (var observation in stream)
{
    window.Add(observation);
    if (window.IsFull)
    {
        // Compute a rolling statistic on the current 21-observation window.
        double latest = window[window.Count - 1];  // newest element
        double rollingMean = window.Average();
    }
}

// Unconstrained T — not limited to numeric payloads.
var eventLog = new RollingWindow<(DateTime When, string What)>(capacity: 100);
```

### Generic-math — caller chooses the precision

Every algorithm ships a generic surface. The two examples below run the same Cholesky factorisation and the same OLS estimator at two different `T`, illustrating the "caller chooses the precision" proposition the tier model delivers.

#### Example 1 — Tier A + √ factorisation at `T = double`

```csharp
using Boutquin.Numerics.LinearAlgebra;

// Curve-bootstrap pipeline: double-precision Cholesky on a rate-model SPD matrix.
double[,] spd =
{
    { 4.0, 12.0, -16.0 },
    { 12.0, 37.0, -43.0 },
    { -16.0, -43.0, 98.0 },
};

var chol = new CholeskyDecomposition<double>();
double[,] L = chol.Decompose(spd);
double[] x = chol.Solve(L, [1.0, 2.0, 3.0]);
// Natural double-precision result; Sqrt inside the factorisation is Math.Sqrt.
// No decimal infrastructure touched — the pipeline stays in double end-to-end.
```

#### Example 2 — Tier A + √ factorisation at `T = decimal` (NIST-bar precision)

```csharp
using Boutquin.Numerics.Solvers;

// Portfolio-grade OLS: instantiate at T = decimal for a full 28-digit decimal QR
// factorisation on every one of the 11 NIST StRD linear-regression problems.
// Well-conditioned designs (Norris, NoInt1/2, Longley, Wampler1/2/3) clear the
// 1e-9 coefficient bar; ill-conditioned ones cap higher (Pontius 5e-2, Filip 5e-4,
// Wampler4 1e-8, Wampler5 1e-6) because the decimal path does not layer extended-
// precision residual refinement on top. Normal-equation OLS returns zero correct
// digits on the worst cases.
decimal[,] wampler5X = /* 21 observations of x, x², …, x⁵ — cond ≈ 5·10¹³ */;
decimal[]  wampler5Y = /* … */;

var ols = new OrdinaryLeastSquares<decimal>();
var fit = ols.Fit(wampler5X, wampler5Y);
// Coefficients agree with NIST certified values to 1e-9 relative — Sqrt in the QR
// back-substitution is supplied by NumericPrecision<decimal> (Newton-Raphson to
// full 28-digit precision); no decimal → double cast anywhere on the critical path.
// See tests/.../Solvers/NistStRD/NistLinearRegressionDecimalTests.cs for the gate.
```

For tier definitions, per-folder assignments, and the forbidden patterns that govern when a `decimal → double` cast is acceptable, see [`docs/tier-constraints.md`](docs/tier-constraints.md).

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                  Boutquin.Numerics  (generic <T>)                  │
│                                                                    │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────────┐   │
│  │ Random       │  │ Distributions   │  │ Solvers              │   │
│  │  [Tier C]    │  │  [Tier C]       │  │  Root [Tier B]:      │   │
│  │ IRandomSrc<T>│  │ NormalDist<T>   │  │   BrentSolver<T>     │   │
│  │ Pcg64<T>     │  │ CumulativeN<T>  │  │   Bisection<T>       │   │
│  │ Xoshiro<T>   │  │ InverseNormal<T>│  │   NewtonRaphson<T>   │   │
│  │ GaussianS<T> │  │                 │  │   Secant<T> Muller<T>│   │
│  │              │  │                 │  │  Nonlin LS [Tier B]: │   │
│  │              │  │                 │  │   LevenbergMarq<T>   │   │
│  │              │  │                 │  │  Linear LS [A+√]:    │   │
│  │              │  │                 │  │   OrdinaryLSq<T>     │   │
│  │              │  │                 │  │  Portfolio QP [A+√]: │   │
│  │              │  │                 │  │   ActiveSetQpSolver<T>│  │
│  └──────┬───────┘  └────────┬────────┘  └──────────────────────┘   │
│         │                   │                                      │
│  ┌──────┴───────┐    ┌──────┴───────────────────┐                  │
│  │ MonteCarlo   │    │ Interpolation            │                  │
│  │  [A / C]     │    │  [A / A+√ / B]           │                  │
│  │ BootstrapME<T│    │ IInterpolator<T>         │                  │
│  │ Stationary<T>│    │ Linear<T> FlatForward<T> │                  │
│  │ Mbb<T> Wild<T│    │ LogLinear<T> TwoPoint<T> │                  │
│  │ Subsampler<T>│    │ MonotoneCubic<T>         │                  │
│  │ Percentile<T>│    │ MonotoneConvex<T>        │                  │
│  │ Sobol<T>     │    │ CubicSpline<T> (5 BC's)  │                  │
│  │ Halton<T>    │    │ MonotoneCubicSpline<T>   │                  │
│  │              │    │ InterpolatorFactory<T>   │                  │
│  └──────┬───────┘    └──────────────────────────┘                  │
│         │                                                          │
│  ┌──────▼──────────────────────────────────────────────┐           │
│  │ LinearAlgebra  [Tier A + √]                         │           │
│  │  CholeskyDecomposition<T> (+ pivoted)               │           │
│  │  JacobiEigenDecomposition<T>                        │           │
│  │  GaussianElimination<T> (GERCP)                     │           │
│  │  NearestPsdProjection<T> (EigenClip/Higham/IsPsd)   │           │
│  │  PrincipalComponentAnalysis<T>                      │           │
│  └──────┬──────────────────────────────────────────────┘           │
│         │                                                          │
│  ┌──────▼──────────────────────────────────────────────────────┐   │
│  │ Statistics                                                  │   │
│  │  [Tier A]  ICovarianceEstimator<T> + ReturnsMatrix<T>       │   │
│  │            Sample / EWMA / LW (3 targets) / QIS / OAS       │   │
│  │            Denoised / Detoned / TracyWidom / DSCE           │   │
│  │            Pearson<T> / Spearman<T> / Kendall<T>            │   │
│  │            Welford<T> + bivariate Pearson                   │   │
│  │  [A + √]   SampleSkewness<T> / SampleExcessKurtosis<T>      │   │
│  │  [A w/ T-tail]  NeweyWest<T> / DSR<T> / HaircutSharpe<T> /  │   │
│  │                 ProbabilityOfBacktestOverfitting<T>         │   │
│  │  [Tier B]       FisherZTransform<T> / DistanceCorrelation<T>│   │
│  │                 GeneralizationScore<T> / MinTRL<T>          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Collections  (no tier — data structure, unconstrained T)    │   │
│  │  RollingWindow<T>  — fixed-capacity circular buffer         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  Internal: NumericPrecision<T> dispatcher, CovarianceHelpers<T>,   │
│            InterpolationHelper.FindInterval<T>                     │
│                                                                    │
│  Legacy concrete-typed facades (BrentSolver, CholeskyDecomposition,│
│    LedoitWolfShrinkageEstimator, …) remain live as thin forwarding │
│    shims over the generic types at T ∈ { double, decimal }.        │
└────────────────────────────────────────────────────────────────────┘
                              │
                              │ no Boutquin.* dependencies (architecture test)
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  Boutquin.MarketData, Boutquin.Curves, Boutquin.Trading,           │
│  Boutquin.OptionPricing, ETFWealthIQ                               │
│  ──────                                                            │
│  Each consumes Boutquin.Numerics for its math primitives.          │
└────────────────────────────────────────────────────────────────────┘
```

The architecture follows a strict bottom-of-stack rule — Numerics has zero references to any other `Boutquin.*` package and is enforced at build time by `tests/Boutquin.Numerics.ArchitectureTests/`. Every other Boutquin package may consume Numerics; Numerics consumes none of them.

## Complexity Reference

Big-O complexity for the major operations. T = number of observations, N = number of assets, K = number of iterations or block size. The **Tier** column names the generic-math constraint class (see `CLAUDE.md` → Floating-Point Type Selection).

| Component | Tier | Build | Query | Space | Reference |
|-----------|------|-------|-------|-------|-----------|
| **`BisectionSolver<T>`** | B | — | O(log(range/tol)) | O(1) | Burden-Faires §2.1 |
| **`BrentSolver<T>`** | B | — | O(log(range/tol)) worst, O(K) typical | O(1) | Brent 1973; Oliveira et al. 2024 |
| **`NewtonRaphsonSolver<T>`** | B | — | O(K) quadratic local | O(1) | Burden-Faires §2.3 |
| **`LevenbergMarquardtSolver<T>`** | B | — | O(K·(m·n² + n³)) | O(m·n + n²) | Marquardt 1963; Nielsen 1999 |
| **Householder QR (pivoted, internal)** | A + √ | O(m·n²) | — | O(m·n) | Golub & Van Loan §5.2 |
| **`OrdinaryLeastSquares<T>`** | A + √ | O(m·n² + n³) | — | O(m·n) | Golub & Van Loan §5.3; Björck 1996 |
| **`ActiveSetQpSolver<T>.SolveMinVariance`** | A + √ | O(K·n³), K ≤ 2N | — | O(n²) | Nocedal-Wright §16.5 (active-set QP) |
| **`ActiveSetQpSolver<T>.SolveMeanVariance`** | A + √ | O(K·n³), K ≤ 3N+3 | — | O(n²) | Nocedal-Wright §16.5; Markowitz 1952 |
| **`LinearInterpolator<T>`** | A | — | O(log n) per query | O(n) | classical |
| **`CubicSplineInterpolator<T>`** | A + √ | O(n) | O(log n) per query | O(n) | de Boor 1978 |
| **`MonotoneCubicSpline<T>`** | A + √ | O(n) | O(log n) per query | O(n) | Fritsch-Carlson 1980 |
| **`MonotoneConvexInterpolator<T>`** | A | — | O(n) per query (recomputes tangents) | O(n) | Hagan-West 2006 |
| **`CholeskyDecomposition<T>` (unpivoted)** | A + √ | O(n³) | O(n²) solve | O(n²) | Higham 2002 |
| **`CholeskyDecomposition<T>.DecomposePivoted` (rank R)** | A + √ | O(n²·R) | O(n·R) solve | O(n·R) | arXiv:2507.20678 |
| **`JacobiEigenDecomposition<T>`** | A + √ | O(n³ · sweeps) | — | O(n²) | Jacobi 1846 |
| **`PrincipalComponentAnalysis<T>.Decompose`** | A + √ | O(n³ · sweeps) | O(n² · k) project | O(n²) | Jolliffe 2002 |
| **`PrincipalComponentAnalysis<T>.FromReturns`** | A + √ | O(T·N² + N³ · sweeps) | O(N² · k) project | O(N²) | Jolliffe 2002 |
| **`GaussianElimination<T>` (GERCP)** | A | O(n³) | — | O(n²) | arXiv:2505.02023 |
| **`NearestPsdProjection<T>.EigenClip`** | A + √ | O(n³) | — | O(n²) | Higham 1988 |
| **`NearestPsdProjection<T>.Higham`** | A + √ | O(n³ · iters) | — | O(n²) | Higham 2002 |
| **`SampleSkewness<T>`** | A + √ | O(n) | — | O(1) | Joanes-Gill 1998 (bias-adjusted G₁) |
| **`SampleExcessKurtosis<T>`** | A + √ | O(n) | — | O(1) | Joanes-Gill 1998 (bias-adjusted G₂) |
| **`SampleCovarianceEstimator<T>`** | A | O(N²·T) | — | O(N²) | classical |
| **`ExponentiallyWeightedCovarianceEstimator<T>`** | A | O(N²·T) | — | O(N²) | RiskMetrics |
| **`LedoitWolfShrinkageEstimator<T>` family** | A | O(N²·T) | — | O(N²) | LW 2004 |
| **`QuadraticInverseShrinkageEstimator<T>`** | A + √ | O(N²·T + N²) | — | O(N²) | LW 2022 |
| **`DenoisedCovarianceEstimator<T>`** | A + √ | O(N²·T + N³) | — | O(N²) | LdP 2018 |
| **`PearsonCorrelation<T>`** | A | O(T) per pair | — | O(1) | classical |
| **`DistanceCorrelation<T>`** | B | O(T²) | — | O(T²) | Székely 2007 |
| **`WelfordMoments<T>`** | A | O(T) total | O(1) per add | O(1) | Welford 1962 |
| **`NeweyWestVariance<T>` HAC** | A w/ t-tail | O(T·L) | — | O(T) | Newey-West 1987 |
| **`ProbabilityOfBacktestOverfitting<T>` via CSCV** | A w/ t-tail | O(C(S, S/2) · N · T) | — | O(N·S + folds) | Bailey-LdP 2014 |
| **`BootstrapMonteCarloEngine<T>` (IID)** | A + √ | O(B·T) | — | O(B) | classical |
| **`StationaryBootstrapResampler<T>`** | A | O(B·T) | — | O(B + T) | Politis-Romano 1994 |
| **`PolitisWhiteBlockLength.Estimate<T>`** | A | O(T²) worst, O(T·log T) typical | — | O(T) | Politis-White 2004 |
| **`FastDoubleBootstrap`** (double-only) | C | O(B²) inner+outer | — | O(B) | Davidson-MacKinnon 2007 |
| **`HarrellDavisPercentile.Compute<T>`** | A | O(n) | — | O(n) | Harrell-Davis 1982 |
| **`SobolSequence<T>` (d-dim)** | C | O(d) per point | — | O(d · log Nmax) | Joe-Kuo 2008 |
| **`HaltonSequence<T>` (d-dim)** | C | O(d · log N) per point | — | O(d) | Halton 1960 |
| **`Pcg64RandomSource<T>` / `Xoshiro256StarStarRandomSource<T>`** | C | — | O(1) per draw | O(1) state | O'Neill 2014 / Blackman-Vigna 2018 |
| **`RollingWindow<T>`** | — (data structure) | O(1) amortised per `Add` | O(1) indexer | O(capacity) | classical circular buffer |

## Design Decisions

The non-obvious design choices and their rationale:

1. **The public surface is generic over `T`; the caller picks the precision.** Every algorithm lives at a tier (A, A+√, B, C, or A with transcendental tail) — see [`docs/tier-constraints.md`](docs/tier-constraints.md) for the definitions and per-folder assignments. Tier A and A+√ algorithms accept `T ∈ {double, float, Half, decimal}`; Tier B algorithms require `IFloatingPointIeee754<T>` and accept `{double, float, Half}`; Tier C algorithms carry a polynomial-approximation internal ceiling calibrated for `double` and expose `T` at the public surface only for pipeline composability. The forbidden patterns from the old `decimal` / `double` split still apply — no cast-through overloads, no `decimal → double → decimal` in an inner loop, no `decimal` Tier B facade — but the motivation for a library-side type choice is gone: instantiation is a caller decision. Legacy concrete-typed facades (`BrentSolver`, `CholeskyDecomposition`, …) remain live as forwarding shims for pre-migration consumers; obsoletion is a future enhancement with no shipped date.

2. **`ReturnsMatrix` resolves layout at the API boundary, not via copies.** Trading uses `decimal[][]` (asset-major). Numerics' canonical layout is `decimal[,]` T×N. The struct holds either layout, exposes a `(time, asset)` indexer, and only copies when an estimator explicitly calls `AsTimeByAsset()` and the input was jagged. Default interface methods on `ICovarianceEstimator` make this transparent — callers just pass either type.

3. **`FromSeed(...)` static factories instead of optional-param constructors.** The `PublicAPI.Unshipped.txt` analyzer enforces RS0026 (no two overloads carrying optional parameters). Resamplers and MC engines have a primary `(...required..., IRandomSource)` constructor and a static `FromSeed(...required..., int? seed = null)` factory that wraps a `Pcg64RandomSource`. Same shape across `BootstrapResampler`, `BootstrapMonteCarloEngine`, `StationaryBootstrapResampler`, `MovingBlockBootstrapResampler`, `WildBootstrapResampler`, `FastDoubleBootstrap`.

4. **No `IRootSolver` marker interface.** The old single `Solve(f) → result` contract didn't fit both bracketed and unbracketed solvers cleanly. They live on independent interfaces; consumers that need polymorphism instantiate the concrete type they want. Reduces interface bloat at zero ergonomic cost — no consumer in the ecosystem actually wanted "any solver."

5. **PCG-64 is the default RNG, not xoshiro256\*\*.** Both pass BigCrush and are faster than `System.Random`. PCG's stream-independence guarantee (independent streams from one seed via `streamId`) is more useful for parallel MC than xoshiro's marginal speed advantage. `Xoshiro256StarStarRandomSource` is exposed for callers who want `Jump()`-based partitioning instead.

6. **Embedded Sobol direction tables, not external file load.** Joe-Kuo's full table goes to 21,201 dimensions; we ship the first 32, which covers the OptionPricing v2 use case. Avoiding an external dependency (or a 200kB embedded resource) keeps the package small. If we ever need higher-dim Sobol, the table extends linearly.

7. **`HarrellDavisPercentile` carries its own incomplete-beta evaluation.** Numerical Recipes §6.4 continued-fraction implementation embedded in the file. Avoids depending on MathNet.Numerics for one function.

8. **Default interface methods are not directly callable on sealed concrete estimators.** This is a C# language constraint, not a design choice. Callers who want the `Estimate(ReturnsMatrix)` overload must cast: `ICovarianceEstimator e = new SampleCovarianceEstimator(); e.Estimate(matrix);`. Documented in the `ReturnsMatrix` XML doc with examples.

9. **PSD contract is documented per-estimator, not enforced.** `ICovarianceEstimator.Estimate` does not promise a PSD result — Sample / LW / QIS / Denoised / Tracy-Widom / DSCE are PSD by construction; EWMA can fail with a small λ; Detoned can invert eigenvalue ordering at α = 1. Callers needing a hard guarantee wrap with `NearestPsdProjection.EigenClip(...)`. Documented at the interface level so consumers know which estimators are safe to feed directly into a Cholesky.

10. **`DeflatedSharpeRatio.ComputeFromReturns` (not a second `Compute` overload).** Two `Compute` overloads each carrying optional parameters would violate RS0026. The returns-driven convenience method gets a distinct name.

11. **LM uses absolute `initialDamping = λ₀` per Moré 1978, not Nielsen's scaled form `τ·max(diag(JᵀJ))`.** Earlier drafts tried the scaled form, which failed hard on NIST StRD Hahn1 (`max(diag)` ~ 1e19 at Start 2 pushed the step below float resolution, producing zero progress and `DampingOverflow` before any acceptance). The Marquardt scaled-diagonal damping inside the step equation `(JᵀJ + λ·diag(JᵀJ))·δ = −Jᵀr` already delivers per-parameter scale-invariance; double-scaling at initialization is unnecessary and harmful on ill-balanced Jacobians. LM's function tolerance is relative (`costDelta / priorCost ≤ ftol`) per MINPACK lmder — an absolute ftol would fire prematurely on low-cost problems like Lanczos3 (where optimal cost ~ 1e-8 triggers convergence at ~1% relative reduction against a 1e-10 absolute bar).

12. **PCA sign convention: largest-magnitude component is non-negative.** Eigenvectors are determined up to sign; without an explicit convention, consumers see arbitrary sign flips across runs. For each eigenvector column, the column is flipped so its largest-magnitude entry is ≥ 0. Deterministic across runs on identical data. Eigenvalues and eigenvector columns are sorted descending so PC1 is always the largest-variance direction — matches Hull Ch 33.6's yield-curve level/slope/curvature convention. `PcaResult.Mean` is populated by `FromReturns` (it has observation data to center) and empty for `Decompose(covarianceMatrix)` (no observation data available); `Project` centers internally when `Mean` is populated and assumes caller-pre-centered input otherwise.

13. **OLS uses Householder QR with column pivoting and decimal-residual refinement, not the normal equations.** The classical closed-form `β̂ = (XᵀX)⁻¹·Xᵀy` requires forming `XᵀX`, which squares the condition number of the design matrix. NIST StRD includes `Wampler5` at cond ≈ 5·10¹³ where the normal-equation estimator loses every digit of the response while QR preserves the original conditioning end-to-end. Column pivoting (LAPACK DGEQP3-style) additionally handles polynomial designs like `Filip` where the column norms span twelve orders of magnitude. Residual computation uses `decimal` (~28 significant digits) and feeds back into a `double` back-substitution — the same mixed-precision refinement pattern SAS/R/MATLAB use via Bailey's QD package (double-double), expressed in the `decimal` infrastructure Boutquin.Numerics already has. See NIST's own `c-Wampler5.shtml` methodology note citing Bailey's MPFUN for why double-precision implementations must use extended-precision intermediates on ill-conditioned problems.

14. **Every load-bearing public type has at least one external-reference cross-check.** The test suite has three tiers: `Boutquin.Numerics.Tests.Unit` asserts algorithm mechanics against self-generated truth, `Boutquin.Numerics.ArchitectureTests` asserts structural invariants (zero `Boutquin.*` deps, sealed types, PublicAPI hygiene), and `Boutquin.Numerics.Tests.Verification` asserts agreement with a Python reference — NIST StRD, `statsmodels` / `scipy` / `numpy` / `scikit-learn` / `arch` / `mpmath`, or real Fama-French factor data. The verification-coverage standard is that every type which ships a numerical contract must have at least one Python cross-check; a deferral is acceptable only with an explicit in-code `<remarks>` note stating why the external reference is not yet available. When multiple implementations of the same algorithm exist in the Python ecosystem, we prefer an **independent** implementation (different authors / codebase / arithmetic backend) over a hand-port — hand-ports share algorithmic choices with the C# class and can silently reproduce the same bug on both sides. When only a hand-port is available, the paper and equation number are cited in the generator comment. This policy retrofitted ~35 coverage gaps across solvers, covariance estimators, bootstrap resamplers, scalar statistics, interpolators, and root solvers — see the subsystem-level **Validation** footnotes in the [Features](#features) section above for the specific references and tolerance tiers per type.

15. **Why generic math: the caller's precision, not the library's.** The original Boutquin.Numerics shipped with a hard-bound type-per-folder model — `Solvers/` was `double`, `LinearAlgebra/` was `decimal`, and so on. The choice was defensible at the time: it pre-dated `INumber<T>` / `IFloatingPoint<T>` stabilising in the BCL, and it let the library ship with one coherent precision story per subsystem. Three pressures made it untenable over 2026. First, consumers repeatedly needed the opposite type — a curve-bootstrap team wanted `double` Cholesky for speed, a portfolio team wanted `decimal` OLS for precision — and cast-through overloads silently capped precision at the decimal-to-double boundary, misleading the caller. Second, ill-conditioned NIST linear-regression problems (`Filip`, `Wampler4`, `Wampler5`) stalled the shipped `double`-QR OLS at 6–8 digits of precision where the caller might reasonably ask for more, and the `decimal` arithmetic infrastructure to do better already existed in the library but wasn't wired to the `double` OLS. Third, the type choice had to be made in exactly the place the library is worst at making it — at ship time, not at call site. Generic math resolves all three: the algorithm is written once in `T`, instantiated by the caller at `double` / `float` / `Half` / `decimal` (Tier A / A+√), and Tier B algorithms fail at compile time rather than silently cast. `OrdinaryLeastSquares<decimal>` runs the QR, `Qᵀ·y`, upper-triangular solve, and `(XᵀX)⁻¹` end-to-end in 28-digit `decimal`, giving callers a 1e-9 coefficient bar on well-conditioned problems and mathematically principled per-problem bars on the ill-conditioned ones (see the OLS Validation line above); `CholeskyDecomposition<double>` lets curve-bootstrap callers run at IEEE speed without a facade; `BrentSolver<float>` enables memory-bound Monte Carlo pipelines that never had an option before. The `NumericPrecision<T>` dispatcher inside `Internal/` keeps the `decimal`-`Sqrt` path (Newton-Raphson) and IEEE `Sqrt` paths in one file, specialised by the JIT per-`T`. See [`docs/tier-constraints.md`](docs/tier-constraints.md) for the full tier model.

## Documentation

User-facing guides live in [`docs/`](docs/):
- [`docs/solvers.md`](docs/solvers.md) — root-finder family overview and the `LevenbergMarquardtSolver` reference (damping rule, termination semantics, parameter covariance).
- [`docs/linear-algebra.md`](docs/linear-algebra.md) — factorizations + `PrincipalComponentAnalysis` conventions and a Treasury-curve usage note.

In-code XML doc-comments on every public API surface (IntelliSense-visible) remain the primary reference for contract-level details.

## Versioning & Release

- **MinVer-driven** — versions come from git tags (`v0.x.y`) on the public release repo, never from .csproj.
- **Local pack** — `dotnet pack /p:MinVerVersionOverride=0.6.0-local --output nupkg` for cross-repo development. Outputs land in `nupkg/` (not `nupkgs/`).
- **Dual-repo workflow** — daily development on the private `Boutquin.Numerics.Dev` repo (`origin`), releases squash-pushed to the public `Boutquin.Numerics` repo (`public`). NEVER tag the private repo — tags trigger NuGet publish but the private repo lacks the `NUGET_API_KEY` secret.

## Test Infrastructure

- **`Boutquin.Numerics.Tests.Unit`** — xUnit + FluentAssertions, 540 tests covering algorithm correctness, numerical precision tolerances, edge cases (zero variance, degenerate inputs, fewer than 3 observations), and reproducibility (seed determinism for all RNG/MC code). Includes full NIST Statistical Reference Dataset (StRD) coverage across three suites — **9 univariate-summary** problems for `WelfordMoments<T>` (`NumAcc1-4`, `Lew`, `Lottery`, `Mavro`, `Michelso`, `PiDigits`) each asserting mean/variance/standard-deviation agreement with NIST-certified values to 14 significant digits plus a bisect-and-merge round-trip for `WelfordMoments<T>.Merge`; **27 nonlinear-regression** problems for `LevenbergMarquardtSolver<T>` (all 26 NIST problems plus the original reference set), each converging from NIST Start 2 with finite-difference Jacobian; **11 linear-regression** problems for `OrdinaryLeastSquares<T>.Fit` at `T = double` (`Norris`, `Pontius`, `NoInt1/2`, `Filip`, `Longley`, `Wampler1-5`) under per-problem tolerances calibrated to each design's condition number, **plus the same 11 problems at `T = decimal`** via [`NistLinearRegressionDecimalTests.cs`](tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionDecimalTests.cs) holding the 1e-9 bar on the 7 well-conditioned problems and principled higher bars on the 4 ill-conditioned ones (Pontius / Filip / Wampler4 / Wampler5) — 47 NIST reference problems, 58 executions total. Additional infrastructure added by the generic-math migration: a **`GenericParity/` suite** with per-subsystem test files (Solvers, LinearAlgebra, Statistics, MonteCarlo, Random, Distributions, Interpolation) asserting every generic type instantiated at the legacy shipped `T` is bit-identical to its pre-migration concrete implementation; a **`CrossType/` suite** with [`CholeskyCrossTypeTests.cs`](tests/Boutquin.Numerics.Tests.Unit/CrossType/CholeskyCrossTypeTests.cs) and [`LedoitWolfCrossTypeTests.cs`](tests/Boutquin.Numerics.Tests.Unit/CrossType/LedoitWolfCrossTypeTests.cs) asserting `T = double` and `T = decimal` instantiations agree to 12 / 10 significant digits respectively on shared inputs. Also includes 9 Fama-French 5-factor PCA tests verifying standardization mechanics on canonical factor returns, and a `GC.GetAllocatedBytesForCurrentThread` regression harness (`LevenbergMarquardtAllocationTests`) asserting the `LevenbergMarquardtSolver<double>` inner iteration allocates exactly zero bytes across 1000 warmup-stabilized invocations when a `LevenbergMarquardtBuffers<double>` pool is reused.
- **`Boutquin.Numerics.ArchitectureTests`** — NetArchTest, 5 tests enforcing:
  - Numerics namespaces contain no `Boutquin.*` references
  - All public types are `sealed` unless they're interfaces or records
  - Internal helpers don't leak to `PublicAPI.Shipped.txt`
- **`Boutquin.Numerics.Tests.Verification`** — xUnit, 92 cross-language tests (~4× the 23-test baseline that shipped prior to the verification-coverage audit). A Python suite under `tests/Verification/` generates reference JSON vectors using `numpy`, `scipy`, `scipy.stats` (skew / kurtosis), `scipy.optimize.minimize('SLSQP')` (QP solver), `statsmodels`, `scikit-learn`, `arch` (stationary / moving-block bootstrap and `optimal_block_length`), and `mpmath` (independent Muller implementation). C# tests load the vectors and assert Numerics kernels agree within tiered precisions (`1e-10` exact for shrinkage estimators and textbook closed forms, `1e-8` for Harrell-Davis-class incomplete-beta routines, `1e-6` for iterative-algorithm outputs propagating LAPACK-vs-Jacobi eigendecomposition noise, `1e-4` for statistical-envelope checks on resamplers). Python generators are committed with fixed seeds; re-run `python tests/Verification/generate_vectors.py` to regenerate any vector after a reference-library version bump. Every new public type ships with a Python cross-check OR an explicit in-code `<remarks>` note stating why the external reference is not yet available (see Design Decision #14 above).
- **`PublicAPI` analyzer** — every public symbol must be declared in `PublicAPI.Unshipped.txt` before it can be added; build fails otherwise. Forces explicit API surface tracking and prevents accidental public-API expansion.
- **`Boutquin.Numerics.BenchMark`** — BenchmarkDotNet harness under `benchmarks/`. One class per namespace with `[MemoryDiagnoser]` to track allocations; see `benchmarks/README.md` for interpretation.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) first.

### Reporting Bugs

Open an issue on the [Issues](https://github.com/boutquin/Boutquin.Numerics/issues) page with:
- A clear and descriptive title
- Steps to reproduce the issue
- Expected and actual behavior
- Code snippets and reference values, if applicable

### Contributing Code

1. Fork the repository and clone locally
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes following the [.NET conventions](https://github.com/boutquin/Boutquin.Numerics/blob/main/CLAUDE.md)
4. Add tests covering the new behavior
5. Add public-API entries to `src/Boutquin.Numerics/PublicAPI.Unshipped.txt`
6. Verify `dotnet build` (warnings-as-errors), `dotnet test`, `dotnet format --verify-no-changes`
7. Commit with clear messages: `git commit -m "Add feature X"`
8. Push and open a pull request

## Disclaimer

Boutquin.Numerics is open-source software provided under the Apache 2.0 License. It is a foundational numerical methods library intended for educational, research, and engineering use.

**This software does not constitute financial advice.** Boutquin.Numerics provides numerical algorithms — root finders, distributions, interpolators, linear algebra, covariance estimators, statistics, Monte Carlo, and random number generation — that are consumed by higher-level financial libraries in the Boutquin ecosystem (MarketData, Analytics, Trading, OptionPricing). The library itself has zero domain dependencies and makes no claims about the suitability of any computed value for trading, investment, risk, or regulatory purposes. Before using numerical results produced with this library in production, validate against independent references, confirm tolerances are appropriate for your use case, and consult qualified professionals who understand your specific requirements and regulatory obligations.

## License

Apache 2.0 — see [LICENSE.txt](LICENSE.txt) for details.

## Contact

For inquiries, please open an issue or reach out via [GitHub Discussions](https://github.com/boutquin/Boutquin.Numerics/discussions).

## Acknowledgments

The library implements algorithms published by:
- Olivier Ledoit and Michael Wolf (linear and nonlinear shrinkage, three targets, QIS)
- Marcos López de Prado (denoising, detoning, deflated Sharpe, PBO, MinTRL)
- Dimitris Politis, Joseph Romano, Halbert White, Michael Wolf (stationary bootstrap, automatic block length, subsampling)
- Russell Davidson, James MacKinnon, Emmanuel Flachaire, Enno Mammen (wild and fast double bootstrap)
- Campbell Harvey, Yan Liu, Heqing Zhu (multiple-testing Sharpe haircut)
- Iain Johnstone, Joël Bun, Jean-Philippe Bouchaud, Marc Potters (Tracy-Widom and RMT denoising review)
- Gábor Székely, Maria Rizzo, Nail Bakirov (distance correlation)
- B. P. Welford, Tony Chan, Gene Golub, Randall LeVeque (online stable moments)
- Whitney Newey, Kenneth West (HAC variance)
- Frank Harrell, C. E. Davis (smooth percentile estimator)
- Stephen Joe, Frances Kuo, John Halton (low-discrepancy sequences)
- Melissa O'Neill (PCG family); David Blackman, Sebastiano Vigna (xoshiro family); Daniel Lemire (bounded-integer rejection sampling)
- George Marsaglia, T. A. Bray (polar Gaussian sampler)
- Nicholas Higham (nearest PSD and nearest correlation matrix)
- Refinements from arXiv:2507.20678 (pivoted Cholesky), arXiv:2505.02023 (GERCP), arXiv:2504.05068 (Laikov erf), arXiv:2507.05083 (Q-Spline), arXiv:2402.01324 (monotone cubic endpoints)
