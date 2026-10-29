# Boutquin.Numerics

[![NuGet](https://img.shields.io/nuget/v/Boutquin.Numerics.svg)](https://www.nuget.org/packages/Boutquin.Numerics)
[![License](https://img.shields.io/github/license/boutquin/Boutquin.Numerics)](https://github.com/boutquin/Boutquin.Numerics/blob/main/LICENSE.txt)
[![Build](https://github.com/boutquin/Boutquin.Numerics/actions/workflows/pr-verify.yml/badge.svg)](https://github.com/boutquin/Boutquin.Numerics/actions/workflows/pr-verify.yml)

A foundational numerical methods library for the Boutquin financial-software ecosystem. Pure math — zero domain dependencies — implementing root finders, distributions, interpolators, linear algebra, covariance estimators, statistics primitives, Monte Carlo bootstrap variants, quasi-random sequences, and deterministic random number generators. Built with .NET 10 and strict `TreatWarningsAsErrors`.

Every algorithm exposes a generic-math surface (`INumber<T>` / `IFloatingPoint<T>`). The caller — not the library — picks the working type. `BrentSolver<double>`, `CholeskyDecomposition<decimal>`, and `OrdinaryLeastSquares<decimal>` all instantiate the same shared implementation at the precision regime that fits the pipeline. Tier constraints are documented on every type and in [`docs/tier-constraints.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/tier-constraints.md). Concrete-typed facades marked `BNU0001` (including `BrentSolver` and `CholeskyDecomposition`) are deprecated in 2.0.0 and scheduled for removal in v3.0; the migration guide lists the retained Monte Carlo/QMC conveniences. See [`docs/migration-to-generic-math.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md) for migration guidance, including v2 breaking changes beyond the facades.

The library sits below MarketData, Curves, Trading, OptionPricing in the dependency graph — algorithms here can be reused without dragging in domain types.

## Solution Structure

| Project | NuGet Package | Description |
|---------|---------------|-------------|
| **Boutquin.Numerics** | `Boutquin.Numerics` | All public API: solvers, distributions, interpolation, linear algebra, statistics, Monte Carlo, random |
| **Boutquin.Numerics.Tests.Unit** | — | xUnit + FluentAssertions coverage for algorithms, edge cases, and precision contracts |
| **Boutquin.Numerics.ArchitectureTests** | — | NetArchTest tests enforcing zero `Boutquin.*` dependencies and namespace purity |
| **Boutquin.Numerics.Tests.Verification** | — | Cross-language tests asserting C# kernels match Python reference vectors generated from numpy / scipy / scipy.stats / scipy.optimize / statsmodels / scikit-learn / arch / mpmath |
| **Boutquin.Numerics.BenchMark** | — | BenchmarkDotNet harness covering solvers, interpolation, linear algebra, covariance, bootstrap, RNG, and QMC |
| **Boutquin.Numerics.Examples** | — | 20 runnable examples covering every feature area — mirrors the code snippets in this README and serves as a live compilation gate. Run with `dotnet run -- --example <id>` or omit `--example` to run all. |

## Features

Every generic public type carries a **tier label** in its XML `<remarks>` summarising the minimum constraint stack it requires and the numeric types it accepts. The tier vocabulary — A, A+√, B, C, and A with transcendental tail — is defined in [`docs/tier-constraints.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/tier-constraints.md).

### Solvers

#### Root Finding

Two interface families based on the bracket requirement, with no shared marker interface — concrete callers pick the appropriate variant. All root finders are **Tier B** (`T : IFloatingPointIeee754<T>`; `double` guaranteed, `float` best-effort, `Half` unsupported — see [Supported numeric types](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/tier-constraints.md#supported-numeric-types)):

- **`IBracketedRootSolver<T>`** — requires a sign-changing bracket `[lo, hi]`
  - **`BisectionSolver<T>`** — guaranteed linear convergence; the safety net
  - **`BrentSolver<T>`** — inverse-quadratic and secant interpolation with bisection fallback (Brent 1973) and a **forced-bisection safeguard**: an iteration that fails to halve the bracket is followed by a bisection step, so the bracket halves at least every two iterations — worst case `2⌈log₂((b−a)/tol)⌉ + 2` iterations. The safeguard can add evaluations even on smooth functions.
  - **`NewtonRaphsonSolver<T>`** (safeguarded) — bracketed variant falls back to bisection when Newton diverges
- **`IUnbracketedRootSolver<T>`** — requires only an initial guess
  - **`SecantSolver<T>`** — superlinear without derivative, with `Solve(f, p0, p1)` two-point overload
  - **`MullerSolver<T>`** — finds real roots by quadratic interpolation, with `Solve(f, x0, x1, x2)` three-point overload
  - **`NewtonRaphsonSolver<T>`** (pure) — quadratic convergence with optional analytical derivative

`RootSolverResult<T>` reports `Converged`, `Iterations`, `FinalResidual`, and `EstimatedError` (bracket half-width or step size).

Concrete-typed types marked `BNU0001` (including `BrentSolver` and `NewtonRaphsonSolver`) are deprecated in 2.0.0 and scheduled for removal in v3.0. Migrate to the generic surface: `BrentSolver<double>`, `NewtonRaphsonSolver<double>`, etc. See [`docs/migration-to-generic-math.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md) for a migration guide with before/after examples. Tier B on `T = decimal` (so `BrentSolver<decimal>`, `LevenbergMarquardtSolver<decimal>`, etc. become available) is a future enhancement — it requires implementing 28-digit Remez / Chebyshev / CORDIC approximations for `Log`, `Exp`, `Sin`, `Cos`, `Atan`, `Tanh`, `Pow` inside the `NumericPrecision<T>` dispatcher; no shipped date, triggered by a consumer that surfaces the need. See [`docs/tier-constraints.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/tier-constraints.md) for the tier model.

**Validation:** every solver (Bisection, Brent, Newton-Raphson bracketed + unbracketed, Secant, Muller) converges to the certified mathematical root (`√2`, `e`, Dottie number, etc.) on the standard test-function pool at 1e-6 absolute. Muller's reference roots are independently validated at generation time against `mpmath.findroot(..., solver='muller')` — arbitrary-precision Muller from a different codebase, catching shared-implementation-bug risk that a numpy hand-port of Faires-Burden §2.6 would not. `Solvers_ParityTests.cs` pins scalar solvers to historical or approved post-hardening bit vectors; its LM Rosenbrock case directly compares the generic solver with the legacy facade.

#### Multivariate Least Squares

For `r: ℝⁿ → ℝᵐ`, minimizing `½ Σᵢ rᵢ(θ)²`. **Tier B** (`T : IFloatingPointIeee754<T>`):

- **`IMultivariateLeastSquaresSolver<T>`** — contract for nonlinear least-squares solvers using the half-SSE convention (gradient is `Jᵀr`, Gauss-Newton Hessian is `JᵀJ`)
- **`LevenbergMarquardtSolver<T>`** — Marquardt scaled-diagonal damping `(JᵀJ + λ·diag(JᵀJ))·δ = −Jᵀr` (invariant under per-parameter rescaling) with Nielsen 1999 gain-ratio update (IMM-REP-1999-05), `ν` starting at 2 — `λ ← λ·max(1/3, 1−(2ρ−1)³); ν ← 2` on acceptance (`ρ > 0`), `λ ← λ·ν; ν ← 2ν` on rejection. Finite-difference Jacobian when no analytic form is provided. Optional box bounds via projection. Relative function tolerance per MINPACK lmder for scale-invariance as cost → 0. **Validation:** all 27 NIST StRD nonlinear-regression reference problems converge from NIST Start 2 with finite-difference Jacobian under per-tier tolerances; `scipy.optimize.least_squares(method='lm')` parity at 1e-5 relative on four smooth regimes disjoint from NIST (exponential decay, sum-of-sinusoids, Gompertz, logistic).
- **`LevenbergMarquardtBuffers<T>`** — caller-owned scratch pool enabling zero-allocation iteration with non-allocating writer callbacks. Pool-free and pool-accepting `Solve` overloads produce bit-identical results; the pooled form is appropriate for hot-path consumers (Monte Carlo calibration, bootstrap loops, real-time refits). Grow-only `Reset(parameterCount, residualCount)` supports fitting models of varying sizes against a single pool. Not thread-safe — one pool per thread. See the `Hot-path usage — pooled buffers` section in [`docs/solvers.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/solvers.md).

`MultivariateSolverResult<T>` reports `Parameters`, `FinalCost = ½ Σᵢ rᵢ²`, `FinalResiduals`, `ParameterCovariance` (`σ̂²·(JᵀJ)⁻¹` at convergence), `Iterations` (accepted-step count), `Converged`, `TerminationReason` (one of `FunctionToleranceReached`, `ParameterToleranceReached`, `GradientToleranceReached`, `MaxIterationsReached`, `DampingOverflow`), and `BoundsActive` (orthogonal — true when any parameter sits on a user-supplied bound at termination).

#### Derivative-Free Minimization

- **`NelderMeadMinimizer<T>.Minimize(objective, initialGuess, options?)`** — Tier A simplex minimizer for smooth-enough scalar objectives when derivatives are unavailable. `SimplexResult<T>` reports `Argmin`, `MinValue`, `Iterations`, and `Converged`. Use `LevenbergMarquardtSolver<T>` instead when the objective is a residual vector.

#### Linear Least Squares

For `y = X·β + ε`, minimizing `‖y − X·β‖₂²`. **Tier A + √** (`T : IFloatingPoint<T>` with `Sqrt` via `NumericPrecision<T>`; `double` and `decimal` guaranteed, `float` best-effort, `Half` unsupported):

- **`OrdinaryLeastSquares<T>.Fit(x, y, includeIntercept)`** — Householder QR with column pivoting (LAPACK DGEQP3-style). At `T = double`, Björck's augmented-system iterative refinement corrects the coefficients and the residual together, with the residual blocks accumulated in `decimal` on power-of-two-rescaled data, reducing the residual-dependent error amplification of coefficient-only refinement. Attainable accuracy still depends on conditioning, input rounding and representable range. At `T = decimal`, the full factorisation — QR, `Qᵀ·y`, upper-triangular solve, and `(XᵀX)⁻¹` — runs in 28-digit `decimal`. Preserves the conditioning of `X` end-to-end, unlike normal-equation approaches that form `XᵀX` and square the condition number. Handles the ill-conditioned `Longley` and `Filip` designs and the noisy Wampler family. `Wampler4` and `Wampler5` share the same degree-five design (condition number about 6.4·10⁶) and differ in response noise; their coefficient sensitivity also depends on that noise. **Validation:** all 11 NIST StRD linear-regression reference problems — at `T = double`, 1e-12 on coefficients and 1e-11 on standard errors on every problem except `Filip` (1e-7 and 1e-6), including Wampler5; NIST's [certification methodology](https://www.itl.nist.gov/div898/strd/lls/data/LINKS/c-Wampler5.shtml) uses 500-digit calculations rounded to 15 significant digits and anticipates small differences in ordinary-precision results; at `T = decimal`, 1e-13 on every problem except `Filip` (1e-6, limited by its 15-digit tabulated inputs), with normalised Householder reflectors to reduce loss from column scaling. `statsmodels.api.OLS` parity at 1e-10 coefficient / 1e-8 derived-quantity relative tolerance across three regimes (well-conditioned random, polynomial degree 5, no-intercept).

`OlsResult<T>` reports `Coefficients` (β̂, length `p + 1` with intercept), `StandardErrors` (√diag(σ̂²·(XᵀX)⁻¹)), `Residuals` (y − X·β̂), `ResidualSumOfSquares`, `ResidualStandardDeviation` (σ̂), `DegreesOfFreedom` (n − p), `RSquared` (centred TSS when `includeIntercept=true`, uncentred otherwise — matches NIST StRD's convention for the `NoInt1`/`NoInt2` problems), and `CovarianceMatrix` (`Cov(β̂) = σ̂²·(XᵀX)⁻¹`, symmetric positive semi-definite at full column rank).

#### Portfolio Optimization (Quadratic Programming)

Two standard long-only portfolio problems under the sum-to-one constraint and per-asset box bounds. **Tier A + √** (`T : IFloatingPoint<T>` via `CholeskyDecomposition<T>`; `double` and `decimal` guaranteed, `float` best-effort, `Half` rejected because its thresholds underflow):

- **`ActiveSetQpSolver<T>.SolveMinVariance(covariance, minWeight, maxWeight)`** — minimum-variance portfolio: `min w′Σw  s.t.  1′w = 1,  lb ≤ w ≤ ub`. Primal-feasible active-set method (Nocedal–Wright Algorithm 16.3 on one budget row and a box): start from a feasible point, step in a budget null-space basis of the free variables, add at most the blocking bound, release the variable with the most negative multiplier relative to its own row scale, and fall back to exact rational iterates where rounded arithmetic cannot decide. The result is checked against an independent KKT certificate before it is returned; a result that cannot be certified throws `InvalidOperationException` naming the failed component, and so does a problem whose deciding block of the covariance is numerically singular at the working precision (`48·N·u·κ ≥ 1` for the exact 1-norm condition number `κ` of the equilibrated block, `u` the unit roundoff; for example near-duplicate assets; for `decimal` the equilibration is limited to exact radix-10 shifts that keep each scaled entry within 28 decimal places, so a covariance with 27 or 28 significant digits can be refused inside this margin of its unit-scale form). Internal pre-checks in `double` (Cholesky-based proofs, in `double` and double-double arithmetic, that the covariance is positive definite or of the order of its first leading principal minor that is not positive, and a certified bound on the conditioning of the deciding block) decide only what an exact test would decide; when they cannot certify the outcome the exact test decides. For `T = decimal` they convert a matrix block to `double`, yet every returned weight is computed in `T`. The rounded route allows `8N + 8` working-set changes and the exact route `12N + 12`; these are defensive limits, not convergence guarantees. For an infeasible bound box (`n · minWeight > 1` or `n · maxWeight < 1`), it relaxes only the offending bound to include the equal-weight point `1/n` (`minWeight` is lowered or `maxWeight` is raised); feasible inputs are unchanged. It throws for invalid matrices or non-convergence.
- **`ActiveSetQpSolver<T>.SolveMeanVariance(covariance, means, riskAversion, minWeight, maxWeight)`** — mean-variance portfolio: `max w′μ − (λ/2)·w′Σw  s.t.  1′w = 1,  lb ≤ w ≤ ub`. Same primal-feasible active set on the equivalent problem `min (λ/2)·w′Σw − μ′w`, with the gradient `λΣw − μ` formed exactly, the same certified result and the same budgets. Positive `riskAversion` relaxes an infeasible bound box in the same way as `SolveMinVariance`. `riskAversion = 0` instead uses the pure-LP path and rejects infeasible bounds; that path fills the budget in `T` while `(n + 2) · |1 − n · minWeight| ≤ 1024` and in exact arithmetic above it, whatever the shape of the box. **Validation:** 5 portfolio cases (including a boxed 3-asset MinVar that exercises the cross-covariance path) matched against `scipy.optimize.minimize(method='SLSQP')` at 5e-6 per-weight.

### Distributions

**Tier C** (`T : IFloatingPoint<T>`; internal polynomial-approximation tables are calibrated for `double` — `T` at the public surface is cosmetic, used so consumers can compose into generic pipelines without a type-swap):

- **`NormalDistribution<T>`** — `Pdf(x)` with pre-computed `1/√(2π)` constant; `Cdf(x)` and `InverseCdf(p)` convenience methods
- **`CumulativeNormal<T>`** — double-based CDF using Cephes rational approximations and a continued-fraction tail, with the Gaussian exponent's rounding remainder retained. Independent high-precision tests check relative tail accuracy; correctly rounded results are not guaranteed.
- **`InverseNormal<T>`** — Acklam (2000) rational approximation polished with two Newton steps. A centered residual preserves precision immediately beside the median; symmetry protects the upper tail. Accuracy remains limited by floating-point rounding.

Legacy static `NormalDistribution` / `CumulativeNormal` / `InverseNormal` classes remain live as forwarding facades over the `T = double` generic form.

### Interpolation

- **`IInterpolator<T>`** stateless interface with concrete implementations:
  - **`LinearInterpolator<T>`**, **`TwoPointLinearInterpolator<T>`** — **Tier A** (arithmetic-only); `double` and `decimal` guaranteed, `float` best-effort, `Half` unsupported
  - **`LogLinearInterpolator<T>`**, **`FlatForwardInterpolator<T>`** — **Tier B** (require `Log` / `Exp` on `T`; `double` guaranteed, `float` best-effort, `Half` unsupported, `decimal` does not compile)
  - **`MonotoneCubicInterpolator<T>`** — **Tier A + √** (Fritsch-Carlson tangents preserving monotonicity; three-point interior tangents, and endpoint tangents from the one-sided three-point formula, set to zero when that estimate has the wrong sign)
  - **`MonotoneConvexInterpolator<T>`** — **Tier A** (Hagan-West 2006, Applied Mathematical Finance 13(2) pp. 89–129). Preserves **non-negative instantaneous forward rates** across the full maturity range for finite, non-decreasing NCR (normalized cumulative return, `−ln(DF)`) node arrays; decreasing NCR input is rejected. The relevant no-arbitrage constraint for discount-factor curves — unlike `MonotoneCubicInterpolator`, which preserves monotonicity of the y-values themselves.
- **`MonotoneCubicSpline<T>`** — pre-computed tangents for O(log n) repeated evaluation (Tier A + √)
- **`CubicSplineInterpolator<T>`** (Tier A + √) with six `SplineBoundary<T>` modes:
  - `Natural`, `Clamped(leftSlope, rightSlope)`, `NotAKnot` — classical
  - `ModifiedNotAKnot` — not-a-knot endpoint curvature scaled by the end-interval spacing ratio; identical to `NotAKnot` on uniform grids and not more accurate than it on non-uniform ones. A library rule, not the revised not-a-knot spline of Jarre 2025 §4
  - `RevisedNotAKnot` — prescribed third-derivative jumps with the factor-5/2 fifth-difference safeguard from Jarre's JCAM 478 (2026) Appendix and authenticated October 29, 2025 author code; requires at least six knots
  - `QSpline` — endpoint-curvature estimates with the journal Appendix’s fifth-difference safeguard; requires at least six knots
- **`InterpolatorFactory<T>`** + **`InterpolatorKind`** enum for runtime kind selection. The factory throws `NotSupportedException` at runtime when `LogLinear` / `FlatForward` (Tier B) is requested at a non-IEEE `T` such as `decimal` — the tier mismatch is surfaced at construction rather than at use site.

**Validation:** Linear / MonotoneCubic (PCHIP) / natural-cubic-spline vs `scipy.interpolate` at 1e-6 relative; LogLinear / FlatForward / TwoPointLinear vs numpy ports of the textbook formulas at 1e-10 absolute; MonotoneCubicSpline N=2 degenerate-case vs `scipy.interpolate.PchipInterpolator` at 1e-10 absolute; MonotoneConvex vs a paper-faithful hand-port of Hagan-West (2006) §2.3–§2.6 at 1e-10 absolute (no library equivalent exists for this finance-specific algorithm, so the reference is hand-ported from the paper with explicit equation citations). Generic parity (`LinearInterpolator<double>` vs legacy `LinearInterpolator`, all eight interpolators) is gated by `Interpolation_ParityTests.cs`.

### Linear Algebra

All matrix routines are **Tier A + √** (`T : IFloatingPoint<T>` with `Sqrt` via `NumericPrecision<T>`; `double` and `decimal` guaranteed, `float` best-effort, `Half` unsupported). The legacy concrete-typed facades preserve existing `decimal[,]` call sites. Most forward to the `T = decimal` instantiation; `JacobiEigenDecomposition` converts the matrix to `double`, performs the rotations at double precision, and converts the result back to `decimal`.

- **`CholeskyDecomposition<T>`** — `Decompose(A)` for symmetric positive-definite matrices; **`DecomposePivoted(A, tolerance, maxRank)`** with diagonal pivoting for positive semi-definite matrices and early termination for low-rank approximations (arXiv:2507.20678); `Solve(L, b)` for triangular back-substitution
- **`JacobiEigenDecomposition<T>.Decompose(A)`** — cyclic Givens rotations, eigenvalues in descending order with eigenvectors as columns. Result record is `EigenResult<T>`.
- **`GaussianElimination<T>.Solve(A, b, seed?)`** — randomized partial row pivoting for square linear systems; pass a seed for reproducible pivot choices
- **`NearestPsdProjection<T>`** — Higham (1988) projection onto the PSD cone:
  - **`EigenClip(A)`** — symmetrize → eigendecompose → apply `max(λ, δ)` → reconstruct (nearest matrix in Frobenius norm with eigenvalues at least the nonnegative `eigenvalueFloor` δ; default zero gives ordinary PSD projection). The floor is separate from numerical convergence tolerance.
  - **`Higham(A)`** — alternating projection between PSD cone and unit-diagonal subspace for the **nearest correlation matrix**
  - **`IsPsd(A, tolerance)`** — eigenvalue-based PSD check with configurable absorption of FP drift
- **`PrincipalComponentAnalysis<T>`** — factor-model eigendecomposition built on `JacobiEigenDecomposition<T>`:
  - **`Decompose(covarianceMatrix)`** — takes a pre-computed covariance (useful for running PCA on top of Ledoit-Wolf shrinkage, POET, Tracy-Widom, etc.); `PcaResult<T>.Mean` is empty and the caller pre-centers inputs to `Project`.
  - **`PrincipalComponentAnalysis.FromReturns(ReturnsMatrix, standardize)`** — decimal-only facade that takes raw returns, computes sample covariance (or correlation when `standardize: true`) internally, and populates `PcaResult.Mean` for downstream projection/reconstruction.
  - **`PcaResult<T>`** exposes `Eigenvalues` (descending — PC1 = largest, matching the Hull Ch 33.6 yield-curve level/slope/curvature convention), `Eigenvectors` (column-stored loadings, sign-flipped so each column's largest-magnitude entry is non-negative for cross-run determinism), `ExplainedVarianceRatio`, `CumulativeExplainedVariance`, `Mean`, plus `Project(data, k)` and `NumComponentsForExplainedVariance(threshold)`.

**Validation:** `CholeskyDecomposition<T>` / `GaussianElimination<T>` / `JacobiEigenDecomposition<T>` / `NearestPsdProjection<T>` vs `scipy.linalg` ground-truth at 1e-8 relative; `PrincipalComponentAnalysis<T>` additionally verified on 60 months of Fama-French 5-factor returns (trace identity under standardization, cumulative variance reaches 1, eigenvector orthonormality at 1e-8). **Cross-type regression** — [`CholeskyCrossTypeTests.cs`](https://github.com/boutquin/Boutquin.Numerics/blob/main/tests/Boutquin.Numerics.Tests.Unit/CrossType/CholeskyCrossTypeTests.cs) runs `CholeskyDecomposition<double>` and `CholeskyDecomposition<decimal>` on a curated 10×10 SPD and asserts their lower factors agree to 12 significant digits — the single test that demonstrates the "caller chooses the precision" proposition is live.

### Statistics

All arithmetic-tier statistics are **Tier A** (`T : IFloatingPoint<T>`; `double` and `decimal` guaranteed, `float` best-effort, `Half` unsupported). A `ReturnsMatrix<T>` value type accepts both layouts and resolves them at the interface boundary, so callers don't transpose:

- **`ReturnsMatrix<T>`** — readonly struct accepting either T×N (`T[,]`) or asset-major (`T[][]`); construct explicitly from either layout with `new ReturnsMatrix<T>(values)`. Implicit conversions belong only to the legacy non-generic `ReturnsMatrix` facade.
- **`ICovarianceEstimator<T>`** — interface with explicit PSD-guarantee documentation and a default-method overload `Estimate(ReturnsMatrix<T>)`

Legacy non-generic `ReturnsMatrix` / `ICovarianceEstimator` / `SampleCovarianceEstimator` / `LedoitWolfShrinkageEstimator` / …  stay live as forwarding facades over the `T = decimal` instantiation.

#### Covariance Estimators

| Estimator | Reference | Use When |
|-----------|-----------|----------|
| **Sample** (T-1 divisor) | classical | baseline; T ≫ N |
| **Exponentially-Weighted Moving Average** | RiskMetrics | non-stationary returns; recent observations should dominate |
| **Ledoit-Wolf** (scaled identity target) | LW 2004, JMA | small T relative to N; minimal structure |
| **Ledoit-Wolf Constant Correlation** | LW 2003, JPM | equity universes with similar pairwise correlations |
| **Ledoit-Wolf Single Factor** | LW 2003, JEF | factor-structured equity data with a market beta |
| **Quadratic Inverse Shrinkage (QIS)** | LW 2022, Bernoulli 28(3) | concentration ratio c = p/n ≳ 0.1; nonlinear shrinkage strictly dominates linear |
| **Oracle Approximating Shrinkage (OAS)** | Chen-Wiesel-Eldar-Hero 2010, arXiv:0907.4698 | Known-zero-mean Gaussian data; raw second moment divided by the observation count |
| **Denoised** (Marčenko-Pastur) | López de Prado 2020 (*Machine Learning for Asset Managers*), §2.5 | high-dim portfolios; replace noise eigenvalues with their mean |
| **Detoned** | López de Prado 2020, §2.6 | denoising + market-component (PC1) removal; singular when a market component is fully removed with positive residual diagonal; small-asset/all-noise paths and identity fallback need not be singular |
| **Tracy-Widom Denoised** | Johnstone 2001; Bun-Bouchaud-Potters 2017, arXiv:1610.08104 | finite-N denoising with sharper signal/noise threshold than MP edge |
| **Doubly Sparse (DSCE)** | Declared repository variant | MP eigenvalue clipping and one-shot absolute signal-eigenvector thresholding with renormalisation |
| **NERCOME** | Lam 2016, AoS, Equations (4.5)–(4.9); Abadir-Distaso-Žikeš 2014, JoE (antecedent) | replicated split-sample estimator; averages reconstructed matrices and selects the split from the published candidate grid |
| **POET** | Fan-Liao-Mincheva 2013, JRSS-B, arXiv:1201.0175 | factor-model data; K leading PCs plus residual covariance soft-thresholded at the adaptive eq. 3.2 level |

DSCE is PSD by construction. It does not implement the Kutateladze–Seregina estimator:
there is no truncated power iteration, rank or support-size input, least-squares
eigenvalue re-estimation, or residual thresholding. Its NumPy vectors verify the
repository variant's behavior.

Every shrinkage/structured estimator is generic over `T : IFloatingPoint<T>` (Tier A for arithmetic shrinkage; Tier A + √ for estimators that normalise by standard deviations). `LedoitWolfShrinkageEstimator<T>`, `OracleApproximatingShrinkageEstimator<T>`, `PoetCovarianceEstimator<T>`, etc. all accept `T ∈ {double, float, Half, decimal}`; `PoetCovarianceEstimator<T>`, `NercomeCovarianceEstimator<T>` and `DoublySparseEstimator<T>` also require `T : struct`, so their optional `T?` tuning parameters are nullable and an omitted or `default` argument takes the documented value. Legacy facades (`LedoitWolfShrinkageEstimator`, …) forward to the `T = decimal` instantiation.

NERCOME defaults to 50 distinct random first subsets per candidate, capped by the number of possible subsets, with deterministic seed 0. It floors, clamps and deduplicates the seven published candidates, retaining the first candidate on a score tie. It uses known-zero-mean second moments, averages full reconstructed matrices in their respective eigenbases, and minimizes the squared norm of the averaged residual. Exact rational projector tests independently check these equations on small exhaustive panels; the larger NumPy benchmark checks the translation and sampling policy. Explicit split-fraction constructors select a fixed split size with the same replication procedure.

**Validation:** every concrete estimator is cross-checked against a Python reference in `tests/Verification/vectors/covariance.json` across three regimes (small well-conditioned N=8/T=120, moderate N=30/T=60, overconcentrated N=50/T=40). Tiered tolerances reflect each estimator's algorithmic structure — shrinkage-only estimators (Sample, LW-family, OAS) agree at 1e-10 relative (no eigendecomposition to propagate LAPACK-vs-Jacobi noise); eigendecomposition-based estimators (Denoised, Detoned, Tracy-Widom, QIS, DoublySparse) at 1e-5 relative; POET at 1e-4 (threshold-nonlinearity amplification); NERCOME at 1e-1 on the well-conditioned regime only (rank-near-deficient splits in the other regimes make the reconstruction legitimately non-unique in the null-space basis). Python references: the exact OAS equations (2) and (23), with known-zero-mean raw second moments divided by T and the finite-N corrections, `sklearn.covariance.ledoit_wolf` for the scaled-identity target, NumPy ports of the authors' published LW-CC (`covCor`), LW-SF (`covMarket`) and QIS (LW 2022 Appendix A) code, NumPy ports of the implemented Denoised / Detoned / TW / NERCOME / DSCE methods (for NERCOME / QIS / Denoised / Detoned / TW / DSCE these check C#-to-NumPy translation parity, not paper fidelity), and a POET reference derived directly from Fan-Liao-Mincheva (2013) by a different computational route. **Cross-type regression** — [`LedoitWolfCrossTypeTests.cs`](https://github.com/boutquin/Boutquin.Numerics/blob/main/tests/Boutquin.Numerics.Tests.Unit/CrossType/LedoitWolfCrossTypeTests.cs) runs `LedoitWolfShrinkageEstimator<double>` and `LedoitWolfShrinkageEstimator<decimal>` on a 20-asset 500-observation returns matrix and asserts shrinkage intensities agree to 10 digits — the arithmetic-only shrinkage formula is type-invariant by construction, and the test proves it on the shipped code.

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

**Validation:** both estimators agree with `scipy.stats.skew` / `scipy.stats.kurtosis` at 1e-10 absolute across 6 synthetic cases (normal, gamma, negated-gamma, uniform, t(3), small-n) via [`SampleMomentsVerificationTests.cs`](https://github.com/boutquin/Boutquin.Numerics/blob/main/tests/Boutquin.Numerics.Tests.Verification/SampleMomentsVerificationTests.cs).

#### Backtest Statistics

Hybrid-tier statistics are **Tier A with transcendental tail** — the inner loop is arithmetic in `T`, and a final scalar `Log` / `Sqrt` / inverse-normal applies once. At `T = decimal`, the tail casts to `double`, runs the BCL function, and casts back (a single scalar, not an inner loop — see [`docs/tier-constraints.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/tier-constraints.md)). At `T = double`, the tail is native.

- **`DeflatedSharpeRatio<T>`** (Tier A w/ transcendental tail) — Bailey-LdP 2014 multiple-testing correction; `ComputeWithTrialMoments(...)` includes the annualized trial-Sharpe mean and variance in the expected maximum. `Compute(...)`, `ComputeWithTrialVariance(...)`, and `ComputeFromReturns(...)` retain the explicit zero-mean trial-null convention. Result: `DsrResult<T>`.
- **`HaircutSharpe<T>`** (Tier A w/ transcendental tail) — Harvey-Liu 2015 ("Backtesting", *Journal of Portfolio Management* 42(1)) with Bonferroni / Holm / BHY corrections from the Harvey-Liu-Zhu 2016 multiple-testing framework; `ComputePanel(...)` accepts the full Sharpe panel, and `AdjustPValues(...)` applies ordered Holm and BHY adjustments to the full p-value panel. Scalar `Compute(...)` assumes every other trial has p = 1. Result: `HaircutSharpeResult<T>`.
- **`MinimumTrackRecordLength<T>`** (Tier B) — Bailey-LdP 2012 (*Journal of Risk* 15(2)); inverse of PSR at a specified benchmark — minimum sample length needed for an observed Sharpe to be statistically distinguishable from a benchmark
- **`ProbabilityOfBacktestOverfitting<T>`** (Tier A w/ transcendental tail) — Bailey-Borwein-LdP-Zhu 2014 (*Journal of Computational Finance* 20(4)); CSCV (Combinatorially-Symmetric Cross-Validation) over a strategy panel, with PBO defined as the fraction of splits where the in-sample winner ranks at or below the out-of-sample median. Result: `PboResult<T>`.
- **`NeweyWestVariance<T>`** (Tier A) — Bartlett HAC variance (Newey-West 1987); `LongRunCovariance(...)` implements the Newey-West 1994 data-dependent selector with VAR(1) prewhitening and recolouring, and `MeanVarianceAutomatic(...)` applies it to a sample mean. `AutomaticLags(int)` remains a pilot-window heuristic; `MeanVariance(..., lags)` retains caller-supplied lags.
- **`GtInspiredScore<T>`** (Tier B) — GT-inspired library search loss (lower is better), using sample SD, cumulative arithmetic-return R² against time, zero-target downside RMS over all observations, and additive epsilon. Inspired by Sheppert (2026, arXiv:2602.00080); no author parity or empirical generalization guarantee is claimed. Result: `GtInspiredScoreResult<T>`.

**Validation:** `DeflatedSharpeRatio` / `MinimumTrackRecordLength` vs the embedded `dsr.json` vectors at 1e-8 relative; `HaircutSharpe` vs `haircut_fisher.json`, derived from Harvey-Liu (2015) eqs. 1–5 with `scipy.stats.t` and including the paper's worked example, at 1e-9; `ProbabilityOfBacktestOverfitting` vs a numpy port of Bailey et al. 2014 CSCV at 1e-8 relative with stable-sort tie-breaking matching the C# `OrderBy`; `NeweyWestVariance` vs `statsmodels.api.OLS(y, ones).fit().get_robustcov_results('HAC', ...)` at 1e-10 absolute; `GtInspiredScore` vs an independent NumPy oracle of the declared library variant, including `Score`, `Mu`, `Z`, `RSquared`, and `DownsideDeviation`, at 1e-10 absolute per component; `FisherZTransform` and `DistanceCorrelation` vs their analytic closed forms.

#### Volatility and dependence

- **`GarchOneOneEstimator<T>.Fit(returns, options?)`** — Tier B Gaussian quasi-maximum-likelihood (QMLE) GARCH(1,1) fit; `GarchFitResult<T>.ForecastVariance(horizon)` produces conditional-variance forecasts.
- **`GraphicalLassoSolver<T>.Solve(sampleCovariance, lambda, options?)`** — Tier B sparse precision-matrix estimator. `SolveWithBic` selects from a supplied lambda grid.
- **`LowerTailDependence<T>.Estimate(x, y, options?)`** — Tier A lower-tail co-crash estimator, with conditional and empirical-copula forms. Leave `TailOptions<T>.UseGarchFilter` at `false`: pre-whitening is not yet supported and throws if requested.

Runnable calls for these APIs are in examples 19 and 20 and the concise setup is in [`docs/solvers.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/solvers.md) and [`docs/statistics.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/statistics.md).

### Monte Carlo

#### Bootstrap Variants

Bootstrap resamplers are **Tier A** (arithmetic resampling / order statistics). `BootstrapMonteCarloEngine<T>` is Tier A + √ (standard-error computation uses `Sqrt`). `FastDoubleBootstrap` is a non-generic, decimal-only arithmetic bootstrap. `SobolSequence<T>` and `HaltonSequence<T>` are Tier C (direction numbers calibrated for `double`).

| Variant | Generic | Tier | Block size | Stationarity | Reference |
|---------|---------|------|------------|--------------|-----------|
| **`BootstrapMonteCarloEngine<T>`** (IID) | ✓ | A + √ | n/a | strictly stationary on IID | classical |
| **`BootstrapResampler<T>`** (circular block) | ✓ | A | fixed | approximately stationary | Politis-Romano 1992 |
| **`MovingBlockBootstrapResampler<T>`** | ✓ | A | fixed, no wrap | edge-effect biased | Künsch 1989; Liu-Singh 1992 |
| **`StationaryBootstrapResampler<T>`** | ✓ | A | geometric | strictly stationary | Politis-Romano 1994 |
| **`WildBootstrapResampler<T>`** | ✓ | A | n/a | preserves residual structure | Mammen 1993; Davidson-Flachaire 2008 |
| **`Subsampler.Run<T>`** | ✓ | A | b ≪ T overlapping | raw statistics; caller supplies centering and scaling | Politis-Romano-Wolf 1999 |
| **`FastDoubleBootstrap`** | decimal-only | A operations | nested 1×1 | bias-corrected p-values | Davidson-MacKinnon 2007 |

- **`PolitisWhiteBlockLength.EstimateDouble(series)`** (non-generic, double-working transcendental selector) — data-driven optimal block length for stationary bootstrap (Politis-White 2004, corrected stationary constant and companion lag window from Patton–Politis–White 2009)
- **`BootstrapMonteCarloEngine<T>.RunMulti(...)`** — multi-statistic single-pass overload; computes K statistics on each resampled path simultaneously, avoiding K separate passes for bundles like `(Sharpe, Sortino, MaxDD, IR)`
- **`WildBootstrapWeights`** enum: `Mammen` (two-point, third-moment matching), `Rademacher` (±1 symmetric — Davidson-Flachaire default), `Gaussian`

Legacy concrete-typed facades (`BootstrapResampler`, `StationaryBootstrapResampler`, …) remain live as forwarding shims over the `T = decimal` generic form.

**Validation:** `StationaryBootstrapResampler` and `MovingBlockBootstrapResampler` resampled-mean distributions fall within the 95% band produced by `arch.bootstrap.StationaryBootstrap` / `MovingBlockBootstrap` references at 3000 trials; `WildBootstrapResampler` Mammen and Rademacher weight distributions match the analytic moments within 4·SE at 20 000 trials; `FastDoubleBootstrap.PValue` matches a numpy implementation of the Davidson-MacKinnon 2007 definition within 8% absolute across all three `PValueTail` conventions (right, left, two-sided), at the sample mean and at an upper-tail observed statistic; `Subsampler` is deterministic and compared bit-identically to a numpy port; `PolitisWhiteBlockLength` matches an independent NumPy implementation of its corrected companion lag window and full-sample covariance convention to 1e-10 on positively and negatively autocorrelated AR(1) series. Six seeded 1,000-trial experiments also check corrected Table 2 against statistical bounds; the author's original seeds are unpublished. Exact test inputs, results, source discrepancies, and reconstruction limits are recorded in [the paper replication catalogue](docs/paper-replication-tests.md).

#### Percentile

- **`Percentile.Compute<T>(sorted, p)`** (Tier A) — linear interpolation, NumPy `linear` / R `type=7` convention
- **`HarrellDavisPercentile.Compute<T>(sorted, p)`** (Tier C) — Harrell-Davis 1982 with embedded incomplete-beta evaluation and double-precision Beta weights; variance relative to linear quantiles depends on the sample distribution, and extreme observations can reduce robustness
- **`PSquareEstimator<T>(p)`** (Tier A) — Jain-Chlamtac 1985 online percentile; five-marker parabolic-interpolation estimator with O(1) memory for unbounded streams (streaming bootstrap / very-large MC)

**Validation:** `HarrellDavisPercentile.Compute` vs a `scipy.special.betainc`-driven reference at five quantiles (10/25/50/75/90) at 1e-8 relative — the C# class embeds an incomplete-beta continued fraction with recurrence/Stirling log-gamma evaluation. Independent high-precision beta-integral controls also cover tiny tail masses amplified by large observations, signed cancellation and telescoping interval sums. `PSquareEstimator.Estimate` vs a verbatim numpy port of the Jain-Chlamtac 1985 five-marker state transitions at 1e-10 absolute across three percentiles (0.5/0.9/0.99).

#### Quasi-Monte Carlo

- **`HaltonSequence<T>`** (Tier C) — radical-inverse in distinct primes, up to 50 dimensions; supports `skip` for low-index correlation avoidance
- **`SobolSequence<T>`** (Tier C) — Joe-Kuo (2008) direction numbers, up to 32 dimensions; `Jump` not exposed (use `skip`)

### Random Number Generation

Deterministic, cross-runtime-stable PRNGs replacing `System.Random` for reproducibility. All **Tier C** (`T : IFloatingPoint<T>` at the public surface; rejection sampling and the Gaussian polar method use IEEE bit tricks internally, bounding `T` to `double` / `float` for the sampled-value path; `Half` is unsupported):

- **`IRandomSource<T>`** — interface with `NextULong`, `NextUInt`, `NextDouble`, `NextInt(upperExclusive)` (Lemire arXiv:1805.10941 nearly-divisionless rejection sampling)
- **`Pcg64RandomSource<T>`** — PCG-XSL-RR 128/64 (O'Neill 2014); period 2^128, k-equidistributed to k = 2; distinct non-zero `streamId` values select different odd increments from the same seed for reproducible parallel MC. Different increments do not prove statistical independence
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

// Brent — bracketed, superlinear, with a forced-bisection safeguard.
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
var sr = secant.Solve(x => x * x - 2.0, p0: 0.5, p1: 2.0);
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

double[] xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
double[] ys = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0];

// Stateless interpolators — pass xs/ys at every call.
double linear = LinearInterpolator.Instance.Interpolate(2.5, xs, ys);   // 6.5
double mono   = MonotoneCubicInterpolator.Instance.Interpolate(2.5, xs, ys);

// Pre-computed cubic spline — O(log n) repeated evaluation.
var spline = new CubicSplineInterpolator(xs, ys, SplineBoundary.NotAKnot);
double y25 = spline.Interpolate(2.5);

// Modified not-a-knot (spacing-ratio-scaled end curvature).
var modSpline = new CubicSplineInterpolator(xs, ys, SplineBoundary.ModifiedNotAKnot);

// Q-Spline — safeguarded endpoint curvature from at least six values alone.
var qSpline = new CubicSplineInterpolator(xs, ys, SplineBoundary.QSpline);

// Runtime kind selection.
var dynamic = InterpolatorFactory.Create(InterpolatorKind.MonotoneCubic);

// Hagan-West monotone-convex for yield-curve NCR nodes. Input convention:
// xs[0]=0, ys[0]=0 is the virtual origin; ys[i] = -ln(P(xs[i])) must be non-decreasing.
// Convert the result back to a discount factor via P(x) = exp(-ncr).
double[] curveXs = [0.0, 0.25, 1.0, 5.0, 10.0, 30.0];
double[] curveYs = [0.0, -Math.Log(0.995), -Math.Log(0.97),
                         -Math.Log(0.83),  -Math.Log(0.68), -Math.Log(0.25)];
double ncr2y = MonotoneConvexInterpolator.Instance.Interpolate(2.0, curveXs, curveYs);
double df2y  = Math.Exp(-ncr2y);  // non-negative forward rates for non-decreasing NCR nodes
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
    // Convergence alone does not guarantee an identifiable covariance matrix.
    if (result.ParameterCovariance is { } covariance)
    {
        double sigmaA = Math.Sqrt(covariance[0, 0]);
        double sigmaB = Math.Sqrt(covariance[1, 1]);
    }
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

// Fill residual storage directly so the callback allocates no arrays.
void WriteResiduals(ReadOnlySpan<double> theta, Span<double> residuals)
{
    for (var i = 0; i < xs.Length; i++)
    {
        residuals[i] = ys[i] - theta[0] * Math.Exp(theta[1] * xs[i]);
    }
}

// Reuse scratch storage across solves. Each returned result owns its output arrays.
var pooledSolver = new LevenbergMarquardtSolver<double>(
    maxIterations: 100, functionTolerance: 1e-10, parameterTolerance: 1e-10,
    gradientTolerance: 1e-10, initialDamping: 1e-3);
var buffers = new LevenbergMarquardtBuffers<double>(parameterCount: 2, residualCount: xs.Length);
double[] initialGuess = [1.0, 0.1];
for (var run = 0; run < 100; run++)
{
    var pooled = pooledSolver.Solve(WriteResiduals, initialGuess, buffers);
    // pooled.Parameters — same result bits as the corresponding pool-free writer solve
}
```

### Ordinary Least Squares

```csharp
using Boutquin.Numerics.Solvers;

// NIST StRD "Longley" — cond ≈ 5·10⁷, the canonical ill-conditioned OLS problem.
// Pivoted QR avoids forming the normal equations; accuracy remains condition-dependent.
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
int     k95       = pca.NumComponentsForExplainedVariance(0.95);

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

// Deflated Sharpe Ratio — zero-mean trial-null compatibility overload.
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

// Scalar compatibility: every other trial is assumed to have p = 1.
var hlz = HaircutSharpe.Compute(observedSharpe: 1.5m, numTrials: 1000, backTestYears: 5m);
// hlz.HaircutSharpe, hlz.HaircutAmount, hlz.BonferroniPValue, hlz.HolmPValue, hlz.BhyPValue

// Full trial-panel corrections and a nonzero trial-Sharpe mean.
var panel = HaircutSharpe<decimal>.ComputePanel([1.5m, 1.2m, 0.8m], backTestYears: 5m);
var adjusted = HaircutSharpe<decimal>.AdjustPValues([0.005m, 0.009m, 0.0135m]);
var withTrialMoments = DeflatedSharpeRatio<decimal>.ComputeWithTrialMoments(
    observedSharpe: 1.5m, numTrials: 1000, trialSharpeMean: 0.1m, trialSharpeVariance: 0.25m,
    backTestYears: 5m, skewness: -0.5m, kurtosis: 4.0m);

// Minimum Track Record Length — inverse of DSR.
double minimumObservations = MinimumTrackRecordLength.Compute(observedSharpe: 1.5);

// Probability of Backtest Overfitting via CSCV.
decimal[,] strategyPanel = LoadStrategyReturns(); // [strategy, time]
var pbo = ProbabilityOfBacktestOverfitting.Compute(strategyPanel, splitCount: 16);
// pbo.Pbo, pbo.LogitMedian, pbo.LogitValues

// Sample-mean HAC variance with automatic bandwidth, prewhitening and recolouring.
decimal hacVar = NeweyWestVariance<decimal>.MeanVarianceAutomatic(dailyReturns);
```

### Monte Carlo Bootstrap

```csharp
using Boutquin.Numerics.MonteCarlo;
using Boutquin.Numerics.Random;

decimal[] returns = LoadReturns();

// Statistic callbacks receive a ReadOnlySpan<decimal>, which has no LINQ Average().
static decimal Mean(ReadOnlySpan<decimal> sample)
{
    decimal sum = 0m;
    foreach (var v in sample) sum += v;
    return sum / sample.Length;
}

// IID bootstrap with deterministic seeded PCG64.
var iid = BootstrapMonteCarloEngine.FromSeed(simulationCount: 5000, seed: 42);
var iidResult = iid.Run(returns, paths => Mean(paths));
// iidResult.Mean, .Median, .Percentile5, .Percentile95, .Statistics

// Stationary bootstrap with Politis-White auto block length. The selector has no lower
// floor (serially independent returns can give an estimate below one), while the
// resampler requires a mean block length of at least one.
double blockLength = PolitisWhiteBlockLength.EstimateDouble(
    Array.ConvertAll(returns, v => (double)v));
double meanBlockLength = Math.Max(1.0, blockLength);
var stationary = StationaryBootstrapResampler.FromSeed(meanBlockLength, seed: 42);
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

// Fast Double Bootstrap for bias-corrected p-values. H0: mean return = 0, so each
// bootstrap DGP (the data, and every outer sample) is recentred to mean zero.
var fdb = FastDoubleBootstrap.FromSeed(outerCount: 500, seed: 42);
decimal pValue = fdb.PValueImposingNull(
    returns,
    observed: Math.Abs(returns.Average()),
    paths => Math.Abs(Mean(paths)),
    imposeNull: sample =>
    {
        decimal mean = 0m;
        foreach (var v in sample) mean += v;
        mean /= sample.Length;
        for (var i = 0; i < sample.Length; i++) sample[i] -= mean;
    });

// Harrell-Davis percentile — smooth estimate using all order statistics.
decimal[] sorted = iidResult.Statistics.ToArray();
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

// PCG-64 — different odd increments via distinct non-zero streamId values.
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

double[,] L = CholeskyDecomposition<double>.Decompose(spd);
double[] x = CholeskyDecomposition<double>.Solve(L, [1.0, 2.0, 3.0]);
// Natural double-precision result; Sqrt inside the factorisation is Math.Sqrt.
// No decimal infrastructure touched — the pipeline stays in double end-to-end.
```

#### Example 2 — Tier A + √ factorisation at `T = decimal` (NIST-bar precision)

```csharp
using Boutquin.Numerics.Solvers;

// Portfolio-grade OLS: instantiate at T = decimal for a full 28-digit decimal QR
// factorisation on every one of the 11 NIST StRD linear-regression problems.
// Coefficients hold 1e-13 of the certified values on every problem except Filip
// (1e-6, limited by its 15-digit tabulated inputs). Forming the normal equations
// squares the design condition number and can amplify rounding error.
// Synthetic noise-free degree-five illustration: 21 observations, x = 0..20.
decimal[,] polynomialX = new decimal[21, 5];
decimal[] polynomialY = new decimal[21];
for (var i = 0; i <= 20; i++)
{
    decimal power = 1m;
    decimal response = 1m;
    for (var degree = 0; degree < 5; degree++)
    {
        power *= i;
        polynomialX[i, degree] = power;
        response += (degree + 2) * power;
    }
    polynomialY[i] = response;
}

var fit = OrdinaryLeastSquares<decimal>.Fit(polynomialX, polynomialY);
// Separately, the NIST Wampler5 coefficient test permits 1e-13 relative error. Sqrt in the QR
// back-substitution is supplied by NumericPrecision<decimal> (Newton-Raphson to
// full 28-digit precision); no decimal → double cast anywhere on the critical path.
// See tests/.../Solvers/NistStRD/NistLinearRegressionDecimalTests.cs for the gate.
```

For tier definitions, per-folder assignments, and the forbidden patterns that govern when a `decimal → double` cast is acceptable, see [`docs/tier-constraints.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/tier-constraints.md).

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
│  │              │  │                 │  │  ActiveSetQpSolver<T>│   │
│  └──────┬───────┘  └────────┬────────┘  └──────────────────────┘   │
│         │                   │                                      │
│  ┌──────┴───────┐    ┌──────┴───────────────────┐                  │
│  │ MonteCarlo   │    │ Interpolation            │                  │
│  │  [A / C]     │    │  [A / A+√ / B]           │                  │
│  │ BootstrapME<T│    │ IInterpolator<T>         │                  │
│  │ Stationary<T>│    │ Linear<T> FlatForward<T> │                  │
│  │ Mbb<T> Wild<T│    │ LogLinear<T> TwoPoint<T> │                  │
│  │ Subsampler   │    │ MonotoneCubic<T>         │                  │
│  │ Percentile   │    │ MonotoneConvex<T>        │                  │
│  │ Sobol<T>     │    │ CubicSpline<T> (6 BC's)  │                  │
│  │ Halton<T>    │    │ MonotoneCubicSpline<T>   │                  │
│  │              │    │ InterpolatorFactory<T>   │                  │
│  └──────┬───────┘    └──────────────────────────┘                  │
│         │                                                          │
│  ┌──────▼──────────────────────────────────────────────┐           │
│  │ LinearAlgebra  [Tier A + √]                         │           │
│  │  CholeskyDecomposition<T> (+ pivoted)               │           │
│  │  JacobiEigenDecomposition<T>                        │           │
│  │  GaussianElimination<T> (partial row pivoting)      │           │
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
│  │            NeweyWestVariance<T>                            │   │
│  │  [A + √]   SampleSkewness<T> / SampleExcessKurtosis<T>      │   │
│  │  [A w/ T-tail]  DSR<T> / HaircutSharpe<T> /                  │   │
│  │                 ProbabilityOfBacktestOverfitting<T>         │   │
│  │  [Tier B]       FisherZTransform<T> / DistanceCorrelation<T>│   │
│  │                 GtInspiredScore<T> / MinTRL<T>          │   │
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

Big-O complexity for the major operations. T = number of observations, N = number of assets, K = number of iterations or block size. For Fast Double Bootstrap, B is the outer resample count, n is the sample length, and Cstat(n) is the cost of one statistic call. The **Tier** column names the generic-math constraint class (see [`docs/tier-constraints.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/tier-constraints.md)).

| Component | Tier | Build | Query | Space | Reference |
|-----------|------|-------|-------|-------|-----------|
| **`BisectionSolver<T>`** | B | — | O(log(range/tol)) | O(1) | Burden-Faires §2.1 |
| **`BrentSolver<T>`** | B | — | O(log(range/tol)) worst (at most 2⌈log₂(range/tol)⌉ + 2 iterations), O(K) typical | O(1) | Brent 1973 |
| **`NewtonRaphsonSolver<T>`** | B | — | O(K) quadratic local | O(1) | Burden-Faires §2.3 |
| **`LevenbergMarquardtSolver<T>`** | B | — | O(K·(m·n² + n³)) | O(m·n + n²) | Marquardt 1963; Nielsen 1999 |
| **Householder QR (pivoted, internal)** | A + √ | O(m·n²) | — | O(m·n) | Golub & Van Loan §5.2 |
| **`OrdinaryLeastSquares<T>`** | A + √ | O(m·n² + n³) | — | O(m·n) | Golub & Van Loan §5.3; Björck 1996 |
| **`ActiveSetQpSolver<T>.SolveMinVariance`** | A + √ | O(K·n³) with the certified pre-checks, K ≤ 8N+8 (rounded route; exact route K ≤ 12N+12); an exact test, run only where no certified check decides (a block near or past the conditioning margin, a covariance whose smallest eigenvalue is below the double-double resolution), grows about as n⁵ (m⁴ per conditioning block) | — | O(n²) | Nocedal-Wright §16.5 (active-set QP) |
| **`ActiveSetQpSolver<T>.SolveMeanVariance`** | A + √ | O(K·n³) with the certified pre-checks, K ≤ 8N+8 (rounded route; exact route K ≤ 12N+12); an exact test, run only where no certified check decides (a block near or past the conditioning margin, a covariance whose smallest eigenvalue is below the double-double resolution), grows about as n⁵ (m⁴ per conditioning block) | — | O(n²) | Nocedal-Wright §16.5; Markowitz 1952 |
| **`LinearInterpolator<T>`** | A | — | O(log n) per query | O(n) | classical |
| **`CubicSplineInterpolator<T>`** | A + √ | O(n) | O(log n) per query | O(n) | de Boor 1978 |
| **`MonotoneCubicSpline<T>`** | A + √ | O(n) | O(log n) per query | O(n) | Fritsch-Carlson 1980 |
| **`MonotoneConvexInterpolator<T>`** | A | — | O(n) per query (recomputes tangents) | O(n) | Hagan-West 2006 |
| **`CholeskyDecomposition<T>` (unpivoted)** | A + √ | O(n³) | O(n²) solve | O(n²) | Higham 2002 |
| **`CholeskyDecomposition<T>.DecomposePivoted` (rank R)** | A + √ | O(n²·R) | O(n·R) solve | O(n·R) | arXiv:2507.20678 |
| **`JacobiEigenDecomposition<T>`** | A + √ | O(n³ · sweeps) | — | O(n²) | Jacobi 1846 |
| **`PrincipalComponentAnalysis<T>.Decompose`** | A + √ | O(n³ · sweeps) | O(n² · k) project | O(n²) | Jolliffe 2002 |
| **`PrincipalComponentAnalysis.FromReturns` (decimal facade)** | A + √ | O(T·N² + N³ · sweeps) | O(N² · k) project | O(N²) | Jolliffe 2002 |
| **`GaussianElimination<T>`** | A | O(n³) | — | O(n²) | partial row pivoting |
| **`NearestPsdProjection<T>.EigenClip`** | A + √ | O(n³) | — | O(n²) | Higham 1988 |
| **`NearestPsdProjection<T>.Higham`** | A + √ | O(n³ · iters) | — | O(n²) | Higham 2002 |
| **`SampleSkewness<T>`** | A + √ | O(n) | — | O(1) | Joanes-Gill 1998 (bias-adjusted G₁) |
| **`SampleExcessKurtosis<T>`** | A + √ | O(n) | — | O(1) | Joanes-Gill 1998 (bias-adjusted G₂) |
| **`SampleCovarianceEstimator<T>`** | A | O(N²·T) | — | O(N²) | classical |
| **`ExponentiallyWeightedCovarianceEstimator<T>`** | A | O(N²·T) | — | O(N²) | RiskMetrics |
| **`LedoitWolfShrinkageEstimator<T>` family** | A | O(N²·T) | — | O(N²) | LW 2004 |
| **`QuadraticInverseShrinkageEstimator<T>`** | A + √ | O(N²·T + N³·sweeps) | — | O(N²) | LW 2022 |
| **`DenoisedCovarianceEstimator<T>`** | A + √ | O(N²·T + N³) | — | O(N²) | LdP 2018 |
| **`PearsonCorrelation<T>`** | A | O(T) per pair | — | O(1) | classical |
| **`DistanceCorrelation<T>`** | B | O(T²) | — | O(T²) | Székely 2007 |
| **`WelfordMoments<T>`** | A | O(T) total | O(1) per add | O(1) | Welford 1962 |
| **`NeweyWestVariance<T>` HAC** | A | O(T·L) | — | O(T) | Newey-West 1987 |
| **`ProbabilityOfBacktestOverfitting<T>` via CSCV** | A w/ t-tail | O(C(S, S/2) · N · T) | — | O(N·S + folds) | Bailey-LdP 2014 |
| **`BootstrapMonteCarloEngine<T>` (IID)** | A + √ | O(B·T) | — | O(B) | classical |
| **`StationaryBootstrapResampler<T>`** | A | O(B·T) | — | O(B + T) | Politis-Romano 1994 |
| **`PolitisWhiteBlockLength.EstimateDouble`** | double, transcendental | O(T^{3/2}) (autocorrelations to lag ⌈√T⌉ + 5) | — | O(T) | Politis-White 2004 |
| **`FastDoubleBootstrap`** (decimal-only) | A operations | O(Bn + B·Cstat(n) + B log B), 2B statistic calls | — | O(B + n) | Davidson-MacKinnon 2007 |
| **`HarrellDavisPercentile.Compute<T>`** | C | O(n) | — | O(n) | Harrell-Davis 1982; double-precision Beta weights |
| **`SobolSequence<T>` (d-dim)** | C | O(d) per point | — | O(d · log Nmax) | Joe-Kuo 2008 |
| **`HaltonSequence<T>` (d-dim)** | C | O(d · log N) per point | — | O(d) | Halton 1960 |
| **`Pcg64RandomSource<T>` / `Xoshiro256StarStarRandomSource<T>`** | C | — | O(1) per draw | O(1) state | O'Neill 2014 / Blackman-Vigna 2018 |
| **`RollingWindow<T>`** | — (data structure) | O(1) amortised per `Add` | O(1) indexer | O(capacity) | classical circular buffer |

## Design Decisions

The non-obvious design choices and their rationale:

1. **The public surface is generic over `T`; the caller picks the precision.** Every algorithm lives at a tier (A, A+√, B, C, or A with transcendental tail) — see [`docs/tier-constraints.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/tier-constraints.md) for the definitions and per-folder assignments. Tier A and A+√ algorithms compile for `T ∈ {double, float, Half, decimal}`, but the library guarantees `double` and `decimal` only, treats `float` as best-effort and does not support `Half` (the [Supported numeric types](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/tier-constraints.md#supported-numeric-types) contract, with its operating domain); Tier B algorithms require `IFloatingPointIeee754<T>` and accept `{double, float, Half}`; Tier C algorithms carry a polynomial-approximation internal ceiling calibrated for `double` and expose `T` at the public surface only for pipeline composability. New overloads must not cast through another working type, put `decimal → double → decimal` in an inner loop, or add a `decimal` Tier B facade. The existing legacy Jacobi facade retains its disclosed double working precision; the generic form makes instantiation a caller decision. Concrete-typed facades marked `BNU0001` (including `BrentSolver` and `CholeskyDecomposition`) are deprecated in 2.0.0 and scheduled for removal in v3.0; the migration guide lists the retained Monte Carlo/QMC conveniences. See [`docs/migration-to-generic-math.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md) for migration patterns.

2. **`ReturnsMatrix` resolves layout at the API boundary and copies defensively.** Trading uses `decimal[][]` (asset-major). Numerics' canonical layout is `decimal[,]` T×N. The struct accepts either layout and exposes a `(time, asset)` indexer. Construction copies the supplied arrays, and each `AsTimeByAsset()` call materializes a new T×N copy for either input layout. Default interface methods on `ICovarianceEstimator` make the layout transparent — callers just pass either type.

3. **`FromSeed(...)` static factories instead of optional-param constructors.** The `PublicAPI.Unshipped.txt` analyzer enforces RS0026 (no two overloads carrying optional parameters). Resamplers and MC engines have a primary `(...required..., IRandomSource)` constructor and a static `FromSeed(...required..., int? seed = null)` factory that wraps a `Pcg64RandomSource`. Same shape across `BootstrapResampler`, `BootstrapMonteCarloEngine`, `StationaryBootstrapResampler`, `MovingBlockBootstrapResampler`, `WildBootstrapResampler`, `FastDoubleBootstrap`.

4. **No `IRootSolver` marker interface.** The old single `Solve(f) → result` contract didn't fit both bracketed and unbracketed solvers cleanly. They live on independent interfaces; consumers that need polymorphism instantiate the concrete type they want. Reduces interface bloat at zero ergonomic cost — no consumer in the ecosystem actually wanted "any solver."

5. **PCG-64 is the default RNG, not xoshiro256\*\*.** The authors report empirical tests for the reference generator families; this library has not reproduced BigCrush. PCG exposes distinct non-zero stream identifiers for different odd increments, which does not guarantee statistical independence. `Xoshiro256StarStarRandomSource` is exposed for callers who want `Jump()`-based partitioning instead.

6. **Embedded Sobol direction tables, not external file load.** Joe-Kuo's full table goes to 21,201 dimensions; we ship the first 32, which covers the OptionPricing v2 use case. Avoiding an external dependency (or a 200kB embedded resource) keeps the package small. If we ever need higher-dim Sobol, the table extends linearly.

7. **`HarrellDavisPercentile` carries its own incomplete-beta evaluation.** The file embeds an incomplete-beta continued fraction and log-gamma recurrence followed by the [NIST DLMF 5.11.1 Stirling expansion](https://dlmf.nist.gov/5.11#E1). Tiny double tail masses retain a logarithmic representation through their observation products. Avoids depending on MathNet.Numerics for one function.

8. **Default interface methods are not directly callable on sealed concrete estimators.** This is a C# language constraint, not a design choice. Callers who want the `Estimate(ReturnsMatrix)` overload must cast: `ICovarianceEstimator e = new SampleCovarianceEstimator(); e.Estimate(matrix);`. Documented in the `ReturnsMatrix` XML doc with examples.

9. **PSD contract is documented per-estimator, not enforced.** `ICovarianceEstimator.Estimate` does not promise a PSD result — Sample / LW (all three targets) / OAS / NERCOME / DSCE are PSD by construction; EWMA is PSD for every accepted λ ∈ (0, 1); QIS is positive definite unless an asset is constant or exactly collinear (those directions stay at zero); Denoised / Detoned / Tracy-Widom keep the eigenvalue reconstruction PSD and rescale it to a unit diagonal, which preserves PSD (Detoned is singular when a market component is fully removed with positive residual diagonal; small-asset/all-noise paths and identity fallback need not be singular); POET's thresholding can break positivity. Callers needing a hard guarantee wrap with `NearestPsdProjection.EigenClip(...)`. Documented at the interface level so consumers know which estimators are safe to feed directly into a Cholesky.

10. **`DeflatedSharpeRatio.ComputeFromReturns` (not a second `Compute` overload).** Two `Compute` overloads each carrying optional parameters would violate RS0026. The returns-driven convenience method gets a distinct name.

11. **LM uses an absolute `initialDamping = λ₀`, with the default `1e-3` starting-value recipe from Numerical Recipes §15.5.2. This is distinct from Moré's trust-radius initialization and the identity-damped Nielsen choice `τ·max(diag(JᵀJ))`.** Earlier drafts tried the scaled form, which failed hard on NIST StRD Hahn1 (`max(diag)` ~ 1e19 at Start 2 pushed the step below float resolution, producing zero progress and `DampingOverflow` before any acceptance). The Marquardt scaled-diagonal damping inside the step equation `(JᵀJ + λ·diag(JᵀJ))·δ = −Jᵀr` already delivers per-parameter scale-invariance; double-scaling at initialization is unnecessary and harmful on ill-balanced Jacobians. LM tests both actual and predicted relative reductions after accepted and rejected trials, together with MINPACK lmder's requirement that half the gain ratio be at most one — an absolute ftol would fire prematurely on low-cost problems like Lanczos3 (where optimal cost ~ 1e-8 triggers convergence at ~1% relative reduction against a 1e-10 absolute bar).

12. **PCA sign convention: largest-magnitude component is non-negative.** Eigenvectors are determined up to sign; without an explicit convention, consumers see arbitrary sign flips across runs. For each eigenvector column, the column is flipped so its largest-magnitude entry is ≥ 0. Deterministic across runs on identical data. Eigenvalues and eigenvector columns are sorted descending so PC1 is always the largest-variance direction — matches Hull Ch 33.6's yield-curve level/slope/curvature convention. `PcaResult.Mean` is populated by `FromReturns` (it has observation data to center) and empty for `Decompose(covarianceMatrix)` (no observation data available); `Project` centers internally when `Mean` is populated and assumes caller-pre-centered input otherwise.

13. **OLS uses Householder QR with column pivoting and decimal-residual refinement, not the normal equations.** The classical closed-form `β̂ = (XᵀX)⁻¹·Xᵀy` requires forming `XᵀX`, which squares the condition number of the design matrix. The Wampler degree-five design has condition number about 6.4·10⁶; Wampler5 adds large response noise. QR avoids squaring the design condition number, while extended residual arithmetic addresses the coefficient sensitivity of this noisy problem. Column pivoting (LAPACK DGEQP3-style) additionally handles polynomial designs like `Filip` where the column norms span twelve orders of magnitude. Refinement follows Björck's augmented-system method (coefficients and residual corrected together), with the residual blocks accumulated in `decimal` (~28 significant digits, after exact power-of-two rescaling) and fed back into `double` solves with the stored QR factor. NIST's certification methodology uses multiple-precision calculations accurate to 500 digits and rounds the certified results to 15 significant digits. It notes that ordinary-precision computations may differ; it does not prescribe a required intermediate precision for other implementations.

14. **Every load-bearing public type has at least one external-reference cross-check.** The test suite has three tiers: `Boutquin.Numerics.Tests.Unit` asserts algorithm mechanics against self-generated truth, `Boutquin.Numerics.ArchitectureTests` asserts structural invariants (zero `Boutquin.*` deps, sealed types, PublicAPI hygiene), and `Boutquin.Numerics.Tests.Verification` asserts agreement with a Python reference — NIST StRD, `statsmodels` / `scipy` / `numpy` / `scikit-learn` / `arch` / `mpmath`, or real Fama-French factor data. The verification-coverage standard is that every type which ships a numerical contract must have at least one Python cross-check; a deferral is acceptable only with an explicit in-code `<remarks>` note stating why the external reference is not yet available. When multiple implementations of the same algorithm exist in the Python ecosystem, we prefer an **independent** implementation (different authors / codebase / arithmetic backend) over a hand-port — hand-ports share algorithmic choices with the C# class and can silently reproduce the same bug on both sides. When only a hand-port is available, the paper and equation number are cited in the generator comment. This policy retrofitted ~35 coverage gaps across solvers, covariance estimators, bootstrap resamplers, scalar statistics, interpolators, and root solvers — see the subsystem-level **Validation** footnotes in the [Features](#features) section above for the specific references and tolerance tiers per type.

15. **Why generic math: the caller's precision, not the library's.** The original Boutquin.Numerics shipped with a hard-bound type-per-folder model — `Solvers/` was `double`, `LinearAlgebra/` was `decimal`, and so on. The choice was defensible at the time: it pre-dated `INumber<T>` / `IFloatingPoint<T>` stabilising in the BCL, and it let the library ship with one coherent precision story per subsystem. Three pressures made it untenable over 2026. First, consumers repeatedly needed the opposite type — a curve-bootstrap team wanted `double` Cholesky for speed, a portfolio team wanted `decimal` OLS for precision — and cast-through overloads silently capped precision at the decimal-to-double boundary, misleading the caller. Second, ill-conditioned NIST linear-regression problems (`Filip`, `Wampler4`, `Wampler5`) stalled the shipped `double`-QR OLS at 6–8 digits of precision where the caller might reasonably ask for more, and the `decimal` arithmetic infrastructure to do better already existed in the library but wasn't wired to the `double` OLS. Third, the type choice had to be made in exactly the place the library is worst at making it — at ship time, not at call site. Generic math resolves all three: the algorithm is written once in `T`, instantiated by the caller at `double` / `float` / `Half` / `decimal` (Tier A / A+√), and Tier B algorithms fail at compile time rather than silently cast. `OrdinaryLeastSquares<decimal>` runs the QR, `Qᵀ·y`, upper-triangular solve, and `(XᵀX)⁻¹` end-to-end in 28-digit `decimal`, holding the NIST coefficient bar at 1e-13 on ten problems and 1e-6 on Filip (see the OLS Validation line above); `CholeskyDecomposition<double>` lets curve-bootstrap callers run at IEEE speed without a facade; `BrentSolver<float>` enables memory-bound Monte Carlo pipelines that never had an option before. The `NumericPrecision<T>` dispatcher inside `Internal/` keeps the `decimal`-`Sqrt` path (Newton-Raphson) and IEEE `Sqrt` paths in one file, specialised by the JIT per-`T`. See [`docs/tier-constraints.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/tier-constraints.md) for the full tier model.

## Documentation

User-facing guides live in [`docs/`](https://github.com/boutquin/Boutquin.Numerics/tree/main/docs):
- [`docs/solvers.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/solvers.md) — root-finder family overview and the `LevenbergMarquardtSolver` reference (damping rule, termination semantics, parameter covariance).
- [`docs/linear-algebra.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/linear-algebra.md) — factorizations + `PrincipalComponentAnalysis` conventions and a Treasury-curve usage note.
- [`docs/statistics.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/statistics.md) — sample moments, covariance estimators, backtest statistics, Monte Carlo bootstrap, and percentile estimation.
- [`docs/migration-to-generic-math.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md) — step-by-step migration from legacy concrete-typed facades to the generic surface, with before/after examples for each subsystem and suppression options for the `BNU0001` warning.
- [`docs/tier-constraints.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/tier-constraints.md) — four-tier constraint model (`A`, `A+√`, `B`, `C`) governing which numeric type each algorithm accepts.

In-code XML doc-comments on every public API surface (IntelliSense-visible) remain the primary reference for contract-level details.

## Versioning & Release

- **MinVer-driven** — versions come from release tags (`vX.Y.Z`), not from project files.
- **Local pack** — `dotnet pack /p:MinVerVersionOverride=2.0.0-local --output nupkg` for local development. Outputs land in `nupkg/`.

## Test Infrastructure

- **`Boutquin.Numerics.Tests.Unit`** — xUnit + FluentAssertions, covering algorithm correctness, numerical tolerances, edge cases, reproducibility, generic parity, and cross-type regression.
- **`Boutquin.Numerics.ArchitectureTests`** — NetArchTest, enforcing:
  - Numerics namespaces contain no `Boutquin.*` references
  - All public types are `sealed` unless they're interfaces or records
  - Internal helpers don't leak to `PublicAPI.Shipped.txt`
- **`Boutquin.Numerics.Tests.Verification`** — xUnit cross-language checks against fixed reference vectors generated with `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `arch`, and `mpmath`.
- **`PublicAPI` analyzer** — every public symbol must be declared in `PublicAPI.Unshipped.txt` before it can be added; build fails otherwise. Forces explicit API surface tracking and prevents accidental public-API expansion.
- **`Boutquin.Numerics.BenchMark`** — BenchmarkDotNet harness under `benchmarks/`. One class per namespace with `[MemoryDiagnoser]` to track allocations; see `benchmarks/README.md` for interpretation.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](https://github.com/boutquin/Boutquin.Numerics/blob/main/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](https://github.com/boutquin/Boutquin.Numerics/blob/main/CODE_OF_CONDUCT.md) first.

### Reporting Bugs

Open an issue on the [Issues](https://github.com/boutquin/Boutquin.Numerics/issues) page with:
- A clear and descriptive title
- Steps to reproduce the issue
- Expected and actual behavior
- Code snippets and reference values, if applicable

### Contributing Code

1. Fork the repository and clone locally
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes following [CONTRIBUTING.md](https://github.com/boutquin/Boutquin.Numerics/blob/main/CONTRIBUTING.md) and the [tier constraints](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/tier-constraints.md)
4. Add tests covering the new behavior
5. Add public-API entries to `src/Boutquin.Numerics/PublicAPI.Unshipped.txt`
6. Verify `dotnet build` (warnings-as-errors), `dotnet test`, `dotnet format --verify-no-changes`
7. Commit with clear messages: `git commit -m "Add feature X"`
8. Push and open a pull request

## Disclaimer

Boutquin.Numerics is open-source software provided under the Apache 2.0 License. It is a foundational numerical methods library intended for educational, research, and engineering use.

**This software does not constitute financial advice.** Boutquin.Numerics provides numerical algorithms — root finders, distributions, interpolators, linear algebra, covariance estimators, statistics, Monte Carlo, and random number generation — that are consumed by higher-level financial libraries in the Boutquin ecosystem (MarketData, Curves, Trading, OptionPricing). The library itself has zero domain dependencies and makes no claims about the suitability of any computed value for trading, investment, risk, or regulatory purposes. Before using numerical results produced with this library in production, validate against independent references, confirm tolerances are appropriate for your use case, and consult qualified professionals who understand your specific requirements and regulatory obligations.

## Migration

Facades marked `BNU0001` are deprecated in 2.0.0 (warning, not error) and scheduled for removal in v3.0. The eight Monte Carlo/QMC convenience facades listed in the migration guide do not carry that warning or that removal announcement. The facades continue to work in 2.x; see the [migration guide](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md) for the separate v2 selector, span, GT-Score, and removed-API changes.

**Quick migration** — append `<double>` or `<decimal>` to the class name:

```csharp
// Before
var solver = new BrentSolver();
var cov    = new SampleCovarianceEstimator().Estimate(matrix);

// After
var solver = new BrentSolver<double>(1e-10);
var cov    = new SampleCovarianceEstimator<decimal>().Estimate(matrix);
```

**Project-wide suppression** (buy time before a full migration):

```xml
<PropertyGroup>
  <NoWarn>$(NoWarn);BNU0001</NoWarn>
</PropertyGroup>
```

Full before/after examples for every subsystem (solvers, linear algebra, statistics, interpolators), suppression options, and the v3.0 removal timeline are in [`docs/migration-to-generic-math.md`](https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md).

## License

Apache 2.0 — see [LICENSE.txt](https://github.com/boutquin/Boutquin.Numerics/blob/main/LICENSE.txt) for details.

## Contact

For inquiries, please [open an issue](https://github.com/boutquin/Boutquin.Numerics/issues).

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
- Refinements from arXiv:2507.20678 (pivoted Cholesky), arXiv:2507.05083 (Q-Spline), and Cephes erf/erfc rational approximations (see [third-party notices](THIRD-PARTY-NOTICES.txt))
