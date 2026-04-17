# Solvers

Boutquin.Numerics ships four families of solvers: scalar root finders, multivariate nonlinear least squares, ordinary least squares, and active-set quadratic programming. The public surface is generic over `T` — the caller picks the working type. Tier constraints are documented in [`docs/tier-constraints.md`](tier-constraints.md).

| Family | Generic type | Tier | Accepted `T` |
|--------|--------------|------|---------------|
| Bracketed root finders | `BisectionSolver<T>`, `BrentSolver<T>`, `NewtonRaphsonSolver<T>` | Tier B | `double`, `float`, `Half` |
| Unbracketed root finders | `SecantSolver<T>`, `MullerSolver<T>`, `NewtonRaphsonSolver<T>` | Tier B | `double`, `float`, `Half` |
| Multivariate nonlinear LS | `LevenbergMarquardtSolver<T>` + `LevenbergMarquardtBuffers<T>` | Tier B | `double`, `float`, `Half` |
| Linear LS | `OrdinaryLeastSquares<T>` + `OlsResult<T>` | Tier A + √ | `double`, `float`, `Half`, `decimal` |
| Portfolio QP | `ActiveSetQpSolver<T>` (MinVar + MeanVar) | Tier A + √ | `double`, `float`, `Half`, `decimal` |

Legacy concrete-typed types (`BrentSolver`, `OrdinaryLeastSquares`, etc.) remain live as forwarding facades over the generic types at `T = double` — existing callers need no change. Facade obsoletion is a future enhancement contingent on every downstream consumer migrating to the generic surface; no shipped date.

## Scalar root finders

For <code>f: ℝ → ℝ</code>, solving <code>f(x) = 0</code>.

| Solver | Bracketing? | Convergence | Use when |
|--------|-------------|-------------|----------|
| `BisectionSolver<T>` | required | linear | a guaranteed bound is more important than speed |
| `BrentSolver<T>` | required | superlinear | general-purpose bracketed root (default pick) |
| `NewtonRaphsonSolver<T>` | optional bracket | quadratic with bracket, possibly divergent without | analytic derivative is available or acceptable via finite differences |
| `SecantSolver<T>` | unbracketed | superlinear | derivative is unavailable and a good initial guess exists |
| `MullerSolver<T>` | unbracketed | superlinear | near a root with complex-adjacent behavior |

All scalar root finders are Tier B: they call transcendental functions on `T` (the caller's `f` is almost always transcendental, and the solvers themselves use `T.Abs`, comparison, and arithmetic). `T : IFloatingPointIeee754<T>` is the required constraint — `double`, `float`, and `Half` are accepted. A `decimal` attempt fails at compile time with `CS0315`; use the legacy concrete-typed facade (which runs at `T = double` internally). Tier B on `T = decimal` is a future enhancement (requires implementing 28-digit Remez / Chebyshev / CORDIC approximations for `Log`, `Exp`, trig, etc.); no shipped date.

### Validation

Every root solver is cross-checked against the certified mathematical root on the standard test-function pool (`x³ − x − 2`, `x² − 2`, `log x − 1`, `cos x − x`, `e⁻ˣ − x`) at 1e-6 absolute — see [`tests/Boutquin.Numerics.Tests.Verification/SolverVerificationTests.cs`](../tests/Boutquin.Numerics.Tests.Verification/SolverVerificationTests.cs). Because the test-function roots are mathematical constants (`√2`, `e`, the Dottie number, etc.), the roots themselves are the source of truth, not any particular library's output. `MullerSolver` additionally has its reference roots validated at generation time against `mpmath.findroot(..., solver='muller')` — mpmath is an arbitrary-precision library with a Muller implementation from a different codebase, so a shared algorithmic bug (sign flip in the "choose denominator with larger magnitude" rule, off-by-one in the sign-of-discriminant branch) cannot silently match between the Numerics C# class and the Python reference. `NewtonRaphsonSolver` is additionally verified on both the bracketed and unbracketed entry points; `SecantSolver` via the single-point `Solve(f, initialGuess)` auto-perturb overload.

Generic parity is gated by [`tests/Boutquin.Numerics.Tests.Unit/GenericParity/Solvers_ParityTests.cs`](../tests/Boutquin.Numerics.Tests.Unit/GenericParity/Solvers_ParityTests.cs): `BrentSolver<double>` is bit-identical to the legacy `BrentSolver` facade on the NIST StRD input pool, and analogous parity for each other solver.

## Multivariate nonlinear least-squares

For <code>r: ℝⁿ → ℝᵐ</code>, minimizing <code>½ Σᵢ rᵢ(θ)²</code>.

### `LevenbergMarquardtSolver<T>`

Industry-standard solver for nonlinear least squares. Interpolates between Gauss-Newton (fast near the optimum) and gradient descent (robust far from it) via a damping parameter <code>λ</code>. Tier B at `T : IFloatingPointIeee754<T>`.

```csharp
using Boutquin.Numerics.Solvers;

// Fit y = a·exp(b·x) to (xs, ys), worked in double.
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

var solver = new LevenbergMarquardtSolver<double>();
var result = solver.Solve(
    residuals: Residuals,
    initialGuess: [1.0, 0.1]);

if (result.Converged)
{
    var a = result.Parameters[0];
    var b = result.Parameters[1];
    // Standard errors from the Gauss-Newton asymptotic approximation.
    var sigmaA = Math.Sqrt(result.ParameterCovariance![0, 0]);
    var sigmaB = Math.Sqrt(result.ParameterCovariance[1, 1]);
}
```

**Working at `T = float` for memory-bound Monte Carlo calibrations:**

```csharp
float[] ResidualsFloat(float[] theta) { /* … */ }

var solverF = new LevenbergMarquardtSolver<float>();
var fit = solverF.Solve(ResidualsFloat, [1.0f, 0.1f]);
```

**Analytic Jacobian** (faster and more accurate than central finite differences when available):

```csharp
double[,] Jacobian(double[] theta)
{
    var j = new double[xs.Length, 2];
    for (var i = 0; i < xs.Length; i++)
    {
        var e = Math.Exp(theta[1] * xs[i]);
        j[i, 0] = -e;                 // ∂rᵢ/∂a
        j[i, 1] = -theta[0] * xs[i] * e; // ∂rᵢ/∂b
    }
    return j;
}

var analytic = solver.Solve(Residuals, [1.0, 0.1], Jacobian);
```

**Box bounds** (projection after each accepted step):

```csharp
var bounded = solver.Solve(
    Residuals,
    initialGuess: [1.0, 0.1],
    jacobian: null,
    lowerBounds: [0.0, -1.0],
    upperBounds: [10.0, 1.0]);
// bounded.BoundsActive == true iff any parameter sits on a supplied bound at termination.
```

**Key design choices:**

- **Half-SSE cost:** `FinalCost = ½ Σᵢ rᵢ²`. The gradient is `Jᵀr` (not `2Jᵀr`), matching the Gauss-Newton Hessian approximation `H ≈ JᵀJ`.
- **Marquardt scaled-diagonal damping:** the step equation is `(JᵀJ + λ·diag(JᵀJ))·δ = −Jᵀr`, invariant under per-parameter rescaling of the residual function. Important when parameters have widely different magnitudes.
- **Nielsen damping update** (Madsen-Nielsen-Tingleff 2004, §3.2): on a successful step, scale <code>λ</code> by <code>max(1/3, 1 − (2ρ − 1)³)</code> where <code>ρ</code> is the gain ratio; on a rejected step, <code>λ ← λ·ν; ν ← 2ν</code>. Converges faster on ill-conditioned problems than the fixed <code>λ/ν</code> Marquardt rule.
- **Absolute initial damping:** `λ₀ = initialDamping` (Moré 1978), default <code>1e-3</code>. Per-parameter scale invariance is already delivered by the Marquardt scaled-diagonal form inside the step equation; double-scaling at initialization (the Nielsen <code>τ · max(diag(JᵀJ))</code> variant) was tried and reverted — it produced zero-progress failures on ill-scaled problems like NIST Hahn1 where <code>max(diag(JᵀJ))</code> at the starting point pushed λ above the step resolution of double precision.
- **Relative function tolerance** (MINPACK lmder): `FunctionToleranceReached` fires when `(priorCost − newCost) / priorCost ≤ functionTolerance` across a successful step. Relative rather than absolute because absolute tolerance fires prematurely on low-cost problems — e.g., NIST Lanczos3's optimal cost is ~1e-8, so an absolute 1e-10 threshold triggers at ~1% relative reduction.
- **Box bounds** (optional): projected LM — after each step, clip parameters into `[lowerBounds, upperBounds]`. Convergence status is evaluated on the free components; if any component is on a bound, `BoundsActive` is set on the result.
- **Deterministic:** no RNG, no parallelism. Same inputs produce bit-identical outputs.

**Jacobian:**

- Supply an analytic Jacobian for best accuracy and speed. Shape is `m × n` with `J[i, j] = ∂rᵢ/∂θⱼ`.
- When omitted, the solver uses central finite differences with adaptive per-component step <code>h = max(1e-8, |θⱼ|·√ε)</code>.

**Termination:**

Exactly one `LmTerminationReason` is reported:

| Reason | Meaning |
|--------|---------|
| `FunctionToleranceReached` | relative cost reduction `(priorCost − newCost) / priorCost` fell at or below `functionTolerance` on an accepted step |
| `ParameterToleranceReached` | step norm fell below `parameterTolerance · (‖θ‖ + parameterTolerance)` |
| `GradientToleranceReached` | `‖Jᵀr‖∞ ≤ gradientTolerance` |
| `MaxIterationsReached` | iteration budget exhausted (`Converged = false`) |
| `DampingOverflow` | `λ` saturated the ceiling without finding an acceptable step (`Converged = false`) |

`BoundsActive` is orthogonal — it can be true with any termination reason.

**Parameter covariance:**

When `Converged = true` and the problem is overdetermined (`m > n`), the solver reports `ParameterCovariance = σ̂² · (JᵀJ)⁻¹` where `σ̂² = 2·FinalCost / (m − n) = RSS / (m − n)`. This is the Gauss-Newton asymptotic approximation; use for standard-error bars and correlation structure, not for exact inference.

### Hot-path usage — pooled buffers

One-shot calibrations (fitting a single model once) use the `Solve` overload shown above and incur a small per-solve allocation for scratch buffers. Hot-path consumers — Monte Carlo calibration, bootstrap loops, real-time refit engines — should pre-allocate a `LevenbergMarquardtBuffers<T>` pool once and pass it to every solve. The pool eliminates inner-loop managed-heap allocation, so only the returned `MultivariateSolverResult<T>` and its owned arrays are produced per solve.

```csharp
using Boutquin.Numerics.Solvers;

// Construct once, outside the hot loop.
var buffers = new LevenbergMarquardtBuffers<double>(parameterCount: 6, residualCount: 24);
var solver = new LevenbergMarquardtSolver<double>();

foreach (var resample in bootstrap.Resamples)
{
    var result = solver.Solve(resample.Residuals, initialGuess, buffers);
    // ... record result.Parameters, result.FinalCost, etc.
}
```

**Rules:**

- **One pool per thread.** `LevenbergMarquardtBuffers<T>` is not thread-safe. Concurrent solves sharing a pool produce undefined results. Each worker thread should own its own pool.
- **Resize with `Reset` when dimensions change.** Fitting models of varying sizes against the same pool is supported via `buffers.Reset(n, m)`. The policy is grow-only — capacity is monotonic over the pool's lifetime, so repeated resets to smaller sizes do not churn allocations.
- **Match the dimensions.** If `buffers.ParameterCount` does not match the initial-guess length, or if the residual function returns a vector whose length differs from `buffers.ResidualCount`, `Solve` throws `ArgumentException` with a message telling you which `Reset(n, m)` call will fix it.
- **Pool-free overload is bit-identical.** Both `Solve` overloads produce identical arithmetic — results match down to floating-point rounding. Choose the pooled overload only when allocation pressure matters; the one-shot overload stays simpler for code that runs a handful of times.
- **Zero-allocation requires a zero-allocation residual callback.** The `Func<T[], T[]>` residual contract is unchanged; consumers targeting the strictest zero-alloc bar should return a cached `T[]` from their residual function rather than a fresh array per call. The solver copies each returned vector into pool storage before any subsequent iteration, so reusing the same array across calls is safe.

**Allocation profile (n = 6, m = 24, `T = double`):**

| Overload | Per-solve allocation |
|----------|----------------------|
| Pool-free `Solve` | Result record + owned arrays + private pool construction |
| Pooled `Solve` | Result record + owned arrays + one-shot covariance scratch at convergence |
| Inner `EvaluateInto + TrySolve` (direct) | 0 bytes across 1000 invocations after warmup |

See [LevenbergMarquardtAllocationTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/LevenbergMarquardtAllocationTests.cs) for the regression harness.

### Validation

Validated against all 26 NIST Statistical Reference Dataset nonlinear-regression benchmark problems — the full suite across every difficulty tier (lower: `Misra1a`, `Chwirut1/2`, `Lanczos1/2`, `Gauss1/2`, `DanWood`, `Misra1b`; average: `Kirby2`, `Hahn1`, `Nelson`, `MGH17`, `Lanczos3`, `Gauss3`, `Misra1c/d`, `Roszman1`, `ENSO`; higher: `MGH09`, `Thurber`, `BoxBOD`, `Rat42`, `MGH10`, `Eckerle4`, `Rat43`, `Bennett5`) — converging from NIST "Start 2" with finite-difference Jacobian under per-tier tolerances. Lanczos1 and Lanczos2 have certified SSE below double precision's practical floor (~10⁻²⁰) and are asserted against that floor rather than a relative bar; parameter agreement remains the authoritative convergence signal for those two. See [`tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs`](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) for the test harness and `NistStRDData.cs` for the transcribed problem set. The NIST harness runs against `LevenbergMarquardtSolver<double>`; parity with the legacy `LevenbergMarquardtSolver` facade is enforced by [`Solvers_ParityTests.cs`](../tests/Boutquin.Numerics.Tests.Unit/GenericParity/Solvers_ParityTests.cs).

Additionally cross-checked against **`scipy.optimize.least_squares(method='lm')`** across four smooth regimes disjoint from the NIST Nonlinear suite (exponential-decay-with-noise, sum-of-two-sinusoids, Gompertz growth curve, logistic) at 1e-5 relative tolerance on both `Parameters` and `FinalCost`. scipy's `lm` method delegates to MINPACK lmder — same algorithmic family as `LevenbergMarquardtSolver<T>`, so precision parity at 5+ digits is achievable on smooth problems. See [`tests/Boutquin.Numerics.Tests.Verification/LevenbergMarquardtVerificationTests.cs`](../tests/Boutquin.Numerics.Tests.Verification/LevenbergMarquardtVerificationTests.cs).

### References

- Marquardt, D. W. (1963). *An algorithm for least-squares estimation of nonlinear parameters*. SIAM J. Appl. Math. 11(2), 431-441.
- Moré, J. J. (1978). *The Levenberg-Marquardt algorithm: implementation and theory*. In: *Numerical Analysis*, Lecture Notes in Mathematics 630, 105-116. The MINPACK lmder implementation referenced for the relative function-tolerance convention.
- Madsen, Nielsen, Tingleff (2004). *Methods for Non-Linear Least Squares Problems* (IMM Technical Report). The Nielsen damping-update rule used here is §3.2.
- NIST/ITL Statistical Reference Datasets — Nonlinear Regression. `https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml`.

## Ordinary least squares

For `y = X·β + ε`, minimizing `‖y − X·β‖₂²`.

### `OrdinaryLeastSquares<T>`

Standard OLS estimator backed by Householder QR with column pivoting and (at `T = double`) mixed-precision iterative refinement. Tier A + √ at `T : IFloatingPoint<T>` with `Sqrt` via `NumericPrecision<T>`. Accepts `double`, `float`, `Half`, and `decimal` — the caller's type choice is the precision dial.

Delivers NIST-bar accuracy on every problem in the NIST StRD linear-regression suite, including the notoriously ill-conditioned `Longley` (cond ≈ 5·10⁷), `Wampler4` (cond ≈ 5·10¹⁰), and `Wampler5` (cond ≈ 5·10¹³) datasets where the classical normal-equation approach loses every digit of the response.

**Worked example — `T = double`, Longley (NIST StRD, cond ≈ 5·10⁷):**

```csharp
using Boutquin.Numerics.Solvers;

// 16 observations of US macroeconomic aggregates, 6 predictors.
var longleyX = new double[16, 6]
{
    {  83.0, 234289, 2356, 1590, 107608, 1947 },
    {  88.5, 259426, 2325, 1456, 108632, 1948 },
    {  88.2, 258054, 3682, 1616, 109773, 1949 },
    // … 13 more observations …
};
var longleyY = new[] { 60323.0, 61122, 60171, /* … */ };

var ols = new OrdinaryLeastSquares<double>();
var fit = ols.Fit(longleyX, longleyY);
// fit.Coefficients[0] ≈ −3,482,258.63459582 (intercept)
// fit.StandardErrors[0] ≈ 890,420.383607373
// fit.ResidualStandardDeviation ≈ 304.854073561965
// fit.RSquared ≈ 0.995479004577296
```

**Worked example — `T = decimal`, full 28-digit QR (NIST StRD `Filip`, cond ≈ 10¹⁰):**

```csharp
// Instantiating OrdinaryLeastSquares<T> at T = decimal runs Householder QR with column
// pivoting, Q^T·y, upper-triangular solve, and (XᵀX)⁻¹ end-to-end in 28-digit decimal.
// Well-conditioned designs clear the 1e-9 coefficient bar; ill-conditioned ones cap
// higher — the achievable accuracy is bounded by cond(X)·u, and the single-precision
// decimal path does not layer extended-precision residual refinement on top the way
// the double-precision path does.
decimal[,] filipX = /* 82 observations of a degree-10 polynomial; cond ≈ 10¹⁰ */;
decimal[]  filipY = /* … */;

var olsDec = new OrdinaryLeastSquares<decimal>();
var fit = olsDec.Fit(filipX, filipY);
// Coefficients agree with NIST certified values to 5e-4 relative — significantly tighter
// than a normal-equation OLS (which loses every digit at cond ≈ 10¹⁰) but looser than
// the 1e-9 bar available on well-conditioned designs. See NistLinearRegressionDecimalTests.cs
// for the per-problem tolerance table and the condition-number citations.
```

**Legacy facade** — the original `OrdinaryLeastSquares.Fit(double[,], double[], bool)` static surface is preserved as a forwarding facade over `OrdinaryLeastSquares<double>`; existing callers need no change. Facade obsoletion is a future enhancement with no shipped date.

**Key design choices:**

- **Householder QR + column pivoting**, not normal equations. `β̂ = (XᵀX)⁻¹·Xᵀy` doubles the condition number by forming `XᵀX`; QR preserves the conditioning of `X` end-to-end. Column pivoting (LAPACK `DGEQP3`-style) additionally handles polynomial designs where later columns have norms many orders of magnitude larger than earlier ones (`Filip` column norms span `10¹⁰–10⁰` across `[x, x², …, x¹⁰]`).
- **Caller-chosen precision.** The generic type parameter `T` is the precision dial: `T = double` runs the QR in `double` and applies `decimal`-residual iterative refinement, which is the path that performs best on the most ill-conditioned designs (cond > 10¹²); `T = decimal` runs the full factorisation in 28-digit `decimal`, which clears the 1e-9 NIST coefficient bar on the 7 well-conditioned problems and holds principled higher per-problem bars on the 4 ill-conditioned ones (Pontius, Filip, Wampler4, Wampler5). `T = float` is available for memory-bound consumers willing to give up precision for throughput.
- **Decimal-residual mixed-precision iterative refinement (at `T = double`).** `r = y − X·β̂` is computed in `decimal` (~28 significant digits), cast back to `double`, and fed into a double-precision back-substitution for the correction step. This lifts the refinement stall floor from `κ·u` (the double-precision limit) toward `u`. Implemented using the `decimal` arithmetic Boutquin.Numerics already has, obviating an external extended-precision package; semantically equivalent to the double-double refinement step SAS, R, and MATLAB use via Bailey's QD package.
- **Intercept toggle.** `includeIntercept: true` (default) prepends a column of ones to `X` and returns `Coefficients` of length `p + 1` with the intercept at index 0. `includeIntercept: false` passes `X` through unchanged and returns `p` coefficients — the correct choice for NIST's `NoInt1` and `NoInt2` problems or any physically zero-intercept model.
- **R² convention.** Centred total sum of squares `Σ(yᵢ − ȳ)²` when `includeIntercept = true`, uncentred `Σ yᵢ²` otherwise. Matches NIST StRD's convention and what R's `lm()` and Python's `statsmodels.OLS` publish.
- **Rank-deficient detection.** QR returns `false` when any column of `X` (after pivoting) has a numerically zero trailing norm, and `Fit` throws `InvalidOperationException` rather than returning a degenerate `OlsResult<T>`. Collinear predictors must be resolved upstream (ridge, pseudo-inverse, or column elimination).

**`OlsResult<T>` contract:**

| Member | Meaning |
|--------|---------|
| `Coefficients` | `β̂`, length `p` (or `p + 1` when an intercept is included). Returned in the caller's original column order even when internal pivoting shuffles columns during QR. |
| `StandardErrors` | `√diag(σ̂²·(XᵀX)⁻¹)`, indexed in the same order as `Coefficients`. |
| `Residuals` | `y − X·β̂`, length `m`. Computed in `T`; at `T = double` the residual step uses an internal `decimal` promotion for stability on ill-conditioned problems. |
| `ResidualSumOfSquares` | `Σᵢ rᵢ²`. |
| `ResidualStandardDeviation` | `σ̂ = √(RSS / (m − p))`. Returns 0 when residual DoF is zero or negative (saturated design). |
| `DegreesOfFreedom` | `m − p` (including intercept in `p` when one is requested). |
| `RSquared` | `1 − RSS / TSS`, centred or uncentred per the intercept flag. |
| `CovarianceMatrix` | `Cov(β̂) = σ̂²·(XᵀX)⁻¹`, symmetric positive semi-definite at full column rank. |

### Validation

Validated against all 11 NIST Statistical Reference Dataset linear-regression benchmark problems — `Norris`, `Pontius`, `NoInt1`, `NoInt2`, `Filip`, `Longley`, `Wampler1–5` — at both `T = double` and `T = decimal`.

At `T = double`, per-problem accuracy bars are documented in [`NistLinearRegressionTests.cs`](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionTests.cs), calibrated against what double-precision QR with mixed-precision decimal-residual refinement actually achieves. Well-conditioned designs (Norris, NoInt, Longley, Wampler1/2/3) clear the 1e-9 bar; the ill-conditioned ones carry the double-precision stability floor bars from NIST's own [`c-Wampler5.shtml`](https://www.itl.nist.gov/div898/strd/lls/data/LINKS/c-Wampler5.shtml) methodology note (Filip 1e-7 at cond ≈ 10¹⁰, Wampler4 1e-8 at cond ≈ 5·10¹⁰, Wampler5 1e-6 at cond ≈ 5·10¹³).

At `T = decimal`, the same 11 problems run under per-problem tolerances documented in [`NistLinearRegressionDecimalTests.cs`](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionDecimalTests.cs): 1e-9 on the 7 well-conditioned problems, 5e-2 on Pontius (cond ≈ 10⁷), 5e-4 on Filip (cond ≈ 10¹⁰), 1e-8 on Wampler4, 1e-6 on Wampler5. The decimal QR wins on well-conditioned designs (same 1e-9 bar in a single precision regime, no residual-refinement layer) but caps higher than the double path on the most ill-conditioned designs because it does not layer extended-precision residual refinement on top — the achievable accuracy is bounded by cond(X)·u where u is the working precision's unit round-off. A caller who needs tighter-than-double bars on cond > 10¹² problems today uses the `T = double` facade with its mixed-precision refinement; genuinely arbitrary-precision OLS for the extreme regime is out of scope.

At `T = double`, additionally cross-checked against **`statsmodels.api.OLS`** across three regimes — well-conditioned random (cond ≈ 10²), polynomial degree 5 (cond ≈ 10⁶), and no-intercept — at 1e-10 relative on coefficients and 1e-8 relative on derived quantities (`StandardErrors`, `ResidualSumOfSquares`, `ResidualStandardDeviation`, `RSquared`). See [`tests/Boutquin.Numerics.Tests.Verification/OrdinaryLeastSquaresVerificationTests.cs`](../tests/Boutquin.Numerics.Tests.Verification/OrdinaryLeastSquaresVerificationTests.cs).

### References

- Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed. Ch 5 (QR factorization), Ch 20 (least squares).
- Björck, Å. (1996). *Numerical Methods for Least Squares Problems*. SIAM.
- Longley, J. W. (1967). *An appraisal of least squares programs for the electronic computer from the point of view of the user*. *JASA* 62, 819–841.
- Wampler, R. H. (1970). *A report on the accuracy of some widely used least squares computer programs*. *JASA* 65, 549–565.
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. §20 (least squares).
- NIST/ITL Statistical Reference Datasets — Linear Regression. `https://www.itl.nist.gov/div898/strd/lls/lls.shtml`. The methodology page at [`c-Wampler5.shtml`](https://www.itl.nist.gov/div898/strd/lls/data/LINKS/c-Wampler5.shtml) documents NIST's use of Bailey's 500-digit MPFUN for certification. `OrdinaryLeastSquares<decimal>` runs in 28-digit decimal (far from NIST's 500-digit reference arithmetic, and without the residual-refinement layer the double path uses), so the per-problem NIST-certified values remain the authoritative bar — this library's test tolerances are set to what the respective working-precision regime achieves, not to the full NIST reference precision.
- Bailey, D. H. *MPFUN / QD multi-precision packages*. `https://www.netlib.org/mpfun/` — the extended-precision arithmetic SAS, R, and MATLAB use for iterative-refinement residuals on ill-conditioned OLS problems.

## Portfolio-optimization QP

For `w ∈ ℝⁿ`, two standard long-only portfolio problems under a sum-to-one constraint and per-asset box bounds.

### `ActiveSetQpSolver<T>`

Cholesky-based active-set solver for the two standard long-only portfolio QPs. Tier A + √ at `T : IFloatingPoint<T>`; delegates to `CholeskyDecomposition<T>` for the inner linear solves. Accepts `double`, `float`, `Half`, `decimal` — `decimal` is the precision default for the legacy non-generic `ActiveSetQpSolver` facade.

**Minimum-variance portfolio** — `min w′Σw  s.t.  1′w = 1, lb ≤ w ≤ ub`:

```csharp
using Boutquin.Numerics.Solvers;

decimal[,] covariance = LoadAssetCovariance();  // N×N symmetric positive-definite

decimal[] weights = ActiveSetQpSolver.SolveMinVariance(
    covariance,
    minWeight: 0.0m,
    maxWeight: 0.4m);
// weights.Sum() ≈ 1, each 0 ≤ w_i ≤ 0.4
```

**Mean-variance portfolio** — `max w′μ − (λ/2)·w′Σw  s.t.  1′w = 1, lb ≤ w ≤ ub`:

```csharp
decimal[] expectedReturns = LoadExpectedReturns();

decimal[] weights = ActiveSetQpSolver.SolveMeanVariance(
    covariance,
    means: expectedReturns,
    riskAversion: 3.0m,
    minWeight: 0.0m,
    maxWeight: 0.4m);
// riskAversion = 0 reduces MeanVar to a pure linear programme:
// max w′μ s.t. 1′w = 1, lb ≤ w ≤ ub.
```

**Generic form** — pick the working type at the call site:

```csharp
// T = double for hot-path Monte Carlo calibration / backtesting loops.
double[] w = ActiveSetQpSolver<double>.SolveMeanVariance(
    covarianceD, meansD, riskAversion: 3.0, minWeight: 0.0, maxWeight: 0.4);

// T = decimal for portfolio-grade precision end-to-end (no decimal → double cast).
decimal[] wDec = ActiveSetQpSolver<decimal>.SolveMinVariance(
    covarianceM, minWeight: 0.0m, maxWeight: 0.4m);
```

**Key design choices:**

- **Active-set scaffolding.** Each iteration: (1) identify the free set (variables not yet pinned to a bound), (2) solve the Lagrangian for the unconstrained-except-for-sum problem via `CholeskyDecomposition<T>` on the free sub-block, (3) project the candidate solution onto the box bounds and identify the most-violated bound, (4) pin that variable, (5) check KKT multipliers on the currently-fixed set and release any whose multiplier changed sign. Terminates in at most **2N iterations** for `SolveMinVariance` and at most **3N + 3 iterations** for `SolveMeanVariance`.
- **Cross-covariance term.** When one or more variables are fixed at their bounds, the reduced sub-problem's gradient includes a cross-covariance contribution `Σ_cross^T · w_fixed` from the fixed variables' effect on the free block. `ActiveSetQpSolver<T>` applies this correctly via an additional Cholesky back-substitution `z_c = Σ_sub⁻¹ · c_free` alongside the standard `Σ_sub⁻¹ · 1` solve, so `SolveMinVariance` and `SolveMeanVariance` both produce correct results regardless of whether the active set is empty, partially active, or near-fully active at termination.
- **Pure-LP branch.** `riskAversion = 0` in `SolveMeanVariance` short-circuits to a bounded-simplex enumeration that returns the long-only maximum-expected-return portfolio — useful as a sanity check and as a limit-case benchmark.
- **No RNG, no parallelism.** Deterministic; same inputs produce bit-identical outputs.

**Exceptions:**

- `ArgumentException` — covariance matrix not square, or `means.Length` does not match the covariance dimension.
- `InvalidOperationException` — covariance is degenerate (`Σ⁻¹·1` sums to zero under the degeneracy tolerance `1e-20`), or the solver fails to converge within its iteration budget.

### Validation

Cross-checked against **`scipy.optimize.minimize(method='SLSQP')`** across 5 portfolio cases at 1e-6 per-weight in [`QpSolverVerificationTests.cs`](../tests/Boutquin.Numerics.Tests.Verification/QpSolverVerificationTests.cs) / [`generate_qp_solver_vectors.py`](../tests/Verification/generate_qp_solver_vectors.py). Cases include:

- A boxed 3-asset MinVar (exercises the cross-covariance sub-problem path — the configuration where a fixed-bound variable contributes to the free-block gradient).
- Mean-variance at `λ ∈ {1, 3, 10}` on 5-asset and 10-asset covariance matrices.
- A degenerate pure-LP case (`riskAversion = 0`) on a 4-asset input.

The 10 unit tests in [`ActiveSetQpSolverTests.cs`](../tests/Boutquin.Numerics.Tests.Unit/Solvers/ActiveSetQpSolverTests.cs) additionally cover the uniform-weight output on a scaled-identity covariance, the sum-to-one invariant, termination on single-asset inputs, degenerate-covariance rejection, and the `riskAversion = 0` pure-LP branch.

### References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Ch 16 (quadratic programming); §16.5 covers the active-set method this implementation follows.
- Markowitz, H. (1952). *Portfolio Selection*. *The Journal of Finance* 7(1), 77–91 — the mean-variance problem setup.
- Sharpe, W. F. (1963). *A Simplified Model for Portfolio Analysis*. *Management Science* 9(2), 277–293 — single-index / factor-model background relevant when `Σ` is a Ledoit-Wolf-shrunk or POET-denoised covariance estimate.
