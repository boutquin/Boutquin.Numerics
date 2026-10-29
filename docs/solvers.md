# Solvers

Boutquin.Numerics ships five solver families: scalar root finders, derivative-free minimization, multivariate nonlinear least squares, ordinary least squares, and active-set quadratic programming. The public surface is generic over `T` — the caller picks the working type. Tier constraints are documented in [`docs/tier-constraints.md`](tier-constraints.md).

| Family | Generic type | Tier | Accepted `T` |
|--------|--------------|------|---------------|
| Bracketed root finders | `BisectionSolver<T>`, `BrentSolver<T>`, `NewtonRaphsonSolver<T>` | Tier B | `double`, `float`, `Half` |
| Unbracketed root finders | `SecantSolver<T>`, `MullerSolver<T>`, `NewtonRaphsonSolver<T>` | Tier B | `double`, `float`, `Half` |
| Multivariate nonlinear LS | `LevenbergMarquardtSolver<T>` + `LevenbergMarquardtBuffers<T>` | Tier B | `double`, `float` |
| Derivative-free minimization | `NelderMeadMinimizer<T>` + `SimplexResult<T>` | Tier A | `double`, `float`, `Half`, `decimal` |
| Linear LS | `OrdinaryLeastSquares<T>` + `OlsResult<T>` | Tier A + √ | `double`, `float`, `Half`, `decimal` |
| Portfolio QP | `ActiveSetQpSolver<T>` (MinVar + MeanVar) | Tier A + √ | `double`, `float`, `decimal` |

Legacy concrete-typed types (`BrentSolver`, `OrdinaryLeastSquares`, etc.) remain live as forwarding facades over their historical default types. They are deprecated with `BNU0001` and will be removed in v3.0; see the [migration guide](migration-to-generic-math.md).

## Scalar root finders

For <code>f: ℝ → ℝ</code>, solving <code>f(x) = 0</code>.

| Solver | Bracketing? | Convergence | Use when |
|--------|-------------|-------------|----------|
| `BisectionSolver<T>` | required | linear | a guaranteed bound is more important than speed |
| `BrentSolver<T>` | required | interpolation with guaranteed bracket contraction | general-purpose bracketed root (default pick) |
| `NewtonRaphsonSolver<T>` | optional bracket | quadratic with bracket, possibly divergent without | analytic derivative is available or acceptable via finite differences |
| `SecantSolver<T>` | unbracketed | superlinear | derivative is unavailable and a good initial guess exists |
| `MullerSolver<T>` | unbracketed | superlinear | near a simple real root, using three-point quadratic interpolation |

All scalar root finders are Tier B: they call transcendental functions on `T` (the caller's `f` is almost always transcendental, and the solvers themselves use `T.Abs`, comparison, and arithmetic). `T : IFloatingPointIeee754<T>` is the required constraint — `double`, `float`, and `Half` are accepted. A `decimal` attempt fails at compile time with `CS0315`; use the legacy concrete-typed facade (which runs at `T = double` internally). Tier B on `T = decimal` is a future enhancement (requires implementing 28-digit Remez / Chebyshev / CORDIC approximations for `Log`, `Exp`, trig, etc.); no shipped date.

The bracketed Newton solver follows the `rtsafe` stopping rules: a strictly small computed step or representational stagnation can stop with a large residual. Endpoint equality alone does not reject a Newton candidate. Its unbracketed entry point retains a separate residual safeguard. Secant uses the `rtsec` strict computed-step test without requiring a residual sign change. Neither flag alone certifies a root; inspect `FinalResidual` and the problem's local behavior.

Bisection uses the `rtbis` strict halved-step test. Equality does not stop, and successful step-based termination returns the endpoint whose function value is nonpositive. The residual tolerance and exact endpoint acceptance are library extensions. On iteration exhaustion the result still reports the midpoint of the remaining interval. Floating-point stagnation can limit coordinate accuracy even while the algorithmic signed step continues halving.

### Validation

Every root solver is cross-checked against the certified mathematical root on the standard test-function pool (`x³ − x − 2`, `x² − 2`, `log x − 1`, `cos x − x`, `e⁻ˣ − x`) at 1e-6 absolute — see [`tests/Boutquin.Numerics.Tests.Verification/SolverVerificationTests.cs`](../tests/Boutquin.Numerics.Tests.Verification/SolverVerificationTests.cs). Because the test-function roots are mathematical constants (`√2`, `e`, the Dottie number, etc.), the roots themselves are the source of truth, not any particular library's output. `MullerSolver` additionally has its reference roots validated at generation time against `mpmath.findroot(..., solver='muller')` — mpmath is an arbitrary-precision library with a Muller implementation from a different codebase, so a shared algorithmic bug (sign flip in the "choose denominator with larger magnitude" rule, off-by-one in the sign-of-discriminant branch) cannot silently match between the Numerics C# class and the Python reference. `NewtonRaphsonSolver` is additionally verified on both the bracketed and unbracketed entry points; `SecantSolver` via the single-point `Solve(f, initialGuess)` auto-perturb overload.

Additional analytic checks in [`Solvers_ParityTests.cs`](../tests/Boutquin.Numerics.Tests.Unit/GenericParity/Solvers_ParityTests.cs) compare scalar roots with known mathematical values, the LM Rosenbrock solution with `(1, 1)`, and OLS with an exact affine fit. These checks do not require historical result bits or iteration counts.

## Derivative-free minimization

### `NelderMeadMinimizer<T>`

Use Nelder-Mead when the objective is scalar, continuous enough for simplex search, and derivatives
are unavailable or unreliable. It is Tier A (`T : IFloatingPoint<T>`); set `SimplexOptions<T>` when
you need a different iteration budget, tolerances, or initial simplex step.

Tolerances must be finite and non-negative. Custom coefficients require reflection α > 0,
expansion γ > max(1, α), and contraction ρ and shrink σ strictly between 0 and 1.
Equal-valued accepted points follow existing equals; after shrink, the old best retains priority
among tied best vertices, following Lagarias et al. (1998), §2. Convergence means the simplex's
coordinate diameter or objective spread meets its tolerance; it does not guarantee a global minimum.
The initialization and stopping rules differ from the 1965 paper's multi-orientation experiments,
so its evaluation-count tables are not predictions for this API.

```csharp
using Boutquin.Numerics.Solvers;

SimplexResult<double> result = NelderMeadMinimizer<double>.Minimize(
    point =>
    {
        double x = point[0] - 2.0;
        double y = point[1] + 1.0;
        return x * x + y * y;
    },
    initialGuess: [0.0, 0.0]);

double[] argmin = result.Argmin;  // approximately [2, -1]
double minimum = result.MinValue;
```

For nonlinear least-squares residuals, prefer `LevenbergMarquardtSolver<T>`: it exploits the
residual-vector structure and provides convergence diagnostics and parameter covariance.

## Multivariate nonlinear least-squares

For <code>r: ℝⁿ → ℝᵐ</code>, minimizing <code>½ Σᵢ rᵢ(θ)²</code>.

### `LevenbergMarquardtSolver<T>`

LM supports `double` and `float`. Its constructor rejects `Half` with `NotSupportedException` because the positive finite damping thresholds `1e-12` and `1e16` are unavailable. Supply an analytic Jacobian when finite-difference accuracy at `float` is insufficient.

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

var solver = new LevenbergMarquardtSolver<double>(
    maxIterations: 200, functionTolerance: 1e-10, parameterTolerance: 1e-10,
    gradientTolerance: 1e-10, initialDamping: 1e-3);
var result = solver.Solve(
    residuals: Residuals,
    initialGuess: [1.0, 0.1]);

if (result.Converged)
{
    var a = result.Parameters[0];
    var b = result.Parameters[1];
    // Convergence alone does not guarantee an identifiable covariance matrix.
    if (result.ParameterCovariance is { } covariance)
    {
        var sigmaA = Math.Sqrt(covariance[0, 0]);
        var sigmaB = Math.Sqrt(covariance[1, 1]);
    }
}
```

**Working at `T = float` for memory-bound Monte Carlo calibrations:**

```csharp
float[] ResidualsFloat(float[] theta) { /* … */ }

var solverF = new LevenbergMarquardtSolver<float>(
    maxIterations: 200, functionTolerance: 1e-5f, parameterTolerance: 1e-5f,
    gradientTolerance: 1e-5f, initialDamping: 1e-3f);
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

- **Arithmetic range:** the accumulated gradient `Jᵀr` is only a stopping-test quantity (the damped step is solved in range-scaled variables and never needs its absolute value). A gradient whose exact value lies beyond the representable range is treated as `±∞`: it is larger than every finite `gradientTolerance`, so it never reports `GradientToleranceReached`, its sign still decides whether a bound blocks the coordinate, and the solve proceeds to the representable optimum. A nonzero gradient that underflows to zero is likewise never certified by `gradientTolerance = 0`. `InvalidOperationException` is raised only when the problem itself is not representable: a residual or analytic-Jacobian value that is non-finite (including a representable trial point whose residuals overflow), or a NaN gradient sum. A damped step that is itself not representable, or a trial point that it carries beyond the range, is instead rejected and grows the damping like any rejected step. Damping raised that way is sized by the range rather than by the model, so from the first such rejection until an accepted step brings `λ` back to the level at which it occurred, `FunctionToleranceReached` and `ParameterToleranceReached` cannot fire. A problem whose optimum can be reached only through steps the range rejects therefore ends with `Converged = false` (normally `DampingOverflow`) rather than a converged report at a non-stationary point. That guarantee covers range rejection only: when no trial step is rejected for range, the MINPACK tolerance tests apply unchanged, so such a solve can end with `Converged = true` and `FunctionToleranceReached` or `ParameterToleranceReached` at a non-stationary point of a numerically rank-deficient problem whose exact optimum lies beyond the range, because the null direction is below the resolution of the damped step. `Converged = true` from those two stops is therefore not a proof of stationarity; check the gradient at the returned parameters when stationarity matters. Rescale residuals or parameters before retrying in those cases.
- **Half-SSE cost:** `FinalCost = ½ Σᵢ rᵢ²`. The gradient is `Jᵀr` (not `2Jᵀr`), matching the Gauss-Newton Hessian approximation `H ≈ JᵀJ`.
- **Marquardt scaled-diagonal damping:** the step equation is `(JᵀJ + λ·diag(JᵀJ))·δ = −Jᵀr`, invariant under per-parameter rescaling of the residual function. Important when parameters have widely different magnitudes.
- **Nielsen damping update** (Nielsen 1999, eq. 2.5 with β = 2, γ = 3, p = 3; restated in Madsen-Nielsen-Tingleff 2004, §3.2): <code>ν</code> starts at 2; on a successful step (<code>ρ > 0</code>), scale <code>λ</code> by <code>max(1/3, 1 − (2ρ − 1)³)</code> and reset <code>ν ← 2</code>, where <code>ρ</code> is the gain ratio; on a rejected step, <code>λ ← λ·ν; ν ← 2ν</code>. Converges faster on ill-conditioned problems than the fixed <code>λ/ν</code> Marquardt rule.
- **Absolute initial damping:** `λ₀ = initialDamping`, default <code>1e-3</code> following the starting-value recipe in Numerical Recipes §15.5.2. This is not Moré's trust-radius initialization. Per-parameter scale invariance is already delivered by the Marquardt scaled-diagonal form inside the step equation; double-scaling at initialization (the Nielsen <code>τ · max(diag(JᵀJ))</code> variant) was tried and reverted — it produced zero-progress failures on ill-scaled problems like NIST Hahn1 where <code>max(diag(JᵀJ))</code> at the starting point pushed λ above the step resolution of double precision.
- **Relative function tolerance** (MINPACK lmder convention): `FunctionToleranceReached` fires only when BOTH the actual relative cost reduction `|(priorCost − newCost) / priorCost|` AND the quadratic model's predicted relative reduction fall at or below `functionTolerance` on either an accepted or rejected trial. A rejected trial leaves the current parameters unchanged. Checking the actual reduction alone accepts a step whose cost happened to land near its starting magnitude for the wrong reason (e.g. an equal-magnitude, opposite-sign residual) even while the model still predicted a large improvement — a non-stationary point reported as converged. Relative rather than absolute because absolute tolerance fires prematurely on low-cost problems — e.g., NIST Lanczos3's optimal cost is ~1e-8, so an absolute 1e-10 threshold triggers at ~1% relative reduction. Like MINPACK's literal ftol test, the stop also gates on the gain ratio: a third clause requires `½·ratio ≤ 1`, where `ratio` is the actual-to-predicted cost reduction of the trial. Both relative reductions can fall below `functionTolerance` while the quadratic model still under-predicts the true reduction by more than a factor of two (gain ratio above 2); such a trial does not certify convergence and the iteration continues. All four `Solve` paths (array and writer callbacks, with and without a pooled buffer) apply the same three-clause rule.
- **Initial damping and stopping:** `initialDamping` must be positive and below the damping ceiling `1e16`. If it starts above one, function and parameter stops wait until damping has fallen to at most one. This prevents strong initial regularization from making a tiny step certify the starting point. Later model-driven damping uses the usual stopping tests.
- **Iteration and gradient units:** `maxIterations` bounds all trials, including rejected ones; the result counts accepted steps. `gradientTolerance` is absolute in residual-times-Jacobian units. Scaling residuals by `s` requires scaling that tolerance by `s²`.
- **Box bounds** (optional): solve the damped system on free coordinates, assigning zero steps to fixed parameters and active bounds with outward descent directions. Clip trials that first encounter a bound before evaluating their residual and model reduction. Convergence status is evaluated on the free components; if any component is on a bound, `BoundsActive` is set on the result.
- **Deterministic:** no RNG, no parallelism. Same inputs produce bit-identical outputs.

**Jacobian:**

- Supply an analytic Jacobian for best accuracy and speed. Shape is `m × n` with `J[i, j] = ∂rᵢ/∂θⱼ`.
- When omitted, the solver uses central finite differences with adaptive per-component step <code>h = max(1e-8, |θⱼ|·∛ε)</code> for nonzero parameters and <code>h = ∛ε</code> at zero, using the actual represented parameter spacing in the denominator. The cube-root rule balances central-difference truncation and roundoff (Numerical Recipes, 3rd ed., §5.7); the near-zero characteristic scale remains a heuristic. At an exact bound, the feasible one-sided stencil has first-order truncation error and instead uses <code>h = max(1, |θⱼ|)·√ε</code>, reduced to the available interval.

**Termination:**

Exactly one `LmTerminationReason` is reported:

| Reason | Meaning |
|--------|---------|
| `FunctionToleranceReached` | absolute actual relative cost reduction AND the model's nonnegative predicted relative reduction both fell at or below `functionTolerance`, and half the gain ratio is at most one (`½·ratio ≤ 1`), on an accepted or rejected trial; rejected parameters remain unchanged, and neither a clipped step nor one taken under range-limited damping can certify convergence |
| `ParameterToleranceReached` | step norm fell below `parameterTolerance · (‖θ‖ + parameterTolerance)`; neither a clipped step nor one taken under range-limited damping can certify it |
| `GradientToleranceReached` | `‖Jᵀr‖∞ ≤ gradientTolerance` |
| `MaxIterationsReached` | iteration budget exhausted (`Converged = false`) |
| `DampingOverflow` | `λ` saturated the ceiling without finding an acceptable step (`Converged = false`) |

`BoundsActive` is orthogonal — it can be true with any termination reason.

**Parameter covariance:**

When `Converged = true` and the problem is overdetermined (`m > n`), the solver reports `ParameterCovariance = σ̂² · (JᵀJ)⁻¹` where `σ̂² = 2·FinalCost / (m − n) = RSS / (m − n)`. This is the Gauss-Newton asymptotic approximation; use for standard-error bars and correlation structure, not for exact inference.

Covariance is `null` when the normal matrix cannot be factored as positive definite or a covariance entry exceeds the numeric range of `T`. Check for `null` before computing standard errors, even when the solve converged. The variance scale is applied to the triangular inverse before forming covariance, so an unrepresentable unscaled inverse does not by itself make a finite covariance unavailable.

### Hot-path usage — pooled buffers

One-shot calibrations (fitting a single model once) use the `Solve` overload shown above and incur a small per-solve allocation for scratch buffers. Hot-path consumers — Monte Carlo calibration, bootstrap loops, real-time refit engines — should pre-allocate a `LevenbergMarquardtBuffers<T>` pool once and pass it to every solve. The pool eliminates the solver's inner-loop managed-heap allocation. Each solve still allocates its returned `MultivariateSolverResult<T>`, owned arrays, and any one-shot covariance scratch. Residual and analytic Jacobian callbacks must also be non-allocating to preserve the loop guarantee.

```csharp
using Boutquin.Numerics.Solvers;

// Construct once, outside the hot loop.
var buffers = new LevenbergMarquardtBuffers<double>(parameterCount: 6, residualCount: 24);
var solver = new LevenbergMarquardtSolver<double>(
    maxIterations: 200, functionTolerance: 1e-10, parameterTolerance: 1e-10,
    gradientTolerance: 1e-10, initialDamping: 1e-3);

foreach (var resample in bootstrap.Resamples)
{
    var result = solver.Solve(resample.Residuals, initialGuess, buffers);
    // ... record result.Parameters, result.FinalCost, etc.
}
```

**Rules:**

- **One pool per thread.** `LevenbergMarquardtBuffers<T>` is not thread-safe. Concurrent solves sharing a pool produce undefined results. Each worker thread should own its own pool.
- **Resize with `Reset` when dimensions change.** Fitting models of varying sizes against the same pool is supported via `buffers.Reset(n, m)`. Work-buffer capacity is grow-only. Changing `n` allocates a new exact-length parameter array for `Func<T[], ...>` callbacks, even when `n` and `m` remain within the pool's capacity; changing only `m` within capacity reuses the existing arrays.
- **Match the dimensions.** If `buffers.ParameterCount` does not match the initial-guess length, or if the residual function returns a vector whose length differs from `buffers.ResidualCount`, `Solve` throws `ArgumentException` with a message telling you which `Reset(n, m)` call will fix it.
- **Pool-free overload is bit-identical.** Both `Solve` overloads produce identical arithmetic — results match down to floating-point rounding. Choose the pooled overload only when allocation pressure matters; the one-shot overload stays simpler for code that runs a handful of times.
- **Zero-allocation requires a zero-allocation residual callback.** The `Func<T[], T[]>` residual contract is unchanged; consumers targeting the strictest zero-alloc bar should return a cached `T[]` from their residual function rather than a fresh array per call. The solver copies each returned vector into pool storage before any subsequent iteration, so reusing the same array across calls is safe.

### Zero-allocation residuals — `LmResidualWriter<T>`

For callers that need true zero managed-heap allocation during the solve loop, two `Solve` overloads accept buffer-fill delegates instead of `Func<T[], T[]>`:

```csharp
// LmResidualWriter<T>: fill residuals in-place from the parameter span.
delegate void LmResidualWriter<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
    where T : IFloatingPointIeee754<T>;

// LmJacobianWriter<T>: fill flat row-major Jacobian in-place.
// jacobian[i * n + j] = ∂rᵢ/∂θⱼ
delegate void LmJacobianWriter<T>(ReadOnlySpan<T> parameters, Span<T> jacobian)
    where T : IFloatingPointIeee754<T>;
```

**Pool-free writer overload** (residual count known up-front):

```csharp
var solver = new LevenbergMarquardtSolver<double>(
    maxIterations: 200, functionTolerance: 1e-10, parameterTolerance: 1e-10,
    gradientTolerance: 1e-10, initialDamping: 1e-3);

// Supply residualCount = m so the pool can be sized before the first evaluation.
var result = solver.Solve(
    residuals: (theta, r) =>
    {
        r[0] = ys[0] - theta[0] * Math.Exp(theta[1] * xs[0]);
        r[1] = ys[1] - theta[0] * Math.Exp(theta[1] * xs[1]);
        // … fill all m elements of r
    },
    initialGuess: [1.0, 0.1],
    residualCount: m);
```

**Pooled writer overload** (true zero-allocation hot loop):

```csharp
var buffers = new LevenbergMarquardtBuffers<double>(parameterCount: 2, residualCount: m);
var solver  = new LevenbergMarquardtSolver<double>(
    maxIterations: 200, functionTolerance: 1e-10, parameterTolerance: 1e-10,
    gradientTolerance: 1e-10, initialDamping: 1e-3);

foreach (var resample in bootstrap.Resamples)
{
    var result = solver.Solve(
        residuals: (theta, r) => { /* fill r in-place */ },
        initialGuess: initialGuess,
        buffers: buffers,
        jacobian: (theta, j) =>
        {
            // Flat row-major Jacobian: j[i * 2 + 0] = ∂rᵢ/∂a, j[i * 2 + 1] = ∂rᵢ/∂b
        });
}
```

The writer overloads produce bit-identical results to the `Func<T[], T[]>` overloads. Choose the writer form only when profiling shows allocation pressure in the solve loop.

**Allocation profile (n = 6, m = 24, `T = double`):**

| Overload | Per-solve allocation |
|----------|----------------------|
| Pool-free `Solve` | Result record + owned arrays + private pool construction |
| Pooled `Solve` | Result record + owned arrays + one-shot covariance scratch at convergence |
| Inner `EvaluateInto + TrySolve` (direct) | 0 bytes across 1000 invocations after warmup |

The [complete-solve slope tests](../tests/Boutquin.Numerics.Tests.Unit/Solvers/Arena20260928LmAllocationSlopeTests.cs) compare two versus seven independently counted rejected iterations in both pooled callback forms, requiring exactly zero additional allocation. [LevenbergMarquardtAllocationTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/LevenbergMarquardtAllocationTests.cs) separately checks the whole-solve budget and the allocation-free inner kernel.

### Validation

Validated against all 27 NIST Statistical Reference Dataset nonlinear-regression benchmark problems — the full suite across every difficulty tier (lower: `Misra1a`, `Chwirut1/2`, `Lanczos3`, `Gauss1/2`, `DanWood`, `Misra1b`; average: `Kirby2`, `Hahn1`, `Nelson`, `MGH17`, `Lanczos1/2`, `Gauss3`, `Misra1c/d`, `Roszman1`, `ENSO`; higher: `MGH09`, `Thurber`, `BoxBOD`, `Rat42`, `MGH10`, `Eckerle4`, `Rat43`, `Bennett5`) — converging from NIST "Start 2" with finite-difference Jacobian under per-tier tolerances. Lanczos1 and Lanczos2 have certified SSE below double precision's practical floor (~10⁻²⁰) and are asserted against that floor rather than a relative bar; parameter agreement remains the authoritative convergence signal for those two. See [`tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs`](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) for the test harness and `NistStRDData.cs` for the transcribed problem set. The NIST harness runs against `LevenbergMarquardtSolver<double>`; a separate Rosenbrock case compares the legacy facade in [`Solvers_ParityTests.cs`](../tests/Boutquin.Numerics.Tests.Unit/GenericParity/Solvers_ParityTests.cs).

Additionally cross-checked against **`scipy.optimize.least_squares(method='lm')`** across four smooth regimes disjoint from the NIST Nonlinear suite (exponential-decay-with-noise, sum-of-two-sinusoids, Gompertz growth curve, logistic) at 1e-5 relative tolerance on both `Parameters` and `FinalCost`. scipy's `lm` method delegates to MINPACK lmder — same algorithmic family as `LevenbergMarquardtSolver<T>`, so precision parity at 5+ digits is achievable on smooth problems. See [`tests/Boutquin.Numerics.Tests.Verification/LevenbergMarquardtVerificationTests.cs`](../tests/Boutquin.Numerics.Tests.Verification/LevenbergMarquardtVerificationTests.cs).

### References

- Marquardt, D. W. (1963). *An algorithm for least-squares estimation of nonlinear parameters*. SIAM J. Appl. Math. 11(2), 431-441.
- Moré, J. J. (1978). *The Levenberg-Marquardt algorithm: implementation and theory*. In: *Numerical Analysis*, Lecture Notes in Mathematics 630, 105-116. The MINPACK lmder implementation referenced for the relative function-tolerance convention.
- Nielsen, H. B. (1999). *Damping Parameter in Marquardt's Method*. Technical Report IMM-REP-1999-05, Technical University of Denmark. The source of the damping-update rule: eq. 2.5, with β = 2, γ = 3, p = 3 recommended on p. 14.
- Madsen, Nielsen, Tingleff (2004). *Methods for Non-Linear Least Squares Problems* (IMM Technical Report). The Nielsen damping-update rule used here is §3.2.
- NIST/ITL Statistical Reference Datasets — Nonlinear Regression. `https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml`.

## Ordinary least squares

For `y = X·β + ε`, minimizing `‖y − X·β‖₂²`.

### `OrdinaryLeastSquares<T>`

Standard OLS estimator backed by Householder QR with column pivoting and (at `T = double`) mixed-precision iterative refinement. Tier A + √ at `T : IFloatingPoint<T>` with `Sqrt` via `NumericPrecision<T>`. Accepts `double`, `float`, `Half`, and `decimal` — the caller's type choice is the precision dial.

At `T = double`, matches the NIST StRD certified coefficients to 1e-12 relative and standard errors to 1e-11 on every linear-regression problem except `Filip` (1e-7 and 1e-6), including `Longley` and the noisy Wampler family. Wampler4 and Wampler5 share a degree-five design with condition number about 6.4·10⁶; their different response noise changes coefficient sensitivity without changing that design condition number.

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

var fit = OrdinaryLeastSquares<double>.Fit(longleyX, longleyY);
// fit.Coefficients[0] ≈ −3,482,258.63459582 (intercept)
// fit.StandardErrors[0] ≈ 890,420.383607373
// fit.ResidualStandardDeviation ≈ 304.854073561965
// fit.RSquared ≈ 0.995479004577296
```

**Worked example — `T = decimal`, full 28-digit QR (NIST StRD `Filip`, cond ≈ 10¹⁰):**

```csharp
// Instantiating OrdinaryLeastSquares<T> at T = decimal runs Householder QR with column
// pivoting, Q^T·y, upper-triangular solve, and (XᵀX)⁻¹ end-to-end in 28-digit decimal,
// with normalised Householder reflectors to reduce loss from column scaling.
decimal[,] filipX = /* 82 observations of a degree-10 polynomial; cond ≈ 10¹⁰ */;
decimal[]  filipY = /* … */;

var fit = OrdinaryLeastSquares<decimal>.Fit(filipX, filipY);
// Coefficients agree with NIST certified values to 1e-6 relative, limited by the
// 15-digit tabulated inputs — a normal-equation OLS loses every digit at cond ≈ 10¹⁰.
// Every other NIST problem holds 1e-13. See NistLinearRegressionDecimalTests.cs.
```

**Legacy facade** — `OrdinaryLeastSquares.Fit(double[,], double[], bool)` remains a forwarding facade over `OrdinaryLeastSquares<double>` through v2, but is deprecated with `BNU0001` and will be removed in v3.0. See the [migration guide](migration-to-generic-math.md).

**Key design choices:**

- **Householder QR + column pivoting**, not normal equations. `β̂ = (XᵀX)⁻¹·Xᵀy` squares the condition number by forming `XᵀX`; QR preserves the conditioning of `X` end-to-end. Column pivoting (LAPACK `DGEQP3`-style) additionally handles polynomial designs where later columns have norms many orders of magnitude larger than earlier ones (`Filip` column norms span `10¹⁰–10⁰` across `[x, x², …, x¹⁰]`).
- **Caller-chosen precision.** The generic type parameter `T` is the precision dial: `T = double` runs the QR in `double` and applies `decimal`-residual iterative refinement; `T = decimal` runs the full factorisation in 28-digit `decimal` with every Householder reflector normalised, which holds the NIST certified coefficients to 1e-13 on every problem except `Filip` (1e-6, limited by its 15-digit tabulated inputs). `T = float` is available for memory-bound consumers willing to give up precision for throughput.
- **Augmented-system mixed-precision iterative refinement (at `T = double`).** The QR solution is refined with Björck's augmented-system refinement (Björck 1996, Algorithm 2.9.1; Golub & Van Loan §5.3.8), which corrects `β̂` and the residual `r` together from `f = y − r − X·β̂` and `g = −Xᵀ·r`. Both blocks are accumulated in `decimal` after an exact power-of-two rescaling of each column, of `y` and, for `g`, of `r` itself, and scaled `double` inputs are converted through their binary significands. Extreme dynamic ranges can still lose small terms or trigger the documented fallback. Refining `β̂` alone helps only nearly consistent systems; the augmented form makes `κ(X)` rather than `κ(X)²·‖r‖` govern the attainable accuracy. `decimal`'s 96-bit significand does not meet the doubled residual precision (106 bits for `double`) sufficient under Higham (2002) §20.5's remaining hypotheses. This is not a necessary accuracy threshold, and missing it does not by itself establish any particular error; attainable accuracy remains problem- and arithmetic-dependent. Residuals, the residual sum of squares and every statistic derived from it are accumulated in `double` from the refined residuals, so very small squares can underflow and extreme derived statistics can exceed the working range.
- **Intercept toggle.** `includeIntercept: true` (default) prepends a column of ones to `X` and returns `Coefficients` of length `p + 1` with the intercept at index 0. `includeIntercept: false` passes `X` through unchanged and returns `p` coefficients — the correct choice for NIST's `NoInt1` and `NoInt2` problems or any physically zero-intercept model.
- **R² convention.** Centred total sum of squares `Σ(yᵢ − ȳ)²` when `includeIntercept = true`, uncentred `Σ yᵢ²` otherwise. Matches NIST StRD's convention and what R's `lm()` and Python's `statsmodels.OLS` publish.
- **Rank-deficient detection.** QR returns `false` when a pivot column's trailing norm (its part orthogonal to the columns already factored) is at most `max(m, p)·ε` times that column's own norm, and `Fit` throws `InvalidOperationException` rather than returning a degenerate `OlsResult<T>`. Collinear predictors must be resolved upstream (ridge, pseudo-inverse, or column elimination).

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

At `T = double`, the accuracy bars in [`NistLinearRegressionTests.cs`](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionTests.cs) are 1e-12 on coefficients and 1e-11 on standard errors for every problem except `Filip` (cond ≈ 10¹⁰; 1e-7 and 1e-6), including Longley and the Wampler family, whose certified Wampler4 and Wampler5 coefficients the augmented-system refinement recovers exactly. NIST's [`c-Wampler5.shtml`](https://www.itl.nist.gov/div898/strd/lls/data/LINKS/c-Wampler5.shtml) methodology note explains why double-precision implementations need extended-precision intermediates on these problems.

At `T = decimal`, the same 11 problems run under the tolerances in [`NistLinearRegressionDecimalTests.cs`](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionDecimalTests.cs): 1e-13 on the coefficients of every problem except `Filip` (1e-6), whose design is the double-rounded powers of a degree-10 polynomial entered at 15 significant digits. Every Householder reflector is normalised at `T = decimal`, because `decimal`'s fixed 28-place scale resolves an absolute step: an unnormalised reflector on a column far from unit magnitude (Pontius's `x²` column, or any data scaled by 1e-10) loses most of its significant digits.

At `T = double`, additionally cross-checked against **`statsmodels.api.OLS`** across three regimes — well-conditioned random (cond ≈ 10²), polynomial degree 5 (cond ≈ 10⁶), and no-intercept — at 1e-10 relative on coefficients and 1e-8 relative on derived quantities (`StandardErrors`, `ResidualSumOfSquares`, `ResidualStandardDeviation`, `RSquared`). See [`tests/Boutquin.Numerics.Tests.Verification/OrdinaryLeastSquaresVerificationTests.cs`](../tests/Boutquin.Numerics.Tests.Verification/OrdinaryLeastSquaresVerificationTests.cs).

### References

- Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed. Ch 5 (orthogonalization and least squares).
- Björck, Å. (1996). *Numerical Methods for Least Squares Problems*. SIAM.
- Longley, J. W. (1967). *An appraisal of least squares programs for the electronic computer from the point of view of the user*. *JASA* 62, 819–841.
- Wampler, R. H. (1970). *A report on the accuracy of some widely used least squares computer programs*. *JASA* 65, 549–565.
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. §20 (least squares).
- NIST/ITL Statistical Reference Datasets — Linear Regression. `https://www.itl.nist.gov/div898/strd/lls/lls.shtml`. The methodology page at [`c-Wampler5.shtml`](https://www.itl.nist.gov/div898/strd/lls/data/LINKS/c-Wampler5.shtml) documents NIST's use of Bailey's 500-digit MPFUN for certification. `OrdinaryLeastSquares<decimal>` runs in 28-digit decimal (far from NIST's 500-digit reference arithmetic, and without the residual-refinement layer the double path uses), so the per-problem NIST-certified values remain the authoritative bar — this library's test tolerances are set to what the respective working-precision regime achieves, not to the full NIST reference precision.
- Bailey, D. H. *MPFUN / QD multi-precision packages*. `https://www.netlib.org/mpfun/` — background on multiprecision arithmetic; this library uses the .NET `decimal` implementation for its extended residual calculations.

## Portfolio-optimization QP

For `w ∈ ℝⁿ`, two standard long-only portfolio problems under a sum-to-one constraint and per-asset box bounds.

### `ActiveSetQpSolver<T>`

Primal-feasible active-set solver for the two standard long-only portfolio QPs, with a certified result. Tier A + √ at `T : IFloatingPoint<T>`; the working-precision steps use `CholeskyDecomposition<T>`. Accepts `double`, `float`, and `decimal`: `double` and `decimal` are guaranteed, `float` is best-effort (it compiles and runs, and a departure seen only at `float` is not a defect) — `decimal` is the precision default for the legacy non-generic `ActiveSetQpSolver` facade. Both solve entry points reject `Half` with `NotSupportedException` because the positive finite thresholds `1e-20` and `1e-10` underflow in that type. Results are deterministic: identical inputs give bit-identical outputs.

**Minimum-variance portfolio** — `min w′Σw  s.t.  1′w = 1, lb ≤ w ≤ ub`:

```csharp
using Boutquin.Numerics.Solvers;

decimal[,] covariance = LoadAssetCovariance();  // N×N symmetric positive-definite

decimal[] weights = ActiveSetQpSolver.SolveMinVariance(
    covariance,
    minWeight: 0.0m,
    maxWeight: 0.4m);
// weights.Sum() ≈ 1, each 0 ≤ w_i ≤ 0.4 for N ≥ 3 assets (N · 0.4 ≥ 1).
// With N = 2 the budget cannot be met under a 0.4 cap, so the cap is raised to 1/N = 0.5.
```

**Infeasible bounds.** The bounds can satisfy the budget `1′w = 1` only when `N · minWeight ≤ 1 ≤ N · maxWeight`. `SolveMinVariance`, and `SolveMeanVariance` with `riskAversion > 0`, relax only the offending bound to the equal-weight point `1/N` and solve with the other bound unchanged; feasible bounds are never altered. `SolveMeanVariance` with `riskAversion = 0` rejects infeasible bounds with `ArgumentException` instead of relaxing them.

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

// T = decimal for portfolio-grade precision: every returned weight is computed and certified in `decimal`.
// Internal pre-checks run in double and decide only what an exact test would decide (see below).
decimal[] wDec = ActiveSetQpSolver<decimal>.SolveMinVariance(
    covarianceM, minWeight: 0.0m, maxWeight: 0.4m);
```

**Key design choices:**

- **Primal-feasible active set.** The problem is equilibrated once (the diagonal scaling `D = diag(√Σᵢᵢ)`, rounded to a radix power, so the working matrix has a unit-scale diagonal; for `decimal` the shifts are limited to exact radix-10 ones that keep every scaled entry within 28 decimal places, so a covariance with 27 or 28 significant digits is not always brought to a unit-scale diagonal) and a feasible point is built first: every weight at its lower bound, the remaining budget distributed up to the upper bounds in index order. Each iteration then solves the equality-constrained subproblem of the free variables (those not held at a bound) in a **budget null-space basis**: the budget row is eliminated with a reference free variable and the reduced Hessian is factored by Cholesky, so the budget-orthogonal part of the step is computed directly and never as a difference of two large solves. A step-length rule adds at most the blocking bound, so every iterate stays feasible. When the step is zero, the multiplier of every held variable is compared with its own row's scale (never with a maximum over all assets) and the most negative one is released, ties going to the smallest index. A release must lower the exact objective of the subspace minimiser, otherwise it is undone and the variable is excluded until the objective next decreases, which is what terminates degenerate problems (identical assets, repeated blocks, tied means, coinciding bounds). The method follows Nocedal–Wright (2006) Algorithm 16.3 specialised to one budget row and a box.
- **Exact route.** Where rounded arithmetic cannot decide a step (a working set whose reduced Hessian is not positive definite at the working precision, a refinement that does not converge, or a result the certificate rejects while no release remains), the solver continues from the same working set on exact rational iterates, under the same step-length, exclusion and tie-break rules, and rounds the result once to `T`. The rounded route allows `8N + 8` working-set changes; the exact route has a fresh budget of `12N + 12`. These are defensive limits, not convergence theorems; exhausting the exact route's budget is a refusal. Unchanged leading Cholesky columns and fraction-free factors are reused across working sets. An exact factor is extended when a variable is released, retains its unchanged prefix when a bound is added, and is rebuilt when the budget reference changes. An affine right-hand-side update is reused only after its relation to the newly released column has been checked exactly.
- **Certified result.** Before it is returned, the result is checked against an independent KKT certificate (`1′w = 1`, the bounds, stationarity of the free variables, and the multiplier signs of the held ones), each component measured against its own row's scale and the working type's resolution. A result is never returned unchecked, and a failed result is never replaced by a different candidate chosen with a looser measure. A problem whose determining block is numerically singular at the working precision (`48·N·u·κ ≥ 1` for the exact 1-norm condition number `κ` of the equilibrated block, `u` the unit roundoff) is refused instead of being answered with a vector that no residual could verify. For `decimal` the equilibrated block is the one produced by the limited shifts described above, so a covariance with 27 or 28 significant digits can be refused inside this margin of its unit-scale form, by up to the ratio of its diagonal scales.
- **Double-side pre-checks.** Internal checks run in `double` before an exact test and decide only what that test would decide, never a value that reaches a weight. The positive-definiteness test of the equilibrated covariance has two certified stages. The first proves the matrix positive definite from a `double` Cholesky factorisation with an a-priori bound on its rounding error. Where it cannot (an ill-conditioned or nearly singular covariance, and every covariance that must be refused), the second factors the matrix in double-double arithmetic and bounds the residual of that factor rigorously: it either proves the matrix positive definite, or proves a leading block positive definite and evaluates, in exact integer arithmetic, a vector `x` with `x′Σx ≤ 0` on the next order, which fixes the order of the first leading principal minor that is not positive, the order the refusal names. The conditioning check runs on the free block of every new working set and on the block that decides the result; it can prove a block of order m strictly inside the margin using an upper bound on the block's condition number from a `double` approximate inverse, or a proved lower bound `s` on its smallest eigenvalue, which gives `κ₁ ≤ √m·‖A‖₁/s`. The eigenvalue bound comes from a double-double Cholesky factor with a rigorous residual bound, of the whole equilibrated covariance once per solve (by Cauchy's interlacing theorem it bounds every principal block) or of the block itself. For `double`, a column `x` of the approximate inverse can also prove a refusal: `‖A⁻¹‖₁ ≥ ‖x‖₁/‖Ax‖₁`, with all three norms and the margin comparison evaluated exactly on the original block. For `T = decimal` each bound converts the block (or the whole equilibrated covariance) to `double` or double-double, with an error that the bound includes. When no bound settles the outcome, the exact test decides, so the accepted and refused problems, and the exception messages, are those of the exact tests. Every returned weight is computed in `T` (`decimal` for the portfolio-grade path); the `double` side only gates the cost: the certified checks are O(n³), while the exact tests grow about as n⁵ and m⁴ on multi-digit integers. The exact conditioning test still runs for a block that is near or past the margin, with `48·N·u·κ₁` above about `1/(2√m)` (for `decimal` at N = m = 100, a condition number above about 10²³), when no exact norm witness proves the refusal; it can cost seconds for a block of order 100. The exact positive-definiteness test still runs for a covariance with a leading block whose smallest eigenvalue is below the double-double shift, at most `32·(n + 3)·n·2⁻¹⁰⁶` of the largest diagonal entry (about 4·10⁻²⁷ at n = 100, 4·10⁻²⁵ at n = 1000; for `decimal` this is past the conditioning guard's margin for every order below about 24,000), and for one whose first non-positive leading minor is exactly zero without a short exact dependence among the assets (a repeated asset, or an asset that is a combination of others with coefficients on a short binary or decimal grid, is decided without it); it costs seconds at n = 100.
- **Mean-variance.** `riskAversion > 0` runs the same machinery on `min (λ/2)·w′Σw − μ′w`: the gradient `λΣw − μ` is formed exactly, and the step, release rule, certificate and conditioning guard are those of the minimum-variance solve.
- **Pure-LP branch.** `riskAversion = 0` in `SolveMeanVariance` short-circuits to a greedy allocation: initialize all weights at the lower bound, then distribute the remaining budget in descending expected-return order up to each upper bound. This returns the long-only maximum-expected-return portfolio — useful as a sanity check and as a limit-case benchmark. The branch reads the covariance only to validate it (square, symmetric, finite), so positive definiteness is required only for `riskAversion > 0`. The fill runs in `T` while the budget `(n + 2) · |1 − n · minWeight|` is at most 1024, and in exact arithmetic above it, rounding the one partially filled weight once; a budget of exactly `1024 / (n + 2)` is still filled in `T`. The formula alone selects the exact fill: it holds for the large short bounds whose budget rounding would absorb, and equally for ordinary boxes (`n = 100`, `minWeight = −0.1`), for every box with `minWeight = 0` and 1023 or more assets, and for positive-floor boxes whose budget is large enough.
- **Deterministic.** Same inputs produce bit-identical outputs. Large exact factors may update independent integer rows concurrently, using at most four workers; each column completes before the next starts, so scheduling changes no arithmetic result. The solver uses no RNG.

**Exceptions:**

- `ArgumentNullException` — `covariance` is `null`.
- `ArgumentException` — the covariance matrix has zero dimension, is not square, is not symmetric, or has a non-finite entry; `means.Length` does not match the covariance dimension; a mean is non-finite; `minWeight` is greater than `maxWeight`; or, for `SolveMeanVariance` with `riskAversion = 0` only, the bounds cannot satisfy `1′w = 1` (`N · minWeight > 1` or `N · maxWeight < 1`).
- `ArgumentOutOfRangeException` — `riskAversion` is negative or non-finite, or `minWeight` or `maxWeight` is non-finite.
- `InvalidOperationException` — for `riskAversion > 0` (and always for `SolveMinVariance`), the covariance is not positive definite; or a result could not be certified or the solve exhausted a documented budget. The message names the cause: a failed certificate component (`budget`, `bound`, `stationarity` or `multiplier sign`, for a result that fails the certificate while no release remains); a determining block that is numerically singular at the working precision; a working-set block that is exactly singular in the exact route (a positive-definite covariance rules this out); or the exact route's `12N + 12` working-set budget was exhausted. A product of the result's gradient that is not representable in `T` is reported the same way, which can only happen for inputs outside the operating domain; for `T = decimal`, an `OverflowException` raised inside the solve is reported as this exception. For `riskAversion = 0`, an allocation whose budget or capacity is not finite is reported the same way.
- `NotSupportedException` — `T` cannot represent the positive finite thresholds (`Half`).

### Validation

Cross-checked against **`scipy.optimize.minimize(method='SLSQP')`** across 5 portfolio cases at 5e-6 per-weight in [`QpSolverVerificationTests.cs`](../tests/Boutquin.Numerics.Tests.Verification/QpSolverVerificationTests.cs) / [`generate_qp_solver_vectors.py`](../tests/Verification/generate_qp_solver_vectors.py). Cases include:

- A boxed 3-asset MinVar (a held variable contributes to the free-block gradient).
- Unconstrained 3-asset and boxed 5-asset MinVar.
- Unconstrained 3-asset MeanVar at `λ = 2` and boxed 5-asset MeanVar at `λ = 3`.

The generator requires a successful SLSQP result; an iteration-limited result is refused. The pure-LP branch is covered by unit tests, not by these five reference vectors.

The unit tests in [`ActiveSetQpSolverTests.cs`](../tests/Boutquin.Numerics.Tests.Unit/Solvers/ActiveSetQpSolverTests.cs) additionally cover the uniform-weight output on a scaled-identity covariance, the sum-to-one invariant, termination on single-asset inputs, degenerate-covariance rejection, and the `riskAversion = 0` pure-LP branch.

### References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Ch 16 (quadratic programming); §16.5 (Algorithm 16.3) is the active-set method specialised here to one budget row and a box.
- Markowitz, H. (1952). *Portfolio Selection*. *The Journal of Finance* 7(1), 77–91 — the mean-variance problem setup.
- Sharpe, W. F. (1963). *A Simplified Model for Portfolio Analysis*. *Management Science* 9(2), 277–293 — single-index / factor-model background relevant when `Σ` is a Ledoit-Wolf-shrunk or POET-denoised covariance estimate.
