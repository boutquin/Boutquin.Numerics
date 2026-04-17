# Tier constraints

Boutquin.Numerics is generic over `T`. Every algorithm that touches a floating-point value documents the minimum generic-math interface stack it needs, and asks for no more. The constraint class — what this document calls a **tier** — is the contract between the library and the caller: it names which numeric types are legal, which BCL interfaces must be available on `T`, and which operations the algorithm's critical path relies on.

The tier label appears as the first `<para>` of every generic public type's XML `<remarks>` block so it shows up in IntelliSense. This document is the canonical reference for what each tier means.

## The four tiers

### Tier A — Arithmetic-only

```csharp
where T : IFloatingPoint<T>
```

The algorithm uses only `+`, `−`, `*`, `/`, comparison, `Abs`, rounding / ceiling / floor / truncation, and the BCL constants `T.Zero` / `T.One`. Works for `double`, `float`, `Half`, `decimal`, and any future `T` that implements `IFloatingPoint<T>`.

Representative files: [`Statistics/SampleCovarianceEstimator<T>.cs`](../src/Boutquin.Numerics/Statistics/SampleCovarianceEstimator.cs), [`Statistics/WelfordMoments<T>.cs`](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs), [`Statistics/PearsonCorrelation<T>.cs`](../src/Boutquin.Numerics/Statistics/PearsonCorrelation.cs), [`Statistics/ReturnsMatrix<T>.cs`](../src/Boutquin.Numerics/Statistics/ReturnsMatrix.cs), [`Interpolation/LinearInterpolator<T>.cs`](../src/Boutquin.Numerics/Interpolation/LinearInterpolator.cs), [`Interpolation/MonotoneConvexInterpolator<T>.cs`](../src/Boutquin.Numerics/Interpolation/MonotoneConvexInterpolator.cs).

### Tier A + √ — Arithmetic plus square root

```csharp
where T : IFloatingPoint<T>    // Sqrt via NumericPrecision<T>
```

Tier A plus a single `Sqrt` call. Works natively for IEEE types (`double`, `float`, `Half`) and for `decimal` via the internal `NumericPrecision<T>` dispatcher at [`src/Boutquin.Numerics/Internal/NumericPrecision.cs`](../src/Boutquin.Numerics/Internal/NumericPrecision.cs), which delegates to `Math.Sqrt` / `MathF.Sqrt` / `Half.Sqrt` / a Newton-Raphson `decimal` implementation based on the `T` chosen at instantiation.

Representative files: [`LinearAlgebra/CholeskyDecomposition<T>.cs`](../src/Boutquin.Numerics/LinearAlgebra/CholeskyDecomposition.cs), [`LinearAlgebra/JacobiEigenDecomposition<T>.cs`](../src/Boutquin.Numerics/LinearAlgebra/JacobiEigenDecomposition.cs), [`LinearAlgebra/PrincipalComponentAnalysis<T>.cs`](../src/Boutquin.Numerics/LinearAlgebra/PrincipalComponentAnalysis.cs), [`LinearAlgebra/NearestPsdProjection<T>.cs`](../src/Boutquin.Numerics/LinearAlgebra/NearestPsdProjection.cs), [`Solvers/OrdinaryLeastSquares<T>.cs`](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs), [`Solvers/ActiveSetQpSolver<T>.cs`](../src/Boutquin.Numerics/Solvers/ActiveSetQpSolver.cs), [`Interpolation/MonotoneCubicInterpolator<T>.cs`](../src/Boutquin.Numerics/Interpolation/MonotoneCubicInterpolator.cs), [`Statistics/SampleSkewness<T>.cs`](../src/Boutquin.Numerics/Statistics/SampleSkewness.cs), [`Statistics/SampleExcessKurtosis<T>.cs`](../src/Boutquin.Numerics/Statistics/SampleExcessKurtosis.cs).

### Tier B — Transcendentals

```csharp
where T : IFloatingPointIeee754<T>
```

`IFloatingPointIeee754<T>` carries `IRootFunctions<T>`, `IExponentialFunctions<T>`, `ILogarithmicFunctions<T>`, `ITrigonometricFunctions<T>`, `IHyperbolicFunctions<T>`, and `IPowerFunctions<T>`. Tier B algorithms call `Log`, `Exp`, `Sin`, `Cos`, `Atan`, `Tanh`, `Pow`, or any non-algebraic function on `T`.

Works for `double`, `float`, `Half`. Does **not** work for `decimal` — a `BrentSolver<decimal>` attempt fails at compile time with `CS0315`, the correct discoverable diagnostic. If you need `decimal` arithmetic underneath a Tier B algorithm today, use the legacy concrete-typed facade (which runs at `T = double` internally) and accept that the transcendental body of the computation runs in double precision.

Representative files: [`Solvers/BrentSolver<T>.cs`](../src/Boutquin.Numerics/Solvers/BrentSolver.cs), [`Solvers/LevenbergMarquardtSolver<T>.cs`](../src/Boutquin.Numerics/Solvers/LevenbergMarquardtSolver.cs), [`Statistics/FisherZTransform<T>.cs`](../src/Boutquin.Numerics/Statistics/FisherZTransform.cs), [`Statistics/DistanceCorrelation<T>.cs`](../src/Boutquin.Numerics/Statistics/DistanceCorrelation.cs), [`Statistics/GeneralizationScore<T>.cs`](../src/Boutquin.Numerics/Statistics/GeneralizationScore.cs), [`Statistics/MinimumTrackRecordLength<T>.cs`](../src/Boutquin.Numerics/Statistics/MinimumTrackRecordLength.cs).

### Tier C — Polynomial-approximation-bound

```csharp
where T : IFloatingPoint<T>    // T is cosmetic at the public surface
```

Algorithms whose internal precision ceiling is a published polynomial-approximation or bit-pattern table calibrated only for `double`-precision arithmetic: Acklam's inverse-CDF coefficients, the `erfc` continued-fraction coefficients used by `CumulativeNormal`, Sobol direction numbers published as `uint32` bit patterns designed for the `double` lattice `[0, 1)`, and the Halton scrambling constants.

These accept a generic `T` at the public surface so consumers can compose them into generic pipelines without an explicit type-swap; internally they cast `T` to `double`, run the approximation, and cast back. The cast loses at most one ULP of `T` and is below the approximation's published precision ceiling — this is the **only** place in the library where a cast is permitted inside an inner-loop-capable path, and it is permitted because the polynomial-approximation table is itself not more precise than `double` can represent.

Representative files: [`Distributions/NormalDistribution<T>.cs`](../src/Boutquin.Numerics/Distributions/NormalDistribution.cs), [`Distributions/CumulativeNormal<T>.cs`](../src/Boutquin.Numerics/Distributions/CumulativeNormal.cs), [`Distributions/InverseNormal<T>.cs`](../src/Boutquin.Numerics/Distributions/InverseNormal.cs), [`MonteCarlo/SobolSequence<T>.cs`](../src/Boutquin.Numerics/MonteCarlo/SobolSequence.cs), [`MonteCarlo/HaltonSequence<T>.cs`](../src/Boutquin.Numerics/MonteCarlo/HaltonSequence.cs), [`Random/Pcg64RandomSource<T>.cs`](../src/Boutquin.Numerics/Random/Pcg64RandomSource.cs).

### Tier A with transcendental tail (sub-rule of Tier A)

Not a separate tier. The algorithm body is Tier A. The very final step reduces to a scalar quantity (a single `T`) and applies `Log` / `Sqrt` / inverse-normal once. When `T` is IEEE, the tail is native `T.Log` / `T.Sqrt`. When `T = decimal`, the tail casts to `double`, applies the BCL function, and casts back. The cast applies to a scalar, not a vector or matrix, and no inner loop runs on the `double` side.

Representative files: [`Statistics/NeweyWestVariance<T>.cs`](../src/Boutquin.Numerics/Statistics/NeweyWestVariance.cs), [`Statistics/DeflatedSharpeRatio<T>.cs`](../src/Boutquin.Numerics/Statistics/DeflatedSharpeRatio.cs), [`Statistics/HaircutSharpe<T>.cs`](../src/Boutquin.Numerics/Statistics/HaircutSharpe.cs), [`Statistics/ProbabilityOfBacktestOverfitting<T>.cs`](../src/Boutquin.Numerics/Statistics/ProbabilityOfBacktestOverfitting.cs).

## Tier assignments by folder

| Folder | Tier | Notes |
|--------|------|-------|
| `Solvers/` root finders (`BrentSolver<T>`, `BisectionSolver<T>`, `NewtonRaphsonSolver<T>`, `MullerSolver<T>`, `SecantSolver<T>`) | Tier B | Caller's `Func<T, T>` is almost always transcendental; constraint is `IFloatingPointIeee754<T>`. |
| `Solvers/LevenbergMarquardtSolver<T>` | Tier B | Residuals + Jacobian in `T`; step equation is arithmetic, but the caller's residual function is transcendental. |
| `Solvers/OrdinaryLeastSquares<T>` | Tier A + √ | QR back-substitution uses only `+ − × ÷ √`. Instantiating at `T = decimal` is the high-precision path for NIST `Filip` / `Wampler4` / `Wampler5`. |
| `Solvers/ActiveSetQpSolver<T>` | Tier A + √ | Cholesky-based active-set QP for `SolveMinVariance` and `SolveMeanVariance`; both reduce to Cholesky back-substitutions per iteration via `CholeskyDecomposition<T>`. |
| `Distributions/` | Tier C | `erfc` + Acklam rational approximations published only to `double` precision. |
| `Interpolation/` linear, flat-forward, cubic-spline, monotone-convex | Tier A or Tier A + √ | `MonotoneCubicInterpolator<T>` uses `Sqrt` (harmonic-mean formula); `MonotoneConvexInterpolator<T>` (Hagan-West 2006) is pure arithmetic despite the cubic-polynomial segment integration; linear variants are pure arithmetic. |
| `Interpolation/LogLinearInterpolator<T>` | Tier B | Requires `Log` / `Exp` on `T`; IEEE only. |
| `LinearAlgebra/` (`CholeskyDecomposition<T>`, `GaussianElimination<T>`, `JacobiEigenDecomposition<T>`, `NearestPsdProjection<T>`, `PrincipalComponentAnalysis<T>`) | Tier A + √ | Each algorithm uses only `+ − × ÷ √`. Jacobi avoids trig by design. |
| `MonteCarlo/` bootstraps, percentiles | Tier A | Order statistics and arithmetic resampling. |
| `MonteCarlo/` Sobol, Halton, FastDoubleBootstrap | Tier C | Direction numbers and scrambling constants calibrated for `double`. |
| `Random/` | Tier C | Rejection sampling uses IEEE bit tricks and inverse-normal. |
| `Statistics/` sample moments, covariance estimators, correlation | Tier A | Arithmetic aggregation — sums of products with weights summing exactly to 1. |
| `Statistics/SampleSkewness<T>`, `Statistics/SampleExcessKurtosis<T>` | Tier A + √ | Two-pass standardized third/fourth moments. Welford mean/variance in pass 1 uses `Sqrt` via `NumericPrecision<T>` for the standard-deviation divisor; pass 2 accumulates cubed/fourth-power standardized deviations. Bias-correction factors match Excel `SKEW`/`KURT` and `scipy.stats.skew(bias=False)`/`scipy.stats.kurtosis(fisher=True, bias=False)`. |
| `Statistics/` shrinkage and structured estimators (Ledoit-Wolf × 3, POET, NERCOME, doubly-sparse, denoised, detoned, Tracy-Widom, OAS, EWMA, QIS) | Tier A or Tier A + √ | Arithmetic operations on the sample covariance, with `Sqrt` for estimators that normalise by standard deviations. |
| `Statistics/NeweyWestVariance<T>`, `DeflatedSharpeRatio<T>`, `HaircutSharpe<T>`, `ProbabilityOfBacktestOverfitting<T>` | Tier A with transcendental tail | Inner loop arithmetic in `T`; final scalar `Log` / `Sqrt` / inverse-normal. |
| `Statistics/FisherZTransform<T>`, `DistanceCorrelation<T>`, `GeneralizationScore<T>`, `MinimumTrackRecordLength<T>` | Tier B | Primary formula is `log` / `atanh` / normal-CDF / `sqrt`-intensive end-to-end. |

Legacy concrete-typed facades (e.g. `BrentSolver` forwarding to `BrentSolver<double>`, `CholeskyDecomposition` forwarding to `CholeskyDecomposition<decimal>`) remain live for every shipped public type so downstream consumers continue to build unchanged.

**Non-tier types.** `Collections/RollingWindow<T>` is a data structure, not a numerical algorithm. It has no tier label because it performs no floating-point arithmetic; its generic parameter `T` is unconstrained and accepts any payload. Types in this category are listed here for completeness: they live in the public API but the tier rules above do not apply to them.

## The `decimal` + transcendentals gap

The BCL provides `IFloatingPoint<decimal>`, `IFloatingPointConstants<decimal>`, and the algebraic operators on `decimal` — but it does not provide `IRootFunctions<decimal>` or any of the transcendental-function interfaces. This is correct: there is no canonical `Log`, `Exp`, or `Sin` on `decimal`; the BCL deliberately leaves that gap for numeric-library authors to fill if they need to.

Boutquin.Numerics fills the `Sqrt` gap only. The internal `NumericPrecision<T>` dispatcher provides a generic `Sqrt(T)` for `T : IFloatingPoint<T>` via compile-time-constant `typeof(T)` branching (the JIT elides the branch for non-decimal types at steady state):

- `T = double` → `Math.Sqrt`
- `T = float` → `MathF.Sqrt`
- `T = Half` → `Half.Sqrt`
- `T = decimal` → Newton-Raphson `decimal`-`Sqrt` seeded from a `double` initial guess, converging to the full 28-digit `decimal` precision in 1–2 iterations.

This design follows the `System.Numerics.Tensors` precedent for typeof-based dispatch in generic-math primitives ([dotnet/runtime#74055](https://github.com/dotnet/runtime/pull/74055)).

The library does **not** provide `Log` / `Exp` / `Sin` / `Cos` / `Atan` / `Tanh` / `Pow` on `decimal`. Tier B algorithms therefore fail at compile time for `T = decimal`, and the `CS0315` diagnostic is the correct discoverable behaviour — the alternative (a silent cast-through to `double`) would mislead the caller about the precision of the result. A `decimal` caller who needs a Tier B routine uses the legacy concrete-typed facade (which runs at `T = double` internally) and accepts the `double`-precision ceiling on the transcendental body. Implementing 28-digit Remez / Chebyshev / CORDIC approximations inside an expanded `NumericPrecision<decimal>` helper would lift that ceiling; that is a substantial engineering project with no shipped date.

## When a `decimal → double` cast is acceptable

The general rule is "never cast mid-pipeline". The Tier A-with-transcendental-tail sub-rule is the single principled exception:

A `decimal → double` cast is acceptable **if and only if**:

1. The cast applies to a **scalar reduced quantity**, not to a vector/matrix or an inner-loop value.
2. The operation on the `double` side is a **transcendental with no decimal equivalent** (`Log`, inverse-normal, chi-square CDF, `Sqrt` where `NumericPrecision<decimal>.Sqrt` is not already warranted).
3. The result either stays in `double` (the method's public return type is `double`) or is cast back once at the very end.
4. No inner loop or accumulation happens on the `double` side.

Concrete example — `ProbabilityOfBacktestOverfitting<T>` at `T = decimal`:

```csharp
var omega = T.CreateChecked(rank) / T.CreateChecked(n + 1);                // decimal aggregation via Tier A
logits[c] = ScalarLogViaDouble(omega / (T.One - omega));                    // scalar transcendental finish
```

The `omega` ratio was accumulated in `T`. The final `Log` is a one-shot scalar; the cast back loses ~1e-15 of one scalar, which is below `Log`'s transcendental ceiling. At `T = double`, the same code path takes the native `T.Log` branch with no cast.

## Forbidden patterns

- **No `decimal → double → decimal` cast inside an inner loop or accumulation.** If every iteration of a loop casts to `double`, calls a transcendental, and casts back, the aggregation's precision is bounded by the transcendental ceiling; the `decimal` working type is purely ceremonial. Either refactor the loop to defer the transcendental to a final scalar step (Tier A with transcendental tail), or host the algorithm at Tier B (IEEE only).
- **No `double` overload of a Tier A type that casts internally.** Generic math removes the motivation entirely — a `T = double` instantiation is implemented once in `T`, not by casting a `decimal` implementation. Any remaining concrete-typed facade is a thin forwarding shim over the generic version at its tier-appropriate `T`, never a cast-through.
- **No mixing types at an interior boundary** (e.g., `decimal` weights times `double` returns inside a portfolio moment calculation). Normalise at the public entry point via `T.CreateChecked` or the generic caller's chosen `T`.
- **Do not add a `decimal` overload (or facade) to a Tier B algorithm.** Tier B requires `IFloatingPointIeee754<T>`, which `decimal` does not satisfy. A cast-through overload would silently cap precision at `double`'s ceiling and mislead the caller — the correct discoverable behaviour is the `CS0315` compile error, not a lying overload.

## When to reconsider a tier

Move a file from one tier to another only when one of these is true:

1. The algorithm's critical path removes a transcendental (e.g., an `exp`-based weighting scheme is replaced by a polynomial one) — the ceiling changed, so the tier should too (Tier B → Tier A or Tier A with transcendental tail).
2. The algorithm's critical path adds a transcendental that genuinely cannot be deferred to a scalar finish (e.g., a new inner loop over `T.Log` appears). Promote from Tier A / A+√ to Tier B and accept that the type is no longer usable at `T = decimal`.
3. A consumer legitimately needs the opposite tier and the caller-side cost dominates — resolve by documenting the tier constraint more prominently, not by flipping the file's arithmetic.

Generic math has made "switch the working type" a caller decision. Ship-side tier changes should be rare.

## Why generic math

Before the generic-math migration, each folder of the library was bound to a single floating-point type — `Solvers/` was `double`, `LinearAlgebra/` was `decimal`, and so on. The choice was defensible at the time: it pre-dated `INumber<T>` / `IFloatingPoint<T>` stabilising in the BCL, and it let the library ship with one coherent precision story per subsystem. Three pressures made it untenable:

1. **Consumers repeatedly needed the opposite type.** A curve-bootstrap team wanted `double` Cholesky for speed; a portfolio team wanted `decimal` OLS for precision. Cast-through overloads silently capped precision at the decimal-to-double boundary, misleading the caller.
2. **NIST-grade OLS required `decimal` QR.** The `Filip` / `Wampler4` / `Wampler5` NIST linear-regression problems stalled the shipped `double`-QR OLS at ≤ 8 digits. A full-`decimal` QR reaches the NIST 1e-9 uniform bar, and the fix was already in the library's own `decimal` infrastructure — just not wired to the `double` OLS.
3. **The type choice had to be made in exactly the place the library is worst at making it** — at ship time, not at call site.

Generic math resolves all three: the algorithm is written once in `T`, instantiated by the caller at `double` / `float` / `Half` / `decimal` (Tier A / A + √), and Tier B algorithms fail at compile time rather than silently cast. `OrdinaryLeastSquares<decimal>` runs a full 28-digit decimal QR that clears the 1e-9 NIST coefficient bar on well-conditioned problems and holds principled per-problem bars on the ill-conditioned ones (Pontius, Filip, Wampler4, Wampler5) — the single-precision decimal path does not layer extended-precision residual refinement on top the way the double path does, so the achievable accuracy on the most ill-conditioned designs remains bounded by cond(X)·u. `CholeskyDecomposition<double>` lets curve-bootstrap callers run at IEEE speed without a facade; `BrentSolver<float>` enables memory-bound Monte Carlo pipelines that never had an option before.

The `NumericPrecision<T>` dispatcher keeps the `decimal`-`Sqrt` path (Newton-Raphson) and the IEEE `Sqrt` paths in one file, specialised by the JIT per-`T` with no branch cost at steady state on IEEE types.

## References

- [.NET Generic Math](https://learn.microsoft.com/en-us/dotnet/standard/generics/math) — `INumber<T>`, `IFloatingPoint<T>`, `IFloatingPointIeee754<T>`, `IRootFunctions<T>`, `IExponentialFunctions<T>`, `ILogarithmicFunctions<T>`, `ITrigonometricFunctions<T>`, `IHyperbolicFunctions<T>`, `IPowerFunctions<T>`.
- [dotnet/runtime#74055](https://github.com/dotnet/runtime/pull/74055) — the `System.Numerics.Tensors` precedent for compile-time `typeof(T)` dispatch in generic-math primitives.
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM. Ch 10 (Cholesky), Ch 19 (QR), Ch 24 (sums and Kahan compensation).
- Muller, J.-M. et al. (2018). *Handbook of Floating-Point Arithmetic*, 2nd ed. Birkhäuser. Ch 4 (IEEE 754 semantics), Ch 12 (decimal floating point), Ch 13 (approximation of transcendentals).
