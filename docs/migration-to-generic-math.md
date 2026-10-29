# Migrating from Legacy Facades to Generic Math

Boutquin.Numerics 2.0.0 deprecates the concrete-typed facades marked with diagnostic `BNU0001`.
Those facades delegate to generic implementations, remain callable in 2.x, and are scheduled for
**removal in v3.0**. This guide explains why the change was made, how to suppress the warning
temporarily, and how to migrate each category of caller.

## Why the change?

The original library shipped with one hard-bound type per subsystem (`Solvers/` at `double`,
`LinearAlgebra/` at `decimal`, …). Three pressures made that untenable:

- Consumers repeatedly needed the opposite type (`double` Cholesky for speed, `decimal` OLS for
  precision); the cast-through facades silently capped precision.
- Ill-conditioned NIST linear-regression problems exposed the limits of the early `double`-QR
  implementation. The current double path adds mixed-precision residual refinement, while the
  decimal path factors in decimal arithmetic; working precision alone does not rank their accuracy.
- The type choice was frozen at library ship time rather than resolved at the call site.

Generic math (`IFloatingPoint<T>`, `IFloatingPointIeee754<T>`) eliminates all three: the algorithm
is written once in `T`, and the caller picks the precision regime. See
[`docs/tier-constraints.md`](tier-constraints.md) for the four-tier constraint model (A, A+√, B, C)
that governs which `T` each algorithm accepts.

## Diagnostic ID and suppression

The facades marked obsolete emit diagnostic **`BNU0001`** on use. This is a warning, not an error.
The non-generic `BootstrapMonteCarloEngine`, `BootstrapResampler`, `MovingBlockBootstrapResampler`,
`StationaryBootstrapResampler`, `WildBootstrapResampler`, `HaltonSequence`, `SobolSequence`, and
`PSquareEstimator` remain convenience forms without that warning or the v3.0 removal announcement.
For a `BNU0001` warning, you have three options:

### Option 1 — Migrate now (recommended)

Replace each facade call with its generic equivalent. The patterns are in the sections below.

### Option 2 — Suppress project-wide

Add to your `.csproj`:

```xml
<PropertyGroup>
  <NoWarn>$(NoWarn);BNU0001</NoWarn>
</PropertyGroup>
```

This is what `Boutquin.Numerics.Tests.Unit` and `Boutquin.Numerics.BenchMark` do to keep
legacy-coverage tests building without noise.

### Option 3 — Suppress per file

```csharp
#pragma warning disable BNU0001
var solver = new BrentSolver();
#pragma warning restore BNU0001
```

Use this when you can only migrate part of a file at a time.

## Migration patterns

### Pattern 1 — Root finders (Tier B: `double`, `float`, `Half`)

The root finders require `IFloatingPointIeee754<T>` — they do not work at `T = decimal`. The
default type to choose is `double`.

**Before:**
```csharp
using Boutquin.Numerics.Solvers;

IBracketedRootSolver solver = new BrentSolver(tolerance: 1e-10, maxIterations: 50);
var result = solver.Solve(x => x * x - 2.0, 1.0, 2.0);
Console.WriteLine(result.Root); // √2
```

**After:**
```csharp
using Boutquin.Numerics.Solvers;

IBracketedRootSolver<double> solver = new BrentSolver<double>(tolerance: 1e-10, maxIterations: 50);
var result = solver.Solve(x => x * x - 2.0, 1.0, 2.0);
Console.WriteLine(result.Root); // √2
```

The same `<double>` suffix applies to `BisectionSolver`, `NewtonRaphsonSolver`, `SecantSolver`,
`MullerSolver`, and `LevenbergMarquardtSolver`.

### Pattern 2 — OLS (Tier A+√: `double` or `decimal`)

`OrdinaryLeastSquares` is special: the generic form works at both `double` (with mixed-precision
decimal-residual refinement) and `decimal` (full 28-digit QR, no refinement). Both pass the NIST
StRD suite; choose based on your precision budget.

**Before:**
```csharp
var result = OrdinaryLeastSquares.Fit(x, y, includeIntercept: true);
```

**After (double — fast, mixed-precision refinement):**
```csharp
var result = OrdinaryLeastSquares<double>.Fit(x, y, includeIntercept: true);
```

**After (decimal — 28-digit QR throughout):**
```csharp
var result = OrdinaryLeastSquares<decimal>.Fit(x, y, includeIntercept: true);
```

### Pattern 3 — Linear algebra (Tier A+√: choose the working type)

Most legacy facades computed in `decimal`. To preserve their working precision, use
`<decimal>`; choose `<double>` for IEEE speed.

**Before:**
```csharp
var L = CholeskyDecomposition.Decompose(matrix);
var solution = CholeskyDecomposition.Solve(L, rhs);
```

**After:**
```csharp
var L = CholeskyDecomposition<decimal>.Decompose(matrix);
var solution = CholeskyDecomposition<decimal>.Solve(L, rhs);
```

The same pattern applies to `GaussianElimination`, `NearestPsdProjection`, and
`PrincipalComponentAnalysis`. `JacobiEigenDecomposition` is an exception: its legacy
facade accepts and returns `decimal` but performs the rotations in `double`.
`JacobiEigenDecomposition<double>` preserves that working type (with `double` inputs
and outputs); `JacobiEigenDecomposition<decimal>` instead computes in `decimal` and
may produce more precise results than the legacy facade.

### Pattern 4 — Statistics (Tier A and arithmetic with transcendental tail: `decimal`; Tier B: IEEE types)

Most statistics types were decimal-pinned. The generic surface preserves the default.

**Before:**
```csharp
var moments = new WelfordMoments();
foreach (var r in returns) moments.Add(r);
Console.WriteLine(moments.Mean);

var cov = new SampleCovarianceEstimator().Estimate(returnsMatrix);
```

**After:**
```csharp
var moments = new WelfordMoments<decimal>();
foreach (var r in returns) moments.Add(r);
Console.WriteLine(moments.Mean);

var cov = new SampleCovarianceEstimator<decimal>().Estimate(returnsMatrix.AsTimeByAsset());
```

`ReturnsMatrix<T>` requires explicit construction from `T[,]` or `T[][]`, for
example `new ReturnsMatrix<decimal>(assetMajor)`. The implicit array conversions
exist only on the legacy non-generic `ReturnsMatrix` facade. To use the generic
interface's wrapper overload, call it through `ICovarianceEstimator<decimal>`.

`NeweyWestVariance` is Tier A: its covariance sums are arithmetic-only, with no
transcendental tail. `DeflatedSharpeRatio`, `HaircutSharpe`, and
`ProbabilityOfBacktestOverfitting` are Tier A with a transcendental tail.
All four legacy facades use
`decimal`; migrate to `<decimal>` to preserve that working type. Distance correlation,
Fisher Z-transform, generalization score, and minimum track-record length are Tier B
and require an IEEE type such as `double`.

In this example, `returns` is a `decimal[,]` strategy-by-time matrix; `sr`, `t`
(backtest years), `skewness`, and `kurtosis` are `decimal`, and `n` is the `int`
number of trials.

**Before:**
```csharp
decimal pbo = ProbabilityOfBacktestOverfitting.Compute(returns, splitCount: 16).Pbo;
decimal dsr = DeflatedSharpeRatio.Compute(sr, n, t, skewness, kurtosis).DeflatedSharpe;
```

**After:**
```csharp
var pboResult = ProbabilityOfBacktestOverfitting<decimal>.Compute(returns, splitCount: 16);
decimal pbo = pboResult.Pbo;

decimal dsr = DeflatedSharpeRatio<decimal>.Compute(sr, n, t, skewness, kurtosis).DeflatedSharpe;
```

### Pattern 5 — Interpolators and factory

**Before:**
```csharp
IInterpolator interp = InterpolatorFactory.Create(InterpolatorKind.Linear);
double y = interp.Interpolate(x, xs, ys);
```

**After:**
```csharp
IInterpolator<double> interp = InterpolatorFactory<double>.Create(InterpolatorKind.Linear);
double y = interp.Interpolate(x, xs, ys);
```

If you held a reference to the singleton instances directly:

**Before:**
```csharp
IInterpolator i = LinearInterpolator.Instance;
```

**After:**
```csharp
IInterpolator<double> i = LinearInterpolator<double>.Instance;
```

The same suffix applies to `LogLinearInterpolator`, `FlatForwardInterpolator`,
`MonotoneCubicInterpolator`, `MonotoneConvexInterpolator`, `CubicSplineInterpolator`,
`MonotoneCubicSpline`, and `TwoPointLinearInterpolator`.

> **`MonotoneCubicInterpolator<T>`, `CubicSplineInterpolator<T>`, `MonotoneConvexInterpolator<T>`, and `InterpolatorFactory<T>` require `where T : unmanaged, IFloatingPoint<T>`.**
> This is stricter than most Tier A types (which only require `IFloatingPoint<T>`). The
> `unmanaged` constraint enables stack allocation of the per-call working buffers, eliminating
> heap pressure on every `Interpolate()` call. All standard numeric types — `double`, `float`,
> `Half`, `decimal` — satisfy `unmanaged`, so the constraint has no practical effect for typical
> callers. It only matters if you have a custom managed type that implements `IFloatingPoint<T>`
> with reference-type fields (rare; the BCL has none). `InterpolatorFactory<T>` already carries
> the same `unmanaged` constraint, so factory-based access is unaffected.

## v2.0 breaking changes

`BNU0001` is not the only v2.0 breaking change. Recompile every consumer against v2.0, then
address the source changes below. Arrays convert implicitly to `ReadOnlySpan<T>`, but a changed
parameter or callback type is binary-breaking even when its call site still compiles.

### Selector parameters are now `double`

Selectors are dimensionless probabilities or confidence levels, not values in the working
numeric type. `Percentile.Compute<T>`, `FisherZTransform<T>.ConfidenceInterval`,
`MinimumTrackRecordLength<T>.Compute`, and `PcaResult<T>.NumComponentsForExplainedVariance`
therefore now take `double` selector arguments.

**Before:**
```csharp
decimal p95 = Percentile.Compute(values, 0.95m);
int components = pca.NumComponentsForExplainedVariance(0.95m);
```

**After:**
```csharp
decimal p95 = Percentile.Compute(values, 0.95);
int components = pca.NumComponentsForExplainedVariance(0.95);
```

### Span-first inputs and callbacks

Bootstrap engines, `Subsampler.Run<T>`, and other data-oriented APIs accept
`ReadOnlySpan<T>` inputs. `RunMulti` and writer-style callbacks use `Span<T>` for their output
buffer. Arrays remain valid inputs; callbacks must accept the span form.

**Before:**
```csharp
double Sharpe(double[] sample) => sample.Average();
double[] values = Subsampler.Run(series, 60, Sharpe);
```

**After:**
```csharp
static double Mean(ReadOnlySpan<double> sample)
{
    double sum = 0;
    for (var i = 0; i < sample.Length; i++) sum += sample[i];
    return sum / sample.Length;
}

double[] values = Subsampler.Run(series, subsampleLength: 60, statistic: Mean);
```

### `GeneralizationScore<T>` is replaced by `GtInspiredScore<T>`

The v2 RC removes `GeneralizationScore<T>`, `GeneralizationScore`, and `GtScoreResult` in favor of
the explicitly named GT-inspired library variant and `GtInspiredScoreResult<T>`
with `Score`, `Mu`, `Z`, `RSquared`, and `DownsideDeviation`. `Score` is a loss: lower is better;
the z > 1 utility branch is negative when its mean is positive, while marginal or underperforming series
receive a positive penalty. It is not an out-of-sample-reliability statistic and it no longer
accepts a trading-days parameter.

**Before:**
```csharp
var result = GeneralizationScore.Compute(returns, tradingDays: 252);
```

**After:**
```csharp
GtInspiredScoreResult<double> result = GtInspiredScore<double>.Compute(
    returns,
    benchmarkMean: 0.0);
double loss = result.Score;
```

The variant uses sample SD, cumulative arithmetic-return R² against time, zero-target downside
RMS across all observations, and additive epsilon protection. The default inclusive minimum
threshold is 50; use the four-argument overload to choose a sampling-frequency threshold.
No author parity or empirical generalization claim is made. The concrete facade is now named
`GtInspiredScore`; it forwards to the generic double implementation and carries `BNU0001`.

### Removed APIs

`PolitisWhiteBlockLength.Estimate<T>` was removed; use the explicit double-precision routine:

```csharp
double blockLength = PolitisWhiteBlockLength.EstimateDouble(returns);
```

The windowed `PearsonCorrelation.Compute` overload was removed. Slice the input span instead:

```csharp
double correlation = PearsonCorrelation<double>.Compute(
    x.AsSpan(start, length),
    y.AsSpan(start, length));
```

### Active-set QP bounds

`ActiveSetQpSolver<T>.SolveMinVariance` and the positive-risk-aversion path of
`SolveMeanVariance` now relax only the offending side of an infeasible box (`n · minWeight > 1`
or `n · maxWeight < 1`) so the interval includes the equal-weight point `1/n`: the minimum is
lowered or the maximum is raised, while the other bound is preserved. Feasible inputs are
unchanged. The `riskAversion = 0` mean-variance path is a bounded linear programme and still
rejects an infeasible box with `ArgumentException`; validate bounds before using that limit case.

## v3.0 removal timeline

The deprecated facades will be removed in **v3.0**. `BNU0001` is a warning, not an error, and
the facades remain callable throughout 2.x; v3.0 will make every removal a compile error. Plan
that migration independently of the v2.0 source and binary changes above.

## Tier model quick reference

| Tier | Constraint | Accepts | When to use |
|------|-----------|---------|-------------|
| A | `IFloatingPoint<T>` | `double`, `float`, `Half`, `decimal` | Arithmetic only |
| A† | `unmanaged, IFloatingPoint<T>` | `double`, `float`, `Half`, `decimal` | Arithmetic only + stackalloc working buffers (`MonotoneConvexInterpolator<T>`) |
| A+√ | `IFloatingPoint<T>` + internal Sqrt dispatcher | `double`, `float`, `Half`, `decimal` | Needs `Sqrt` |
| B | `IFloatingPointIeee754<T>` | `double`, `float`, `Half` | Transcendentals (`Log`, `Exp`, trig) |
| C | `IFloatingPoint<T>` at surface; `double` internally | `double` in practice | Polynomial-approximation tables calibrated for `double` |

† All BCL floating-point types satisfy `unmanaged`; the stricter constraint only excludes
exotic user-defined types with reference-type fields.

Full tier definitions and per-folder assignments are in [`docs/tier-constraints.md`](tier-constraints.md).
