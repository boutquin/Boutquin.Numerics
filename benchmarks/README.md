# Boutquin.Numerics — Performance Benchmarks

BenchmarkDotNet harness comparing the main numerical kernels shipped by this
package. Run against Release builds only — Debug adds an order-of-magnitude
noise floor and BenchmarkDotNet will emit warnings.

## Running

From the repository root:

```bash
# dotnet binary path varies per machine — see CLAUDE.local.md
DOTNET=/usr/local/share/dotnet/dotnet

# Build benchmark project in Release.
$DOTNET build benchmarks/Boutquin.Numerics.BenchMark \
    --configuration Release

# Run every benchmark class in the project.
$DOTNET run --project benchmarks/Boutquin.Numerics.BenchMark \
    --configuration Release -- --filter "*"

# Or pick one class.
$DOTNET run --project benchmarks/Boutquin.Numerics.BenchMark \
    --configuration Release -- \
    --filter "*SolverBenchmarks*"
```

Results land in `BenchmarkDotNet.Artifacts/results/` as
`*.md`, `*.csv`, `*.html`, and `*.json`. Commit the JSON as a baseline after
a significant change so regressions show up in code review.

## Benchmark classes

| Class | What it measures |
|---|---|
| `SolverBenchmarks` | Bisection vs Brent vs Newton vs Secant on a cubic root-finding problem at two tolerances. |
| `InterpolationBenchmarks` | Linear / monotone-cubic / cubic-spline build + 10,000 queries on 10 / 100 / 1,000-knot grids. |
| `LinearAlgebraBenchmarks` | Plain Cholesky / pivoted Cholesky / Jacobi eigendecomposition on SPD matrices of N = 10 / 50 / 200. |
| `CovarianceBenchmarks` | Sample, LW-linear, LW-CC, LW-FM, QIS, OAS, denoised, Tracy-Widom, NERCOME, POET estimators at (T=252,N=10) and (T=1260,N=50). |
| `BootstrapBenchmarks` | IID, stationary, moving-block, wild bootstrap resamplers + single-stat vs 4-stat engine overloads. |
| `RngBenchmarks` | `System.Random`, PCG-64, xoshiro256\*\*, and Marsaglia-polar Gaussian throughput. |
| `QmcBenchmarks` | Sobol vs Halton low-discrepancy sequence throughput in 4 / 8 / 16 dimensions. |

## Interpreting results

- **Baseline** — Each class marks one benchmark with `Baseline = true`. BenchmarkDotNet reports every other row as a ratio relative to that baseline. Ratios > 1.0 mean *slower*.
- **`[MemoryDiagnoser]`** — every class is annotated, so allocations (`Allocated` column) are comparable across runs. Zero-allocation hot paths should stay zero after a change.
- **Decimal vs double** — Covariance kernels operate on `decimal` and are consequently 10–100× slower than the double-based solvers and interpolators. This is expected — portfolio-level numbers demand the precision; Monte Carlo paths do not.
- **Noise floor** — Short kernels (< 100 ns) drift on laptop hardware. Prefer `dotnet run` on a quiet machine over a shared CI runner for anything you plan to regression-track.

## Committing baselines

After validating a new release:

```bash
mkdir -p benchmarks/baselines/0.6.0
cp BenchmarkDotNet.Artifacts/results/*.json benchmarks/baselines/0.6.0/
```

Baselines are **reference-only** — CI does not block on regression, but reviewers
can diff the committed JSON against a fresh run.
