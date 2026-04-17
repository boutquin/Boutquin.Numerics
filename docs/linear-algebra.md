# Linear algebra

Boutquin.Numerics bundles the linear-algebra primitives that the rest of the numerical stack depends on — decomposition, solve, and factor analysis. Every type is generic over `T : IFloatingPoint<T>` with `Sqrt` via the internal `NumericPrecision<T>` dispatcher (Tier A + √). Accepted `T`: `double`, `float`, `Half`, `decimal`. See [`docs/tier-constraints.md`](tier-constraints.md) for the tier model, per-folder assignments, the forbidden patterns, and the scalar-cast sub-rule.

## Factorizations

| Type | Purpose | Tier |
|------|---------|------|
| `CholeskyDecomposition<T>` | `A = L·Lᵀ` for SPD matrices; pivoted variant `Pᵀ·A·P = L·Lᵀ` for PSD with rank revelation | A + √ |
| `GaussianElimination<T>` | `A·x = b` with randomized complete pivoting (GERCP) | A |
| `JacobiEigenDecomposition<T>` | `A = V·diag(λ)·Vᵀ` for real symmetric matrices via cyclic Givens rotations | A + √ |
| `NearestPsdProjection<T>` | Higham (1988) projection onto the PSD cone; `EigenClip`, `Higham`, `IsPsd` | A + √ |

Legacy concrete-typed facades (`CholeskyDecomposition`, `GaussianElimination`, `JacobiEigenDecomposition`, `NearestPsdProjection`) remain live as forwarding shims over the `T = decimal` instantiation — existing `decimal[,]` callers need no change. See the individual type doc-comments for algorithmic details and references.

> Least-squares estimation — both nonlinear (`LevenbergMarquardtSolver<T>`) and linear (`OrdinaryLeastSquares<T>`) — lives under [`docs/solvers.md`](solvers.md). The `OrdinaryLeastSquares<T>` estimator uses an internal Householder QR with column pivoting (kept internal; promotion to a public API is deferred until a second caller emerges). See `docs/solvers.md` for the public contract.

### Caller-chosen precision — worked example

The generic surface makes the working-type choice a caller decision. A curve-bootstrap consumer that prefers `double` speed and a portfolio consumer that needs the 28-digit exact arithmetic of `decimal` both use the same type:

```csharp
using Boutquin.Numerics.LinearAlgebra;

// Fast double-precision Cholesky — natural for curve-bootstrap and scenario pipelines.
double[,] spdDouble =
{
    { 4.0, 12.0, -16.0 },
    { 12.0, 37.0, -43.0 },
    { -16.0, -43.0, 98.0 },
};
var cholD = new CholeskyDecomposition<double>();
double[,] Ld = cholD.Decompose(spdDouble);
double[] xd = cholD.Solve(Ld, [1.0, 2.0, 3.0]);
// Ld ≈ [[2, 0, 0], [6, 1, 0], [-8, 5, 3]] — classical textbook result, double precision.

// 28-digit decimal Cholesky — natural for portfolio covariance pipelines where the
// factorisation is an input to a decimal-denominated optimiser. Sqrt inside the
// factorisation is supplied by NumericPrecision<decimal> (Newton-Raphson to full
// 28-digit precision). No decimal → double cast anywhere on the critical path.
decimal[,] spdDecimal =
{
    { 4m, 12m, -16m },
    { 12m, 37m, -43m },
    { -16m, -43m, 98m },
};
var cholM = new CholeskyDecomposition<decimal>();
decimal[,] Lm = cholM.Decompose(spdDecimal);
decimal[] xm = cholM.Solve(Lm, [1m, 2m, 3m]);
// Lm agrees with Ld to 15 significant digits; the remaining 13 digits of precision
// headroom in decimal absorb the next 13 decades of condition-number growth.

// Pivoted Cholesky for PSD matrices with rank revelation — still generic.
var pivoted = cholM.DecomposePivoted(spdDecimal, tolerance: 1e-10m, maxRank: 2);
// pivoted.Lower (N×Rank), pivoted.Permutation, pivoted.Rank

// Cross-type regression — double and decimal instantiations on the same 10×10 SPD
// agree to 12 significant digits (the smaller of double's precision and decimal's
// precision at the chosen input scale).
// See tests/Boutquin.Numerics.Tests.Unit/CrossType/CholeskyCrossTypeTests.cs.
```

The same pattern applies to every other factorization. `GaussianElimination<T>.Solve`, `JacobiEigenDecomposition<T>.Decompose`, and `NearestPsdProjection<T>.EigenClip` accept the same `T` choice and respect the same tier-A + √ constraint.

### Validation

Every factorization has a Python cross-check in [`tests/Verification/vectors/linalg.json`](../tests/Verification/vectors/linalg.json) and [`psd.json`](../tests/Verification/vectors/psd.json):

- **`CholeskyDecomposition`** — agrees with `scipy.linalg.cholesky` (lower-triangular) to 1e-8 relative on a pool of synthetic SPD matrices across multiple condition numbers; the pivoted variant's rank-revelation output matches `scipy.linalg.cholesky(..., check_finite=True)` plus a manual permutation application. **Cross-type regression:** [`CholeskyCrossTypeTests.cs`](../tests/Boutquin.Numerics.Tests.Unit/CrossType/CholeskyCrossTypeTests.cs) exercises `CholeskyDecomposition<double>` and `CholeskyDecomposition<decimal>` on a curated 10×10 SPD and asserts agreement to 12 significant digits — this is the single test that demonstrates the "caller chooses the precision" proposition is live.
- **`GaussianElimination<T>.Solve`** — agrees with `numpy.linalg.solve` to 1e-8 relative on the same pool; the GERCP seed path is additionally determinism-tested (same seed → same pivot sequence).
- **`JacobiEigenDecomposition<T>.Decompose`** — eigenvalues agree with `numpy.linalg.eigh` to 1e-8 relative; eigenvectors agree up to sign (reconstruction `V·diag(λ)·Vᵀ` is sign-invariant and matches `numpy.linalg.eigh`'s reconstruction to 1e-10 absolute). The small gap between 1e-8 on eigenvalues and 1e-10 on reconstruction reflects the inherent numerical difference between cyclic-Jacobi and LAPACK `dsyevr` — both are iterative, both converge to the same symmetric-eigendecomposition but on different rotation sequences.
- **`NearestPsdProjection<T>.EigenClip` / `Higham`** — agree with a hand-ported numpy implementation of Higham (1988) at 1e-8 relative on a pool of correlation-matrix inputs in `psd.json`.

Generic parity (every migrated type at `T = decimal` is bit-identical to its pre-migration legacy decimal implementation) is gated by [`LinearAlgebra_ParityTests.cs`](../tests/Boutquin.Numerics.Tests.Unit/GenericParity/LinearAlgebra_ParityTests.cs).

`PrincipalComponentAnalysis<T>` carries its own independent validation layer — see the [Validation](#validation-1) subsection below.

## Principal component analysis

### `PrincipalComponentAnalysis<T>`

Factor-model eigendecomposition with explained-variance semantics. Built on `JacobiEigenDecomposition<T>`; adds the loadings/scores interface consumers expect. Tier A + √ at `T : IFloatingPoint<T>`.

```csharp
using Boutquin.Numerics.LinearAlgebra;

// From raw returns (most common path). Decimal for portfolio-grade precision.
var pca = new PrincipalComponentAnalysis<decimal>();
var result = pca.FromReturns(returnsMatrix);

// From a pre-estimated covariance (e.g., Ledoit-Wolf shrunk):
var resultFromCov = pca.Decompose(shrunkCovariance);

// How many PCs for 95% of variance?
var k = result.NumComponentsForExplainedVariance(0.95m);

// Project observations onto the top-k components.
var scores = result.Project(returnsMatrix, numComponents: k);
```

Switch `T` to run the same factor decomposition at `double` speed when the caller is feeding a downstream `double`-denominated pipeline:

```csharp
var pcaD = new PrincipalComponentAnalysis<double>();
var r = pcaD.FromReturns(doublePanelReturns);
```

### Conventions

- **Sort:** eigenvalues descending. PC1 is the largest-variance direction. Matches the Hull Ch 33.6 level/slope/curvature convention for yield curves.
- **Sign:** for each eigenvector column, the column is flipped so the largest-magnitude component is non-negative. Deterministic across runs on identical data.
- **Centering:**
  - `FromReturns` populates `PcaResult<T>.Mean` with per-column means; `Project(data, k)` subtracts `Mean` from each row before projection.
  - `Decompose(covarianceMatrix)` leaves `Mean` empty because no observation data was supplied; in that case the caller must pre-center the data passed to `Project`.
- **Precision:** caller-chosen via `T`. The legacy public API (`decimal[,]` in / `decimal[,]` out) is the `T = decimal` instantiation.

### Typical Treasury-curve use

A daily US Treasury curve panel (tenors as variables, days as observations) decomposes into a level/slope/curvature structure:

| Component | Typical share of variance |
|-----------|---------------------------|
| PC1 (level)     | ~ 70-90% |
| PC2 (slope)     | ~ 5-20% |
| PC3 (curvature) | ~ 1-8% |

The first three components typically explain > 98% of the total variance. This makes PCA the canonical input for parsimonious curve risk models and scenario construction.

### Standardization

Pass `standardize: true` to `FromReturns` to divide each column by its sample standard deviation before covariance estimation — effectively running PCA on the correlation matrix instead of the covariance matrix. Useful when variables have heterogeneous scales (e.g., mixing rates and spreads in one panel, or combining market excess returns with long-short factor returns).

On correlation input, the eigenvalue trace identity becomes `Σ λᵢ = N` (number of variables) exactly, so the explained-variance ratios reduce to `λᵢ / N`. Factors designed to be near-orthogonal (e.g., Fama-French construction) yield eigenvalues near 1 each; a dominant common factor shows up as a first eigenvalue noticeably above 1.

### Validation

The PCA is validated against:
- **Synthetic yield-curve-shape data** — confirms PC1 ≥ 70% variance, PC2 5–25%, PC3 1–10% on a level+slope+curvature panel. See `tests/.../LinearAlgebra/PrincipalComponentAnalysisTests.cs`.
- **Fama-French 5-factor monthly returns** (Mkt-RF, SMB, HML, RMW, CMA) — 60 months of real factor returns (1970-01 through 1974-12) from Kenneth French's data library. Nine structural-invariant assertions verify trace identity, descending eigenvalues, eigenvector orthonormality, projection round-trip recovery, standardization reducing market-factor dominance in PC1, and agreement between `NumComponentsForExplainedVariance` and the cumulative-variance array. See `tests/.../LinearAlgebra/FamaFrenchPcaTests.cs`.

### References

- Jolliffe, I. T. (2002). *Principal Component Analysis*, 2nd ed. Springer.
- Hull, J. C. (2017). *Options, Futures, and Other Derivatives*, 11e. Ch 22.9 (PCA for VaR), Ch 33.6 (PCA for yield-curve factors).
- Fama, E. F. & French, K. R. (2015). *A Five-Factor Asset Pricing Model*. Journal of Financial Economics 116(1), 1–22. Factor data from Kenneth French's data library, `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html`.
