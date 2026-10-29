# Linear algebra

Boutquin.Numerics bundles the linear-algebra primitives that the rest of the numerical stack depends on — decomposition, solve, and factor analysis. Every type is generic over `T : IFloatingPoint<T>` with `Sqrt` via the internal `NumericPrecision<T>` dispatcher (Tier A + √). Accepted `T`: `double`, `float`, `Half`, `decimal`. See [`docs/tier-constraints.md`](tier-constraints.md) for the tier model, per-folder assignments, the forbidden patterns, and the scalar-cast sub-rule.

## Factorizations

| Type | Purpose | Tier |
|------|---------|------|
| `CholeskyDecomposition<T>` | `A = L·Lᵀ` for SPD matrices; pivoted variant `Pᵀ·A·P = L·Lᵀ + E` for PSD, with trailing residual `E` and a threshold-dependent rank estimate | A + √ |
| `GaussianElimination<T>` | `A·x = b` with randomized partial row pivoting (seeded for reproducibility) | A |
| `JacobiEigenDecomposition<T>` | `A = V·diag(λ)·Vᵀ` for real symmetric matrices via cyclic Givens rotations | A + √ |
| `NearestPsdProjection<T>` | Higham (1988) projection onto the PSD cone; `EigenClip`, `Higham`, `IsPsd` | A + √ |

Legacy concrete-typed facades (`CholeskyDecomposition`, `GaussianElimination`, `JacobiEigenDecomposition`, `NearestPsdProjection`) remain live for existing `decimal[,]` callers. Most forward to the `T = decimal` instantiation; the Jacobi facade instead converts to `double`, performs its rotations at double precision, and converts the result back to `decimal`. See the individual type doc-comments for algorithmic details and references.

> Least-squares estimation — both nonlinear (`LevenbergMarquardtSolver<T>`) and linear (`OrdinaryLeastSquares<T>`) — lives under [`docs/solvers.md`](solvers.md). `OrdinaryLeastSquares<T>` is a public static API backed by Householder QR with column pivoting.

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
double[,] Ld = CholeskyDecomposition<double>.Decompose(spdDouble);
double[] xd = CholeskyDecomposition<double>.Solve(Ld, [1.0, 2.0, 3.0]);
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
decimal[,] Lm = CholeskyDecomposition<decimal>.Decompose(spdDecimal);
decimal[] xm = CholeskyDecomposition<decimal>.Solve(Lm, [1m, 2m, 3m]);
// Lm agrees with Ld to 15 significant digits; the remaining 13 digits of precision
// headroom in decimal absorb the next 13 decades of condition-number growth.

// Pivoted Cholesky for PSD matrices with rank revelation — still generic.
var pivoted = CholeskyDecomposition<decimal>.DecomposePivoted(spdDecimal, tolerance: 1e-10m, maxRank: 2);
// pivoted.Lower (N×Rank), pivoted.Permutation, pivoted.Rank

// Cross-type regression — double and decimal instantiations on the same 10×10 SPD
// agree to 12 significant digits (the smaller of double's precision and decimal's
// precision at the chosen input scale).
// See tests/Boutquin.Numerics.Tests.Unit/CrossType/CholeskyCrossTypeTests.cs.
```

The same pattern applies to every other factorization. `GaussianElimination<T>.Solve`, `JacobiEigenDecomposition<T>.Decompose`, and `NearestPsdProjection<T>.EigenClip` accept the same `T` choice and respect the same tier-A + √ constraint.

Jacobi returns descending eigenvalues and matching eigenvector columns. Its selection sort follows Numerical Recipes, third edition, §11.1: at each step it selects the last equal maximum among the remaining columns. This deterministic tie rule does not select a unique mathematical basis for a repeated eigenvalue.


### Validation

The independent verification fixtures contain one 4×4 SPD input in [`linalg.json`](../tests/Verification/vectors/linalg.json) and one input with expected results for both projection methods in [`psd.json`](../tests/Verification/vectors/psd.json). The executable assertions are:

- **Cholesky:** reconstructs the input as `L·Lᵀ` with the `PrecisionNumeric = 1e-6` assertion. The class does not consume a SciPy factor or a pivoted-rank oracle. The separate [`CholeskyCrossTypeTests.cs`](../tests/Boutquin.Numerics.Tests.Unit/CrossType/CholeskyCrossTypeTests.cs) compares double and decimal on a 10×10 SPD matrix at 12 significant digits; this is cross-type parity rather than independent factor verification.
- **Gaussian elimination:** compares the solution with `numpy.linalg.solve` using `PrecisionNumeric = 1e-6`.
- **Jacobi:** compares sorted eigenvalues with `numpy.linalg.eigh` using `PrecisionStatistical = 1e-4`. This verification class does not assert eigenvector reconstruction.
- **EigenClip:** compares the projected matrix with NumPy eigendecomposition and zero-floor clipping using `PrecisionNumeric = 1e-6`.
- **Higham:** compares the nearest correlation matrix with an independent alternating-projection implementation of Higham (2002, Algorithm 3.3), using `PrecisionStatistical = 1e-4`.

These tolerances are the shared comparison parameters used by `CrossLanguageVerificationBase`, not uniform relative-error guarantees across matrix families. Pivoted Cholesky's rank estimate depends on the cutoff; Higham (1990, §6) gives numerically singular matrices whose small singular values are not revealed by diagonal pivots.

The generic implementations are checked against independent reference values and numerical invariants. Legacy facade parity is not an accuracy criterion; in particular, decimal Jacobi arithmetic can return more precise results than the legacy double-based facade.

`PrincipalComponentAnalysis<T>` carries its own independent validation layer — see the [Validation](#validation-1) subsection below.

## Principal component analysis

### `PrincipalComponentAnalysis<T>`

Factor-model eigendecomposition with explained-variance semantics. Built on `JacobiEigenDecomposition<T>`; adds the loadings/scores interface consumers expect. Tier A + √ at `T : IFloatingPoint<T>`.

```csharp
using Boutquin.Numerics.LinearAlgebra;

// From raw decimal returns (most common path). This entry point is on the
// decimal-compatible facade, because ReturnsMatrix itself is decimal-typed.
var result = PrincipalComponentAnalysis.FromReturns(returnsMatrix);

// From a pre-estimated covariance (e.g., Ledoit-Wolf shrunk):
var resultFromCov = PrincipalComponentAnalysis<decimal>.Decompose(shrunkCovariance);

// How many PCs for 95% of variance?
var k = result.NumComponentsForExplainedVariance(0.95);

// Project observations onto the top-k components.
var scores = result.Project(returnsMatrix.AsTimeByAsset(), numComponents: k);
```

Choose `T = double` to decompose a pre-computed `double` covariance matrix for a downstream `double` pipeline:

```csharp
var r = PrincipalComponentAnalysis<double>.Decompose(doubleCovarianceMatrix);
```

### Conventions

- **Sort:** eigenvalues descending. PC1 is the largest-variance direction. Matches Hull (2021, 11th Global Edition, §22.9)'s level/slope/curvature convention for yield curves.
- **Sign:** for each eigenvector column, the column is flipped so the largest-magnitude component is non-negative. Deterministic across runs on identical data.
- **Centering:**
  - The decimal facade's `FromReturns` populates `PcaResult.Mean` with per-column means; `Project(data, k)` subtracts `Mean` from each row before projection.
  - `Decompose(covarianceMatrix)` leaves `Mean` empty because no observation data was supplied; in that case the caller must pre-center the data passed to `Project`.
- **Precision:** `Decompose` is caller-chosen via `T`; the returns-based convenience path is decimal-only.

### Typical Treasury-curve use

A daily US Treasury curve panel (tenors as variables, days as observations) decomposes into a level/slope/curvature structure:

| Component | Typical share of variance |
|-----------|---------------------------|
| PC1 (level)     | ~ 70-90% |
| PC2 (slope)     | ~ 5-20% |
| PC3 (curvature) | ~ 1-8% |

The first three components typically explain > 98% of the total variance. This makes PCA the canonical input for parsimonious curve risk models and scenario construction.

### Standardization

Pass `standardize: true` to `FromReturns` to divide each column by its sample standard deviation before covariance estimation — effectively running PCA on the correlation matrix instead of the covariance matrix. The result exposes those fitted scales through `StandardDeviations` (empty for an unstandardized fit): `Project` centers new observations and divides by the same scales before applying the loadings. Copy `StandardDeviations` together with the other public properties when rebuilding a result. A full-component inverse projection recovers standardized centered data; multiply each column by its fitted sample standard deviation and add its mean to recover original units. Useful when variables have heterogeneous scales (e.g., mixing rates and spreads in one panel, or combining market excess returns with long-short factor returns).

On correlation input, the eigenvalue trace identity becomes `Σ λᵢ = N` (number of variables) exactly, so the explained-variance ratios reduce to `λᵢ / N`. Factors designed to be near-orthogonal (e.g., Fama-French construction) yield eigenvalues near 1 each; a dominant common factor shows up as a first eigenvalue noticeably above 1.

### Validation

The PCA is validated against:
- **Synthetic yield-curve-shape data** — confirms PC1 ≥ 70% variance, PC2 5–25%, PC3 1–10% on a level+slope+curvature panel. See `tests/.../LinearAlgebra/PrincipalComponentAnalysisTests.cs`.
- **Fama-French 5-factor monthly returns** (Mkt-RF, SMB, HML, RMW, CMA) — 60 months of real factor returns (1970-01 through 1974-12) from Kenneth French's data library. Nine structural-invariant assertions verify trace identity, descending eigenvalues, eigenvector orthonormality, projection round-trip recovery, standardization reducing market-factor dominance in PC1, and agreement between `NumComponentsForExplainedVariance` and the cumulative-variance array. See `tests/.../LinearAlgebra/FamaFrenchPcaTests.cs`.

### References

- Jolliffe, I. T. (2002). *Principal Component Analysis*, 2nd ed. Springer.
- Hull, J. C. (2021). *Options, Futures, and Other Derivatives*, 11th Global Edition, §22.9, pp. 534–537 (yield-curve PCA).
- Fama, E. F. & French, K. R. (2015). *A Five-Factor Asset Pricing Model*. Journal of Financial Economics 116(1), 1–22. Factor data from Kenneth French's data library, `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html`.
