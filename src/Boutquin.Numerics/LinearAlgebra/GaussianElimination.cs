// Copyright (c) 2026 Pierre G. Boutquin. All rights reserved.
//
//   Licensed under the Apache License, Version 2.0 (the "License").
//   You may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//
//   See the License for the specific language governing permissions and
//   limitations under the License.
//

using System.Numerics;

using Boutquin.Numerics.Internal;

namespace Boutquin.Numerics.LinearAlgebra;

/// <summary>
/// Gaussian elimination with randomized <em>partial</em> pivoting for solving
/// <c>A · x = b</c>. The pivot search scans only the active column (partial-pivoting
/// cost, <c>O(n)</c> per step) and selects randomly among the column entries that are
/// within a factor of two of the column maximum, a heuristic that decorrelates pivot
/// choices at the cost of leaving partial pivoting's stability guarantee: an eligible
/// pivot down to 0.5·columnMax allows multipliers up to 2, where partial pivoting bounds
/// them at 1.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+√. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>.
/// </para>
/// <para>
/// The factorization is an LU decomposition with partial (row-only) pivoting; for the
/// underlying algorithm and its stability analysis see Golub, G. H. &amp; Van Loan, C. F.
/// (2013), <em>Matrix Computations</em>, 4th ed., §3.4 (Gaussian elimination with partial
/// pivoting). Instead of deterministic partial pivoting (always choosing the column-max
/// element), this routine samples uniformly from the set of column entries whose absolute
/// value is at least 50% of the column maximum. The search examines only the active column,
/// not the trailing submatrix; this is the cost and footprint of partial pivoting, not
/// complete pivoting (which scans the entire trailing submatrix and may also swap columns).
/// The randomized choice is a heuristic that breaks ties and decorrelates pivot ordering;
/// it is <b>outside</b> partial pivoting's stability guarantee. Partial pivoting bounds
/// every multiplier <c>|ℓ| ≤ 1</c> and the growth factor <c>ρ ≤ 2ⁿ⁻¹</c> (Golub &amp; Van
/// Loan §3.4.5, attained by a Wilkinson-type matrix). An eligible pivot down to
/// 0.5·columnMax allows multipliers <c>|ℓ| ≤ 2</c>, so the growth factor can exceed
/// <c>2ⁿ⁻¹</c>, up to <c>3ⁿ⁻¹</c> (audited: growth 9841.5 at n=10 on a Wilkinson-type
/// matrix, vs 512 for partial pivoting). This is a repo extension, not a property of
/// classical partial pivoting.
/// </para>
/// <para>
/// The half-maximum magnitude threshold bounds every candidate pivot to within a
/// factor of two of the column maximum. It is not a percentile: a column can have
/// only one eligible row, in which case the choice is deterministic. Smaller
/// thresholds admit weaker pivots and a larger growth-factor ceiling.
/// </para>
/// <para>
/// The singular-matrix test is scale-RELATIVE: a pivot is treated as zero when
/// <c>|pivot| &lt; 1e-14 · max|aᵢⱼ|</c> (the largest-magnitude coefficient of the
/// input matrix). This keeps a well-conditioned but small-magnitude system from
/// being flagged as singular and still catches a genuinely weak pivot in a
/// large-magnitude system. The all-zero matrix (scale 0) falls back to the
/// absolute <c>1e-14</c> floor and is reported singular.
/// </para>
/// <para>
/// Determinism: pass a non-<see langword="null"/> <c>seed</c> to reproduce
/// pivot choices across runs (tests, regression reports). Output depends on the seed
/// through pivot ordering; on well-scaled inputs the solution is numerically identical
/// up to floating-point noise across seeds. Because the randomized pivot allows larger
/// multipliers than partial pivoting (see the growth-factor caveat above), this is
/// <b>not</b> guaranteed on ill-scaled or near-overflow inputs: different seeds can pick
/// different eligible pivots and so produce different numerical outcomes — including one
/// seed succeeding and another throwing <see cref="InvalidOperationException"/> from
/// intermediate overflow.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class GaussianElimination<T>
    where T : IFloatingPoint<T>
{
    private const int StackAllocThreshold = 1024;

    // Relative singularity threshold: a pivot is treated as zero when its magnitude is
    // below this fraction of the matrix's largest-magnitude coefficient (see Solve). The
    // all-zero matrix falls back to using this as an absolute floor.
    private static readonly T s_pivotThreshold = T.CreateChecked(1e-14);

    // Fraction of the column maximum defining the "top" set for randomized
    // pivot sampling. 0.5 trades a larger growth-factor ceiling (multipliers up to 2,
    // growth up to 3^(n-1) — outside partial pivoting's own bound, see class remarks)
    // for randomization benefit; every candidate is within a factor of two of the
    // column max.
    private static readonly T s_randomPivotFraction = T.CreateChecked(0.5);

    /// <summary>
    /// Solves the linear system A·x = b using Gaussian elimination with randomized pivoting.
    /// </summary>
    /// <param name="a">N×N coefficient matrix (will be copied, not modified).</param>
    /// <param name="b">Right-hand-side vector of length N (will be copied, not modified).</param>
    /// <param name="seed">
    /// Optional RNG seed for deterministic pivot selection in tests.
    /// Pass <see langword="null"/> for non-deterministic randomized pivoting.
    /// </param>
    /// <returns>Solution vector x of length N.</returns>
    /// <remarks>
    /// Stack note: the pivot-candidate array in <see cref="SelectPivotRow"/> is stack-allocated
    /// (via <c>stackalloc</c>) when <c>N ≤ 1024</c>; for larger active columns it rents
    /// a buffer from the shared array pool through a scoped owner that clears it on return.
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="b"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">Matrix is not square, dimensions mismatch, or an input is non-finite.</exception>
    /// <exception cref="InvalidOperationException">Matrix is singular, nearly singular, or a pivot is non-finite.</exception>
    public static T[] Solve(T[,] a, T[] b, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        var n = a.GetLength(0);
        if (a.GetLength(1) != n)
        {
            throw new ArgumentException("Matrix must be square.", nameof(a));
        }

        if (b.Length != n)
        {
            throw new ArgumentException("Right-hand side length must match matrix dimension.", nameof(b));
        }

        // Copy inputs to avoid mutation, tracking the largest-magnitude coefficient so the
        // singularity test can be made scale-RELATIVE. An absolute threshold falsely flags
        // a well-conditioned but small-magnitude system (e.g. 1e-15·I) as singular and would
        // miss a genuinely weak pivot in a large-magnitude system.
        var aug = new T[n, n + 1];
        var matrixScale = T.Zero;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                if (!T.IsFinite(a[i, j]))
                {
                    throw new ArgumentException("Matrix contains a non-finite coefficient.", nameof(a));
                }

                aug[i, j] = a[i, j];
                var mag = T.Abs(a[i, j]);
                if (mag > matrixScale)
                {
                    matrixScale = mag;
                }
            }

            if (!T.IsFinite(b[i]))
            {
                throw new ArgumentException("Right-hand side contains a non-finite value.", nameof(b));
            }

            aug[i, n] = b[i];
        }

        // Below this magnitude a pivot is treated as zero. Relative to the matrix scale,
        // falling back to the absolute floor only for the all-zero matrix (scale == 0).
        var pivotFloor = matrixScale > T.Zero ? s_pivotThreshold * matrixScale : s_pivotThreshold;

        var rng = seed.HasValue ? new System.Random(seed.Value) : new System.Random();

        // Non-IEEE types (decimal) raise OverflowException on the near-singular inputs for which
        // IEEE types produce non-finite values; both surface as the documented
        // InvalidOperationException.
        try
        {
            // Decimal has an absolute 1e-28 grid. Scale a small system upward by
            // an exact radix power before products can erase its elimination terms.
            if (typeof(T) == typeof(decimal) && matrixScale > T.Zero && matrixScale < T.One)
            {
                var exponent = -ScaledArithmetic<T>.ScaleExponent([matrixScale]);
                for (var i = 0; i < n; i++)
                {
                    for (var j = 0; j <= n; j++)
                    {
                        aug[i, j] = ScaledArithmetic<T>.Rescale(aug[i, j], exponent);
                    }
                }

                matrixScale = ScaledArithmetic<T>.Rescale(matrixScale, exponent);
                pivotFloor = s_pivotThreshold * matrixScale;
            }

            try
            {
                return EliminateAndBackSubstitute(aug, n, rng, pivotFloor);
            }
            catch (Exception ex) when (ex is OverflowException or InvalidOperationException)
            {
                // Restore the original system before retrying. Common scaling preserves
                // randomized eligibility and the input-relative pivot floor.
                if (matrixScale <= T.One)
                {
                    throw;
                }

                for (var i = 0; i < n; i++)
                {
                    for (var j = 0; j < n; j++)
                    {
                        if (!T.IsFinite(aug[i, j]))
                        {
                            // Preserve the documented refusal of non-finite pivots.
                            throw;
                        }
                    }
                }

                for (var i = 0; i < n; i++)
                {
                    for (var j = 0; j < n; j++)
                    {
                        aug[i, j] = a[i, j] / matrixScale;
                    }

                    aug[i, n] = b[i] / matrixScale;
                }

                rng = seed.HasValue ? new System.Random(seed.Value) : new System.Random();
                return EliminateAndBackSubstitute(aug, n, rng, pivotFloor / matrixScale);
            }
        }
        catch (OverflowException ex)
        {
            throw new InvalidOperationException(
                "Matrix is singular or nearly singular: elimination or back substitution overflowed.", ex);
        }
    }

    private static T[] EliminateAndBackSubstitute(T[,] aug, int n, System.Random rng, T pivotFloor)
    {
        // Forward elimination with randomized pivoting.
        for (var col = 0; col < n; col++)
        {
            var pivotRow = SelectPivotRow(aug, n, col, rng, pivotFloor, out var columnMax);

            // Singularity is a property of the column maximum, NOT of the randomly-selected pivot
            // row. The randomized partial pivoting may sample any candidate down to 0.5·columnMax, so testing the chosen
            // pivot would flag a non-singular column as singular for some RNG seeds whenever
            // columnMax sits in [pivotFloor, 2·pivotFloor). The maximum is deterministic
            // within this elimination state; previous random pivots can change this
            // state and hence the singularity decision in later columns.
            if (!T.IsFinite(columnMax) || columnMax <= T.Zero || columnMax < pivotFloor)
            {
                throw new InvalidOperationException(
                    $"Matrix is singular or nearly singular (column max {columnMax} at column {col}).");
            }

            // Swap rows if needed.
            if (pivotRow != col)
            {
                for (var j = col; j <= n; j++)
                {
                    (aug[col, j], aug[pivotRow, j]) = (aug[pivotRow, j], aug[col, j]);
                }
            }

            // Eliminate below.
            for (var i = col + 1; i < n; i++)
            {
                var factor = aug[i, col] / aug[col, col];
                for (var j = col; j <= n; j++)
                {
                    aug[i, j] -= factor * aug[col, j];
                }
            }
        }

        // Back substitution.
        var x = new T[n];
        for (var i = n - 1; i >= 0; i--)
        {
            var sum = aug[i, n];
            for (var j = i + 1; j < n; j++)
            {
                sum -= aug[i, j] * x[j];
            }

            x[i] = sum / aug[i, i];

            // A pivot that clears the singularity floor can still be small enough that back
            // substitution overflows; IEEE types would otherwise return ±Infinity/NaN silently.
            if (!T.IsFinite(x[i]))
            {
                throw new InvalidOperationException(
                    $"Matrix is nearly singular: back substitution overflowed at row {i}.");
            }
        }

        return x;
    }

    private static int SelectPivotRow(T[,] aug, int n, int col, System.Random rng, T pivotFloor, out T columnMax)
    {
        // Find the maximum absolute value in the column below the diagonal.
        var maxVal = T.Abs(aug[col, col]);
        var maxRow = col;
        for (var i = col + 1; i < n; i++)
        {
            var absVal = T.Abs(aug[i, col]);
            if (absVal > maxVal)
            {
                maxVal = absVal;
                maxRow = i;
            }
        }

        // Surface the column max so the caller's singularity test is independent of the random
        // pivot choice below, but not of previous random pivots (see Solve).
        columnMax = maxVal;

        if (maxVal <= T.Zero || maxVal < pivotFloor)
        {
            return maxRow;
        }

        // Normalize candidates before comparing: multiplying a subnormal maximum by
        // the fraction can round to zero and admit an exact-zero pivot.
        using var scratch = new PooledScratch<int>((n - col) > StackAllocThreshold ? checked(1 * (n - col)) : 0);
        var rented = scratch.Buffer;
        {
            Span<int> candidates = (n - col) > StackAllocThreshold ? rented.AsSpan(0, n - col) : stackalloc int[n - col];
            var count = 0;
            for (var i = col; i < n; i++)
            {
                if (T.Abs(aug[i, col]) / maxVal >= s_randomPivotFraction)
                {
                    candidates[count++] = i;
                }
            }

            // Randomly select among candidates.
            return count switch
            {
                0 => maxRow,
                1 => candidates[0],
                _ => candidates[rng.Next(count)],
            };
        }
    }
}

/// <summary>
/// Legacy facade delegating to <see cref="GaussianElimination{T}"/> instantiated
/// at <c>T = decimal</c>. Preserves source compatibility for existing callers.
/// </summary>
[Obsolete(
    "Use GaussianElimination<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class GaussianElimination
{
    /// <inheritdoc cref="GaussianElimination{T}.Solve"/>
    public static decimal[] Solve(decimal[,] a, decimal[] b, int? seed = null)
        => GaussianElimination<decimal>.Solve(a, b, seed);
}
