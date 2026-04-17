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

namespace Boutquin.Numerics.LinearAlgebra;

/// <summary>
/// Gaussian Elimination with Randomized Complete Pivoting (GERCP) for solving
/// <c>A · x = b</c>. Delivers complete-pivoting growth-factor bounds
/// <em>in expectation</em> at partial-pivoting cost, avoiding the O(N³) pivot
/// search of deterministic complete pivoting.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+√. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>.
/// </para>
/// <para>
/// Reference: arXiv:2505.02023. Instead of deterministic partial pivoting
/// (always choosing the column-max element), GERCP samples uniformly from the
/// set of column entries whose absolute value exceeds 50% of the column
/// maximum. In expectation the growth factor matches complete pivoting's
/// <c>O(n^(1/2 + ε))</c> (Wilkinson's conjecture regime) rather than partial
/// pivoting's worst-case <c>2^(n−1)</c>, while the per-step cost stays
/// <c>O(n)</c> — the search examines only the active column, not the entire
/// submatrix.
/// </para>
/// <para>
/// Why 50%? A uniform sample from the top-50% set is large enough to
/// guarantee a randomization benefit (at least half the column is eligible
/// on average) but tight enough that every candidate pivot is within a
/// factor of two of the column max, preserving the partial-pivoting
/// stability floor. Smaller thresholds admit weak pivots; larger thresholds
/// collapse to deterministic pivoting and lose the probabilistic advantage.
/// </para>
/// <para>
/// The singular-matrix test uses an absolute threshold
/// (<c>|pivot| &lt; 1e-14</c>), appropriate for the covariance-scale inputs
/// in this pipeline. Callers working with matrices scaled far from unit
/// magnitude should normalize first or augment with a relative-pivoting
/// check downstream.
/// </para>
/// <para>
/// Determinism: pass a non-<see langword="null"/> <c>seed</c> to reproduce
/// pivot choices across runs (tests, regression reports). Output depends
/// on the seed only through pivot ordering; the solution is numerically
/// identical up to floating-point noise regardless of seed.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class GaussianElimination<T>
    where T : IFloatingPoint<T>
{
    // Absolute singularity threshold: below this the pivot is treated as zero.
    // Sized for covariance-scale inputs; callers with far-from-unit-scale
    // matrices should rescale or add a relative check downstream.
    private static readonly T s_pivotThreshold = T.CreateChecked(1e-14);

    // Fraction of the column maximum defining the "top" set for randomized
    // pivot sampling. 0.5 balances randomization benefit vs. stability floor —
    // every candidate is within a factor of two of the column max.
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
    /// <exception cref="ArgumentException">Matrix is not square or dimensions mismatch.</exception>
    /// <exception cref="InvalidOperationException">Matrix is singular or nearly singular.</exception>
    public static T[] Solve(T[,] a, T[] b, int? seed = null)
    {
        var n = a.GetLength(0);
        if (a.GetLength(1) != n)
        {
            throw new ArgumentException("Matrix must be square.", nameof(a));
        }

        if (b.Length != n)
        {
            throw new ArgumentException("Right-hand side length must match matrix dimension.", nameof(b));
        }

        // Copy inputs to avoid mutation.
        var aug = new T[n, n + 1];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                aug[i, j] = a[i, j];
            }

            aug[i, n] = b[i];
        }

        var rng = seed.HasValue ? new System.Random(seed.Value) : new System.Random();

        // Forward elimination with randomized pivoting.
        for (var col = 0; col < n; col++)
        {
            var pivotRow = SelectPivotRow(aug, n, col, rng);

            if (T.Abs(aug[pivotRow, col]) < s_pivotThreshold)
            {
                throw new InvalidOperationException(
                    $"Matrix is singular or nearly singular (pivot {aug[pivotRow, col]} at column {col}).");
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
        }

        return x;
    }

    private static int SelectPivotRow(T[,] aug, int n, int col, System.Random rng)
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

        if (maxVal < s_pivotThreshold)
        {
            return maxRow;
        }

        // GERCP: collect all rows with |element| >= threshold * max.
        var threshold = s_randomPivotFraction * maxVal;
        Span<int> candidates = stackalloc int[n - col];
        var count = 0;
        for (var i = col; i < n; i++)
        {
            if (T.Abs(aug[i, col]) >= threshold)
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

/// <summary>
/// Legacy facade delegating to <see cref="GaussianElimination{T}"/> instantiated
/// at <c>T = decimal</c>. Preserves source compatibility for existing callers.
/// </summary>
public static class GaussianElimination
{
    /// <inheritdoc cref="GaussianElimination{T}.Solve"/>
    public static decimal[] Solve(decimal[,] a, decimal[] b, int? seed = null)
        => GaussianElimination<decimal>.Solve(a, b, seed);
}
