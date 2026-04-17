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
/// Result of a pivoted Cholesky decomposition: P^T·A·P = L·L^T where
/// only the first <see cref="Rank"/> columns of L are non-zero.
/// </summary>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
/// <param name="Lower">Lower-triangular factor (N×Rank).</param>
/// <param name="Permutation">Pivot permutation array — row i of the permuted matrix corresponds to row Permutation[i] of the original.</param>
/// <param name="Rank">Numerical rank (number of pivots accepted before tolerance was reached).</param>
public sealed record PivotedCholeskyResult<T>(T[,] Lower, int[] Permutation, int Rank)
    where T : IFloatingPoint<T>;

/// <summary>
/// Result of a pivoted Cholesky decomposition: P^T·A·P = L·L^T where
/// only the first <see cref="Rank"/> columns of L are non-zero.
/// </summary>
/// <param name="Lower">Lower-triangular factor (N×Rank).</param>
/// <param name="Permutation">Pivot permutation array — row i of the permuted matrix corresponds to row Permutation[i] of the original.</param>
/// <param name="Rank">Numerical rank (number of pivots accepted before tolerance was reached).</param>
public sealed record PivotedCholeskyResult(decimal[,] Lower, int[] Permutation, int Rank);

/// <summary>
/// Cholesky factorization <c>A = L · Lᵀ</c> for symmetric positive-definite
/// (SPD) matrices, with a rank-revealing pivoted variant <c>PᵀAP = LLᵀ</c> for
/// symmetric positive-semidefinite (PSD) inputs.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+√. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Square-root operations use
/// <see cref="NumericPrecision{T}.Sqrt"/> to support <c>decimal</c> (which lacks
/// <see cref="IRootFunctions{TSelf}"/>).
/// </para>
/// <para>
/// The standard (unpivoted) factorization processes columns left-to-right and
/// requires strict positive-definiteness; it throws
/// <see cref="InvalidOperationException"/> as soon as a non-positive pivot is
/// encountered. Work is O(N³/3), memory is O(N²); the factor is
/// unconditionally stable on SPD inputs (Higham 2002, §10).
/// </para>
/// <para>
/// The pivoted variant (Higham 1990; refinement in arXiv:2507.20678) performs
/// diagonal pivoting — at each step the column with the largest residual
/// diagonal (the Schur-complement diagonal) is selected. This makes the
/// factorization rank-revealing: when the largest remaining diagonal falls
/// below <c>tolerance</c>, the algorithm terminates early, producing a
/// numerically-rank-deficient L with only <c>Rank</c> non-zero columns. A
/// running <c>diagRemaining</c> vector tracks the Schur-complement diagonals
/// without recomputing them from scratch each step, keeping the operation
/// count at O(N³/3) despite the pivot search.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class CholeskyDecomposition<T>
    where T : IFloatingPoint<T>
{
    /// <summary>
    /// Computes the standard (unpivoted) Cholesky decomposition A = L·Lᵀ.
    /// </summary>
    /// <param name="a">Symmetric positive-definite N×N matrix (only the lower triangle is read).</param>
    /// <returns>Lower-triangular factor L.</returns>
    /// <exception cref="ArgumentException">Matrix is not square.</exception>
    /// <exception cref="InvalidOperationException">Matrix is not positive definite.</exception>
    public static T[,] Decompose(T[,] a)
    {
        var n = a.GetLength(0);
        if (a.GetLength(1) != n)
        {
            throw new ArgumentException("Matrix must be square.", nameof(a));
        }

        var lower = new T[n, n];

        for (var j = 0; j < n; j++)
        {
            var sum = T.Zero;
            for (var k = 0; k < j; k++)
            {
                sum += lower[j, k] * lower[j, k];
            }

            var diag = a[j, j] - sum;
            if (diag <= T.Zero)
            {
                throw new InvalidOperationException(
                    $"Matrix is not positive definite (non-positive diagonal {diag} at index {j}).");
            }

            lower[j, j] = NumericPrecision<T>.Sqrt(diag);

            for (var i = j + 1; i < n; i++)
            {
                sum = T.Zero;
                for (var k = 0; k < j; k++)
                {
                    sum += lower[i, k] * lower[j, k];
                }

                lower[i, j] = (a[i, j] - sum) / lower[j, j];
            }
        }

        return lower;
    }

    /// <summary>
    /// Solves A·x = b given the lower-triangular Cholesky factor L (where A = L·Lᵀ)
    /// via forward substitution (L·y = b) and back substitution (Lᵀ·x = y).
    /// </summary>
    /// <param name="lower">Lower-triangular factor from <see cref="Decompose"/>.</param>
    /// <param name="b">Right-hand-side vector.</param>
    /// <returns>Solution vector x.</returns>
    public static T[] Solve(T[,] lower, T[] b)
    {
        var n = lower.GetLength(0);
        if (b.Length != n)
        {
            throw new ArgumentException("Vector length must match matrix dimension.", nameof(b));
        }

        // Forward substitution: L·y = b.
        var y = new T[n];
        for (var i = 0; i < n; i++)
        {
            var sum = b[i];
            for (var j = 0; j < i; j++)
            {
                sum -= lower[i, j] * y[j];
            }

            y[i] = sum / lower[i, i];
        }

        // Back substitution: Lᵀ·x = y.
        var x = new T[n];
        for (var i = n - 1; i >= 0; i--)
        {
            var sum = y[i];
            for (var j = i + 1; j < n; j++)
            {
                sum -= lower[j, i] * x[j];
            }

            x[i] = sum / lower[i, i];
        }

        return x;
    }

    /// <summary>Convenience overload using tolerance = <c>T.Zero</c> and no rank limit.</summary>
    /// <summary>Convenience overload using tolerance = <c>T.Zero</c> and no rank limit.</summary>
    public static PivotedCholeskyResult<T> DecomposePivoted(T[,] a)
        => DecomposePivoted(a, T.Zero, null);

    /// <summary>
    /// Computes the pivoted Cholesky decomposition selecting the largest remaining diagonal at each step.
    /// Supports positive semi-definite matrices and early termination.
    /// </summary>
    /// <param name="a">Symmetric positive semi-definite N-by-N matrix.</param>
    /// <param name="tolerance">Diagonal values below this threshold are treated as zero.</param>
    /// <param name="maxRank">Maximum number of pivots (columns of L). Null means no limit.</param>
    /// <returns>Lower-triangular factor, permutation array, and numerical rank.</returns>
    public static PivotedCholeskyResult<T> DecomposePivoted(
        T[,] a, T tolerance, int? maxRank)
    {
        var tol = tolerance;
        var n = a.GetLength(0);
        if (a.GetLength(1) != n)
        {
            throw new ArgumentException("Matrix must be square.", nameof(a));
        }

        var rankLimit = maxRank ?? n;

        // Work on a copy.
        var work = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                work[i, j] = a[i, j];
            }
        }

        var perm = new int[n];
        for (var i = 0; i < n; i++)
        {
            perm[i] = i;
        }

        var lower = new T[n, n];
        var rank = 0;

        // Track remaining Schur complement diagonals for pivot selection.
        var diagRemaining = new T[n];
        for (var i = 0; i < n; i++)
        {
            diagRemaining[i] = work[i, i];
        }

        for (var j = 0; j < Math.Min(n, rankLimit); j++)
        {
            // Find the largest remaining diagonal from j..n-1.
            var maxDiag = diagRemaining[j];
            var maxIdx = j;
            for (var i = j + 1; i < n; i++)
            {
                if (diagRemaining[i] > maxDiag)
                {
                    maxDiag = diagRemaining[i];
                    maxIdx = i;
                }
            }

            if (maxDiag <= tol)
            {
                break;
            }

            // Swap rows/columns in work matrix, diagRemaining, and permutation.
            if (maxIdx != j)
            {
                SwapRowsAndCols(work, n, j, maxIdx);
                SwapRowsPartial(lower, j, j, maxIdx);
                (perm[j], perm[maxIdx]) = (perm[maxIdx], perm[j]);
                (diagRemaining[j], diagRemaining[maxIdx]) = (diagRemaining[maxIdx], diagRemaining[j]);
            }

            // Standard Cholesky step on column j using the permuted work matrix.
            var sum = T.Zero;
            for (var k = 0; k < j; k++)
            {
                sum += lower[j, k] * lower[j, k];
            }

            var diag = work[j, j] - sum;
            if (diag <= T.Zero)
            {
                break;
            }

            lower[j, j] = NumericPrecision<T>.Sqrt(diag);

            for (var i = j + 1; i < n; i++)
            {
                sum = T.Zero;
                for (var k = 0; k < j; k++)
                {
                    sum += lower[i, k] * lower[j, k];
                }

                lower[i, j] = (work[i, j] - sum) / lower[j, j];
            }

            // Update remaining diagonals for next pivot selection.
            for (var i = j + 1; i < n; i++)
            {
                diagRemaining[i] -= lower[i, j] * lower[i, j];
            }

            rank++;
        }

        return new PivotedCholeskyResult<T>(lower, perm, rank);
    }

    private static void SwapRowsAndCols(T[,] m, int n, int i, int j)
    {
        for (var k = 0; k < n; k++)
        {
            (m[i, k], m[j, k]) = (m[j, k], m[i, k]);
        }

        for (var k = 0; k < n; k++)
        {
            (m[k, i], m[k, j]) = (m[k, j], m[k, i]);
        }
    }

    private static void SwapRowsPartial(T[,] m, int cols, int i, int j)
    {
        for (var k = 0; k < cols; k++)
        {
            (m[i, k], m[j, k]) = (m[j, k], m[i, k]);
        }
    }
}

/// <summary>
/// Legacy facade delegating to <see cref="CholeskyDecomposition{T}"/> instantiated
/// at <c>T = decimal</c>. Preserves source compatibility for existing callers.
/// </summary>
public static class CholeskyDecomposition
{
    /// <inheritdoc cref="CholeskyDecomposition{T}.Decompose"/>
    public static decimal[,] Decompose(decimal[,] a)
        => CholeskyDecomposition<decimal>.Decompose(a);

    /// <inheritdoc cref="CholeskyDecomposition{T}.Solve"/>
    public static decimal[] Solve(decimal[,] lower, decimal[] b)
        => CholeskyDecomposition<decimal>.Solve(lower, b);

    /// <summary>
    /// Computes the pivoted Cholesky decomposition, selecting the largest remaining diagonal at each step.
    /// </summary>
    public static PivotedCholeskyResult DecomposePivoted(
        decimal[,] a, decimal tolerance = 0m, int? maxRank = null)
    {
        var result = CholeskyDecomposition<decimal>.DecomposePivoted(a, tolerance, maxRank);
        return new PivotedCholeskyResult(result.Lower, result.Permutation, result.Rank);
    }
}
