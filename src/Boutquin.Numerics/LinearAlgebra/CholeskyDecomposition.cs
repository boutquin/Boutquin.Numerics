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
/// Result of a pivoted Cholesky decomposition: P^T·A·P = L·L^T + E,
/// where E is the trailing Schur-complement residual (up to rounding) and L has
/// <see cref="Rank"/> columns. E vanishes for a complete exact-rank factorization in exact arithmetic.
/// </summary>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed record PivotedCholeskyResult<T>
    where T : IFloatingPoint<T>
{
    private T[,] _lower;
    private int[] _permutation;

    /// <summary>Initializes a result with owned copies of the factor and permutation.</summary>
    /// <param name="Lower">Lower-triangular factor (N×Rank).</param>
    /// <param name="Permutation">Pivot permutation array — row i of the permuted matrix corresponds to row Permutation[i] of the original.</param>
    /// <param name="Rank">Numerical rank (number of pivots accepted before tolerance was reached).</param>
    public PivotedCholeskyResult(T[,] Lower, int[] Permutation, int Rank)
    {
        _lower = (T[,])(Lower ?? throw new ArgumentNullException(nameof(Lower))).Clone();
        _permutation = (int[])(Permutation ?? throw new ArgumentNullException(nameof(Permutation))).Clone();
        this.Rank = Rank;
    }

    /// <summary>Lower-triangular factor (N×Rank).</summary>
    public T[,] Lower
    {
        get => (T[,])_lower.Clone();
        init => _lower = (T[,])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>Pivot permutation array.</summary>
    public int[] Permutation
    {
        get => (int[])_permutation.Clone();
        init => _permutation = (int[])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>Numerical rank.</summary>
    public int Rank { get; init; }

    /// <summary>Deconstructs this result into defensive snapshots and its rank.</summary>
    public void Deconstruct(out T[,] Lower, out int[] Permutation, out int Rank)
    {
        Lower = (T[,])_lower.Clone();
        Permutation = (int[])_permutation.Clone();
        Rank = this.Rank;
    }
}

/// <summary>
/// Result of a pivoted Cholesky decomposition: P^T·A·P = L·L^T + E,
/// where E is the trailing Schur-complement residual (up to rounding) and L has
/// <see cref="Rank"/> columns. E vanishes for a complete exact-rank factorization in exact arithmetic.
/// </summary>
public sealed record PivotedCholeskyResult
{
    private decimal[,] _lower;
    private int[] _permutation;

    /// <summary>Initializes a result with owned copies of the factor and permutation.</summary>
    /// <param name="Lower">Lower-triangular factor (N×Rank).</param>
    /// <param name="Permutation">Pivot permutation array — row i of the permuted matrix corresponds to row Permutation[i] of the original.</param>
    /// <param name="Rank">Numerical rank (number of pivots accepted before tolerance was reached).</param>
    public PivotedCholeskyResult(decimal[,] Lower, int[] Permutation, int Rank)
    {
        _lower = (decimal[,])(Lower ?? throw new ArgumentNullException(nameof(Lower))).Clone();
        _permutation = (int[])(Permutation ?? throw new ArgumentNullException(nameof(Permutation))).Clone();
        this.Rank = Rank;
    }

    /// <summary>Lower-triangular factor (N×Rank).</summary>
    public decimal[,] Lower
    {
        get => (decimal[,])_lower.Clone();
        init => _lower = (decimal[,])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>Pivot permutation array.</summary>
    public int[] Permutation
    {
        get => (int[])_permutation.Clone();
        init => _permutation = (int[])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>Numerical rank.</summary>
    public int Rank { get; init; }

    /// <summary>Deconstructs this result into defensive snapshots and its rank.</summary>
    public void Deconstruct(out decimal[,] Lower, out int[] Permutation, out int Rank)
    {
        Lower = (decimal[,])_lower.Clone();
        Permutation = (int[])_permutation.Clone();
        Rank = this.Rank;
    }
}

/// <summary>
/// Cholesky factorization <c>A = L · Lᵀ</c> for symmetric positive-definite
/// (SPD) matrices, with a diagonally pivoted variant <c>PᵀAP = LLᵀ + E</c> for
/// symmetric positive-semidefinite (PSD) inputs. The residual E may be nonzero
/// when a tolerance or rank limit truncates the factorization.
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
/// The pivoted variant (Higham 1990; the existing Var baseline discussed in arXiv:2507.20678) performs
/// diagonal pivoting — at each step the column with the largest residual
/// diagonal (the Schur-complement diagonal) is selected. This gives a
/// threshold-dependent rank estimate: the largest remaining diagonal is compared
/// with an absolute cutoff. The default cutoff is n·u times the original largest
/// diagonal (the first pivot); an explicit tolerance supplies the absolute cutoff.
/// Higham (1990, §6) gives numerically singular counterexamples whose small singular
/// values are not revealed by these pivots. Early termination produces L with
/// only <c>Rank</c> columns and a generally nonzero trailing Schur-complement residual,
/// so <c>P^T A P = L L^T + E</c> up to rounding. A
/// running <c>diagRemaining</c> vector tracks the Schur-complement diagonals
/// without recomputing them from scratch each step, keeping the operation
/// count at O(N³/3) despite the pivot search.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class CholeskyDecomposition<T>
    where T : IFloatingPoint<T>
{
    // Below this magnitude a factor diagonal is treated as zero in Solve. A factor produced by
    // Decompose is always strictly positive on the diagonal, so this only fires for an externally
    // supplied rank-deficient factor — where dividing by it would yield a silent NaN/Infinity.
    private static readonly T s_solveDiagonalFloor = T.CreateChecked(1e-300) == T.Zero
        ? T.Zero
        : T.CreateChecked(1e-300);

    private static readonly T s_two = T.CreateChecked(2);

    // Machine epsilon for T: the smallest positive value eps such that (T.One + eps) != T.One.
    // Half this spacing is the round-to-nearest unit roundoff u in Higham's
    // (1990) default pivoted-Cholesky stopping tolerance (see DefaultTolerance below).
    private static readonly T s_machineEpsilon = ComputeMachineEpsilon();

    // decimal does not float below magnitude one: it rounds to an absolute grid of 1e-28, its smallest positive value,
    // where the relative unit roundoff no longer bounds a rounding error. DefaultTolerance adds n units of that grid,
    // so it never rounds to an absolute-zero cutoff. IEEE types keep Higham's relative rule unchanged (zero here).
    private static readonly T s_absoluteGrid = typeof(T) == typeof(decimal)
        ? T.CreateChecked(new decimal(1, 0, 0, false, 28))
        : T.Zero;

    private static T ComputeMachineEpsilon()
    {
        var eps = T.One;
        while (T.One + eps / s_two != T.One)
        {
            eps /= s_two;
        }

        return eps;
    }

    private static void ThrowIfSingularDiagonal(T diagonal, int index)
    {
        if (T.Abs(diagonal) <= s_solveDiagonalFloor)
        {
            throw new InvalidOperationException(
                $"Cholesky factor has a zero (or near-zero) diagonal at index {index} (value {diagonal}); " +
                "the factor is rank-deficient. Pass a factor produced by Decompose on a positive-definite matrix.");
        }
    }

    /// <summary>
    /// Computes the standard (unpivoted) Cholesky decomposition A = L·Lᵀ.
    /// </summary>
    /// <param name="a">Symmetric positive-definite N×N matrix (only the lower triangle is read).</param>
    /// <returns>Lower-triangular factor L.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="a"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">Matrix is not square or its lower triangle contains a non-finite entry.</exception>
    /// <exception cref="InvalidOperationException">Matrix is not positive definite or factorization produces a non-finite value.</exception>
    public static T[,] Decompose(T[,] a)
    {
        ArgumentNullException.ThrowIfNull(a);

        var n = a.GetLength(0);
        if (a.GetLength(1) != n)
        {
            throw new ArgumentException("Matrix must be square.", nameof(a));
        }

        // A non-positive-definite input can overflow the factor for decimal, which raises
        // OverflowException where IEEE types reach the non-finite guards below; both surface as
        // the documented InvalidOperationException.
        try
        {
            return DecomposeCore(a, n);
        }
        catch (OverflowException ex)
        {
            throw new InvalidOperationException(
                "Matrix is not positive definite: the Cholesky factorization overflowed.", ex);
        }
    }

    // The caller has proved that the leading principal block is byte-identical to the one already factored.
    // Reusing its columns preserves the arithmetic order of a fresh factorisation, including all new rows.
    internal static T[,] DecomposeWithPrefix(T[,] a, T[,] previous, int prefix)
    {
        try
        {
            return DecomposeCore(a, a.GetLength(0), previous, prefix);
        }
        catch (OverflowException ex)
        {
            throw new InvalidOperationException("Matrix is not positive definite: the Cholesky factorization overflowed.", ex);
        }
    }

    private static T[,] DecomposeCore(T[,] a, int n, T[,]? previous = null, int prefix = 0)
    {
        var lower = new T[n, n];
        for (var i = 0; i < prefix; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                lower[i, j] = previous![i, j];
            }
        }

        for (var j = 0; j < n; j++)
        {
            if (j >= prefix)
            {
                if (!T.IsFinite(a[j, j]))
                {
                    throw new ArgumentException("Matrix contains a non-finite entry.", nameof(a));
                }

                var sum = T.Zero;
                for (var k = 0; k < j; k++)
                {
                    sum += lower[j, k] * lower[j, k];
                }

                var diag = a[j, j] - sum;
                if (!T.IsFinite(diag) || diag <= T.Zero)
                {
                    throw new InvalidOperationException(
                        $"Matrix is not positive definite (non-positive diagonal {diag} at index {j}).");
                }

                lower[j, j] = NumericPrecision<T>.Sqrt(diag);
            }

            for (var i = Math.Max(j + 1, prefix); i < n; i++)
            {
                if (!T.IsFinite(a[i, j]))
                {
                    throw new ArgumentException("Matrix contains a non-finite entry.", nameof(a));
                }

                var sum = T.Zero;
                for (var k = 0; k < j; k++)
                {
                    sum += lower[i, k] * lower[j, k];
                }

                lower[i, j] = (a[i, j] - sum) / lower[j, j];
                if (!T.IsFinite(lower[i, j]))
                {
                    throw new InvalidOperationException("Cholesky factor contains a non-finite entry.");
                }
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
    /// <exception cref="ArgumentNullException"><paramref name="lower"/> or <paramref name="b"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException"><paramref name="lower"/> is not square, vector length does not match matrix dimension, or an input is non-finite.</exception>
    /// <exception cref="InvalidOperationException">The factor has a zero (or near-zero) diagonal entry, or substitution produces a non-finite result.</exception>
    public static T[] Solve(T[,] lower, T[] b)
    {
        try
        {
            return SolveCore(lower, b);
        }
        catch (OverflowException exception)
        {
            throw new InvalidOperationException("Cholesky solve exceeded the numeric range.", exception);
        }
    }

    private static T[] SolveCore(T[,] lower, T[] b)
    {
        ArgumentNullException.ThrowIfNull(lower);
        ArgumentNullException.ThrowIfNull(b);

        var n = lower.GetLength(0);
        if (lower.GetLength(1) != n)
        {
            throw new ArgumentException("Cholesky factor must be square.", nameof(lower));
        }

        if (b.Length != n)
        {
            throw new ArgumentException("Vector length must match matrix dimension.", nameof(b));
        }

        for (var i = 0; i < n; i++)
        {
            if (!T.IsFinite(b[i]))
            {
                throw new ArgumentException("Right-hand side contains a non-finite value.", nameof(b));
            }

            for (var j = 0; j <= i; j++)
            {
                if (!T.IsFinite(lower[i, j]))
                {
                    throw new ArgumentException("Cholesky factor contains a non-finite entry.", nameof(lower));
                }
            }
        }

        // Forward substitution: L·y = b. A zero (or near-zero) factor diagonal means the supplied
        // factor is rank-deficient — the back/forward substitution would divide by zero and return
        // a silent NaN/Infinity solution. Guard each diagonal and throw a clear exception instead.
        var y = new T[n];
        for (var i = 0; i < n; i++)
        {
            ThrowIfSingularDiagonal(lower[i, i], i);

            y[i] = SubstituteRow(lower, b[i], y, i, 0, i, false);
            if (!T.IsFinite(y[i]))
            {
                throw new InvalidOperationException("Cholesky solve produced a non-finite forward result.");
            }
        }

        // Back substitution: Lᵀ·x = y.
        var x = new T[n];
        for (var i = n - 1; i >= 0; i--)
        {
            x[i] = SubstituteRow(lower, y[i], x, i, i + 1, n, true);
            if (!T.IsFinite(x[i]))
            {
                throw new InvalidOperationException("Cholesky solve produced a non-finite solution.");
            }
        }

        return x;
    }

    private static T SubstituteRow(T[,] lower, T rhs, T[] solution, int row, int start, int end, bool transpose)
    {
        var sum = rhs;
        var overflowed = false;
        try
        {
            for (var j = start; j < end; j++)
            {
                sum -= (transpose ? lower[j, row] : lower[row, j]) * solution[j];
            }
        }
        catch (OverflowException)
        {
            // Retry a decimal intermediate overflow in scaled term units.
            overflowed = true;
        }

        if (!overflowed && T.IsFinite(sum))
        {
            return sum / lower[row, row];
        }

        return SubstituteRowScaled(lower, rhs, solution, row, start, end, transpose);
    }

    // Retry an overflowing row with the right-hand side and the factor entries divided by a common
    // power of the radix R (exact for every IFloatingPoint type: 10 for decimal, 2 for binary), the
    // smallest R for which every term (L[row, j] / R) * x[j] is finite. The terms are then divided by
    // their own largest magnitude so every addend lies in [-1, 1] however large the cancelling terms
    // are. Dividing each factor entry by the row's largest ENTRY instead (the right-hand side usually)
    // rounds unit-sized entries to zero on decimal's absolute grid or to subnormals in IEEE, which
    // either refuses a representable solution or silently drops terms from the sum.
    private static T SubstituteRowScaled(
        T[,] lower, T rhs, T[] solution, int row, int start, int end, bool transpose)
    {
        var radix = T.CreateChecked(T.Radix);
        var divisor = T.One;
        var terms = new T[end - start];
        while (true)
        {
            if (TryScaledRow(lower, rhs, solution, row, start, end, transpose, divisor, terms, out var scaled))
            {
                // The scaled row is in range, so the restored value leaves it only when the
                // solution itself is outside the working range.
                return scaled * divisor;
            }

            var next = divisor * radix;
            if (!T.IsFinite(next))
            {
                throw new OverflowException("Cholesky row cannot be evaluated within the numeric range.");
            }

            divisor = next;
        }
    }

    private static bool TryScaledRow(
        T[,] lower,
        T rhs,
        T[] solution,
        int row,
        int start,
        int end,
        bool transpose,
        T divisor,
        T[] terms,
        out T scaled)
    {
        scaled = T.Zero;
        try
        {
            var rhsUnits = rhs / divisor;
            var magnitude = T.Abs(rhsUnits);
            for (var j = start; j < end; j++)
            {
                var term = ((transpose ? lower[j, row] : lower[row, j]) / divisor) * solution[j];
                if (!T.IsFinite(term))
                {
                    return false;
                }

                terms[j - start] = term;
                magnitude = T.Max(magnitude, T.Abs(term));
            }

            if (magnitude == T.Zero)
            {
                return true;
            }

            var normalized = rhsUnits / magnitude;
            for (var j = start; j < end; j++)
            {
                normalized -= terms[j - start] / magnitude;
            }

            return TryScaledQuotient(normalized, magnitude, lower[row, row], out scaled);
        }
        catch (OverflowException)
        {
            // Decimal throws where IEEE arithmetic produces infinity; retry with a larger divisor.
            return false;
        }
    }

    // normalized * magnitude / diagonal in whichever association stays in range: the product first,
    // then the quotient first when the product leaves the range of a large magnitude.
    private static bool TryScaledQuotient(T normalized, T magnitude, T diagonal, out T quotient)
    {
        quotient = T.Zero;
        try
        {
            quotient = normalized * magnitude / diagonal;
            if (T.IsFinite(quotient))
            {
                return true;
            }
        }
        catch (OverflowException)
        {
            // Retry below with the quotient formed first.
        }

        try
        {
            quotient = normalized / diagonal * magnitude;
            return T.IsFinite(quotient);
        }
        catch (OverflowException)
        {
            return false;
        }
    }

    /// <summary>
    /// Convenience overload using Higham's (1990) recommended default stopping tolerance
    /// (the relative test <c>ε·â₁₁⁽¹⁾</c> with <c>ε = n·u</c>, eq. 4.3 — see
    /// <c>DefaultTolerance</c>; at <see cref="decimal"/> it also adds <c>n · 1e-28</c>, decimal's
    /// absolute grid, so it never rounds to zero) and no rank limit.
    /// </summary>
    public static PivotedCholeskyResult<T> DecomposePivoted(T[,] a)
        => DecomposePivoted(a, DefaultTolerance(a), null);

    /// <summary>
    /// Computes Higham's (1990) recommended default stopping tolerance for the pivoted
    /// decomposition: the relative test <c>ε·â₁₁⁽¹⁾</c> with <c>ε = n·u</c> (eq. 4.3), where
    /// <c>u</c> is half <typeparamref name="T"/>'s spacing at one (round-to-nearest unit roundoff) and <c>â₁₁⁽¹⁾</c> is the first
    /// (largest) pivot — the maximum diagonal entry of the ORIGINAL matrix, since pivoted
    /// Cholesky always selects the largest remaining diagonal first. An absolute, zero-default
    /// tolerance (the prior default) admits floating-point rounding residue on a rank-deficient
    /// input as a spurious extra pivot; see the <c>DecomposePivoted(T[,])</c> overload. At
    /// <see cref="decimal"/>, which rounds to an absolute grid of <c>1e-28</c> below magnitude one,
    /// <c>n · u · max(diag(a))</c> can lie below that grid and round to zero; the default therefore
    /// adds <c>n · 1e-28</c> there, so it never becomes an absolute-zero cutoff.
    /// </summary>
    /// <param name="a">Symmetric positive semi-definite N-by-N matrix.</param>
    /// <returns>
    /// <c>n · u · max(diag(a))</c> (plus <c>n · 1e-28</c> at <see cref="decimal"/>), or <c>T.Zero</c>
    /// when <paramref name="a"/> is not square — the caller's own validation then reports the
    /// correct exception.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="a"/> is <see langword="null"/>.</exception>
    internal static T DefaultTolerance(T[,] a)
    {
        ArgumentNullException.ThrowIfNull(a);

        var n = a.GetLength(0);
        if (a.GetLength(1) != n)
        {
            // Not square: let DecomposePivoted's own validation raise the correct exception.
            return T.Zero;
        }

        var maxDiag = T.Zero;
        for (var i = 0; i < n; i++)
        {
            if (T.IsFinite(a[i, i]) && a[i, i] > maxDiag)
            {
                maxDiag = a[i, i];
            }
        }

        var relative = (T.CreateChecked(n) * s_machineEpsilon / s_two) * maxDiag;
        return s_absoluteGrid == T.Zero ? relative : relative + (T.CreateChecked(n) * s_absoluteGrid);
    }

    /// <summary>
    /// Computes the pivoted Cholesky decomposition selecting the largest remaining diagonal at each step.
    /// Supports positive semi-definite matrices and early termination.
    /// </summary>
    /// <param name="a">Symmetric positive semi-definite N-by-N matrix.</param>
    /// <param name="tolerance">Diagonal values below this threshold are treated as zero.</param>
    /// <param name="maxRank">Maximum number of pivots (columns of L). Null means no limit.</param>
    /// <returns>Lower-triangular factor, permutation array, and numerical rank.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="a"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">Matrix is not square or contains a non-finite entry.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxRank"/> is negative.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="tolerance"/> is non-finite.</exception>
    /// <exception cref="InvalidOperationException">Factorization produces a non-finite value.</exception>
    public static PivotedCholeskyResult<T> DecomposePivoted(
        T[,] a, T tolerance, int? maxRank)
    {
        ArgumentNullException.ThrowIfNull(a);

        var tol = tolerance;
        if (!T.IsFinite(tol))
        {
            throw new ArgumentOutOfRangeException(nameof(tolerance), tolerance, "Tolerance must be finite.");
        }

        var n = a.GetLength(0);
        if (a.GetLength(1) != n)
        {
            throw new ArgumentException("Matrix must be square.", nameof(a));
        }

        var rankLimit = maxRank ?? n;
        if (rankLimit < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxRank), maxRank, "Maximum rank cannot be negative.");
        }

        // Work on a copy.
        var work = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                if (!T.IsFinite(a[i, j]))
                {
                    throw new ArgumentException("Matrix contains a non-finite entry.", nameof(a));
                }

                work[i, j] = a[i, j];
            }
        }

        // Same overflow-to-contract mapping as Decompose: an indefinite decimal input overflows
        // the factor or the Schur update before the non-finite guards can fire.
        try
        {
            return DecomposePivotedCore(work, n, tol, rankLimit);
        }
        catch (OverflowException ex)
        {
            throw new InvalidOperationException(
                "Pivoted Cholesky factorization overflowed: the matrix is not positive semi-definite.", ex);
        }
    }

    private static PivotedCholeskyResult<T> DecomposePivotedCore(T[,] work, int n, T tol, int rankLimit)
    {
        var perm = new int[n];
        for (var i = 0; i < n; i++)
        {
            perm[i] = i;
        }

        var lower = new T[n, Math.Min(n, rankLimit)];
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

            if (!T.IsFinite(maxDiag))
            {
                throw new InvalidOperationException("Schur-complement diagonal is non-finite.");
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

            // Standard Cholesky step on column j. The accepted pivot diagonal is the maintained
            // Schur-complement value diagRemaining[j] (already net of prior column contributions),
            // NOT work[j,j] − Σ lower[j,k]², which re-derives it from the UN-permuted original
            // diagonal. After the pivot swap above diagRemaining[j] == maxDiag (the largest
            // remaining Schur diagonal); reading work[j,j] corrupts the factor whenever the pivot
            // order is non-identity, because the chosen column's original diagonal differs from its
            // Schur-reduced value. Apply the same value to the diag<=0 positive-definiteness guard.
            var diag = diagRemaining[j];
            if (diag <= T.Zero)
            {
                break;
            }

            lower[j, j] = NumericPrecision<T>.Sqrt(diag);

            for (var i = j + 1; i < n; i++)
            {
                var sum = T.Zero;
                for (var k = 0; k < j; k++)
                {
                    sum += lower[i, k] * lower[j, k];
                }

                lower[i, j] = (work[i, j] - sum) / lower[j, j];
                if (!T.IsFinite(lower[i, j]))
                {
                    throw new InvalidOperationException("Cholesky factor contains a non-finite entry.");
                }
            }

            // Update remaining diagonals for next pivot selection.
            for (var i = j + 1; i < n; i++)
            {
                diagRemaining[i] -= lower[i, j] * lower[i, j];
                if (!T.IsFinite(diagRemaining[i]))
                {
                    throw new InvalidOperationException("Schur-complement diagonal is non-finite.");
                }
            }

            rank++;
        }

        if (rank != lower.GetLength(1))
        {
            var trimmed = new T[n, rank];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < rank; j++)
                {
                    trimmed[i, j] = lower[i, j];
                }
            }

            lower = trimmed;
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
[Obsolete(
    "Use CholeskyDecomposition<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
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
    /// <param name="a">Symmetric positive semi-definite N-by-N matrix.</param>
    /// <param name="tolerance">
    /// Diagonal values below this threshold are treated as zero. The default, <c>0m</c>, requests
    /// Higham's (1990) recommended relative stopping rule (eq. 4.3: <c>ε·â₁₁⁽¹⁾</c> with <c>ε = n·u</c>
    /// — see <see cref="CholeskyDecomposition{T}.DecomposePivoted(T[,])"/>) rather than a literal
    /// absolute-zero cutoff, which admits floating-point rounding residue on a rank-deficient input
    /// as a spurious extra pivot. Because decimal rounds to an absolute grid of <c>1e-28</c> below
    /// magnitude one, the default adds <c>n · 1e-28</c> to that rule, so it never rounds to zero on a
    /// small-scale matrix. Pass an explicit positive value for a literal absolute cutoff.
    /// </param>
    /// <param name="maxRank">Maximum number of pivots (columns of L). Null means no limit.</param>
    public static PivotedCholeskyResult DecomposePivoted(
        decimal[,] a, decimal tolerance = 0m, int? maxRank = null)
    {
        var tol = tolerance == 0m ? CholeskyDecomposition<decimal>.DefaultTolerance(a) : tolerance;
        var result = CholeskyDecomposition<decimal>.DecomposePivoted(a, tol, maxRank);
        var lower = result.Lower;
        var permutation = result.Permutation;
        return new PivotedCholeskyResult(lower, permutation, result.Rank);
    }
}
