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

namespace Boutquin.Numerics.LinearAlgebra.Internal;

/// <summary>
/// Compact Householder QR decomposition for dense overdetermined systems.
/// Factors <c>A (m x n)</c> into <c>Q . R</c> where <c>Q</c> is orthogonal and
/// <c>R</c> is upper-triangular, storing the reflectors implicitly so that a
/// single pass computes both the decomposition and the transformed right-hand
/// side <c>c = Qt . y</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Square-root operations use
/// <see cref="NumericPrecision{T}.Sqrt"/> to support <c>decimal</c> (which lacks
/// <see cref="IRootFunctions{TSelf}"/>).
/// </para>
/// <para>
/// Used internally by <see cref="Boutquin.Numerics.Solvers.OrdinaryLeastSquares{T}"/>. Kept
/// internal until a second caller emerges; when it does, this type may be promoted to
/// public API under a dedicated spec.
/// </para>
/// <para>
/// References: Golub &amp; Van Loan, <i>Matrix Computations</i>, 4th ed., Algorithm 5.2.1
/// (Householder QR) and section 5.3.2 (QR-based least squares). The per-column reflector
/// follows LAPACK's <c>DLARFG/DGEQRF</c> convention:
/// <list type="bullet">
/// <item><description><c>alpha = -sign(x0) . ||x||2</c></description></item>
/// <item><description><c>v = (x0 - alpha, x1, ..., x(m-k-1))</c></description></item>
/// <item><description><c>tau = 2 / (vtv)</c></description></item>
/// <item><description><c>H = I - tau . v . vt</c></description></item>
/// </list>
/// Each reflector is applied in-place to <c>A</c>'s trailing columns and to <c>y</c>.
/// After <c>n</c> reflectors, <c>A</c>'s upper-triangular block holds <c>R</c> and
/// <c>y[0..n-1]</c> holds the leading block of <c>c = Qty</c>, which is all that's
/// needed for least-squares back-substitution.
/// </para>
/// <para>
/// Stability: preserves the conditioning of <c>A</c> exactly, unlike normal-equation
/// approaches that form <c>AtA</c> and double the condition number. This matters for
/// NIST StRD's <c>Wampler4</c> (cond ~ 5e10) and <c>Wampler5</c> (cond ~ 5e13) where
/// the normal-equation approach loses the response signal entirely while QR recovers
/// all ten published digits.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
internal static class HouseholderQr<T>
    where T : IFloatingPoint<T>
{
    private static readonly T s_two = T.CreateChecked(2);

    /// <summary>
    /// Factors <paramref name="a"/> in-place as <c>Q . R . Pt</c> with column pivoting,
    /// transforms <paramref name="y"/> in-place to <c>Qt . y</c>, and records the column
    /// permutation <paramref name="permutation"/>. On return, <paramref name="a"/>
    /// contains the upper-triangular <c>R</c> in its leading <c>n x n</c> block and the
    /// Householder reflectors in its strict lower triangle; the first <c>n</c> entries
    /// of <paramref name="y"/> are the coefficients needed for solving <c>R . beta = c</c>,
    /// and the remaining entries <c>y[n..m-1]</c> are the orthogonal residual
    /// components (their sum of squares is the residual sum of squares).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Column pivoting (LAPACK DGEQP3 style) selects the trailing-submatrix column with
    /// the largest 2-norm at each step and swaps it into the pivot position. This turns
    /// Householder QR into a rank-revealing decomposition and is essential for polynomial
    /// designs such as NIST's <c>Filip</c> (degree 10 with predictor range 0.5-10) and
    /// <c>Wampler4/5</c> (degree 5, cond ~ 1e10-1e13), where unpivoted QR loses 4-6
    /// digits and rank-revealing QR keeps 10+. Callers who care about the pivot order
    /// pass a <paramref name="permutation"/> buffer of length <c>n</c> initialised as
    /// the identity <c>[0, 1, ..., n-1]</c>; callers who don't can pass any length-<c>n</c>
    /// buffer and ignore the output.
    /// </para>
    /// </remarks>
    /// <param name="a">Design matrix, dimensions <c>m x n</c>. Overwritten.</param>
    /// <param name="y">Response vector, length <c>m</c>. Overwritten.</param>
    /// <param name="diagonal">Output buffer of length <c>n</c> receiving the diagonal of <c>R</c> (alpha values).</param>
    /// <param name="permutation">
    /// Output buffer of length <c>n</c>. After <see cref="Factor"/>, <c>permutation[k]</c>
    /// is the original column of <paramref name="a"/> that now occupies position <c>k</c>.
    /// </param>
    /// <returns>
    /// <see langword="true"/> when every column of <c>A</c> yielded a non-degenerate reflector;
    /// <see langword="false"/> if the trailing submatrix at some step has a zero column norm
    /// (the matrix is rank-deficient and OLS is ill-defined). Callers surface this as an
    /// argument or invalid-operation exception rather than a degenerate result.
    /// </returns>
    public static bool Factor(T[,] a, T[] y, T[] diagonal, int[] permutation)
    {
        var m = a.GetLength(0);
        var n = a.GetLength(1);

        for (var k = 0; k < n; k++)
        {
            permutation[k] = k;
        }

        for (var k = 0; k < n; k++)
        {
            // Column pivoting: among columns k..n-1, find the one with the largest
            // 2-norm restricted to rows k..m-1, and swap it into position k.
            var pivot = k;
            var pivotNormSq = ColumnNormSquared(a, k, k, m);
            for (var j = k + 1; j < n; j++)
            {
                var candidate = ColumnNormSquared(a, j, k, m);
                if (candidate > pivotNormSq)
                {
                    pivotNormSq = candidate;
                    pivot = j;
                }
            }

            if (pivot != k)
            {
                SwapColumns(a, k, pivot, m);
                (permutation[k], permutation[pivot]) = (permutation[pivot], permutation[k]);
            }

            // Compute ||x||2 where x = a[k..m-1, k]. Use two-pass accumulation to keep
            // the norm numerically sound when the column spans many orders of magnitude
            // (Longley has entries from 1e0 to 1e5).
            var xMax = T.Zero;
            for (var i = k; i < m; i++)
            {
                var abs = T.Abs(a[i, k]);
                if (abs > xMax)
                {
                    xMax = abs;
                }
            }

            if (xMax == T.Zero)
            {
                diagonal[k] = T.Zero;
                return false;
            }

            var sumSquares = T.Zero;
            for (var i = k; i < m; i++)
            {
                var scaled = a[i, k] / xMax;
                sumSquares += scaled * scaled;
            }

            var xNorm = xMax * NumericPrecision<T>.Sqrt(sumSquares);

            // alpha = -sign(x0) . ||x||. A zero x0 takes the conventional +||x|| to avoid a
            // zero-signum branch; either sign is mathematically admissible, but the
            // sign-flip convention keeps v's leading component far from cancellation.
            var x0 = a[k, k];
            var alpha = !T.IsNegative(x0) ? -xNorm : xNorm;
            diagonal[k] = alpha;

            // v = (x0 - alpha, x1, ..., x(m-k-1)) stored in-place over column k. Skip the
            // reflector entirely when v is numerically zero (column already aligned
            // with the axis).
            var v0 = x0 - alpha;
            a[k, k] = v0;

            var vtv = v0 * v0;
            for (var i = k + 1; i < m; i++)
            {
                vtv += a[i, k] * a[i, k];
            }

            if (vtv == T.Zero)
            {
                continue;
            }

            // Guard against numeric types (decimal) whose range is too narrow to represent
            // tau = 2/vtv. Near-zero vtv arises from near-collinear columns after one or
            // more Householder reflections: the residual column is nearly in the span of
            // previous columns, so the design matrix is effectively rank-deficient.
            // Declaring rank deficiency here is correct and prevents OverflowException.
            // T.MaxValue is not available via IFloatingPoint<T>; use decimal.MaxValue as
            // a type-portable upper bound (≈7.9e28) — safe for double and exact for decimal.
            var tauThreshold = s_two / T.CreateChecked(decimal.MaxValue);
            if (vtv < tauThreshold)
            {
                diagonal[k] = T.Zero;
                return false;
            }

            var tau = s_two / vtv;

            // Apply H = I - tau.v.vt to trailing columns a[k..m-1, j] for j = k+1..n-1.
            for (var j = k + 1; j < n; j++)
            {
                var vDotA = v0 * a[k, j];
                for (var i = k + 1; i < m; i++)
                {
                    vDotA += a[i, k] * a[i, j];
                }

                var scale = tau * vDotA;
                a[k, j] -= scale * v0;
                for (var i = k + 1; i < m; i++)
                {
                    a[i, j] -= scale * a[i, k];
                }
            }

            // Apply the same reflector to y.
            var vDotY = v0 * y[k];
            for (var i = k + 1; i < m; i++)
            {
                vDotY += a[i, k] * y[i];
            }

            var scaleY = tau * vDotY;
            y[k] -= scaleY * v0;
            for (var i = k + 1; i < m; i++)
            {
                y[i] -= scaleY * a[i, k];
            }
        }

        return true;
    }

    /// <summary>
    /// Solves <c>R . x = c</c> where <c>R</c>'s strict upper triangle is stored in the
    /// upper triangle of <paramref name="a"/> and its diagonal is <paramref name="diagonal"/>.
    /// The first <paramref name="n"/> entries of <paramref name="c"/> carry the right-hand
    /// side. Returns the solution in the permuted order dictated by the QR's column
    /// pivoting; the caller un-permutes via <paramref name="permutation"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException">Any diagonal entry is zero (rank-deficient R).</exception>
    public static T[] SolveUpperTriangular(T[,] a, T[] diagonal, T[] c, int n, int[] permutation)
    {
        var permuted = new T[n];
        for (var i = n - 1; i >= 0; i--)
        {
            var sum = c[i];
            for (var j = i + 1; j < n; j++)
            {
                sum -= a[i, j] * permuted[j];
            }

            var dii = diagonal[i];
            if (dii == T.Zero)
            {
                throw new InvalidOperationException(
                    $"Upper-triangular R has a zero diagonal at row {i}; design matrix is rank-deficient.");
            }

            permuted[i] = sum / dii;
        }

        // Un-permute: x[permutation[k]] = permuted[k].
        var x = new T[n];
        for (var k = 0; k < n; k++)
        {
            x[permutation[k]] = permuted[k];
        }

        return x;
    }

    /// <summary>
    /// Computes <c>(XtX)^-1</c> in the original (un-permuted) coordinate system from the
    /// QR decomposition's permuted <c>R</c> factor. This is the un-scaled covariance of
    /// the OLS estimator — multiply by the residual variance <c>sigma-hat-sq</c> to recover
    /// <c>Cov(beta-hat) = sigma-hat-sq . (XtX)^-1</c>.
    /// </summary>
    /// <remarks>
    /// Let <c>X . P = Q . R</c> where <c>P</c> is the column permutation. Then
    /// <c>XtX = P . RtR . Pt</c> and <c>(XtX)^-1 = P . (RtR)^-1 . Pt</c>. We build
    /// <c>(RtR)^-1 = R^-1 . R^-t</c> in the permuted frame, then un-permute both axes.
    /// </remarks>
    /// <param name="a">QR-factored matrix (upper triangle holds <c>R</c> above the diagonal).</param>
    /// <param name="diagonal">Diagonal of <c>R</c> (length <c>n</c>).</param>
    /// <param name="permutation">Column permutation from <see cref="Factor"/>.</param>
    /// <returns>Symmetric <c>n x n</c> matrix equal to <c>(XtX)^-1</c> in original column order.</returns>
    public static T[,] BuildXtXInverse(T[,] a, T[] diagonal, int[] permutation)
    {
        var n = diagonal.Length;

        // Invert R by column-by-column back-substitution of R . Z = I.
        var rInverse = new T[n, n];
        try
        {
            for (var col = 0; col < n; col++)
            {
                for (var i = n - 1; i >= 0; i--)
                {
                    var sum = i == col ? T.One : T.Zero;
                    for (var j = i + 1; j < n; j++)
                    {
                        sum -= a[i, j] * rInverse[j, col];
                    }

                    var dii = diagonal[i];
                    if (dii == T.Zero)
                    {
                        throw new InvalidOperationException(
                            $"Upper-triangular R has a zero diagonal at row {i}; design matrix is rank-deficient.");
                    }

                    rInverse[i, col] = sum / dii;
                }
            }

            // Compute (RtR)^-1 = R^-1 . R^-t in the permuted frame.
            var inversePermuted = new T[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = i; j < n; j++)
                {
                    var dot = T.Zero;
                    // Math.Max(i, j) is int max, not T max — j >= i always here, so k starts at j.
                    for (var k = Math.Max(i, j); k < n; k++)
                    {
                        dot += rInverse[i, k] * rInverse[j, k];
                    }

                    inversePermuted[i, j] = dot;
                    inversePermuted[j, i] = dot;
                }
            }

            // Un-permute: (XtX)^-1[permutation[i], permutation[j]] = inversePermuted[i, j].
            var result = new T[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    result[permutation[i], permutation[j]] = inversePermuted[i, j];
                }
            }

            return result;
        }
        catch (OverflowException)
        {
            // An arithmetic overflow during R^{-1} computation means the R diagonal
            // contains a value so small that 1/R[k,k] exceeds the numeric type's
            // representable range (e.g., decimal's ~7.9e28). This signals that the
            // design matrix is numerically rank-deficient for this numeric type — the
            // normal equations are ill-conditioned beyond what the type can represent.
            throw new InvalidOperationException(
                "R-inverse computation overflowed; design matrix is numerically rank-deficient. " +
                "The column space may contain nearly collinear columns.");
        }
    }

    /// <summary>
    /// Applies <c>Qt</c> (the orthogonal factor from <see cref="Factor"/>) in-place to
    /// <paramref name="r"/>. Re-traces the stored reflectors; after this call,
    /// <paramref name="r"/>[0..n-1] holds the coefficients needed for
    /// <see cref="SolveUpperTriangular"/> and <paramref name="r"/>[n..m-1] holds the
    /// residual components orthogonal to the column space.
    /// </summary>
    /// <param name="a">QR-factored design matrix from <see cref="Factor"/> (strict lower triangle holds the reflectors).</param>
    /// <param name="diagonal">Diagonal of <c>R</c> (length <c>n</c>).</param>
    /// <param name="r">Vector of length <c>m</c>. Overwritten with <c>Qt . r</c>.</param>
    public static void ApplyQTranspose(T[,] a, T[] diagonal, T[] r)
    {
        var m = a.GetLength(0);
        var n = diagonal.Length;

        for (var k = 0; k < n; k++)
        {
            // Reflector H_k: v[0] stored in a[k, k]; v[1..m-k-1] stored in a[k+1..m-1, k].
            // alpha = diagonal[k]. tau = 2/(vtv). Skip when v is effectively zero.
            var v0 = a[k, k];

            var vtv = v0 * v0;
            for (var i = k + 1; i < m; i++)
            {
                vtv += a[i, k] * a[i, k];
            }

            if (vtv == T.Zero)
            {
                continue;
            }

            var tau = s_two / vtv;
            var vDotR = v0 * r[k];
            for (var i = k + 1; i < m; i++)
            {
                vDotR += a[i, k] * r[i];
            }

            var scale = tau * vDotR;
            r[k] -= scale * v0;
            for (var i = k + 1; i < m; i++)
            {
                r[i] -= scale * a[i, k];
            }
        }
    }

    private static T ColumnNormSquared(T[,] a, int col, int rowStart, int rowEnd)
    {
        // Two-pass scaled sum-of-squares guards against overflow on polynomial designs
        // whose later columns (x^10 on NIST Filip) span 11 orders of magnitude.
        var xMax = T.Zero;
        for (var i = rowStart; i < rowEnd; i++)
        {
            var abs = T.Abs(a[i, col]);
            if (abs > xMax)
            {
                xMax = abs;
            }
        }

        if (xMax == T.Zero)
        {
            return T.Zero;
        }

        var sum = T.Zero;
        for (var i = rowStart; i < rowEnd; i++)
        {
            var scaled = a[i, col] / xMax;
            sum += scaled * scaled;
        }

        return sum * xMax * xMax;
    }

    private static void SwapColumns(T[,] a, int left, int right, int rowCount)
    {
        for (var i = 0; i < rowCount; i++)
        {
            (a[i, left], a[i, right]) = (a[i, right], a[i, left]);
        }
    }
}

/// <summary>
/// Legacy facade delegating all static methods to <see cref="HouseholderQr{T}"/>
/// instantiated at <c>T = double</c>. Preserves source compatibility for existing
/// callers that reference the non-generic <c>HouseholderQr</c>.
/// </summary>
internal static class HouseholderQr
{
    /// <inheritdoc cref="HouseholderQr{T}.Factor"/>
    public static bool Factor(double[,] a, double[] y, double[] diagonal, int[] permutation)
        => HouseholderQr<double>.Factor(a, y, diagonal, permutation);

    /// <inheritdoc cref="HouseholderQr{T}.SolveUpperTriangular"/>
    public static double[] SolveUpperTriangular(double[,] a, double[] diagonal, double[] c, int n, int[] permutation)
        => HouseholderQr<double>.SolveUpperTriangular(a, diagonal, c, n, permutation);

    /// <inheritdoc cref="HouseholderQr{T}.BuildXtXInverse"/>
    public static double[,] BuildXtXInverse(double[,] a, double[] diagonal, int[] permutation)
        => HouseholderQr<double>.BuildXtXInverse(a, diagonal, permutation);

    /// <inheritdoc cref="HouseholderQr{T}.ApplyQTranspose"/>
    public static void ApplyQTranspose(double[,] a, double[] diagonal, double[] r)
        => HouseholderQr<double>.ApplyQTranspose(a, diagonal, r);
}
