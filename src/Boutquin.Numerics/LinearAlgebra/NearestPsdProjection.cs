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
/// Projects a symmetric matrix onto the cone of positive-semidefinite (PSD)
/// matrices. Two methods are exposed:
/// <list type="bullet">
/// <item><description><see cref="EigenClip(T[,])"/> — one-shot spectral truncation: symmetrize, eigendecompose, clip negative eigenvalues to zero, reconstruct. Cheapest option; the result is the closest PSD matrix in Frobenius norm when no additional constraints are imposed.</description></item>
/// <item><description><see cref="Higham(T[,], int, T)"/> — Higham's alternating projection between the PSD cone and the unit-diagonal cone for <em>correlation</em> matrices, a.k.a. NearPD. Use this when the output must also have unit diagonal (correlation matrices) or preserve a prescribed diagonal.</description></item>
/// </list>
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+√. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Delegates eigendecomposition
/// to <see cref="JacobiEigenDecomposition{T}"/>.
/// </para>
/// <para>
/// References:
/// <list type="bullet">
/// <item><description>Higham, N. J. (1988). "Computing a Nearest Symmetric Positive Semidefinite Matrix." Linear Algebra and its Applications, 103, 103–118.</description></item>
/// <item><description>Higham, N. J. (2002). "Computing the Nearest Correlation Matrix — A Problem from Finance." IMA Journal of Numerical Analysis, 22(3), 329–343.</description></item>
/// <item><description>Qi, H. &amp; Sun, D. (2006). "A Quadratically Convergent Newton Method for Computing the Nearest Correlation Matrix." SIAM Journal on Matrix Analysis, 28(2), 360–385.</description></item>
/// </list>
/// </para>
/// <para>
/// Consumers that only need "my covariance must be PSD before I Cholesky it"
/// should prefer <see cref="EigenClip(T[,])"/> — it is exact (closest PSD in
/// Frobenius norm) and requires a single eigendecomposition. Consumers that
/// need a <em>correlation</em> matrix (unit diagonal, off-diagonal in [−1, 1])
/// should use <see cref="Higham(T[,], int, T)"/>.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class NearestPsdProjection<T>
    where T : IFloatingPoint<T>
{
    private static readonly T s_half = T.CreateChecked(0.5);
    private static readonly T s_defaultHighamTolerance = T.CreateChecked(1e-12);
    private static readonly T s_defaultIsPsdTolerance = T.CreateChecked(-1e-12);

    /// <summary>Convenience overload using tolerance = <c>T.Zero</c>.</summary>
    public static T[,] EigenClip(T[,] a) => EigenClip(a, T.Zero);

    /// <summary>
    /// One-shot spectral truncation. Symmetrizes the input, clips negative
    /// eigenvalues to the specified tolerance, reconstructs V·diag(λ⁺)·Vᵀ.
    /// Returns the closest PSD matrix in Frobenius norm.
    /// </summary>
    /// <param name="a">Symmetric N×N matrix.</param>
    /// <param name="tolerance">Eigenvalues below this threshold are clipped to zero.</param>
    /// <returns>A symmetric PSD matrix of the same shape as <paramref name="a"/>.</returns>
    /// <exception cref="ArgumentException">Matrix is not square.</exception>
    public static T[,] EigenClip(T[,] a, T tolerance)
    {
        ArgumentNullException.ThrowIfNull(a);
        var tol = tolerance;
        var n = a.GetLength(0);
        if (a.GetLength(1) != n)
        {
            throw new ArgumentException("Matrix must be square.", nameof(a));
        }

        // Symmetrize defensively.
        var sym = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            sym[i, i] = a[i, i];
            for (var j = i + 1; j < n; j++)
            {
                var avg = (a[i, j] + a[j, i]) * s_half;
                sym[i, j] = avg;
                sym[j, i] = avg;
            }
        }

        var eigen = JacobiEigenDecomposition<T>.Decompose(sym);
        var values = (T[])eigen.Values.Clone();
        for (var i = 0; i < n; i++)
        {
            if (values[i] < tol)
            {
                values[i] = T.Zero;
            }
        }

        var result = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                var sum = T.Zero;
                for (var k = 0; k < n; k++)
                {
                    sum += values[k] * eigen.Vectors[i, k] * eigen.Vectors[j, k];
                }

                result[i, j] = sum;
                result[j, i] = sum;
            }
        }

        return result;
    }

    /// <summary>
    /// Higham's alternating projection to compute the nearest correlation
    /// matrix. Alternates between projection onto the PSD cone (via eigen
    /// clipping) and projection onto the unit-diagonal subspace, converging
    /// to the unique nearest correlation matrix in Frobenius norm.
    /// </summary>
    /// <param name="a">Symmetric N×N matrix (input correlation estimate, possibly invalid).</param>
    /// <param name="maxIterations">Iteration cap. Default 100.</param>
    /// <param name="tolerance">Convergence tolerance on the Frobenius norm of the iterate delta. Default 1e-12.</param>
    /// <returns>Nearest correlation matrix (symmetric, PSD, unit diagonal).</returns>
    /// <exception cref="ArgumentException">Matrix is not square.</exception>
    public static T[,] Higham(T[,] a, int maxIterations, T tolerance)
    {
        ArgumentNullException.ThrowIfNull(a);
        var n = a.GetLength(0);
        if (a.GetLength(1) != n)
        {
            throw new ArgumentException("Matrix must be square.", nameof(a));
        }

        var y = new T[n, n];
        var deltaS = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            y[i, i] = a[i, i];
            for (var j = i + 1; j < n; j++)
            {
                var avg = (a[i, j] + a[j, i]) * s_half;
                y[i, j] = avg;
                y[j, i] = avg;
            }
        }

        var prev = (T[,])y.Clone();
        for (var iter = 0; iter < maxIterations; iter++)
        {
            // R = Y - ΔS
            var r = new T[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    r[i, j] = y[i, j] - deltaS[i, j];
                }
            }

            // X = P_S(R): project onto PSD cone (eigenvalue clip at zero).
            var x = EigenClip(r);

            // ΔS = X - R
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    deltaS[i, j] = x[i, j] - r[i, j];
                }
            }

            // Y = P_U(X): unit-diagonal projection.
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    y[i, j] = i == j ? T.One : x[i, j];
                }
            }

            // Convergence test.
            var delta = T.Zero;
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    var d = y[i, j] - prev[i, j];
                    delta += d * d;
                }
            }

            if (delta < tolerance * tolerance)
            {
                return y;
            }

            Array.Copy(y, prev, n * n);
        }

        return y;
    }

    /// <summary>Convenience overload using maxIterations=100 and tolerance=1e-12.</summary>
    public static T[,] Higham(T[,] a) => Higham(a, 100, s_defaultHighamTolerance);

    /// <summary>Convenience overload using tolerance=-1e-12 (absorbing FP drift).</summary>
    public static bool IsPsd(T[,] a) => IsPsd(a, s_defaultIsPsdTolerance);

    /// <summary>
    /// Returns <see langword="true"/> if the input is numerically PSD. A
    /// matrix is considered PSD when its smallest eigenvalue exceeds
    /// <paramref name="tolerance"/>.
    /// </summary>
    public static bool IsPsd(T[,] a, T tolerance)
    {
        ArgumentNullException.ThrowIfNull(a);
        var n = a.GetLength(0);
        if (a.GetLength(1) != n)
        {
            return false;
        }

        var eigen = JacobiEigenDecomposition<T>.Decompose(a);
        for (var i = 0; i < n; i++)
        {
            if (eigen.Values[i] < tolerance)
            {
                return false;
            }
        }

        return true;
    }
}

/// <summary>
/// Legacy facade delegating to <see cref="NearestPsdProjection{T}"/> instantiated
/// at <c>T = decimal</c>. Preserves source compatibility for existing callers.
/// </summary>
public static class NearestPsdProjection
{
    /// <inheritdoc cref="NearestPsdProjection{T}.EigenClip(T[,], T)"/>
    public static decimal[,] EigenClip(decimal[,] a, decimal tolerance = 0m)
        => NearestPsdProjection<decimal>.EigenClip(a, tolerance);

    /// <inheritdoc cref="NearestPsdProjection{T}.Higham(T[,], int, T)"/>
    public static decimal[,] Higham(decimal[,] a, int maxIterations, decimal tolerance)
        => NearestPsdProjection<decimal>.Higham(a, maxIterations, tolerance);

    /// <inheritdoc cref="NearestPsdProjection{T}.Higham(T[,])"/>
    public static decimal[,] Higham(decimal[,] a) => NearestPsdProjection<decimal>.Higham(a);

    /// <inheritdoc cref="NearestPsdProjection{T}.IsPsd(T[,])"/>
    public static bool IsPsd(decimal[,] a) => NearestPsdProjection<decimal>.IsPsd(a);

    /// <inheritdoc cref="NearestPsdProjection{T}.IsPsd(T[,], T)"/>
    public static bool IsPsd(decimal[,] a, decimal tolerance) => NearestPsdProjection<decimal>.IsPsd(a, tolerance);
}
