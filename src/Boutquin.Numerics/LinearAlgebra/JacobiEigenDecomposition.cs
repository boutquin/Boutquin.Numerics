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
/// Result of an eigendecomposition: eigenvalues in descending order with
/// corresponding eigenvectors as columns.
/// </summary>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
/// <param name="Values">Eigenvalues sorted in descending order.</param>
/// <param name="Vectors">Eigenvector matrix — column j is the eigenvector for Values[j].</param>
public sealed record EigenResult<T>(T[] Values, T[,] Vectors)
    where T : IFloatingPoint<T>;

/// <summary>
/// Result of an eigendecomposition: eigenvalues in descending order with
/// corresponding eigenvectors as columns.
/// </summary>
/// <param name="Values">Eigenvalues sorted in descending order.</param>
/// <param name="Vectors">Eigenvector matrix — column j is the eigenvector for Values[j].</param>
public sealed record EigenResult(decimal[] Values, decimal[,] Vectors);

/// <summary>
/// Jacobi eigendecomposition for real symmetric matrices using cyclic
/// Givens rotations. Works entirely in <typeparamref name="T"/> — the Jacobi
/// rotation avoids trigonometric functions by design, using only arithmetic
/// and square root.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+√. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Square-root operations use
/// <see cref="NumericPrecision{T}.Sqrt"/> to support <c>decimal</c> (which lacks
/// <see cref="IRootFunctions{TSelf}"/>).
/// </para>
/// <para>
/// Convergence criterion: sum of squared off-diagonal elements &lt; tolerance.
/// Maximum 100 sweeps (each sweep visits all upper-triangular pairs).
/// Eigenvalues are returned in descending order with eigenvector columns
/// reordered to match.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class JacobiEigenDecomposition<T>
    where T : IFloatingPoint<T>
{
    private const int MaxSweeps = 100;

    private static readonly T s_two = T.CreateChecked(2);

    private static readonly T s_offDiagonalTolerance = T.CreateChecked(1e-15);
    private static readonly T s_smallRotationThreshold = T.CreateChecked(1e-20);
    private static readonly T s_tinyPhi = T.CreateChecked(1e-20);

    // Overflow guard for the phi^2 computation in the Jacobi rotation.
    // decimal.MaxValue ~ 7.9e28, so phi^2 overflows when |phi| > ~2.8e14.
    // Use 1e13 as a safe threshold with margin.
    private static readonly T s_phiOverflowGuard = T.CreateChecked(1e13);

    /// <summary>
    /// Decomposes a symmetric matrix A into A = V·diag(λ)·Vᵀ.
    /// </summary>
    /// <param name="matrix">Symmetric N×N matrix. Only the upper triangle is read.</param>
    /// <returns>Eigenvalues (descending) and eigenvector columns.</returns>
    /// <exception cref="ArgumentException">Matrix is not square.</exception>
    /// <exception cref="InvalidOperationException">Algorithm did not converge within <see cref="MaxSweeps"/> sweeps.</exception>
    public static EigenResult<T> Decompose(T[,] matrix)
    {
        var n = matrix.GetLength(0);
        if (matrix.GetLength(1) != n)
        {
            throw new ArgumentException("Matrix must be square.", nameof(matrix));
        }

        // Work copy.
        var a = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                a[i, j] = matrix[i, j];
            }
        }

        // Initialize eigenvector matrix to identity.
        var v = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            v[i, i] = T.One;
        }

        for (var sweep = 0; sweep < MaxSweeps; sweep++)
        {
            // Compute sum of squares of off-diagonal elements.
            var offDiagSum = T.Zero;
            for (var p = 0; p < n - 1; p++)
            {
                for (var q = p + 1; q < n; q++)
                {
                    offDiagSum += a[p, q] * a[p, q];
                }
            }

            if (offDiagSum < s_offDiagonalTolerance)
            {
                return BuildResult(a, v, n);
            }

            // Cyclic sweep over all upper-triangular pairs.
            for (var p = 0; p < n - 1; p++)
            {
                for (var q = p + 1; q < n; q++)
                {
                    if (T.Abs(a[p, q]) < s_smallRotationThreshold)
                    {
                        continue;
                    }

                    ApplyRotation(a, v, n, p, q);
                }
            }
        }

        throw new InvalidOperationException(
            $"Jacobi eigendecomposition did not converge within {MaxSweeps} sweeps.");
    }

    private static void ApplyRotation(T[,] a, T[,] v, int n, int p, int q)
    {
        var diff = a[q, q] - a[p, p];
        T t;
        if (diff != T.Zero && T.Abs(a[p, q]) < T.Abs(diff) * s_tinyPhi)
        {
            t = a[p, q] / diff;
        }
        else
        {
            var phi = diff / (s_two * a[p, q]);
            var absPhi = T.Abs(phi);

            // When |phi| is very large, phi^2 can overflow decimal. In that regime
            // sqrt(phi^2 + 1) ~ |phi|, so t ~ 1/(2*|phi|) = a[p,q] / diff.
            // Use the asymptotic form when |phi| exceeds a safe threshold.
            if (absPhi > s_phiOverflowGuard)
            {
                t = T.One / (s_two * absPhi);
            }
            else
            {
                t = T.One / (absPhi + NumericPrecision<T>.Sqrt(phi * phi + T.One));
            }

            if (T.IsNegative(phi))
            {
                t = -t;
            }
        }

        var c = T.One / NumericPrecision<T>.Sqrt(t * t + T.One);
        var s = t * c;
        var tau = s / (T.One + c);
        var temp = t * a[p, q];
        a[p, p] -= temp;
        a[q, q] += temp;
        a[p, q] = T.Zero;

        // Rotate rows and columns of a.
        for (var j = 0; j < p; j++)
        {
            Rotate(a, s, tau, j, p, j, q);
        }

        for (var j = p + 1; j < q; j++)
        {
            Rotate(a, s, tau, p, j, j, q);
        }

        for (var j = q + 1; j < n; j++)
        {
            Rotate(a, s, tau, p, j, q, j);
        }

        // Accumulate eigenvectors.
        for (var j = 0; j < n; j++)
        {
            var vp = v[j, p];
            var vq = v[j, q];
            v[j, p] = vp - s * (vq + tau * vp);
            v[j, q] = vq + s * (vp - tau * vq);
        }
    }

    private static void Rotate(T[,] a, T s, T tau, int i, int j, int k, int l)
    {
        var g = a[i, j];
        var h = a[k, l];
        a[i, j] = g - s * (h + tau * g);
        a[k, l] = h + s * (g - tau * h);
    }

    private static EigenResult<T> BuildResult(T[,] a, T[,] v, int n)
    {
        // Extract eigenvalues from diagonal.
        var eigenvalues = new T[n];
        for (var i = 0; i < n; i++)
        {
            eigenvalues[i] = a[i, i];
        }

        // Sort in descending order, keeping eigenvectors aligned.
        var indices = new int[n];
        for (var i = 0; i < n; i++)
        {
            indices[i] = i;
        }

        // Simple selection sort by descending eigenvalue.
        for (var i = 0; i < n - 1; i++)
        {
            var maxIdx = i;
            for (var j = i + 1; j < n; j++)
            {
                if (eigenvalues[j] > eigenvalues[maxIdx])
                {
                    maxIdx = j;
                }
            }

            if (maxIdx != i)
            {
                (eigenvalues[i], eigenvalues[maxIdx]) = (eigenvalues[maxIdx], eigenvalues[i]);
                (indices[i], indices[maxIdx]) = (indices[maxIdx], indices[i]);
            }
        }

        // Build result with sorted eigenvectors.
        var vectors = new T[n, n];
        for (var col = 0; col < n; col++)
        {
            var srcCol = indices[col];
            for (var row = 0; row < n; row++)
            {
                vectors[row, col] = v[row, srcCol];
            }
        }

        return new EigenResult<T>(eigenvalues, vectors);
    }
}

/// <summary>
/// Legacy facade preserving source compatibility for existing callers.
/// Delegates to <see cref="JacobiEigenDecomposition{T}"/> instantiated at
/// <c>T = double</c> because the Jacobi rotation's intermediate products
/// (phi^2 terms) can overflow <c>decimal</c>'s narrower range (~7.9e28 vs
/// <c>double</c>'s ~1.8e308). Results are converted back to <c>decimal</c>.
/// </summary>
public static class JacobiEigenDecomposition
{
    /// <inheritdoc cref="JacobiEigenDecomposition{T}.Decompose"/>
    public static EigenResult Decompose(decimal[,] matrix)
    {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        if (rows != cols)
        {
            throw new ArgumentException("Matrix must be square.", nameof(matrix));
        }

        var n = rows;

        // Convert to double for the rotation computations.
        var d = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                d[i, j] = (double)matrix[i, j];
            }
        }

        var result = JacobiEigenDecomposition<double>.Decompose(d);

        // Convert back to decimal.
        var values = new decimal[n];
        var vectors = new decimal[n, n];
        for (var i = 0; i < n; i++)
        {
            values[i] = (decimal)result.Values[i];
            for (var j = 0; j < n; j++)
            {
                vectors[i, j] = (decimal)result.Vectors[i, j];
            }
        }

        return new EigenResult(values, vectors);
    }
}
