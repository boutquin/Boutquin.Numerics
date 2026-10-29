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
/// <item><description><see cref="Higham(T[,], int, T)"/> — Higham's alternating projection between the PSD cone and the unit-diagonal cone for <em>correlation</em> matrices, a.k.a. NearPD. Use this when the output must have unit diagonal (correlation matrices); the input diagonal is replaced with ones.</description></item>
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
/// should prefer <see cref="EigenClip(T[,])"/> — it is exact in exact arithmetic
/// (closest PSD in Frobenius norm) and requires a single eigendecomposition. In
/// floating point, an already-PSD input is reproduced up to the underlying
/// <see cref="JacobiEigenDecomposition{T}"/> eigensolver's convergence tolerance
/// (near working precision — see <c>docs/linear-algebra.md</c> for the measured bound),
/// not bit-for-bit. Consumers that need a <em>correlation</em> matrix (unit diagonal,
/// off-diagonal in [−1, 1]) should use <see cref="Higham(T[,], int, T)"/>.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class NearestPsdProjection<T>
    where T : IFloatingPoint<T>
{
    private static readonly T s_half = T.CreateChecked(0.5);
    private static readonly T s_defaultHighamTolerance = T.CreateChecked(1e-12);
    private static readonly T s_defaultIsPsdTolerance = T.CreateChecked(-1e-12);

    /// <summary>Convenience overload using eigenvalue floor <c>T.Zero</c>.</summary>
    public static T[,] EigenClip(T[,] a) => EigenClip(a, T.Zero);

    /// <summary>
    /// Symmetrizes the input and reconstructs V·diag(max(λ, δ))·Vᵀ, the nearest
    /// symmetric matrix in Frobenius norm whose eigenvalues are at least δ.
    /// The eigenvalue floor is independent of the eigensolver's numerical tolerance.
    /// </summary>
    /// <param name="a">Symmetric N×N matrix.</param>
    /// <param name="eigenvalueFloor">Nonnegative minimum eigenvalue δ. Zero gives ordinary PSD projection.</param>
    /// <returns>A symmetric PSD matrix of the same shape as <paramref name="a"/>.</returns>
    /// <exception cref="ArgumentException">Matrix is not square.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="eigenvalueFloor"/> is negative or non-finite.</exception>
    public static T[,] EigenClip(T[,] a, T eigenvalueFloor)
    {
        ArgumentNullException.ThrowIfNull(a);
        var tol = eigenvalueFloor;
        if (!T.IsFinite(tol) || tol < T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(eigenvalueFloor),
                eigenvalueFloor,
                "Eigenvalue floor must be finite and non-negative.");
        }

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
                var avg = SymmetricAverage(a[i, j], a[j, i]);
                sym[i, j] = avg;
                sym[j, i] = avg;
            }
        }

        var scale = T.Zero;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                if (!T.IsFinite(sym[i, j]))
                {
                    throw new ArgumentException("Matrix contains a non-finite entry.", nameof(a));
                }

                scale = T.Max(scale, T.Abs(sym[i, j]));
            }
        }

        // Orthogonal coordinate blocks have independent spectral projections.
        // Normalize each connected block separately so an unrelated large entry
        // cannot erase a small, representable eigenvalue.
        var visited = new bool[n];
        var components = new List<List<int>>();
        for (var root = 0; root < n; root++)
        {
            if (visited[root])
            {
                continue;
            }

            var component = new List<int> { root };
            visited[root] = true;
            for (var cursor = 0; cursor < component.Count; cursor++)
            {
                var row = component[cursor];
                for (var column = 0; column < n; column++)
                {
                    if (!visited[column] && sym[row, column] != T.Zero)
                    {
                        visited[column] = true;
                        component.Add(column);
                    }
                }
            }

            components.Add(component);
        }

        if (components.Count > 1 || n == 1)
        {
            var blocks = new T[n, n];
            foreach (var component in components)
            {
                if (component.Count == 1)
                {
                    var index = component[0];
                    blocks[index, index] = T.Max(sym[index, index], eigenvalueFloor);
                    continue;
                }

                var block = new T[component.Count, component.Count];
                for (var i = 0; i < component.Count; i++)
                {
                    for (var j = 0; j < component.Count; j++)
                    {
                        block[i, j] = sym[component[i], component[j]];
                    }
                }

                var projected = EigenClip(block, eigenvalueFloor);
                for (var i = 0; i < component.Count; i++)
                {
                    for (var j = 0; j < component.Count; j++)
                    {
                        blocks[component[i], component[j]] = projected[i, j];
                    }
                }
            }

            return blocks;
        }

        scale = T.Max(scale, eigenvalueFloor);
        if (scale == T.Zero)
        {
            return sym;
        }

        var scaled = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                scaled[i, j] = sym[i, j] / scale;
            }
        }

        var eigen = JacobiEigenDecomposition<T>.Decompose(scaled);
        var values = eigen.Values;
        var vectors = eigen.Vectors;

        // The floor is a requested eigenvalue in ORIGINAL units. tol / scale can round to zero
        // (decimal's absolute grid, IEEE underflow), so it is used only to decide which normalized
        // eigenvalues are floored; the floored spectrum is reconstructed as
        // scale * sum over kept (lambda v v') + tol * sum over floored (v v'), never as
        // (tol / scale) * scale. The projector sum over an orthonormal subset is bounded by one.
        var scaledFloor = tol / scale;
        var floored = new bool[n];
        for (var i = 0; i < n; i++)
        {
            floored[i] = values[i] <= scaledFloor;
        }

        var result = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                var kept = T.Zero;
                var projector = T.Zero;
                for (var k = 0; k < n; k++)
                {
                    if (floored[k])
                    {
                        projector += vectors[i, k] * vectors[j, k];
                    }
                    else
                    {
                        kept += values[k] * vectors[i, k] * vectors[j, k];
                    }
                }

                var rescaled = (kept * scale) + (tol * projector);
                result[i, j] = rescaled;
                result[j, i] = rescaled;
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
    /// <param name="maxIterations">Iteration cap. Default 100. If the alternating projection does
    /// not converge within this many iterations, the method throws rather than returning an
    /// unconverged iterate. A successful return has unit diagonal and passes the separate
    /// default <c>IsPsd</c> numerical tolerance; this is not an exact PSD certificate.</param>
    /// <param name="tolerance">Convergence tolerance. Iteration stops when <c>‖Y_k − Y_{k-1}‖_F ≤ tolerance</c> (checked using the squared norm, with a scaled comparison for tolerances above one to avoid overflow) and the iterate is numerically PSD. Zero requires an exact fixed point. Default 1e-12.</param>
    /// <returns>A symmetric unit-diagonal approximation to the nearest correlation matrix,
    /// numerically PSD under the default <c>IsPsd</c> acceptance tolerance. Small negative
    /// eigenvalues may remain, and strict Cholesky acceptance is not guaranteed.</returns>
    /// <exception cref="ArgumentException">Matrix is not square.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxIterations"/> is negative, or <paramref name="tolerance"/> is negative or non-finite.</exception>
    /// <exception cref="InvalidOperationException">Alternating projection did not converge within <paramref name="maxIterations"/> iterations. Increase <paramref name="maxIterations"/> or loosen <paramref name="tolerance"/>.</exception>
    public static T[,] Higham(T[,] a, int maxIterations, T tolerance)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentOutOfRangeException.ThrowIfNegative(maxIterations);
        if (!T.IsFinite(tolerance) || tolerance < T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(tolerance),
                tolerance,
                "Tolerance must be finite and non-negative.");
        }

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
                var avg = SymmetricAverage(a[i, j], a[j, i]);
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

            // Normalize before squaring so a finite change cannot overflow its norm.
            var changeScale = tolerance;
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    changeScale = T.Max(changeScale, T.Abs(y[i, j] - prev[i, j]));
                }
            }

            // An unchanged iterate with zero tolerance has a zero norm.
            changeScale = changeScale == T.Zero ? T.One : changeScale;
            var delta = T.Zero;
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    var d = (y[i, j] - prev[i, j]) / changeScale;
                    delta += d * d;
                }
            }

            var scaledTolerance = tolerance / changeScale;
            var withinTolerance = delta <= scaledTolerance * scaledTolerance;
            if (withinTolerance && IsPsd(y))
            {
                return y;
            }

            Array.Copy(y, prev, n * n);
        }

        // The last iterate has not passed both convergence and numerical PSD acceptance.
        // Signal exhaustion rather than return an unchecked approximation.
        throw new InvalidOperationException(
            $"Higham nearest-correlation projection did not converge within {maxIterations} iterations.");
    }

    /// <summary>Convenience overload using maxIterations=100 and tolerance=1e-12.</summary>
    public static T[,] Higham(T[,] a) => Higham(a, 100, s_defaultHighamTolerance);

    /// <summary>
    /// Convenience overload using a tolerance of -1e-12 scaled by the matrix's own magnitude
    /// (absorbing floating-point drift proportionally to scale). At unit-or-smaller scale this
    /// is the historical absolute -1e-12; above unit scale it grows with the matrix's largest
    /// entry, because rounding in the eigendecomposition scales with the matrix's magnitude —
    /// a fixed absolute tolerance would otherwise report an exactly-PSD matrix as non-PSD once
    /// that magnitude pushed rounding past -1e-12 (see <see cref="EigenClip(T[,])"/>, which
    /// applies the same relative-scale reasoning to its own default tolerance).
    /// </summary>
    /// <exception cref="ArgumentException">Matrix is not square.</exception>
    public static bool IsPsd(T[,] a)
    {
        ArgumentNullException.ThrowIfNull(a);
        var n = a.GetLength(0);
        if (a.GetLength(1) != n)
        {
            throw new ArgumentException("Matrix must be square.", nameof(a));
        }

        var scale = T.Zero;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                scale = T.Max(scale, T.Abs(a[i, j]));
            }
        }

        var tolerance = s_defaultIsPsdTolerance * T.Max(scale, T.One);
        return IsPsd(a, tolerance);
    }

    /// <summary>
    /// Returns <see langword="true"/> if the input is numerically PSD. A
    /// matrix is considered PSD when its smallest eigenvalue exceeds
    /// <paramref name="tolerance"/>.
    /// </summary>
    /// <exception cref="ArgumentException">Matrix is not square.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="tolerance"/> is non-finite.</exception>
    public static bool IsPsd(T[,] a, T tolerance)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (!T.IsFinite(tolerance))
        {
            throw new ArgumentOutOfRangeException(
                nameof(tolerance), tolerance, "Tolerance must be finite.");
        }

        var n = a.GetLength(0);
        if (a.GetLength(1) != n)
        {
            throw new ArgumentException("Matrix must be square.", nameof(a));
        }

        // Symmetrize defensively: Jacobi reads only the upper triangle, so floating-point
        // asymmetry (e.g., from alternating projection in Higham) can skew the eigenvalues.
        var sym = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            sym[i, i] = a[i, i];
            for (var j = i + 1; j < n; j++)
            {
                var avg = SymmetricAverage(a[i, j], a[j, i]);
                sym[i, j] = avg;
                sym[j, i] = avg;
            }
        }

        var eigen = JacobiEigenDecomposition<T>.Decompose(sym);
        var values = eigen.Values;
        for (var i = 0; i < n; i++)
        {
            if (values[i] < tolerance)
            {
                return false;
            }
        }

        return true;
    }

    private static T SymmetricAverage(T left, T right)
    {
        if (left == right)
        {
            return left;
        }

        try
        {
            var sum = left + right;
            if (T.IsFinite(sum))
            {
                return sum * s_half;
            }
        }
        catch (OverflowException)
        {
            // Decimal throws where IEEE addition produces infinity.
        }

        return left * s_half + right * s_half;
    }
}

/// <summary>
/// Legacy facade delegating to <see cref="NearestPsdProjection{T}"/> instantiated
/// at <c>T = decimal</c>. Preserves source compatibility for existing callers.
/// </summary>
[Obsolete(
    "Use NearestPsdProjection<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class NearestPsdProjection
{
    /// <inheritdoc cref="NearestPsdProjection{T}.EigenClip(T[,], T)"/>
    public static decimal[,] EigenClip(decimal[,] a, decimal eigenvalueFloor = 0m)
        => NearestPsdProjection<decimal>.EigenClip(a, eigenvalueFloor);

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
