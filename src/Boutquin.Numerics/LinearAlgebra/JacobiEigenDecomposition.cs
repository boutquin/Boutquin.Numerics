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
public sealed record EigenResult<T>
    where T : IFloatingPoint<T>
{
    private T[] _values;
    private T[,] _vectors;

    /// <summary>Initializes a result with owned copies of the supplied arrays.</summary>
    /// <param name="Values">Eigenvalues sorted in descending order.</param>
    /// <param name="Vectors">Eigenvector matrix — column j is the eigenvector for Values[j].</param>
    public EigenResult(T[] Values, T[,] Vectors)
    {
        _values = (T[])(Values ?? throw new ArgumentNullException(nameof(Values))).Clone();
        _vectors = (T[,])(Vectors ?? throw new ArgumentNullException(nameof(Vectors))).Clone();
    }

    /// <summary>Eigenvalues sorted in descending order.</summary>
    public T[] Values
    {
        get => (T[])_values.Clone();
        init => _values = (T[])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>Eigenvector matrix — column j is the eigenvector for Values[j].</summary>
    public T[,] Vectors
    {
        get => (T[,])_vectors.Clone();
        init => _vectors = (T[,])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>Deconstructs this result into defensive snapshots.</summary>
    public void Deconstruct(out T[] Values, out T[,] Vectors)
    {
        Values = (T[])_values.Clone();
        Vectors = (T[,])_vectors.Clone();
    }
}

/// <summary>
/// Result of an eigendecomposition: eigenvalues in descending order with
/// corresponding eigenvectors as columns.
/// </summary>
public sealed record EigenResult
{
    private decimal[] _values;
    private decimal[,] _vectors;

    /// <summary>Initializes a result with owned copies of the supplied arrays.</summary>
    /// <param name="Values">Eigenvalues sorted in descending order.</param>
    /// <param name="Vectors">Eigenvector matrix — column j is the eigenvector for Values[j].</param>
    public EigenResult(decimal[] Values, decimal[,] Vectors)
    {
        _values = (decimal[])(Values ?? throw new ArgumentNullException(nameof(Values))).Clone();
        _vectors = (decimal[,])(Vectors ?? throw new ArgumentNullException(nameof(Vectors))).Clone();
    }

    /// <summary>Eigenvalues sorted in descending order.</summary>
    public decimal[] Values
    {
        get => (decimal[])_values.Clone();
        init => _values = (decimal[])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>Eigenvector matrix — column j is the eigenvector for Values[j].</summary>
    public decimal[,] Vectors
    {
        get => (decimal[,])_vectors.Clone();
        init => _vectors = (decimal[,])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>Deconstructs this result into defensive snapshots.</summary>
    public void Deconstruct(out decimal[] Values, out decimal[,] Vectors)
    {
        Values = (decimal[])_values.Clone();
        Vectors = (decimal[,])_vectors.Clone();
    }
}

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
/// Convergence criterion: the Frobenius norm of the off-diagonal elements falls to or below
/// <c>tolerance · min(‖A‖_F, 1)</c> (the Frobenius norm is invariant under the orthogonal
/// Jacobi rotations, so it is computed once), with <c>tolerance</c> near working precision
/// (1e-27 for decimal, 1e-15 for binary types). The relative factor prevents premature convergence
/// on small-magnitude matrices; capping it at 1 gives an absolute threshold at unit or larger scales.
/// This is compared directly against the off-diagonal norm — the same scale-relative pattern
/// the per-pair rotation-skip floor below uses — rather than against its square, so the bound
/// is expressed in the same units as the reconstruction error it governs. That test is accepted
/// only when every off-diagonal is also small against its own diagonals,
/// <c>|a_pq| &lt;= tolerance · sqrt|a_pp| · sqrt|a_qq|</c>, and a pair below the rotation-skip floor is
/// skipped only under the same condition: on a graded matrix (D·C·D with D spanning many binades) the
/// small block lies far below both absolute thresholds although its off-diagonals are of the order
/// of its own diagonals. This relative test prevents that premature stop. Relative-accuracy
/// bounds for positive-definite graded matrices depend on the conditioning of the scaled matrix
/// (Demmel &amp; Veselić, 1992); the stopping test alone gives no unconditional relative-accuracy
/// guarantee for indefinite or singular matrices. When every diagonal entry is at least
/// <c>min(‖A‖_F, 1)</c> in magnitude it is implied by the Frobenius test. Maximum 100 sweeps
/// (each sweep visits all upper-triangular pairs).
/// </para>
/// <para>
/// Each diagonal entry carries the rounding error of its rotation updates in a low-order part
/// (an error-free two-sum), and rotation angles are formed from the diagonal difference including
/// those parts; the eigenvalues are the diagonal plus its low-order part, rounded once. Without it
/// a cluster of equal or nearly equal diagonals whose off-diagonals are below half an ulp of them
/// (float at any scale, double above unit scale) loses every diagonal update to rounding, the
/// following angles are computed from a difference that no longer carries them, and convergence
/// turns linear: the off-diagonals cannot reach the absolute floor within the sweep cap, and each
/// rotation leaves its discarded update as backward error. With it such a cluster converges
/// quadratically to the eigenvectors of the matrix as given.
/// Eigenvalues are returned in descending order with eigenvector columns
/// reordered to match. Following Numerical Recipes, third edition, §11.1
/// <c>eigsrt</c>, each selection step chooses the last equal maximum in the
/// remaining columns. This fixes tie ordering; it does not make an eigenbasis
/// in a repeated-eigenvalue subspace mathematically unique.
/// </para>
/// <para>
/// Type-precision note: norm accumulation and rotation construction are scaled to avoid
/// overflow and underflow in intermediate squares and diagonal differences. For narrow
/// types such as <c>Half</c>, thresholds below the type's representable range become zero,
/// so convergence requires the remaining off-diagonal entries to round to exact zero.
/// </para>
/// <para>
/// Sweep cap: the 100-sweep limit is a HARD cap, not a best-effort budget. Cyclic Jacobi
/// converges quadratically once the off-diagonals are small, so well-separated spectra reach the
/// convergence threshold in well under 100 sweeps. Matrices with tightly clustered or repeated
/// eigenvalues can stall — the rotation that would zero an off-diagonal between two near-equal
/// diagonals is ill-determined — and may exhaust the cap. When that happens the method throws
/// <see cref="InvalidOperationException"/> rather than returning an unconverged decomposition;
/// callers that expect clustered spectra should perturb the input or use a shift-and-deflate
/// eigensolver instead.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class JacobiEigenDecomposition<T>
    where T : IFloatingPoint<T>
{
    private const int MaxSweeps = 100;

    private static readonly T s_two = T.CreateChecked(2);

    private static readonly T s_offDiagonalTolerance = typeof(T) == typeof(decimal)
        ? T.CreateChecked(1e-27m) : T.CreateChecked(1e-15);

    // Absolute CAP on the scale-relative per-pair rotation-skip floor (see Decompose). For
    // unit-or-larger-magnitude matrices the relative floor (tolerance·‖A‖_F) exceeds this cap, so
    // the skip reverts to the historical absolute 1e-20 and those matrices keep their prior behaviour
    // exactly. For tiny-magnitude matrices the relative floor is far smaller, so rotations that an
    // absolute 1e-20 would have wrongly suppressed now run, restoring convergence.
    private static readonly T s_smallRotationThreshold = T.CreateChecked(1e-20);

    // Ratio threshold for the linear small-rotation fast path: when |a[p,q]| < |diff|·s_tinyPhi the
    // exact rotation t = 1/(|phi| + sqrt(phi²+1)) and its linear approximation t ≈ a[p,q]/diff agree
    // to machine precision (phi = diff/(2·a[p,q]) is then ≳ 1/(2·s_tinyPhi), so sqrt(phi²+1) ≈ |phi|).
    // Calibrated to ~double machine epsilon (1e-15) so this is a REAL fast path rather than the dead
    // branch the prior 1e-20 value produced — 1e-20 is far below eps, so the guard effectively never
    // fired. Below the eps-scale ratio the two t expressions are numerically indistinguishable, so
    // eigenvalues and eigenvectors are unchanged (covered by the existing Jacobi parity tests).
    private static readonly T s_tinyPhi = T.CreateChecked(1e-15);

    /// <summary>
    /// Decomposes a symmetric matrix A into A = V·diag(λ)·Vᵀ.
    /// </summary>
    /// <param name="matrix">Symmetric N×N matrix. Only the upper triangle is read.</param>
    /// <returns>Eigenvalues (descending) and eigenvector columns.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="matrix"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">Matrix is not square, or its upper triangle contains a non-finite entry.</exception>
    /// <exception cref="InvalidOperationException">Algorithm did not converge within <see cref="MaxSweeps"/> sweeps, or a rotation produced an unrepresentable eigenvalue.</exception>
    public static EigenResult<T> Decompose(T[,] matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);

        var n = matrix.GetLength(0);
        if (matrix.GetLength(1) != n)
        {
            throw new ArgumentException("Matrix must be square.", nameof(matrix));
        }

        // Work copy. The public contract reads only the upper triangle, so mirror it into
        // the lower triangle rather than allowing unused lower-triangle storage to affect
        // scale calculations or convergence.
        // Reject non-finite entries in the triangle that is read: a NaN or Infinity would poison
        // the Frobenius norm and make every convergence comparison false, surfacing as a
        // misleading 100-sweep "did not converge" failure instead of an input error.
        var a = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                if (!T.IsFinite(matrix[i, j]))
                {
                    throw new ArgumentException("Matrix contains a non-finite entry.", nameof(matrix));
                }

                a[i, j] = matrix[i, j];
                a[j, i] = matrix[i, j];
            }
        }

        var guarded = ExceedsRotationRange(a, n);

        // Initialize eigenvector matrix to identity.
        var v = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            v[i, i] = T.One;
        }

        // The Frobenius norm is invariant under the orthogonal Jacobi rotations, so it is
        // computed once and used to make the convergence test scale-RELATIVE for small-magnitude
        // matrices, where an absolute off-diagonal threshold converges prematurely. Compute it
        // with scaling so squaring a representable Half value cannot underflow and squaring a
        // large double cannot overflow. The relative factor is capped
        // at 1 so that for unit-or-larger-scale matrices the test reverts exactly to the
        // historical absolute threshold — those already converged accurately, and loosening
        // them would perturb downstream eigenvalue-denoising results.
        var matrixMagnitude = FrobeniusNorm(a, n, offDiagonalOnly: false);
        var convergenceThreshold = s_offDiagonalTolerance * T.Min(matrixMagnitude, T.One);

        // Per-pair rotation-skip floor. The skip must be SCALE-RELATIVE for the same reason the
        // convergence test is: an absolute floor (the historical 1e-20) suppresses every rotation on
        // a tiny-magnitude matrix (all off-diagonals below 1e-20) so the algorithm makes no progress
        // and spuriously throws "did not converge", even on a trivially-diagonalizable, well-separated
        // spectrum. The floor is tied to the matrix magnitude scaled
        // by the off-diagonal tolerance, then capped at the historical absolute 1e-20 so unit-or-
        // larger matrices keep their previous skip behaviour exactly. A magnitude comparison (not a
        // squared one) is used deliberately: squaring a sub-1e-14 off-diagonal underflows to zero in
        // decimal, which would defeat the skip and let ApplyRotation divide by a vanishing pivot.
        // A zero matrix yields a zero floor, harmless because it is already diagonal.
        //
        // The floor must also sit strictly below convergenceThreshold / n. The convergence test
        // bounds the norm of all n(n−1)/2 upper off-diagonals together, which can be up to
        // √(n(n−1)/2) (< n) times the largest skipped element. A floor equal to the threshold (the
        // small-magnitude case of the uncapped term) lets every element be skipped while their
        // combined norm still exceeds the threshold, so no rotation runs and the sweep stalls
        // until MaxSweeps. Dividing by n keeps the skipped remainder under the threshold; for
        // unit-or-larger matrices with n ≤ 1e5 the absolute cap still binds, as before.
        var rotationSkipFloor = T.Min(convergenceThreshold / T.CreateChecked(Math.Max(n, 1)), s_smallRotationThreshold);

        // Both absolute tests above are blind to grading: in D·C·D with D spanning many binades the
        // small block sits far below them although its off-diagonals are O(1) of its own diagonals,
        // so it would be neither rotated nor waited for. Each pair is therefore also measured against
        // its own diagonals, |a_pq| <= tolerance·sqrt|a_pp|·sqrt|a_qq|, and a pair is
        // skipped, or the Frobenius test accepted, only when that relative test also holds. When every
        // diagonal is at least min(‖A‖_F, 1) in magnitude the Frobenius test implies it (each skipped
        // or accepted |a_pq| is below the threshold), so for such matrices the absolute tests alone
        // still decide.
        var diagonalRoots = new T[n];

        // Low-order parts of the diagonal. A rotation moves t·a_pq between two diagonals; when the
        // diagonals are equal or nearly equal and that update is below half an ulp of them (a
        // float cluster at any scale, double above unit scale), rounding discards it, the next
        // angle is computed from a difference that no longer carries it, and convergence turns
        // linear: the off-diagonals cannot reach the absolute floor, and every rotation leaves a
        // backward error of its discarded update. Carrying the rounding error of each diagonal
        // update (an error-free sum) keeps the angles exact to twice the working precision, so the
        // cluster converges quadratically to the eigenvectors of the matrix as given.
        var lowDiagonal = new T[n];

        for (var sweep = 0; sweep < MaxSweeps; sweep++)
        {
            var offDiagonalMagnitude = FrobeniusNorm(a, n, offDiagonalOnly: true);
            for (var i = 0; i < n; i++)
            {
                diagonalRoots[i] = NumericPrecision<T>.Sqrt(T.Abs(a[i, i]));
            }

            if (offDiagonalMagnitude <= convergenceThreshold && AllPairsRelativelyNegligible(a, diagonalRoots, n))
            {
                return BuildResult(a, lowDiagonal, v, n);
            }

            // Cyclic sweep over all upper-triangular pairs.
            for (var p = 0; p < n - 1; p++)
            {
                for (var q = p + 1; q < n; q++)
                {
                    if (T.Abs(a[p, q]) <= rotationSkipFloor
                        && IsRelativelyNegligible(a[p, q], diagonalRoots[p], diagonalRoots[q]))
                    {
                        continue;
                    }

                    try
                    {
                        ApplyRotationCore(a, v, lowDiagonal, n, p, q, guarded);
                    }
                    catch (OverflowException exception)
                    {
                        throw new InvalidOperationException(
                            "Jacobi rotation produced an eigenvalue that is not finite in the working type.", exception);
                    }
                }
            }
        }

        throw new InvalidOperationException(
            $"Jacobi eigendecomposition did not converge within {MaxSweeps} sweeps.");
    }

    // One unguarded rotation of a matrix whose diagonal carries no low-order parts. Decompose itself
    // calls ApplyRotationCore with its low-order diagonal; this entry is kept for the tests that
    // drive a single rotation by reflection.
    internal static void ApplyRotation(T[,] a, T[,] v, int n, int p, int q)
        => ApplyRotationCore(a, v, null, n, p, q, guarded: false);

    private static void ApplyRotationCore(T[,] a, T[,] v, T[]? lowDiagonal, int n, int p, int q, bool guarded)
    {
        var apq = a[p, q];
        var scale = T.Max(T.Max(T.Abs(a[p, p]), T.Abs(a[q, q])), T.Abs(apq));
        var lowDifference = lowDiagonal is null ? T.Zero : lowDiagonal[q] - lowDiagonal[p];
        var scaledDiff = (a[q, q] / scale) - (a[p, p] / scale);
        if (lowDifference != T.Zero && T.Sign(a[p, p]) * T.Sign(a[q, q]) >= 0)
        {
            // Diagonals of one sign subtract without overflow, and exactly when they are within a
            // factor of two of each other, which is where the low-order parts decide the angle.
            scaledDiff = ((a[q, q] - a[p, p]) + lowDifference) / scale;
        }

        var scaledOffDiagonal = apq / scale;
        T t;
        if (a[p, p] == a[q, q] && lowDifference == T.Zero)
        {
            // Equal diagonals require a 45-degree rotation even when normalizing a
            // representable pivot by a large scale rounds it to zero (decimal).
            t = T.One;
        }
        else if (scaledOffDiagonal == T.Zero)
        {
            // The pivot is below the resolution of the normalized pair (decimal's absolute grid,
            // IEEE underflow): the rotation is the identity at working precision, and the pair is
            // annihilated below exactly as the small-angle path does. Forming phi would divide by it.
            t = T.Zero;
        }
        else if (scaledDiff != T.Zero && T.Abs(scaledOffDiagonal) < T.Abs(scaledDiff) * s_tinyPhi)
        {
            t = scaledOffDiagonal / scaledDiff;
        }
        else
        {
            var phi = scaledDiff / (s_two * scaledOffDiagonal);
            var absPhi = T.Abs(phi);

            // For large phi, divide the formula by |phi| before squaring.
            // The reciprocal is at most one, including for narrow types such as Half.
            if (absPhi > T.One)
            {
                var inversePhi = s_two * T.Abs(scaledOffDiagonal / scaledDiff);
                t = inversePhi / (T.One + NumericPrecision<T>.Sqrt(T.One + inversePhi * inversePhi));
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
        var temp = t * apq;
        if (lowDiagonal is null)
        {
            a[p, p] -= temp;
            a[q, q] += temp;
        }
        else
        {
            UpdateDiagonal(a, lowDiagonal, p, -temp);
            UpdateDiagonal(a, lowDiagonal, q, temp);
        }

        if (!T.IsFinite(a[p, p]) || !T.IsFinite(a[q, q]))
        {
            throw new InvalidOperationException("Jacobi rotation cannot produce finite eigenvalues in the working type.");
        }
        a[p, q] = T.Zero;

        // Rotate rows and columns of a.
        if (guarded)
        {
            RotateRowsGuarded(a, s, tau, c, n, p, q);
        }
        else
        {
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

    private static T FrobeniusNorm(T[,] matrix, int n, bool offDiagonalOnly)
    {
        var scale = T.Zero;
        for (var i = 0; i < n; i++)
        {
            var start = offDiagonalOnly ? i + 1 : 0;
            for (var j = start; j < n; j++)
            {
                scale = T.Max(scale, T.Abs(matrix[i, j]));
            }
        }

        if (scale == T.Zero)
        {
            return T.Zero;
        }

        // Both callers compare only against thresholds at most one. Cap before
        // reconstructing the norm, whose true value may exceed the range of T.
        if (scale >= T.One)
        {
            return T.One;
        }

        var scaledSum = T.Zero;
        for (var i = 0; i < n; i++)
        {
            var start = offDiagonalOnly ? i + 1 : 0;
            for (var j = start; j < n; j++)
            {
                var scaled = matrix[i, j] / scale;
                scaledSum += offDiagonalOnly ? s_two * scaled * scaled : scaled * scaled;
            }
        }

        var root = NumericPrecision<T>.Sqrt(scaledSum);
        return scale >= T.One / root ? T.One : scale * root;
    }

    private static bool AllPairsRelativelyNegligible(T[,] a, T[] diagonalRoots, int n)
    {
        for (var p = 0; p < n - 1; p++)
        {
            for (var q = p + 1; q < n; q++)
            {
                if (!IsRelativelyNegligible(a[p, q], diagonalRoots[p], diagonalRoots[q]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    // |a_pq| <= tolerance·sqrt|a_pp|·sqrt|a_qq|. The bound is formed from the square roots so that
    // neither a large nor a graded pair squares out of the range of T.
    private static bool IsRelativelyNegligible(T offDiagonal, T rootP, T rootQ)
        => T.Abs(offDiagonal) <= (s_offDiagonalTolerance * rootP) * rootQ;

    // Adds an increment to diagonal entry i, carrying the rounding error of the sum in the low-order
    // part (Knuth's error-free two-sum, exact under round-to-nearest in any radix barring overflow),
    // then renormalizes so the low-order part stays below half an ulp of the entry.
    private static void UpdateDiagonal(T[,] a, T[] lowDiagonal, int i, T increment)
    {
        var (sum, error) = TwoSum(a[i, i], increment);
        (a[i, i], lowDiagonal[i]) = TwoSum(sum, lowDiagonal[i] + error);
    }

    private static (T Sum, T Error) TwoSum(T x, T y)
    {
        var sum = x + y;
        var yPart = sum - x;
        return (sum, (x - (sum - yPart)) + (y - yPart));
    }

    private static void Rotate(T[,] a, T s, T tau, int i, int j, int k, int l)
    {
        var g = a[i, j];
        var h = a[k, l];
        a[i, j] = g - s * (h + tau * g);
        a[k, l] = h + s * (g - tau * h);
    }

    // The update form g - s*(h + tau*g) adds h + tau*g before scaling by s. For two entries within a
    // factor of two of the working range that sum can leave it although both rotated entries are
    // finite; the rotation's own form c*g - s*h, s*g + c*h has terms bounded by max(|g|, |h|).
    // Only matrices that ExceedsRotationRange flags take this path, so the common case keeps the
    // original form without exception handling.
    private static void RotateRowsGuarded(T[,] a, T s, T tau, T c, int n, int p, int q)
    {
        for (var j = 0; j < p; j++)
        {
            RotateGuarded(a, s, tau, c, j, p, j, q);
        }

        for (var j = p + 1; j < q; j++)
        {
            RotateGuarded(a, s, tau, c, p, j, j, q);
        }

        for (var j = q + 1; j < n; j++)
        {
            RotateGuarded(a, s, tau, c, p, j, q, j);
        }
    }

    private static void RotateGuarded(T[,] a, T s, T tau, T c, int i, int j, int k, int l)
    {
        var g = a[i, j];
        var h = a[k, l];
        var rotatedG = T.Zero;
        var rotatedH = T.Zero;
        var finite = false;
        try
        {
            rotatedG = g - s * (h + tau * g);
            rotatedH = h + s * (g - tau * h);
            finite = T.IsFinite(rotatedG) && T.IsFinite(rotatedH);
        }
        catch (OverflowException)
        {
            // Decimal throws where IEEE arithmetic produces infinity.
        }

        if (!finite)
        {
            rotatedG = (c * g) - (s * h);
            rotatedH = (s * g) + (c * h);
        }

        a[i, j] = rotatedG;
        a[k, l] = rotatedH;
    }

    // Every entry of an orthogonally rotated copy of A is bounded by the Frobenius norm, which is at
    // most n times the largest entry, and h + tau*g by twice that. Checked once per decomposition:
    // true when four times that bound leaves the working range, i.e. when the rotation update may
    // overflow and must take the guarded path.
    private static bool ExceedsRotationRange(T[,] a, int n)
    {
        var largest = T.Zero;
        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                largest = T.Max(largest, T.Abs(a[i, j]));
            }
        }

        try
        {
            return !T.IsFinite(largest * T.CreateChecked(4 * Math.Max(n, 1)));
        }
        catch (OverflowException)
        {
            return true;
        }
    }

    private static EigenResult<T> BuildResult(T[,] a, T[] lowDiagonal, T[,] v, int n)
    {
        // Extract eigenvalues from the diagonal, rounding its low-order parts in once.
        var eigenvalues = new T[n];
        for (var i = 0; i < n; i++)
        {
            eigenvalues[i] = a[i, i] + lowDiagonal[i];
        }

        // Sort in descending order, keeping eigenvectors aligned.
        var indices = new int[n];
        for (var i = 0; i < n; i++)
        {
            indices[i] = i;
        }

        // Numerical Recipes eigsrt: select the last equal maximum and swap
        // the corresponding eigenvector column through its index.
        for (var i = 0; i < n - 1; i++)
        {
            var maxIdx = i;
            for (var j = i + 1; j < n; j++)
            {
                if (eigenvalues[j] >= eigenvalues[maxIdx])
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
/// <c>T = double</c>, retaining the facade's double working precision. Inputs
/// and results are converted between <c>decimal</c> and <c>double</c>; use
/// <c>JacobiEigenDecomposition&lt;decimal&gt;</c> for rotations in decimal arithmetic.
/// </summary>
[Obsolete(
    "Use JacobiEigenDecomposition<double> for the legacy working precision, or JacobiEigenDecomposition<decimal> for decimal rotations. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class JacobiEigenDecomposition
{
    /// <inheritdoc cref="JacobiEigenDecomposition{T}.Decompose"/>
    public static EigenResult Decompose(decimal[,] matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);

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
        var resultValues = result.Values;
        var resultVectors = result.Vectors;

        // Convert back to decimal.
        var values = new decimal[n];
        var vectors = new decimal[n, n];
        for (var i = 0; i < n; i++)
        {
            values[i] = (decimal)resultValues[i];
            for (var j = 0; j < n; j++)
            {
                vectors[i, j] = (decimal)resultVectors[i, j];
            }
        }

        return new EigenResult(values, vectors);
    }
}
