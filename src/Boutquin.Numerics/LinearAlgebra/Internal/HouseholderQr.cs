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
/// (Householder QR) and section 5.3.3 (QR-based least squares). The per-column reflector
/// uses the nonzero-tail sign convention of LAPACK's <c>DLARFG/DGEQRF</c>:
/// <list type="bullet">
/// <item><description><c>alpha = -sign(x0) . ||x||2</c></description></item>
/// <item><description><c>v = (x0 - alpha, x1, ..., x(m-k-1))</c>, normalized by its first component for large columns</description></item>
/// <item><description><c>tau = 2 / (vtv)</c></description></item>
/// <item><description><c>H = I - tau . v . vt</c></description></item>
/// </list>
/// For a one-entry or zero-tail nonzero column, the reflector is the identity:
/// the leading entry is preserved and the effective <c>tau</c> is zero, following
/// <c>DLARFG</c>. A zero stored reflector represents this case for both
/// <c>ApplyQ</c> and <c>ApplyQTranspose</c>.
/// Each reflector is applied in-place to <c>A</c>'s trailing columns and to <c>y</c>.
/// After <c>n</c> reflectors, <c>A</c>'s upper-triangular block holds <c>R</c> and
/// <c>y[0..n-1]</c> holds the leading block of <c>c = Qty</c>, which is all that's
/// needed for least-squares back-substitution.
/// </para>
/// <para>
/// Stability: Householder QR is backward stable under the standard rounding model.
/// The conditioning of the underlying least-squares problem also depends on residual
/// size and can include a squared-condition-number term. QR avoids the additional
/// loss from explicitly forming <c>AtA</c>, particularly for small residuals
/// (Golub &amp; Van Loan, sections 5.3.6–5.3.7).
/// NIST StRD's <c>Wampler1–5</c> share the degree-5 design on <c>x = 0,…,20</c>
/// (cond ≈ 6.4e6). Wampler4/5 increase response noise, not the design condition number.
/// Accurate recovery amid that cancellation uses the iterative
/// refinement in <see cref="Boutquin.Numerics.Solvers.OrdinaryLeastSquares{T}"/>. The
/// NIST StRD tests hold Wampler4/5 to 1e-12 relative of the certified coefficients.
/// </para>
/// <para>
/// Scale handling: a pivot column whose largest entry lies outside
/// <c>[1/s, s]</c>, with <c>s = sqrt(B / (4m))</c>, has its reflector normalised by that
/// entry before any product is formed, so <c>vtv</c> stays in <c>[1, m]</c> and
/// <c>tau = 2 / vtv</c> can neither overflow nor underflow at any column scale. Columns
/// inside that band keep the raw arithmetic. <c>B</c> is <c>decimal.MaxValue</c>
/// (~7.9e28, standing in for <c>T.MaxValue</c>, which <see cref="IFloatingPoint{TSelf}"/>
/// does not expose) when <typeparamref name="T"/> represents it, and otherwise the
/// largest finite power of two of <typeparamref name="T"/> (32768 for <c>Half</c>). A
/// <typeparamref name="T"/> that is not IEEE 754 (<c>decimal</c>) normalises every
/// reflector: its fixed 28-place scale resolves an absolute step rather than a relative
/// one, so a raw <c>vtv</c> far from 1 would lose significant digits in <c>tau</c> and in
/// every product.
/// </para>
/// <para>
/// Rank decision: the pivot column at step <c>k</c> is numerically dependent on the
/// columns already factored when the norm of its trailing part is at most
/// <c>max(m, n) . eps</c> times the norm of the same column in the original matrix
/// (<c>eps</c> is the machine epsilon of <typeparamref name="T"/>), the column-relative
/// form of the <c>max(M, N) . eps</c> rank tolerance used by LAPACK and NumPy. The test
/// depends on the pivot order; changing a column's scale can change that order.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
internal static class HouseholderQr<T>
    where T : IFloatingPoint<T>
{
    private static readonly T s_two = T.CreateChecked(2);

    // Machine epsilon: the smallest power of two e (found by halving from 1) with 1 + e != 1.
    private static readonly T s_machineEpsilon = ComputeMachineEpsilon();

    // IEEE types keep full relative precision at every in-range magnitude. Any other type
    // (decimal: 96-bit integer significand over a fixed 28-place scale) resolves a fixed
    // absolute step, so its reflectors are always normalized to keep vtv and tau O(1).
    private static readonly bool s_alwaysNormalize = !Array.Exists(
        typeof(T).GetInterfaces(),
        static i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IFloatingPointIeee754<>));

    // Largest magnitude the scale band may assume: decimal.MaxValue (IFloatingPoint exposes
    // no MaxValue) when T represents it, else T's largest finite power of two (Half).
    private static readonly T s_overflowBound = ComputeOverflowBound();

    // Headroom factor 4 * m for the column/response overflow tests: a column sum of m products each
    // below the bound squared is kept inside the range with a factor-4 margin. Row counts reach 2^30 - 1
    // in the supported length domain, so the product is formed in long; the int product wraps negative
    // from m = 2^29.
    internal static long RowHeadroomFactor(int m) => 4L * m;

    // Exponent e of T's largest finite power of two 2^e (1023 for double); T's largest finite value
    // is below 2^(e + 1). Used only by the IEEE 754 range decisions.
    private static readonly int s_largestPowerOfTwoExponent = ComputeLargestPowerOfTwoExponent();

    /// <summary>Machine epsilon of <typeparamref name="T"/> (2^-52 for <c>double</c>).</summary>
    internal static T MachineEpsilon => s_machineEpsilon;

    /// <summary>
    /// <see langword="true"/> when <typeparamref name="T"/> keeps full relative precision at
    /// every in-range magnitude (an IEEE 754 type); <see langword="false"/> for a type with a
    /// fixed absolute resolution such as <c>decimal</c>.
    /// </summary>
    internal static bool HasFloatingExponent => !s_alwaysNormalize;

    /// <summary>
    /// Whether a raw sum of squares kept its relative precision: it is a normal number of an
    /// IEEE 754 type, so none of its terms was rounded away below the normal range. A type with
    /// a fixed absolute resolution (<c>decimal</c>) never certifies its raw squares, because
    /// every square far from one is rounded to the same absolute step.
    /// </summary>
    internal static bool SquareSumKeepsPrecision(T sum) => HasFloatingExponent && T.IsNormal(sum);

    /// <summary>
    /// Decides whether a response must be rescaled before it is transformed by <see cref="Factor"/>,
    /// and by what exact factor. The reflectors are orthogonal, so every entry of <c>Qt . y</c> is
    /// bounded by <c>||y||2</c>, and the leading entry can equal it: a response whose Euclidean norm
    /// exceeds the largest finite value cannot be transformed (an IEEE 754 type produces infinity,
    /// <c>decimal</c> throws) even when the least-squares solution is representable. A response with
    /// <c>||y||2 &lt;= B . (1 - 1e-6)</c> returns <see langword="false"/>, which keeps every response
    /// the factorization can already transform on its established arithmetic; the margin covers the
    /// rounding of the norm and of the reflectors. <c>B</c> is <c>decimal.MaxValue</c> for
    /// <c>decimal</c> and the largest finite value (to within a factor <c>1 - 2^-p</c>) of an IEEE 754
    /// type. Otherwise the factor is exact: a power of ten for <c>decimal</c> (a shift of the decimal
    /// point) that brings the norm to about <c>1e27</c>, and a power of two for an IEEE 754 type that
    /// brings it to at most a quarter of <c>B</c>. The centred deviations of the response from its mean
    /// are bounded by the same norm.
    /// </summary>
    /// <param name="y">Response vector.</param>
    /// <param name="down">The factor applied to the response (<c>1</c> when no rescaling is needed).</param>
    /// <param name="up">Its exact reciprocal, applied to the solution.</param>
    /// <returns><see langword="true"/> when the response must be factored as <c>y . down</c>.</returns>
    internal static bool TryResponseScale(ReadOnlySpan<T> y, out T down, out T up)
    {
        down = T.One;
        up = T.One;

        // The norm is only a magnitude decision, so it is formed in double (two-pass scaled, as in
        // ColumnNorm, and as a base-2 logarithm so that it cannot itself overflow); nothing
        // computed here enters a returned value.
        var largest = 0.0;
        foreach (var value in y)
        {
            largest = Math.Max(largest, Math.Abs(double.CreateChecked(value)));
        }

        if (largest == 0.0)
        {
            return false;
        }

        var sum = 0.0;
        foreach (var value in y)
        {
            var scaled = double.CreateChecked(value) / largest;
            sum += scaled * scaled;
        }

        if (HasFloatingExponent)
        {
            // B = 2^(e + 1) . (1 - 2^-p) with 2^e the largest finite power of two. For double the
            // norm itself can exceed double's range, so it is compared as a base-2 logarithm.
            var log2Norm = Math.Log2(largest) + (0.5 * Math.Log2(sum));
            if (log2Norm <= s_largestPowerOfTwoExponent + 1 + Math.Log2(1.0 - 1e-6))
            {
                return false;
            }

            var shift = Math.Max(1, (int)Math.Ceiling(log2Norm - s_largestPowerOfTwoExponent) + 1);
            down = T.CreateChecked(Math.ScaleB(1.0, -shift));
            up = T.CreateChecked(Math.ScaleB(1.0, shift));
            return true;
        }

        // decimal's norm (at most sqrt(m) . 7.9e28) is finite in double.
        var norm = largest * Math.Sqrt(sum);
        if (norm <= double.CreateChecked(s_overflowBound) * (1.0 - 1e-6))
        {
            return false;
        }

        var k = Math.Clamp((int)Math.Ceiling(Math.Log10(norm) - 27.0), 1, 28);
        down = T.CreateChecked(new decimal(1, 0, 0, false, (byte)k));
        var ten = T.CreateChecked(10);
        for (var i = 0; i < k; i++)
        {
            up *= ten;
        }

        return true;
    }

    /// <summary>
    /// Decides whether design column <paramref name="col"/> of <paramref name="a"/> must be rescaled
    /// before <see cref="Factor"/>, by the same rule and exact factors as <see cref="TryResponseScale"/>.
    /// <c>R</c>'s diagonal is a column norm and every entry of a reflected column is bounded by it, so a
    /// column whose Euclidean norm exceeds the largest finite value cannot be factored (an IEEE 754 type
    /// produces infinity and the rank test compares infinity with infinity; <c>decimal</c> throws) even
    /// when the least-squares solution is representable. Scaling column <c>j</c> by <c>d_j</c> scales the
    /// solution's entry <c>j</c> by <c>1 / d_j</c>. Small decimal columns are also scaled upward by
    /// exact powers of ten until their maximum entry is at least one, so their trailing rounding
    /// residues are measured at a scale appropriate for the relative rank test. Scaling may change
    /// the column pivot order.
    /// </summary>
    /// <param name="a">Design matrix, <c>m x n</c>. Not mutated.</param>
    /// <param name="col">Column index.</param>
    /// <param name="down">The factor applied to the column (<c>1</c> when no rescaling is needed).</param>
    /// <param name="up">Its exact reciprocal.</param>
    /// <returns><see langword="true"/> when the column must be factored as <c>a[:, col] . down</c>.</returns>
    internal static bool TryColumnScale(T[,] a, int col, out T down, out T up)
    {
        var column = new T[a.GetLength(0)];
        var maximum = T.Zero;
        for (var i = 0; i < column.Length; i++)
        {
            column[i] = a[i, col];
            maximum = T.Max(maximum, T.Abs(column[i]));
        }

        // Decimal elimination leaves absolute 1e-28 rounding residues. Bring a small
        // column to unit scale before applying a relative rank test, using only exact
        // decimal shifts so independent tiny columns retain their information too.
        if (s_alwaysNormalize && maximum > T.Zero && maximum < T.One)
        {
            var ten = T.CreateChecked(10);
            down = T.One;
            up = T.One;
            while (maximum < T.One)
            {
                maximum *= ten;
                down *= ten;
                up /= ten;
            }

            return true;
        }

        return TryResponseScale(column, out down, out up);
    }

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
    /// designs such as NIST's <c>Filip</c> (degree 10 with predictors approximately -8.8 to -3.1) and
    /// <c>Wampler4/5</c> (degree 5, cond ≈ 6.4e6). Pivoting selects the largest
    /// remaining column norm; it does not promise a fixed digit count. Callers who care about the pivot order
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
    /// <see langword="false"/> if, at some step, the pivot column's trailing norm is zero or
    /// at most <c>max(m, n) . eps</c> times that column's original norm (the matrix is
    /// numerically rank-deficient and OLS is ill-defined). Callers surface this as an
    /// argument or invalid-operation exception rather than a degenerate result.
    /// </returns>
    public static bool Factor(T[,] a, T[] y, T[] diagonal, int[] permutation)
    {
        var m = a.GetLength(0);
        var n = a.GetLength(1);

        var originalNorms = new T[n];
        for (var k = 0; k < n; k++)
        {
            permutation[k] = k;
            originalNorms[k] = ColumnNorm(a, k, 0, m);
        }

        var rankTolerance = T.CreateChecked(Math.Max(m, n)) * s_machineEpsilon;

        for (var k = 0; k < n; k++)
        {
            // Column pivoting: among columns k..n-1, find the one with the largest
            // 2-norm restricted to rows k..m-1, and swap it into position k.
            var pivot = k;
            var pivotNorm = ColumnNorm(a, k, k, m);
            for (var j = k + 1; j < n; j++)
            {
                var candidate = ColumnNorm(a, j, k, m);
                if (candidate > pivotNorm)
                {
                    pivotNorm = candidate;
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

            // Column-relative rank test: the part of this column orthogonal to the columns
            // already factored is rounding residue when it is below max(m, n).eps of the
            // column's own original norm. Being relative, it neither rejects a full-rank
            // column for being small nor accepts a dependent one for being large.
            if (xNorm <= rankTolerance * originalNorms[permutation[k]])
            {
                diagonal[k] = T.Zero;
                return false;
            }

            // alpha = -sign(x0) . ||x||, with sign(+0) = +1 as in LAPACK DLARFG (Fortran
            // SIGN), so a zero x0 gives alpha = -||x||. The sign-flip convention keeps
            // v's leading component x0 - alpha far from cancellation.
            var x0 = a[k, k];
            var hasNonzeroTail = false;
            for (var i = k + 1; i < m; i++)
            {
                if (a[i, k] != T.Zero)
                {
                    hasNonzeroTail = true;
                    break;
                }
            }

            // LAPACK DLARFG returns tau=0 when n<=1 or the tail norm is zero.
            // Preserve alpha and encode the identity as a zero stored reflector;
            // ApplyQ and ApplyQTranspose already skip that representation.
            if (!hasNonzeroTail)
            {
                diagonal[k] = x0;
                a[k, k] = T.Zero;
                continue;
            }

            var alpha = !T.IsNegative(x0) ? -xNorm : xNorm;
            diagonal[k] = alpha;

            // Preserve the original arithmetic at ordinary scales. Outside the safe
            // band, normalize v before forming products that would overflow (large
            // columns) or underflow (tiny columns); normalized vtv lies in [1, m].
            // A type without a floating exponent (decimal) always normalizes: its fixed
            // absolute resolution would otherwise cost digits on any column far from 1.
            var safeMagnitude = NumericPrecision<T>.Sqrt(
                s_overflowBound / (T.CreateChecked(4) * T.CreateChecked(m)));
            var normalized = s_alwaysNormalize || xMax > safeMagnitude || xMax * safeMagnitude < T.One;
            var v0Scaled = normalized ? (x0 / xMax) - (alpha / xMax) : T.Zero;
            var v0 = normalized ? T.One : x0 - alpha;
            a[k, k] = v0;

            var vtv = v0 * v0;
            for (var i = k + 1; i < m; i++)
            {
                if (normalized)
                {
                    a[i, k] = (a[i, k] / xMax) / v0Scaled;
                }

                vtv += a[i, k] * a[i, k];
            }

            if (vtv == T.Zero)
            {
                continue;
            }

            var tau = s_two / vtv;

            // Apply H = I - tau.v.vt to trailing columns a[k..m-1, j] for j = k+1..n-1.
            for (var j = k + 1; j < n; j++)
            {
                var columnScale = T.Zero;
                var reflectorScale = T.Abs(v0);
                for (var i = k; i < m; i++)
                {
                    columnScale = T.Max(columnScale, T.Abs(a[i, j]));
                    reflectorScale = T.Max(reflectorScale, T.Abs(a[i, k]));
                }

                var productScale = T.Max(T.One, T.Max(reflectorScale, tau * reflectorScale));
                if (columnScale > s_overflowBound / T.CreateChecked(RowHeadroomFactor(m)) / productScale)
                {
                    // Cancel the rank-one update at unit column scale before restoring
                    // magnitude; its intermediate may exceed a finite final R entry.
                    var dot = T.Zero;
                    for (var i = k; i < m; i++)
                    {
                        dot += a[i, k] * (a[i, j] / columnScale);
                    }

                    var update = tau * dot;
                    for (var i = k; i < m; i++)
                    {
                        a[i, j] = (a[i, j] / columnScale - update * a[i, k]) * columnScale;
                    }

                    continue;
                }

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
            ApplyReflector(a, k, y);
        }

        return true;
    }

    /// <summary>
    /// Solves <c>R . x = c</c> where <c>R</c>'s strict upper triangle is stored in the
    /// upper triangle of <paramref name="a"/> and its diagonal is <paramref name="diagonal"/>.
    /// The first <paramref name="n"/> entries of <paramref name="c"/> carry the right-hand
    /// side. Returns the solution in the original column order by undoing the QR's
    /// column pivoting using <paramref name="permutation"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException">Any diagonal entry is zero (rank-deficient R).</exception>
    public static T[] SolveUpperTriangular(T[,] a, T[] diagonal, T[] c, int n, int[] permutation)
    {
        var permuted = new T[n];
        for (var i = n - 1; i >= 0; i--)
        {
            var dii = diagonal[i];
            if (dii == T.Zero)
            {
                throw new InvalidOperationException(
                    $"Upper-triangular R has a zero diagonal at row {i}; design matrix is rank-deficient.");
            }

            permuted[i] = SubstituteRow(a, dii, c[i], permuted, i, i + 1, n, false);
            if (!T.IsFinite(permuted[i]))
            {
                throw new InvalidOperationException("QR solution is not finite in the working type.");
            }
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
        => BuildScaledXtXInverse(a, diagonal, permutation, T.One);

    /// <summary>
    /// Builds <c>scale² (XtX)^-1</c> by solving <c>R Z = scale I</c> before
    /// forming <c>Z Zt</c>, avoiding an overflowing unscaled inverse when the
    /// scaled covariance is representable.
    /// </summary>
    public static T[,] BuildScaledXtXInverse(T[,] a, T[] diagonal, int[] permutation, T scale)
        => BuildScaledXtXInverse(a, diagonal, permutation, scale, out _);

    /// <summary>
    /// Builds <c>scale² (XtX)^-1</c> as above and, from the same solve, the square roots of its
    /// diagonal in the original column order. The diagonal entry <c>i</c> is the squared
    /// Euclidean norm of row <c>i</c> of <c>Z</c>, so each square root is that norm: it stays
    /// representable when the squared diagonal under- or overflows (a <c>double</c> standard
    /// error near <c>1e-200</c> or <c>1e200</c>; for <c>decimal</c>, one far below one). Where
    /// the raw diagonal kept its precision it is square-rooted unchanged.
    /// </summary>
    /// <param name="a">QR-factored matrix (upper triangle holds <c>R</c> above the diagonal).</param>
    /// <param name="diagonal">Diagonal of <c>R</c> (length <c>n</c>).</param>
    /// <param name="permutation">Column permutation from <see cref="Factor"/>.</param>
    /// <param name="scale">Standard-deviation scale folded into the triangular solve.</param>
    /// <param name="standardErrors">Receives <c>scale · sqrt(diag((XtX)^-1))</c> in original column order.</param>
    /// <returns>Symmetric <c>n x n</c> matrix equal to <c>scale² (XtX)^-1</c> in original column order.</returns>
    public static T[,] BuildScaledXtXInverse(T[,] a, T[] diagonal, int[] permutation, T scale, out T[] standardErrors)
        => BuildScaledXtXInverse(a, diagonal, permutation, scale, null, out standardErrors);

    /// <summary>
    /// As above for a factored design <c>X . D</c> whose columns were rescaled by
    /// <paramref name="columnScales"/> (<c>D</c>, in original column order, or <see langword="null"/>
    /// for <c>D = I</c>): returns <c>scale² (XtX)^-1 = D . scale² ((XD)t(XD))^-1 . D</c> and the
    /// standard errors of <c>X</c>. Row <c>i</c> of <c>scale . R^-1</c> is multiplied by its column's
    /// <c>d</c> before any product is formed, so each entry is computed at its returned magnitude.
    /// </summary>
    /// <param name="a">QR-factored matrix (upper triangle holds <c>R</c> above the diagonal).</param>
    /// <param name="diagonal">Diagonal of <c>R</c> (length <c>n</c>).</param>
    /// <param name="permutation">Column permutation from <see cref="Factor"/>.</param>
    /// <param name="scale">Standard-deviation scale folded into the triangular solve.</param>
    /// <param name="columnScales">Per-column factors applied to the design before <see cref="Factor"/>, or <see langword="null"/>.</param>
    /// <param name="standardErrors">Receives <c>scale · sqrt(diag((XtX)^-1))</c> in original column order.</param>
    /// <returns>Symmetric <c>n x n</c> matrix equal to <c>scale² (XtX)^-1</c> in original column order.</returns>
    public static T[,] BuildScaledXtXInverse(
        T[,] a, T[] diagonal, int[] permutation, T scale, T[]? columnScales, out T[] standardErrors)
    {
        var n = diagonal.Length;

        // Include the standard-deviation scale in the triangular solve.
        var rInverse = new T[n, n];
        var column = new T[n];
        try
        {
            for (var col = 0; col < n; col++)
            {
                Array.Clear(column);
                for (var i = n - 1; i >= 0; i--)
                {
                    var dii = diagonal[i];
                    if (dii == T.Zero)
                    {
                        throw new InvalidOperationException(
                            $"Upper-triangular R has a zero diagonal at row {i}; design matrix is rank-deficient.");
                    }

                    // The numerator r_ij . z_j can exceed the range although z_i = (rhs - sum) / r_ii
                    // fits; SubstituteRow then divides each term by r_ii before combining them.
                    column[i] = SubstituteRow(a, dii, i == col ? scale : T.Zero, column, i, i + 1, n, false);
                    rInverse[i, col] = columnScales is null ? column[i] : column[i] * columnScales[permutation[i]];
                }
            }

            // Compute (RtR)^-1 = R^-1 . R^-t in the permuted frame.
            var inversePermuted = new T[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = i; j < n; j++)
                {
                    var dot = T.Zero;
                    // The inner loop walks the overlapping tails of rows i and j of R⁻¹.
                    // Because the outer loop runs j from i upward, j ≥ i always holds here,
                    // so the lower bound is simply j (the former Math.Max(i, j) was a no-op).
                    for (var k = j; k < n; k++)
                    {
                        dot += rInverse[i, k] * rInverse[j, k];
                    }

                    if (!T.IsFinite(dot))
                    {
                        // Opposite-signed overflowing products can produce NaN even
                        // when the final covariance has a definite sign or fits T.
                        var left = new T[n - j];
                        var right = new T[n - j];
                        for (var k = j; k < n; k++)
                        {
                            left[k - j] = rInverse[i, k];
                            right[k - j] = rInverse[j, k];
                        }

                        var leftExponent = ScaledArithmetic<T>.ScaleExponent(left);
                        var rightExponent = ScaledArithmetic<T>.ScaleExponent(right);
                        dot = ScaledArithmetic<T>.Rescale(
                            ScaledArithmetic<T>.Dot(left, right, leftExponent, rightExponent),
                            leftExponent + rightExponent);
                    }

                    inversePermuted[i, j] = dot;
                    inversePermuted[j, i] = dot;
                }
            }

            // Un-permute: (XtX)^-1[permutation[i], permutation[j]] = inversePermuted[i, j].
            var result = new T[n, n];
            var errors = new T[n];
            for (var i = 0; i < n; i++)
            {
                errors[permutation[i]] = RowNorm(rInverse, i, n, inversePermuted[i, i]);
                for (var j = 0; j < n; j++)
                {
                    result[permutation[i], permutation[j]] = inversePermuted[i, j];
                }
            }

            standardErrors = errors;
            return result;
        }
        catch (OverflowException ex)
        {
            throw new InvalidOperationException(
                "The inverse factor or covariance exceeds the numeric type's range; rescale the problem.",
                ex);
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
        for (var k = 0; k < diagonal.Length; k++)
        {
            ApplyReflector(a, k, r);
        }
    }

    /// <summary>
    /// Applies <c>Q</c> (the orthogonal factor from <see cref="Factor"/>) in-place to
    /// <paramref name="r"/>: the stored reflectors in reverse order, the inverse of
    /// <see cref="ApplyQTranspose"/>.
    /// </summary>
    /// <param name="a">QR-factored design matrix from <see cref="Factor"/>.</param>
    /// <param name="diagonal">Diagonal of <c>R</c> (length <c>n</c>).</param>
    /// <param name="r">Vector of length <c>m</c>. Overwritten with <c>Q . r</c>.</param>
    public static void ApplyQ(T[,] a, T[] diagonal, T[] r)
    {
        for (var k = diagonal.Length - 1; k >= 0; k--)
        {
            ApplyReflector(a, k, r);
        }
    }

    /// <summary>
    /// Solves <c>Rt . h = Pt . g</c> by forward substitution, where <paramref name="g"/> is
    /// in the original column order and the returned <c>h</c> is in the permuted frame
    /// of <see cref="Factor"/> (the frame in which <c>Qt . r</c> is expressed).
    /// </summary>
    /// <exception cref="InvalidOperationException">Any diagonal entry is zero (rank-deficient R).</exception>
    public static T[] SolveUpperTriangularTranspose(T[,] a, T[] diagonal, T[] g, int n, int[] permutation)
    {
        var h = new T[n];
        for (var i = 0; i < n; i++)
        {
            var dii = diagonal[i];
            if (dii == T.Zero)
            {
                throw new InvalidOperationException(
                    $"Upper-triangular R has a zero diagonal at row {i}; design matrix is rank-deficient.");
            }

            h[i] = SubstituteRow(a, dii, g[permutation[i]], h, i, 0, i, true);
            if (!T.IsFinite(h[i]))
            {
                throw new InvalidOperationException("QR transposed solution is not finite in the working type.");
            }
        }

        return h;
    }

    private static T SubstituteRow(T[,] a, T diagonal, T rhs, T[] solution, int row, int start, int end, bool transpose)
    {
        var sum = rhs;
        var overflowed = false;
        try
        {
            for (var j = start; j < end; j++)
            {
                sum -= (transpose ? a[j, row] : a[row, j]) * solution[j];
            }
        }
        catch (OverflowException)
        {
            // The numerator can exceed the working range although its quotient fits.
            overflowed = true;
        }

        if (!overflowed && T.IsFinite(sum))
        {
            return sum / diagonal;
        }

        // Divide each term of the numerator by the diagonal before combining them, so the
        // quotient's own magnitude is the largest value formed. Normalising the whole row by
        // its largest entry instead (the entries here are of order one beside a numerator of
        // order 1e28) rounds the unit entries and the diagonal to zero at decimal's fixed
        // 28-place resolution and refuses a representable solution.
        var quotient = rhs / diagonal;
        for (var j = start; j < end; j++)
        {
            quotient -= ((transpose ? a[j, row] : a[row, j]) / diagonal) * solution[j];
        }

        return quotient;
    }

    private static void ApplyReflector(T[,] a, int k, T[] r)
    {
        var m = a.GetLength(0);

        // Reflector H_k: v[0] stored in a[k, k]; v[1..m-k-1] stored in a[k+1..m-1, k].
        // tau = 2/(vtv). Skip when v is zero (Factor skipped the reflection too).
        var v0 = a[k, k];

        var vtv = v0 * v0;
        for (var i = k + 1; i < m; i++)
        {
            vtv += a[i, k] * a[i, k];
        }

        if (vtv == T.Zero)
        {
            return;
        }

        var tau = s_two / vtv;
        var responseScale = T.Zero;
        var reflectorScale = T.Zero;
        for (var i = k; i < m; i++)
        {
            responseScale = T.Max(responseScale, T.Abs(r[i]));
            reflectorScale = T.Max(reflectorScale, T.Abs(a[i, k]));
        }

        // Work with a unit-sized response when the rank-one products have
        // insufficient headroom in either v' r or tau * (v' r). Both the
        // reflector and response magnitudes matter for unnormalized reflectors.
        // Restore the response scale only after cancellation.
        var productScale = T.Max(T.One, T.Max(reflectorScale, tau * reflectorScale));
        var scaleResponse = responseScale > s_overflowBound / T.CreateChecked(RowHeadroomFactor(m)) / productScale;
        if (scaleResponse)
        {
            var dot = v0 * (r[k] / responseScale);
            for (var i = k + 1; i < m; i++)
            {
                dot += a[i, k] * (r[i] / responseScale);
            }

            var update = tau * dot;
            for (var i = k; i < m; i++)
            {
                r[i] = (r[i] / responseScale - update * a[i, k]) * responseScale;
            }

            return;
        }

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

    private static T ComputeOverflowBound()
    {
        var decimalMax = T.CreateChecked(decimal.MaxValue);
        if (T.IsFinite(decimalMax))
        {
            return decimalMax;
        }

        var bound = T.One;
        for (var i = 0; i < 16384; i++)
        {
            var next = bound * s_two;
            if (!T.IsFinite(next))
            {
                break;
            }

            bound = next;
        }

        return bound;
    }

    private static int ComputeLargestPowerOfTwoExponent()
    {
        if (s_alwaysNormalize)
        {
            // Not an IEEE 754 type: decimal throws on overflow, and its range decisions use s_overflowBound.
            return 0;
        }

        var bound = T.One;
        var exponent = 0;
        for (var i = 0; i < 16384; i++)
        {
            var next = bound * s_two;
            if (!T.IsFinite(next))
            {
                break;
            }

            bound = next;
            exponent++;
        }

        return exponent;
    }

    private static T ComputeMachineEpsilon()
    {
        var half = T.One / T.CreateChecked(2);
        var epsilon = T.One;
        for (var i = 0; i < 256; i++)
        {
            var next = epsilon * half;
            if (T.One + next == T.One)
            {
                break;
            }

            epsilon = next;
        }

        return epsilon;
    }

    private static T ColumnNorm(T[,] a, int col, int rowStart, int rowEnd)
    {
        // Two-pass scaled sum-of-squares keeps the norm finite when the sum of
        // raw squares would overflow, and handles wide-range polynomial designs.
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

        return xMax * NumericPrecision<T>.Sqrt(sum);
    }

    /// <summary>
    /// Euclidean norm of the upper-triangular row <paramref name="row"/> of
    /// <paramref name="z"/> (columns <paramref name="row"/> to <paramref name="n"/> − 1).
    /// <paramref name="rawSumOfSquares"/> is the plain sum of those squares, returned
    /// square-rooted when it kept its precision; otherwise the entries are divided by the
    /// row's largest magnitude before squaring, so the norm is representable whenever it is.
    /// </summary>
    private static T RowNorm(T[,] z, int row, int n, T rawSumOfSquares)
    {
        if (SquareSumKeepsPrecision(rawSumOfSquares))
        {
            return NumericPrecision<T>.Sqrt(rawSumOfSquares);
        }

        var largest = T.Zero;
        for (var k = row; k < n; k++)
        {
            largest = T.Max(largest, T.Abs(z[row, k]));
        }

        if (largest == T.Zero)
        {
            return T.Zero;
        }

        if (!T.IsFinite(largest))
        {
            // An entry already overflowed: the norm is at least that large, so it saturates
            // with it (as the covariance diagonal does) rather than dividing infinity by itself.
            return largest;
        }

        var sum = T.Zero;
        for (var k = row; k < n; k++)
        {
            var scaled = z[row, k] / largest;
            sum += scaled * scaled;
        }

        return largest * NumericPrecision<T>.Sqrt(sum);
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
