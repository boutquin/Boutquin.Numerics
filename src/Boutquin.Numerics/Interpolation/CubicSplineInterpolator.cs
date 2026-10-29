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

namespace Boutquin.Numerics.Interpolation;

/// <summary>
/// Cubic spline interpolator with C² continuity at all interior knots,
/// supporting six boundary conditions — natural, clamped, classical not-a-knot
/// (de Boor 1978), modified and revised not-a-knot, and Q-spline (Jarre 2025, arXiv:2507.05083).
/// Tridiagonal system solved by the Thomas algorithm in O(n), unconditionally
/// stable for the diagonally dominant systems arising in cubic spline construction
/// (Faires &amp; Burden, §3.5).
/// </summary>
/// <typeparam name="T">The floating-point type for interpolation.</typeparam>
/// <remarks>
/// Tier A: Requires only basic floating-point arithmetic.
/// </remarks>
public sealed class CubicSplineInterpolator<T> where T : unmanaged, IFloatingPoint<T>
{
    private readonly T[] _x;
    private readonly T[] _originalX;
    private readonly T[] _originalY;
    private readonly T[] _a;
    private readonly T[] _b;
    private readonly T[] _c;
    private readonly T[] _d;
    private readonly T _ordinateScale = T.One;
    private readonly bool _scaleOrdinates;
    private readonly T _abscissaScale = T.One;
    private readonly T _coordinateScale = T.One;

    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_six = T.CreateChecked(6);

    private const int StackAllocThreshold = 256;

    /// <summary>
    /// Constructs a natural cubic spline (S''(x₀) = S''(xₙ) = 0) from the given data points.
    /// </summary>
    /// <param name="xs">Strictly increasing x-coordinates. Must contain at least 3 elements.</param>
    /// <param name="ys">Y-values corresponding to each x-coordinate.</param>
    /// <exception cref="ArgumentNullException"><paramref name="xs"/> or <paramref name="ys"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">Fewer than 3 points, mismatched lengths, non-increasing x-values, or non-finite y-values.</exception>
    /// <exception cref="InvalidOperationException">Coordinate normalization cannot preserve distinct knots, or spline coefficients are unrepresentable in the working type.</exception>
    public CubicSplineInterpolator(IReadOnlyList<T> xs, IReadOnlyList<T> ys)
        : this(xs, ys, SplineBoundary<T>.Natural)
    {
    }

    /// <summary>
    /// Constructs a cubic spline with the specified non-generic boundary condition
    /// (<see cref="SplineBoundary.Natural"/>, <see cref="SplineBoundary.CreateClamped"/>,
    /// <see cref="SplineBoundary.NotAKnot"/>, <see cref="SplineBoundary.ModifiedNotAKnot"/>,
    /// <see cref="SplineBoundary.RevisedNotAKnot"/>, or <see cref="SplineBoundary.QSpline"/>); a clamped boundary's slopes stay
    /// <see cref="double"/> while the constructor scales them to its normalized coordinate units and are converted to
    /// <typeparamref name="T"/> once, via <c>T.CreateChecked</c>, after that scaling. A slope whose value in the
    /// constructor's final units is outside <typeparamref name="T"/>'s finite range (above range as well as below it
    /// is judged after scaling, not in the caller's own units) is refused with <see cref="ArgumentOutOfRangeException"/>
    /// at every <typeparamref name="T"/>, including <see cref="decimal"/>, which has no infinity.
    /// </summary>
    /// <param name="xs">Strictly increasing x-coordinates.</param>
    /// <param name="ys">Y-values corresponding to each x-coordinate.</param>
    /// <param name="boundary">Non-generic boundary condition.</param>
    /// <exception cref="ArgumentNullException"><paramref name="boundary"/>, <paramref name="xs"/>, or <paramref name="ys"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">Too few points for the boundary, mismatched lengths, non-increasing x-values, or non-finite y-values.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A clamped slope is not finite, or is outside <typeparamref name="T"/>'s finite range once scaled to the constructor's normalized units.</exception>
    /// <exception cref="InvalidOperationException">Coordinate normalization cannot preserve distinct knots, or spline coefficients are unrepresentable in the working type.</exception>
    public CubicSplineInterpolator(IReadOnlyList<T> xs, IReadOnlyList<T> ys, SplineBoundary boundary)
        : this(xs, ys, ConvertBoundary(boundary))
    {
    }

    /// <summary>
    /// Constructs a cubic spline with the specified generic boundary conditions.
    /// </summary>
    /// <param name="xs">Strictly increasing x-coordinates.</param>
    /// <param name="ys">Y-values corresponding to each x-coordinate.</param>
    /// <param name="boundary">Generic boundary condition.</param>
    /// <exception cref="ArgumentNullException"><paramref name="boundary"/>, <paramref name="xs"/>, or <paramref name="ys"/> is <see langword="null"/>.</exception>
    /// <exception cref="InvalidOperationException">Coordinate normalization cannot preserve distinct knots, or spline coefficients are unrepresentable in the working type.</exception>
    public CubicSplineInterpolator(IReadOnlyList<T> xs, IReadOnlyList<T> ys, SplineBoundary<T> boundary)
    {
        ArgumentNullException.ThrowIfNull(boundary);
        ArgumentNullException.ThrowIfNull(xs);
        ArgumentNullException.ThrowIfNull(ys);

        if (xs.Count != ys.Count)
        {
            throw new ArgumentException("xs and ys must have the same length.", nameof(ys));
        }

        var minPoints = boundary switch
        {
            SplineBoundary<T>.RevisedNotAKnotBoundary => 6,
            SplineBoundary<T>.QSplineBoundary => 6,
            SplineBoundary<T>.NotAKnotBoundary or SplineBoundary<T>.ModifiedNotAKnotBoundary => 4,
            _ => 3,
        };
        if (xs.Count < minPoints)
        {
            throw new ArgumentException(
                $"At least {minPoints} data points are required for {boundary.GetType().Name.Replace("Boundary", "")} boundary.",
                nameof(xs));
        }

        int n = xs.Count;
        _x = new T[n];
        var y = new T[n];
        for (int i = 0; i < n; i++)
        {
            _x[i] = xs[i];
            y[i] = ys[i];
        }

        _originalY = y;
        InterpolationHelper.ValidateSortedAscending<T>(_x, nameof(xs));
        InterpolationHelper.ValidateFiniteOrdinates<T>(y, nameof(ys));

        _originalX = _x;
        _coordinateScale = InterpolationHelper.CoordinateScale<T>(_x);
        if (_coordinateScale != T.One && TryScaleSlopes(boundary, _coordinateScale, out var coordinateBoundary))
        {
            boundary = coordinateBoundary;

            // Queries are bracketed and measured in the caller's own units, so the knots keep an
            // unnormalized copy: dividing a small query by the grid magnitude erases it.
            _originalX = (T[])_x.Clone();
            for (int i = 0; i < n; i++)
            {
                _x[i] /= _coordinateScale;
            }
        }
        else
        {
            _coordinateScale = T.One;
        }

        int m = n - 1;
        T[] h = new T[m];
        for (int i = 0; i < m; i++)
        {
            h[i] = _x[i + 1] - _x[i];
        }

        // The spline is affine-invariant in x, so it is solved on t = (x − x_i) / H with H a power
        // of T's radix near the largest spacing. Unscaled, the second derivatives (∝ Δy/h²) and
        // cubic coefficients (∝ Δy/h³) fall below decimal's 1e-28 grid on wide spacings and
        // silently lose curvature, and the not-a-knot / Q-spline products of spacings overflow
        // double. Scaling by a radix power is exact, so IEEE results on ordinary grids are
        // unchanged bit for bit. Clamped slopes convert to t-units by the same factor.
        if (TryAbscissaScale(h, out var abscissaScale)
            && TryScaleSlopes(boundary, abscissaScale, out var scaledBoundary))
        {
            for (int i = 0; i < m; i++)
            {
                h[i] /= abscissaScale;
            }

            boundary = scaledBoundary;
            _abscissaScale = abscissaScale;
        }

        RequireRepresentableSlopes(boundary);

        // The spline is linear in the ordinates (and clamped slopes), so large finite inputs
        // are solved on data normalized by their magnitude and the result is rescaled. The
        // unnormalized path is kept whenever it succeeds, preserving its rounding.
        if (!TryBuildCoefficients(n, h, y, boundary, out var coefficients))
        {
            var magnitude = OrdinateMagnitude(y, boundary);
            if (magnitude > T.One)
            {
                var normalized = new T[n];
                for (int i = 0; i < n; i++)
                {
                    normalized[i] = y[i] / magnitude;
                }

                if (TryBuildCoefficients(n, h, normalized, NormalizeBoundary(boundary, magnitude), out coefficients))
                {
                    _ordinateScale = magnitude;
                    _scaleOrdinates = true;
                }
            }

            if (!_scaleOrdinates)
            {
                throw new InvalidOperationException("Spline coefficients are not representable in the working type.");
            }
        }

        (_a, _b, _c, _d) = coefficients;
    }

    /// <summary>
    /// Evaluates the spline at the given x-coordinate. Values outside the data range are
    /// extrapolated along the nearest cubic segment (not clamped to the boundary value).
    /// </summary>
    /// <param name="x">The x-coordinate at which to evaluate the spline.</param>
    /// <returns>The interpolated y-value.</returns>
    public T Interpolate(T x)
    {
        InterpolationHelper.ValidateFiniteTarget(x);
        var originalX = x;
        x /= _coordinateScale;

        // Bracket and measure the query in the caller's own units: dividing a small query by the
        // grid magnitude before subtracting would erase it. Without normalization the original
        // knots are the working knots and nothing changes.
        int seg = FindSegment(originalX, _originalX);
        // Coefficient normalization may erase a small ordinate beside a very large
        // neighbour. At a knot the original value is exact and needs no polynomial.
        if (originalX == _originalX[seg])
        {
            return _originalY[seg];
        }
        if (originalX == _originalX[^1])
        {
            return _originalY[^1];
        }
        T offset;
        T dx;
        var offsetExact = true;
        if (_coordinateScale == T.One)
        {
            offset = x - _x[seg];
            dx = offset / _abscissaScale;
        }
        else if (TryLocalOffsetUnit(originalX, seg, out offset, out var unit))
        {
            dx = offset / unit;
        }
        else
        {
            offsetExact = false;
            offset = T.Zero;
            dx = (x - _x[seg]) / _abscissaScale;
        }

        // A coordinate that underflowed (or sits on decimal's grid) has lost the digits of the
        // first-order term dx * b. Take it from the unrounded offset over the length of one
        // coordinate unit; the higher-order terms need no such digits.
        if (offsetExact && offset != T.Zero && InterpolationHelper.IsCoarseFraction(dx)
            && InterpolationHelper.TryScaledProduct(_b[seg], offset, _coordinateScale * _abscissaScale, out var linear))
        {
            var refined = _a[seg] + linear + (dx * dx * (_c[seg] + (dx * _d[seg])));
            return _scaleOrdinates ? refined * _ordinateScale : refined;
        }

        var result = _a[seg] + (dx * (_b[seg] + (dx * (_c[seg] + (dx * _d[seg])))));
        return _scaleOrdinates ? result * _ordinateScale : result;
    }

    // Offset of the query from the segment's left knot in the caller's units, and the length in
    // those units of one working coordinate unit (the radix-power abscissa scale times the grid
    // magnitude). False when either is not representable.
    private bool TryLocalOffsetUnit(T originalX, int seg, out T offset, out T unit)
    {
        unit = T.Zero;
        if (!InterpolationHelper.TryOffset(originalX, _originalX[seg], out offset))
        {
            return false;
        }

        try
        {
            unit = _coordinateScale * _abscissaScale;
            return T.IsFinite(unit) && unit != T.Zero;
        }
        catch (OverflowException)
        {
            // Decimal throws where IEEE types produce infinity.
            return false;
        }
    }

    private static (T[] A, T[] B, T[] C, T[] D) BuildCoefficients(int n, T[] h, T[] y, SplineBoundary<T> boundary)
    {
        T[] sigma = boundary switch
        {
            SplineBoundary<T>.NaturalBoundary => SolveNatural(n, h, y),
            SplineBoundary<T>.ClampedBoundary clamped => SolveClamped(n, h, y, clamped.LeftSlope, clamped.RightSlope),
            SplineBoundary<T>.NotAKnotBoundary => SolveNotAKnot(n, h, y, T.Zero, T.Zero),
            SplineBoundary<T>.RevisedNotAKnotBoundary => SolveNotAKnot(n, h, y,
                EstimateRevisedJump(h, y, fromRight: false), EstimateRevisedJump(h, y, fromRight: true)),
            SplineBoundary<T>.ModifiedNotAKnotBoundary => SolveModifiedNotAKnot(n, h, y),
            SplineBoundary<T>.QSplineBoundary => SolveQSpline(n, h, y),
            _ => throw new InvalidOperationException(
                $"CubicSplineInterpolator<T> reached its SplineBoundary<T> dispatch default for boundary type {boundary.GetType().Name}. " +
                "CubicSplineInterpolator<T> supports only Natural, Clamped, NotAKnot, ModifiedNotAKnot, RevisedNotAKnot, and QSpline; " +
                "reaching this path indicates a drift between the SplineBoundary<T> closed hierarchy and this dispatch layer. " +
                "Extend both in lockstep when a new SplineBoundary<T> subtype ships."),
        };

        int m = n - 1;
        var a = new T[m];
        var b = new T[m];
        var c = new T[m];
        var d = new T[m];

        for (int i = 0; i < m; i++)
        {
            a[i] = y[i];
            c[i] = sigma[i] / s_two;
            d[i] = (sigma[i + 1] - sigma[i]) / (s_six * h[i]);
            b[i] = ((y[i + 1] - y[i]) / h[i]) - ((h[i] * (sigma[i + 1] + (s_two * sigma[i]))) / s_six);
        }

        return (a, b, c, d);
    }

    private static bool TryBuildCoefficients(
        int n, T[] h, T[] y, SplineBoundary<T> boundary, out (T[] A, T[] B, T[] C, T[] D) coefficients)
    {
        try
        {
            coefficients = BuildCoefficients(n, h, y, boundary);
        }
        catch (OverflowException)
        {
            coefficients = default;
            return false;
        }

        for (int i = 0; i < coefficients.A.Length; i++)
        {
            if (!T.IsFinite(coefficients.A[i]) || !T.IsFinite(coefficients.B[i])
                || !T.IsFinite(coefficients.C[i]) || !T.IsFinite(coefficients.D[i]))
            {
                return false;
            }
        }

        return true;
    }

    private static bool TryAbscissaScale(T[] h, out T scale)
    {
        scale = T.One;
        var widest = T.Zero;
        for (int i = 0; i < h.Length; i++)
        {
            if (!T.IsFinite(h[i]))
            {
                return false;
            }

            widest = T.Max(widest, h[i]);
        }

        if (widest <= T.Zero)
        {
            return false;
        }

        // Largest power of the radix not exceeding the widest spacing. Compare against
        // widest / radix rather than scale · radix so the search never overflows.
        var radix = T.CreateChecked(T.Radix);
        while (widest / radix >= scale)
        {
            scale *= radix;
        }

        while (widest < scale)
        {
            scale /= radix;
        }

        if (scale <= T.Zero)
        {
            return false;
        }

        // A radix-power division is not exact at the working type's minimum
        // quantum. Keep the original widths if normalization erases an interval.
        for (int i = 0; i < h.Length; i++)
        {
            if (h[i] / scale == T.Zero)
            {
                return false;
            }
        }

        return true;
    }

    private static bool TryScaleSlopes(SplineBoundary<T> boundary, T scale, out SplineBoundary<T> scaled)
    {
        scaled = boundary;
        if (boundary is not SplineBoundary<T>.ClampedBoundary clamped)
        {
            return true;
        }

        T left;
        T right;
        (double Left, double Right)? doubleScaled = null;
        try
        {
            if (clamped.DoubleSlopes is { } source)
            {
                // The caller supplied double slopes. Multiply them by the scale while they are still doubles and
                // narrow the product once: narrowing first erases a slope below T's range (or keeps only a few
                // bits of it) although the scaled slope, and every spline value it drives, is representable.
                // This is one scalar conversion per slope, not a loop-carried cast. The product stays in double
                // so a later scaling stage multiplies the unrounded value rather than the narrowed one.
                var factor = double.CreateChecked(scale);
                var scaledLeft = source.Left * factor;
                var scaledRight = source.Right * factor;
                left = T.CreateChecked(scaledLeft);
                right = T.CreateChecked(scaledRight);
                doubleScaled = (scaledLeft, scaledRight);
            }
            else
            {
                left = clamped.LeftSlope * scale;
                right = clamped.RightSlope * scale;
            }
        }
        catch (OverflowException)
        {
            return false;
        }

        if (!T.IsFinite(left) || !T.IsFinite(right))
        {
            return false;
        }

        scaled = doubleScaled is { } carried
            ? SplineBoundary<T>.CreateClampedWithDoubleSlopes(left, right, carried.Left, carried.Right)
            : SplineBoundary<T>.CreateClamped(left, right);
        return true;
    }

    // A clamped boundary built from double slopes carries working-type placeholders that can be infinite for a slope
    // above T's range in the caller's units. Scaling may have replaced them with representable values; if a slope is
    // still not finite in T once no further scaling applies, refuse it as the generic factory refuses such a slope.
    private static void RequireRepresentableSlopes(SplineBoundary<T> boundary)
    {
        if (boundary is SplineBoundary<T>.ClampedBoundary clamped)
        {
            SplineBoundary<T>.RequireRepresentableSlopes(clamped.LeftSlope, clamped.RightSlope, clamped.DoubleSlopes);
        }
    }

    private static T OrdinateMagnitude(T[] y, SplineBoundary<T> boundary)
    {
        var magnitude = T.One;
        for (int i = 0; i < y.Length; i++)
        {
            magnitude = T.Max(magnitude, T.Abs(y[i]));
        }

        if (boundary is SplineBoundary<T>.ClampedBoundary clamped)
        {
            magnitude = T.Max(magnitude, T.Max(T.Abs(clamped.LeftSlope), T.Abs(clamped.RightSlope)));
        }

        return magnitude;
    }

    private static SplineBoundary<T> NormalizeBoundary(SplineBoundary<T> boundary, T magnitude) =>
        boundary is SplineBoundary<T>.ClampedBoundary clamped
            ? SplineBoundary<T>.CreateClamped(clamped.LeftSlope / magnitude, clamped.RightSlope / magnitude)
            : boundary;

    private static SplineBoundary<T> ConvertBoundary(SplineBoundary boundary)
    {
        ArgumentNullException.ThrowIfNull(boundary);

        return boundary switch
        {
            SplineBoundary.NaturalBoundary => SplineBoundary<T>.Natural,
            SplineBoundary.NotAKnotBoundary => SplineBoundary<T>.NotAKnot,
            SplineBoundary.ModifiedNotAKnotBoundary => SplineBoundary<T>.ModifiedNotAKnot,
            SplineBoundary.QSplineBoundary => SplineBoundary<T>.QSpline,
            SplineBoundary.RevisedNotAKnotBoundary => SplineBoundary<T>.RevisedNotAKnot,
            SplineBoundary.ClampedBoundary c => SplineBoundary<T>.CreateClampedFromDoubleSlopes(c.LeftSlope, c.RightSlope),
            _ => throw new InvalidOperationException(
                $"CubicSplineInterpolator<T>.ConvertBoundary reached its SplineBoundary dispatch default for boundary type {boundary.GetType().Name}. " +
                "ConvertBoundary supports only Natural, Clamped, NotAKnot, ModifiedNotAKnot, RevisedNotAKnot, and QSpline; " +
                "reaching this path indicates a drift between the SplineBoundary closed hierarchy and this dispatch layer. " +
                "Extend both in lockstep when a new SplineBoundary subtype ships."),
        };
    }

    private static T[] SolveNatural(int n, T[] h, T[] y)
    {
        T[] sigma = new T[n];
        if (n <= 2)
        {
            return sigma;
        }

        int ic = n - 2;
        using var scratch = new PooledScratch<T>(ic > StackAllocThreshold ? checked(4 * ic) : 0);
        var rented = scratch.Buffer;
        {
            Span<T> diag = ic <= StackAllocThreshold ? stackalloc T[ic] : rented.AsSpan(0 * ic, ic);
            Span<T> upper = ic <= StackAllocThreshold ? stackalloc T[ic] : rented.AsSpan(1 * ic, ic);
            Span<T> lower = ic <= StackAllocThreshold ? stackalloc T[ic] : rented.AsSpan(2 * ic, ic);
            Span<T> rhs = ic <= StackAllocThreshold ? stackalloc T[ic] : rented.AsSpan(3 * ic, ic);
            diag.Clear();
            upper.Clear();
            lower.Clear();
            rhs.Clear();

            for (int i = 0; i < ic; i++)
            {
                int k = i + 1;
                diag[i] = s_two * (h[k - 1] + h[k]);
                rhs[i] = s_six * ((y[k + 1] - y[k]) / h[k] - (y[k] - y[k - 1]) / h[k - 1]);
                if (i > 0)
                {
                    lower[i] = h[k - 1];
                }

                if (i < ic - 1)
                {
                    upper[i] = h[k];
                }
            }

            ThomasSolve(diag, upper, lower, rhs, ic);

            for (int i = 0; i < ic; i++)
            {
                sigma[i + 1] = rhs[i];
            }

            return sigma;
        }
    }

    private static T[] SolveClamped(int n, T[] h, T[] y, T leftSlope, T rightSlope)
    {
        using var scratch = new PooledScratch<T>(n > StackAllocThreshold ? checked(3 * n) : 0);
        var rented = scratch.Buffer;
        {
            Span<T> diag = n <= StackAllocThreshold ? stackalloc T[n] : rented.AsSpan(0 * n, n);
            Span<T> upper = n <= StackAllocThreshold ? stackalloc T[n] : rented.AsSpan(1 * n, n);
            Span<T> lower = n <= StackAllocThreshold ? stackalloc T[n] : rented.AsSpan(2 * n, n);
            diag.Clear();
            upper.Clear();
            lower.Clear();
            var rhs = new T[n];  // also serves as the sigma output

            diag[0] = s_two * h[0];
            upper[0] = h[0];
            rhs[0] = s_six * ((y[1] - y[0]) / h[0] - leftSlope);

            for (int i = 1; i < n - 1; i++)
            {
                lower[i] = h[i - 1];
                diag[i] = s_two * (h[i - 1] + h[i]);
                upper[i] = h[i];
                rhs[i] = s_six * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
            }

            lower[n - 1] = h[n - 2];
            diag[n - 1] = s_two * h[n - 2];
            rhs[n - 1] = s_six * (rightSlope - (y[n - 1] - y[n - 2]) / h[n - 2]);

            ThomasSolve(diag, upper, lower, rhs.AsSpan(), n);

            return rhs;
        }
    }

    private static T[] SolveNotAKnot(int n, T[] h, T[] y, T leftJump, T rightJump)
    {
        int ic = n - 2;
        using var scratch = new PooledScratch<T>(ic > StackAllocThreshold ? checked(4 * ic) : 0);
        var rented = scratch.Buffer;
        {
            Span<T> diag = ic <= StackAllocThreshold ? stackalloc T[ic] : rented.AsSpan(0 * ic, ic);
            Span<T> upper = ic <= StackAllocThreshold ? stackalloc T[ic] : rented.AsSpan(1 * ic, ic);
            Span<T> lower = ic <= StackAllocThreshold ? stackalloc T[ic] : rented.AsSpan(2 * ic, ic);
            Span<T> rhs = ic <= StackAllocThreshold ? stackalloc T[ic] : rented.AsSpan(3 * ic, ic);
            diag.Clear();
            upper.Clear();
            lower.Clear();
            rhs.Clear();

            for (int j = 0; j < ic; j++)
            {
                int i = j + 1;
                lower[j] = h[i - 1];
                diag[j] = s_two * (h[i - 1] + h[i]);
                upper[j] = h[i];
                rhs[j] = s_six * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
            }

            // Eliminating endpoint curvatures from the prescribed third-derivative
            // jumps changes only the first and last right-hand sides. Reflection at
            // the right endpoint preserves the sign of the jump.
            rhs[0] -= h[0] * h[0] * leftJump;
            rhs[ic - 1] -= h[n - 2] * h[n - 2] * rightJump;

            diag[0] += h[0] * (h[0] + h[1]) / h[1];
            upper[0] -= h[0] * h[0] / h[1];

            diag[ic - 1] += h[n - 2] * (h[n - 3] + h[n - 2]) / h[n - 3];
            lower[ic - 1] -= h[n - 2] * h[n - 2] / h[n - 3];

            ThomasSolve(diag, upper, lower, rhs, ic);

            T[] sigma = new T[n];
            for (int j = 0; j < ic; j++)
            {
                sigma[j + 1] = rhs[j];
            }

            sigma[0] = ((h[0] + h[1]) * sigma[1] - h[0] * sigma[2]) / h[1] + h[0] * leftJump;

            sigma[n - 1] = ((h[n - 3] + h[n - 2]) * sigma[n - 2] - h[n - 2] * sigma[n - 3]) / h[n - 3] + h[n - 2] * rightJump;

            return sigma;
        }
    }

    private static T EstimateRevisedJump(T[] h, T[] y, bool fromRight)
    {
        // Jarre JCAM 478 (2026), Appendix RNAK steps 4-7, damping factor 5/2.
        // Local reflected coordinates give the same left-end formula at both
        // ends. All divided differences and returned spline values stay in T.
        Span<T> knots = stackalloc T[6];
        Span<T> differences = stackalloc T[6];
        knots[0] = T.Zero;
        differences[0] = fromRight ? y[^1] : y[0];
        for (int i = 1; i < 6; i++)
        {
            knots[i] = knots[i - 1] + h[fromRight ? h.Length - i : i - 1];
            differences[i] = fromRight ? y[y.Length - 1 - i] : y[i];
        }

        for (int order = 1; order < 6; order++)
        {
            for (int i = 5; i >= order; i--)
            {
                differences[i] = (differences[i] - differences[i - 1]) / (knots[i] - knots[i - order]);
            }
        }

        var fourth = differences[4];
        var fifth = differences[5];
        var rho = fourth;
        var damping = T.CreateChecked(2.5m);
        if ((fourth > T.Zero && fifth > T.Zero) || (fourth < T.Zero && fifth < T.Zero))
        {
            var magnitude = T.Max(T.Zero, T.Abs(fourth) - damping * T.Abs(fifth) * (knots[4] - knots[2]));
            rho = fourth < T.Zero ? -magnitude : magnitude;
        }

        return T.CreateChecked(12) * rho * knots[2];
    }

    private static T[] SolveModifiedNotAKnot(int n, T[] h, T[] y)
    {
        int ic = n - 2;
        using var scratch = new PooledScratch<T>(ic > StackAllocThreshold ? checked(4 * ic) : 0);
        var rented = scratch.Buffer;
        {
            Span<T> diag = ic <= StackAllocThreshold ? stackalloc T[ic] : rented.AsSpan(0 * ic, ic);
            Span<T> upper = ic <= StackAllocThreshold ? stackalloc T[ic] : rented.AsSpan(1 * ic, ic);
            Span<T> lower = ic <= StackAllocThreshold ? stackalloc T[ic] : rented.AsSpan(2 * ic, ic);
            Span<T> rhs = ic <= StackAllocThreshold ? stackalloc T[ic] : rented.AsSpan(3 * ic, ic);
            diag.Clear();
            upper.Clear();
            lower.Clear();
            rhs.Clear();

            for (int j = 0; j < ic; j++)
            {
                int i = j + 1;
                lower[j] = h[i - 1];
                diag[j] = s_two * (h[i - 1] + h[i]);
                upper[j] = h[i];
                rhs[j] = s_six * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
            }

            T ratioL = T.Min(h[0], h[1]) / T.Max(h[0], h[1]);
            T alphaL = ratioL;

            diag[0] += alphaL * h[0] * (h[0] + h[1]) / h[1];
            upper[0] -= alphaL * h[0] * h[0] / h[1];

            T ratioR = T.Min(h[n - 3], h[n - 2]) / T.Max(h[n - 3], h[n - 2]);
            T alphaR = ratioR;

            diag[ic - 1] += alphaR * h[n - 2] * (h[n - 3] + h[n - 2]) / h[n - 3];
            lower[ic - 1] -= alphaR * h[n - 2] * h[n - 2] / h[n - 3];

            ThomasSolve(diag, upper, lower, rhs, ic);

            T[] sigma = new T[n];
            for (int j = 0; j < ic; j++)
            {
                sigma[j + 1] = rhs[j];
            }

            if (alphaL > T.Zero && ic >= 2)
            {
                sigma[0] = alphaL * (((h[0] + h[1]) * sigma[1] - h[0] * sigma[2]) / h[1]);
            }

            if (alphaR > T.Zero && ic >= 2)
            {
                sigma[n - 1] = alphaR * (((h[n - 3] + h[n - 2]) * sigma[n - 2] - h[n - 2] * sigma[n - 3]) / h[n - 3]);
            }

            return sigma;
        }
    }

    private static T[] SolveQSpline(int n, T[] h, T[] y)
    {
        using var scratch = new PooledScratch<T>(n > StackAllocThreshold ? checked(3 * n) : 0);
        var rented = scratch.Buffer;
        {
            Span<T> diagFull = n <= StackAllocThreshold ? stackalloc T[n] : rented.AsSpan(0 * n, n);
            Span<T> upperFull = n <= StackAllocThreshold ? stackalloc T[n] : rented.AsSpan(1 * n, n);
            Span<T> lowerFull = n <= StackAllocThreshold ? stackalloc T[n] : rented.AsSpan(2 * n, n);
            diagFull.Clear();
            upperFull.Clear();
            lowerFull.Clear();
            var rhsFull = new T[n];  // also serves as sigma output

            diagFull[0] = T.One;
            rhsFull[0] = EstimateQSplineEndpointCurvature(h, y, fromRight: false);

            for (int i = 1; i < n - 1; i++)
            {
                lowerFull[i] = h[i - 1];
                diagFull[i] = s_two * (h[i - 1] + h[i]);
                upperFull[i] = h[i];
                rhsFull[i] = s_six * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
            }

            diagFull[n - 1] = T.One;
            rhsFull[n - 1] = EstimateQSplineEndpointCurvature(h, y, fromRight: true);

            ThomasSolve(diagFull, upperFull, lowerFull, rhsFull.AsSpan(), n);

            return rhsFull;
        }
    }

    private static T EstimateQSplineEndpointCurvature(T[] h, T[] y, bool fromRight)
    {
        // Jarre (JCAM 478, 2026, Appendix Q steps 2-6; author Oct 29, 2025):
        // estimate the fourth difference and damp it using the fifth difference.
        // Reflecting the right knots reverses odd differences, so the same
        // same-sign safeguard applies at both endpoints.
        Span<T> knots = stackalloc T[6];
        Span<T> differences = stackalloc T[6];
        knots[0] = T.Zero;
        differences[0] = fromRight ? y[^1] : y[0];
        for (int i = 1; i < 6; i++)
        {
            knots[i] = knots[i - 1] + h[fromRight ? h.Length - i : i - 1];
            differences[i] = fromRight ? y[y.Length - 1 - i] : y[i];
        }

        for (int order = 1; order < 6; order++)
        {
            for (int i = 5; i >= order; i--)
            {
                differences[i] = (differences[i] - differences[i - 1]) / (knots[i] - knots[i - order]);
            }
        }

        var fourth = differences[4];
        var fifth = differences[5];
        var rho = fourth;
        if ((fourth > T.Zero && fifth > T.Zero) || (fourth < T.Zero && fifth < T.Zero))
        {
            var magnitude = T.Max(T.Zero, T.Abs(fourth) - T.CreateChecked(2.5m) * T.Abs(fifth) * (knots[2] - knots[1]));
            rho = fourth < T.Zero ? -magnitude : magnitude;
        }

        T d1 = knots[1], d2 = knots[2], d3 = knots[3];
        return s_two * (differences[2] - (d1 + d2) * differences[3]
            + (d1 * d2 + d1 * d3 + d2 * d3) * rho);
    }

    private static void ThomasSolve(Span<T> diag, Span<T> upper, Span<T> lower, Span<T> rhs, int n)
    {
        for (int i = 1; i < n; i++)
        {
            T factor = lower[i] / diag[i - 1];
            diag[i] -= factor * upper[i - 1];
            rhs[i] -= factor * rhs[i - 1];
        }

        rhs[n - 1] /= diag[n - 1];
        for (int i = n - 2; i >= 0; i--)
        {
            rhs[i] = (rhs[i] - (upper[i] * rhs[i + 1])) / diag[i];
        }
    }

    private int FindSegment(T x, T[] knots)
    {
        if (x <= knots[0])
        {
            return 0;
        }

        if (x >= knots[^1])
        {
            return _a.Length - 1;
        }

        int lo = 0;
        int hi = knots.Length - 1;
        while (hi - lo > 1)
        {
            int mid = (lo + hi) / 2;
            if (knots[mid] > x)
            {
                hi = mid;
            }
            else
            {
                lo = mid;
            }
        }

        return lo;
    }
}

/// <summary>
/// Cubic spline interpolator with C² continuity at all interior knots,
/// supporting six boundary conditions — natural, clamped, classical not-a-knot
/// (de Boor 1978), modified and revised not-a-knot, and Q-spline (Jarre 2025, arXiv:2507.05083).
/// Tridiagonal system solved by the Thomas algorithm in O(n), unconditionally
/// stable for the diagonally dominant systems arising in cubic spline construction
/// (Faires &amp; Burden, §3.5).
/// </summary>
/// <remarks>
/// <para>
/// The spline passes through all data points exactly and is C² continuous at every
/// interior knot. Pointwise convergence is O(h⁴) for smooth data away from the
/// boundary; boundary error depends on the condition chosen (O(h²) for natural,
/// O(h⁴) for clamped when the true derivative is supplied, and O(h⁴) for classical
/// and revised not-a-knot on regular grids). ModifiedNotAKnot agrees with classical
/// not-a-knot on uniform grids, but can have O(h³) boundary error on smoothly graded
/// grids and lower order on less regular grids.
/// </para>
/// <para>
/// Boundary-condition selection guidance:
/// <list type="bullet">
///   <item><see cref="SplineBoundary.Natural"/> — fallback when endpoint derivatives are unknown; simple and stable.</item>
///   <item><see cref="SplineBoundary.CreateClamped"/> — optimal when the endpoint derivative is known analytically.</item>
///   <item><see cref="SplineBoundary.NotAKnot"/> — preferred default on uniform / near-uniform grids.</item>
///   <item><see cref="SplineBoundary.ModifiedNotAKnot"/> — spacing-ratio blend of not-a-knot and natural end conditions; identical to not-a-knot on uniform grids and not more accurate than it on non-uniform ones (not Jarre's §4 RNAK).</item>
///   <item><see cref="SplineBoundary.RevisedNotAKnot"/> — third-derivative jumps with the fifth-difference safeguard of Jarre, JCAM 478 (2026), Appendix.</item>
///   <item><see cref="SplineBoundary.QSpline"/> — safeguarded endpoint curvature estimates from six function values and no derivative input.</item>
/// </list>
/// </para>
/// <para>
/// Outside the node range, <see cref="Interpolate"/> extrapolates along the nearest
/// cubic segment rather than clamping to the boundary value. Stateless linear,
/// log-linear, flat-forward and monotone-cubic interpolators clamp instead;
/// monotone-convex interpolation continues its endpoint forward. Prefer this type
/// when repeated evaluations on the same node set are needed (coefficients are
/// computed once at construction); for single-shot use of a monotone or linear
/// scheme, see <see cref="MonotoneCubicInterpolator"/> or <see cref="LinearInterpolator"/>.
/// </para>
/// <para>Minimum 3 data points for natural/clamped, 4 for classical/modified not-a-knot, 6 for Q-spline and revised not-a-knot boundaries. x-values must be strictly increasing.</para>
/// </remarks>
[Obsolete(
    "Use CubicSplineInterpolator<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class CubicSplineInterpolator
{
    private readonly CubicSplineInterpolator<double> _impl;

    /// <summary>
    /// Constructs a natural cubic spline (S''(x0) = S''(xn) = 0) from the given data points.
    /// </summary>
    /// <param name="xs">Strictly increasing x-coordinates. Must contain at least 3 elements.</param>
    /// <param name="ys">Y-values corresponding to each x-coordinate.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when fewer than 3 points are provided or x-values are not strictly increasing.
    /// </exception>
    public CubicSplineInterpolator(IReadOnlyList<double> xs, IReadOnlyList<double> ys)
        : this(xs, ys, SplineBoundary.Natural)
    {
    }

    /// <summary>
    /// Constructs a cubic spline with the specified boundary conditions.
    /// </summary>
    /// <param name="xs">Strictly increasing x-coordinates.</param>
    /// <param name="ys">Y-values corresponding to each x-coordinate.</param>
    /// <param name="boundary">
    /// Boundary condition: <see cref="SplineBoundary.Natural"/>,
    /// <see cref="SplineBoundary.CreateClamped"/>, or <see cref="SplineBoundary.NotAKnot"/>.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when too few points are provided, x-values are not strictly increasing,
    /// or array lengths differ.
    /// </exception>
    public CubicSplineInterpolator(IReadOnlyList<double> xs, IReadOnlyList<double> ys, SplineBoundary boundary)
    {
        _impl = new CubicSplineInterpolator<double>(xs, ys, boundary);
    }

    /// <summary>
    /// Evaluates the spline at the given x-coordinate.
    /// Values outside the data range are extrapolated using the nearest segment.
    /// </summary>
    /// <param name="x">The x-coordinate at which to evaluate the spline.</param>
    /// <returns>The interpolated y-value.</returns>
    public double Interpolate(double x)
    {
        return _impl.Interpolate(x);
    }
}
