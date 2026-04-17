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

namespace Boutquin.Numerics.Interpolation;

/// <summary>
/// Cubic spline interpolator with C² continuity at all interior knots,
/// supporting five boundary conditions — natural, clamped, classical not-a-knot
/// (de Boor 1978), modified not-a-knot and Q-spline (Jarre 2025, arXiv:2507.05083).
/// Tridiagonal system solved by the Thomas algorithm in O(n), unconditionally
/// stable for the diagonally dominant systems arising in cubic spline construction
/// (Faires &amp; Burden, §3.5).
/// </summary>
/// <typeparam name="T">The floating-point type for interpolation.</typeparam>
/// <remarks>
/// Tier A: Requires only basic floating-point arithmetic.
/// </remarks>
public sealed class CubicSplineInterpolator<T> where T : IFloatingPoint<T>
{
    private readonly T[] _x;
    private readonly T[] _a;
    private readonly T[] _b;
    private readonly T[] _c;
    private readonly T[] _d;

    private static readonly T s_zero = T.Zero;
    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_six = T.CreateChecked(6);

    /// <inheritdoc cref="CubicSplineInterpolator(IReadOnlyList{double}, IReadOnlyList{double})"/>
    public CubicSplineInterpolator(IReadOnlyList<T> xs, IReadOnlyList<T> ys)
        : this(xs, ys, SplineBoundary<T>.Natural)
    {
    }

    /// <inheritdoc cref="CubicSplineInterpolator(IReadOnlyList{double}, IReadOnlyList{double}, SplineBoundary)"/>
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
    public CubicSplineInterpolator(IReadOnlyList<T> xs, IReadOnlyList<T> ys, SplineBoundary<T> boundary)
    {
        ArgumentNullException.ThrowIfNull(boundary);

        var minPoints = boundary is SplineBoundary<T>.NotAKnotBoundary
            or SplineBoundary<T>.ModifiedNotAKnotBoundary
            or SplineBoundary<T>.QSplineBoundary ? 4 : 3;
        if (xs.Count < minPoints)
        {
            throw new ArgumentException(
                $"At least {minPoints} data points are required for {boundary.GetType().Name.Replace("Boundary", "")} boundary.",
                nameof(xs));
        }

        if (xs.Count != ys.Count)
        {
            throw new ArgumentException("xs and ys must have the same length.", nameof(ys));
        }

        int n = xs.Count;
        _x = new T[n];
        var y = new T[n];
        for (int i = 0; i < n; i++)
        {
            _x[i] = xs[i];
            y[i] = ys[i];
        }

        for (int i = 1; i < n; i++)
        {
            if (_x[i] <= _x[i - 1])
            {
                throw new ArgumentException("x-values must be strictly increasing.", nameof(xs));
            }
        }

        int m = n - 1;
        T[] h = new T[m];
        for (int i = 0; i < m; i++)
        {
            h[i] = _x[i + 1] - _x[i];
        }

        T[] sigma = boundary switch
        {
            SplineBoundary<T>.NaturalBoundary => SolveNatural(n, h, y),
            SplineBoundary<T>.ClampedBoundary clamped => SolveClamped(n, h, y, clamped.LeftSlope, clamped.RightSlope),
            SplineBoundary<T>.NotAKnotBoundary => SolveNotAKnot(n, h, y),
            SplineBoundary<T>.ModifiedNotAKnotBoundary => SolveModifiedNotAKnot(n, h, y),
            SplineBoundary<T>.QSplineBoundary => SolveQSpline(n, h, y),
            _ => throw new ArgumentException($"Unknown boundary type: {boundary.GetType().Name}", nameof(boundary)),
        };

        _a = new T[m];
        _b = new T[m];
        _c = new T[m];
        _d = new T[m];

        for (int i = 0; i < m; i++)
        {
            _a[i] = y[i];
            _c[i] = sigma[i] / s_two;
            _d[i] = (sigma[i + 1] - sigma[i]) / (s_six * h[i]);
            _b[i] = ((y[i + 1] - y[i]) / h[i]) - ((h[i] * (sigma[i + 1] + (s_two * sigma[i]))) / s_six);
        }
    }

    /// <inheritdoc cref="CubicSplineInterpolator.Interpolate"/>
    public T Interpolate(T x)
    {
        int seg = FindSegment(x);
        T dx = x - _x[seg];
        return _a[seg] + (dx * (_b[seg] + (dx * (_c[seg] + (dx * _d[seg])))));
    }

    private static SplineBoundary<T> ConvertBoundary(SplineBoundary boundary)
    {
        ArgumentNullException.ThrowIfNull(boundary);

        return boundary switch
        {
            SplineBoundary.NaturalBoundary => SplineBoundary<T>.Natural,
            SplineBoundary.NotAKnotBoundary => SplineBoundary<T>.NotAKnot,
            SplineBoundary.ModifiedNotAKnotBoundary => SplineBoundary<T>.ModifiedNotAKnot,
            SplineBoundary.QSplineBoundary => SplineBoundary<T>.QSpline,
            SplineBoundary.ClampedBoundary c => SplineBoundary<T>.CreateClamped(
                T.CreateChecked(c.LeftSlope), T.CreateChecked(c.RightSlope)),
            _ => throw new ArgumentException($"Unknown boundary type: {boundary.GetType().Name}", nameof(boundary)),
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
        T[] diag = new T[ic];
        T[] upper = new T[ic];
        T[] lower = new T[ic];
        T[] rhs = new T[ic];

        for (int i = 0; i < ic; i++)
        {
            int k = i + 1;
            diag[i] = s_two * (h[k - 1] + h[k]);
            rhs[i] = (s_six / h[k] * (y[k + 1] - y[k])) - (s_six / h[k - 1] * (y[k] - y[k - 1]));
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

    private static T[] SolveClamped(int n, T[] h, T[] y, T leftSlope, T rightSlope)
    {
        T[] diag = new T[n];
        T[] upper = new T[n];
        T[] lower = new T[n];
        T[] rhs = new T[n];

        diag[0] = s_two * h[0];
        upper[0] = h[0];
        rhs[0] = (s_six / h[0] * (y[1] - y[0])) - (s_six * leftSlope);

        for (int i = 1; i < n - 1; i++)
        {
            lower[i] = h[i - 1];
            diag[i] = s_two * (h[i - 1] + h[i]);
            upper[i] = h[i];
            rhs[i] = (s_six / h[i] * (y[i + 1] - y[i])) - (s_six / h[i - 1] * (y[i] - y[i - 1]));
        }

        lower[n - 1] = h[n - 2];
        diag[n - 1] = s_two * h[n - 2];
        rhs[n - 1] = (s_six * rightSlope) - (s_six / h[n - 2] * (y[n - 1] - y[n - 2]));

        ThomasSolve(diag, upper, lower, rhs, n);

        return rhs;
    }

    private static T[] SolveNotAKnot(int n, T[] h, T[] y)
    {
        int ic = n - 2;
        T[] diag = new T[ic];
        T[] upper = new T[ic];
        T[] lower = new T[ic];
        T[] rhs = new T[ic];

        for (int j = 0; j < ic; j++)
        {
            int i = j + 1;
            lower[j] = h[i - 1];
            diag[j] = s_two * (h[i - 1] + h[i]);
            upper[j] = h[i];
            rhs[j] = (s_six / h[i] * (y[i + 1] - y[i])) - (s_six / h[i - 1] * (y[i] - y[i - 1]));
        }

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

        sigma[0] = ((h[0] + h[1]) * sigma[1] - h[0] * sigma[2]) / h[1];

        sigma[n - 1] = ((h[n - 3] + h[n - 2]) * sigma[n - 2] - h[n - 2] * sigma[n - 3]) / h[n - 3];

        return sigma;
    }

    private static T[] SolveModifiedNotAKnot(int n, T[] h, T[] y)
    {
        int ic = n - 2;
        T[] diag = new T[ic];
        T[] upper = new T[ic];
        T[] lower = new T[ic];
        T[] rhs = new T[ic];

        for (int j = 0; j < ic; j++)
        {
            int i = j + 1;
            lower[j] = h[i - 1];
            diag[j] = s_two * (h[i - 1] + h[i]);
            upper[j] = h[i];
            rhs[j] = (s_six / h[i] * (y[i + 1] - y[i])) - (s_six / h[i - 1] * (y[i] - y[i - 1]));
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

        T epsilon = T.CreateChecked(1e-15);
        if (alphaL > epsilon && ic >= 2)
        {
            sigma[0] = alphaL * (((h[0] + h[1]) * sigma[1] - h[0] * sigma[2]) / h[1]);
        }

        if (alphaR > epsilon && ic >= 2)
        {
            sigma[n - 1] = alphaR * (((h[n - 3] + h[n - 2]) * sigma[n - 2] - h[n - 2] * sigma[n - 3]) / h[n - 3]);
        }

        return sigma;
    }

    private static T[] SolveQSpline(int n, T[] h, T[] y)
    {
        int m = n - 1;

        T[] midEstimates = new T[m];
        for (int i = 0; i < m; i++)
        {
            T xMid = (y[i] + y[i + 1]) / s_two;

            if (i > 0 && i < m - 1)
            {
                T x0 = s_zero, x1 = h[i - 1], x2 = h[i - 1] + h[i];
                T xm = h[i - 1] + h[i] / s_two;
                T l0 = ((xm - x1) * (xm - x2)) / ((x0 - x1) * (x0 - x2));
                T l1 = ((xm - x0) * (xm - x2)) / ((x1 - x0) * (x1 - x2));
                T l2 = ((xm - x0) * (xm - x1)) / ((x2 - x0) * (x2 - x1));
                xMid = l0 * y[i - 1] + l1 * y[i] + l2 * y[i + 1];
            }
            else if (i == 0 && m >= 2)
            {
                T x0 = s_zero, x1 = h[0], x2 = h[0] + h[1];
                T xm = h[0] / s_two;
                T l0 = ((xm - x1) * (xm - x2)) / ((x0 - x1) * (x0 - x2));
                T l1 = ((xm - x0) * (xm - x2)) / ((x1 - x0) * (x1 - x2));
                T l2 = ((xm - x0) * (xm - x1)) / ((x2 - x0) * (x2 - x1));
                xMid = l0 * y[0] + l1 * y[1] + l2 * y[2];
            }
            else if (i == m - 1 && m >= 2)
            {
                T x0 = s_zero, x1 = h[m - 2], x2 = h[m - 2] + h[m - 1];
                T xm = h[m - 2] + h[m - 1] / s_two;
                T l0 = ((xm - x1) * (xm - x2)) / ((x0 - x1) * (x0 - x2));
                T l1 = ((xm - x0) * (xm - x2)) / ((x1 - x0) * (x1 - x2));
                T l2 = ((xm - x0) * (xm - x1)) / ((x2 - x0) * (x2 - x1));
                xMid = l0 * y[n - 3] + l1 * y[n - 2] + l2 * y[n - 1];
            }

            midEstimates[i] = xMid;
        }

        T[] diagFull = new T[n];
        T[] upperFull = new T[n];
        T[] lowerFull = new T[n];
        T[] rhsFull = new T[n];

        {
            var h2 = h[0] * h[0];
            var delta = T.CreateChecked(48) * ((y[0] + y[1]) / s_two - midEstimates[0]) / h2;
            diagFull[0] = T.CreateChecked(11);
            upperFull[0] = T.CreateChecked(7);
            rhsFull[0] = delta;
        }

        for (int i = 1; i < n - 1; i++)
        {
            lowerFull[i] = h[i - 1];
            diagFull[i] = s_two * (h[i - 1] + h[i]);
            upperFull[i] = h[i];
            rhsFull[i] = (s_six / h[i] * (y[i + 1] - y[i])) - (s_six / h[i - 1] * (y[i] - y[i - 1]));
        }

        {
            var h2 = h[m - 1] * h[m - 1];
            var delta = T.CreateChecked(48) * ((y[n - 2] + y[n - 1]) / s_two - midEstimates[m - 1]) / h2;
            lowerFull[n - 1] = T.CreateChecked(11);
            diagFull[n - 1] = T.CreateChecked(7);
            rhsFull[n - 1] = delta;
        }

        ThomasSolve(diagFull, upperFull, lowerFull, rhsFull, n);

        return rhsFull;
    }

    private static void ThomasSolve(T[] diag, T[] upper, T[] lower, T[] rhs, int n)
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

    private int FindSegment(T x)
    {
        if (x <= _x[0])
        {
            return 0;
        }

        if (x >= _x[^1])
        {
            return _a.Length - 1;
        }

        int lo = 0;
        int hi = _x.Length - 1;
        while (hi - lo > 1)
        {
            int mid = (lo + hi) / 2;
            if (_x[mid] > x)
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
/// supporting five boundary conditions — natural, clamped, classical not-a-knot
/// (de Boor 1978), modified not-a-knot and Q-spline (Jarre 2025, arXiv:2507.05083).
/// Tridiagonal system solved by the Thomas algorithm in O(n), unconditionally
/// stable for the diagonally dominant systems arising in cubic spline construction
/// (Faires &amp; Burden, §3.5).
/// </summary>
/// <remarks>
/// <para>
/// The spline passes through all data points exactly and is C² continuous at every
/// interior knot. Pointwise convergence is O(h⁴) for smooth data away from the
/// boundary; boundary error depends on the condition chosen (O(h²) for natural,
/// O(h⁴) for clamped when the true derivative is supplied, and O(h⁴) for not-a-knot
/// variants on uniform or near-uniform grids).
/// </para>
/// <para>
/// Boundary-condition selection guidance:
/// <list type="bullet">
///   <item><see cref="SplineBoundary.Natural"/> — fallback when endpoint derivatives are unknown; simple and stable.</item>
///   <item><see cref="SplineBoundary.CreateClamped"/> — optimal when the endpoint derivative is known analytically.</item>
///   <item><see cref="SplineBoundary.NotAKnot"/> — preferred default on uniform / near-uniform grids.</item>
///   <item><see cref="SplineBoundary.ModifiedNotAKnot"/> — preferred on highly non-uniform grids (Jarre 2025).</item>
///   <item><see cref="SplineBoundary.QSpline"/> — 4th-order optimal using only function values, no derivatives.</item>
/// </list>
/// </para>
/// <para>
/// Outside the node range, <see cref="Interpolate"/> extrapolates along the nearest
/// cubic segment rather than clamping to the boundary value — differs from the
/// flat-extrapolation convention of the stateless interpolators. Prefer this type
/// when repeated evaluations on the same node set are needed (coefficients are
/// computed once at construction); for single-shot use of a monotone or linear
/// scheme, see <see cref="MonotoneCubicInterpolator"/> or <see cref="LinearInterpolator"/>.
/// </para>
/// <para>Minimum 3 data points for natural/clamped boundaries, 4 for the not-a-knot variants. x-values must be strictly increasing.</para>
/// </remarks>
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
