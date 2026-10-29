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
/// Hagan-West (2006) monotone-convex interpolation. Guarantees non-negative instantaneous
/// forward rates across the full maturity range for finite, non-decreasing normalized cumulative
/// return (NCR) node arrays derived from discount factors.
/// </summary>
/// <typeparam name="T">The floating-point type for interpolation.</typeparam>
/// <remarks>
/// Tier A: Requires only basic floating-point arithmetic. The <c>unmanaged</c> constraint
/// enables stack allocation of the per-call working buffers (<c>h</c>, <c>F</c>, <c>g</c>),
/// eliminating heap pressure for the typical small node counts used in yield-curve construction.
/// All standard numeric types (<c>double</c>, <c>float</c>, <c>Half</c>, <c>decimal</c>) satisfy
/// this constraint.
/// </remarks>
public sealed class MonotoneConvexInterpolator<T> : IInterpolator<T> where T : unmanaged, IFloatingPoint<T>
{
    private const int StackAllocThreshold = 256;

    /// <inheritdoc cref="MonotoneConvexInterpolator.Instance"/>
    public static readonly MonotoneConvexInterpolator<T> Instance = new();

    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_three = T.CreateChecked(3);

    private MonotoneConvexInterpolator() { }

    /// <inheritdoc/>
    public string Name => "MonotoneConvex";

    /// <summary>
    /// Evaluates the monotone-convex interpolant at <paramref name="x"/> given an augmented
    /// node array that includes a virtual node at <c>xs[0] = 0, ys[0] = 0</c> followed by
    /// the actual curve nodes.
    /// </summary>
    /// <param name="x">Target x-coordinate (year fraction). Must be positive for meaningful results.</param>
    /// <param name="xs">
    /// Strictly-increasing node x-coordinates, with <c>xs[0] = 0</c> (the virtual origin node)
    /// and <c>xs[1..N-1]</c> as the actual node times in year fractions.
    /// </param>
    /// <param name="ys">
    /// Node normalized cumulative return (NCR) values: <c>ys[i] = −ln P(xs[i])</c>. Must satisfy
    /// <c>ys[0] = 0</c> (the virtual origin, corresponding to <c>P(0) = 1</c>). All values must be
    /// finite and non-decreasing; otherwise non-negative forward rates cannot be guaranteed.
    /// </param>
    /// <returns>
    /// The interpolated NCR value at <paramref name="x"/>. Convert back to a discount factor via
    /// <c>P(x) = exp(−result)</c>.
    /// </returns>
    /// <remarks>
    /// Algorithm summary (Hagan &amp; West 2006; the paper numbers equations, not sections):
    /// <list type="number">
    ///   <item>Compute per-segment discrete forward rates <c>F[j] = (ys[j] − ys[j−1]) / h[j]</c>,
    ///         where <c>h[j] = xs[j] − xs[j−1]</c>.</item>
    ///   <item>Estimate instantaneous forwards <c>f[i]</c> at each node as a time-weighted
    ///         average of adjacent segment forwards (eq. 30); extrapolate to boundary nodes
    ///         (eqs. 31–32).</item>
    ///   <item>Bound each <c>f[i]</c> to <c>[0, 2·min(F[i], F[i+1])]</c> (eqs. 60–62) so the
    ///         instantaneous forward stays non-negative.</item>
    ///   <item>On the bracketing segment, choose the monotonicity/convexity sector of
    ///         <c>(f[i−1] − F[i], f[i] − F[i])</c> (eqs. 45–56): the basic quadratic in sector (i),
    ///         a flat segment joined to a quadratic in sectors (ii) and (iii), or two quadratics
    ///         meeting at their extremum in sector (iv). Monotone inputs therefore give a monotone
    ///         forward curve.</item>
    ///   <item>Integrate the chosen forward within the segment (eq. 37) to obtain the NCR at
    ///         <paramref name="x"/>. The amelioration step (eqs. 63–80) is not applied.</item>
    /// </list>
    /// Constant-forward extrapolation beyond the node range. Stateless; exposed as the singleton
    /// <see cref="Instance"/>. Every call recomputes tangents from the node arrays.
    /// Reference: Hagan, P. S. and West, G. (2006), "Interpolation Methods for Curve Construction",
    /// Applied Mathematical Finance 13(2), pp. 89–129.
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="xs"/> and <paramref name="ys"/>
    /// have different lengths, the virtual origin is invalid, <paramref name="xs"/> is not
    /// strictly ascending, or <paramref name="ys"/> is non-finite or decreases.</exception>
    /// <exception cref="InvalidOperationException">Coordinate normalization cannot preserve distinct knots in the working type.</exception>
    public T Interpolate(T x, ReadOnlySpan<T> xs, ReadOnlySpan<T> ys)
    {
        if (xs.Length != ys.Length)
        {
            throw new ArgumentException("xs and ys must have the same length.");
        }

        if (xs.Length < 2)
        {
            throw new ArgumentException("The virtual origin and at least one actual node are required.", nameof(xs));
        }

        // The virtual origin node must be (numerically) zero. Use a small absolute
        // tolerance rather than exact equality so callers who compute the origin
        // (e.g. as t - t) and land a few ULPs off zero are not spuriously rejected.
        var originTolerance = T.CreateChecked(1e-12);
        if (T.Abs(xs[0]) > originTolerance)
        {
            throw new ArgumentException("xs[0] must be 0 (the virtual origin node).", nameof(xs));
        }

        if (!T.IsFinite(ys[0]) || T.Abs(ys[0]) > originTolerance)
        {
            throw new ArgumentException("ys[0] must be 0 (the virtual origin node).", nameof(ys));
        }

        InterpolationHelper.ValidateSortedAscending(xs, nameof(xs));
        InterpolationHelper.ValidateFiniteTarget(x);

        for (int i = 1; i < ys.Length; i++)
        {
            if (!T.IsFinite(ys[i]) || ys[i] < ys[i - 1])
            {
                throw new ArgumentException("ys must contain finite, non-decreasing NCR values.", nameof(ys));
            }
        }

        var originalXs = xs;
        var originalX = x;
        var coordinateScale = InterpolationHelper.CoordinateScale(xs);
        if (coordinateScale != T.One)
        {
            var normalized = new T[xs.Length];
            for (var i = 0; i < xs.Length; i++)
            {
                normalized[i] = xs[i] / coordinateScale;
            }

            xs = normalized;
            x /= coordinateScale;
        }

        var ordinateScale = InterpolationHelper.OrdinateScale(ys);
        var scaleOrdinates = ordinateScale != T.One;
        if (scaleOrdinates)
        {
            var normalized = new T[ys.Length];
            for (var i = 0; i < ys.Length; i++)
            {
                normalized[i] = ys[i] / ordinateScale;
            }

            ys = normalized;
        }

        int N = xs.Length; // total nodes (includes virtual node at index 0)
        var derivativeScale = InterpolationHelper.DerivativeScale(xs, ys, out var robust);

        // --- Step 1: Segment widths h[j] and discrete forward rates F[j] for j = 1..N-1 ---
        using var scratch = new PooledScratch<T>((N) > StackAllocThreshold ? checked(3 * (N)) : 0);
        var rented = scratch.Buffer;
        {
            Span<T> h = N <= StackAllocThreshold ? stackalloc T[N] : rented.AsSpan(0 * N, N);
            Span<T> F = N <= StackAllocThreshold ? stackalloc T[N] : rented.AsSpan(1 * N, N);
            h.Clear();
            F.Clear();
            for (int j = 1; j < N; j++)
            {
                h[j] = xs[j] - xs[j - 1];
                F[j] = InterpolationHelper.ScaledSecant(ys[j] - ys[j - 1], h[j], derivativeScale, robust);
            }

            // --- Step 2: Instantaneous forward tangents g[i] at each node ---
            Span<T> g = N <= StackAllocThreshold ? stackalloc T[N] : rented.AsSpan(2 * N, N);
            g.Clear();

            // Interior nodes: time-weighted average of adjacent segment forwards
            for (int i = 1; i < N - 1; i++)
            {
                g[i] = InterpolationHelper.InteriorTangent(h[i], h[i + 1], F[i], F[i + 1], robust);
            }

            // Boundary extrapolation (Hagan-West eqs. 31-32)
            g[0] = N >= 3 ? (s_three * F[1] - g[1]) / s_two : F[1];
            g[N - 1] = N >= 3 ? (s_three * F[N - 1] - g[N - 2]) / s_two : F[N - 1];

            // --- Step 3: Enforce positivity of the node forwards (Hagan-West eqs. 60-62) ---
            g[0] = T.Max(T.Zero, T.Min(g[0], s_two * F[1]));

            for (int i = 1; i < N - 1; i++)
            {
                T limit = T.Min(s_two * F[i], s_two * F[i + 1]);
                g[i] = limit > T.Zero ? T.Max(T.Zero, T.Min(g[i], limit)) : T.Zero;
            }

            g[N - 1] = T.Max(T.Zero, T.Min(g[N - 1], s_two * F[N - 1]));

            // Continue the integrated curve with a constant endpoint forward. A query so close to an
            // endpoint that normalization rounds it onto that endpoint takes this branch too; the
            // tail is then measured from the caller's own units, which keeps its first-order value.
            var normalizedCoordinates = coordinateScale != T.One;
            if (x <= xs[0] || x >= xs[^1])
            {
                var endpoint = x <= xs[0] ? 0 : N - 1;
                T tail;
                if (normalizedCoordinates
                    && InterpolationHelper.TryOffset(originalX, originalXs[endpoint], out var endpointOffset)
                    && InterpolationHelper.TryScaledProduct(g[endpoint], endpointOffset, coordinateScale * derivativeScale, out var scaledTail))
                {
                    tail = scaledTail;
                }
                else
                {
                    tail = InterpolationHelper.IntegratedDerivative(g[endpoint], x - xs[endpoint], derivativeScale);
                }

                var exterior = ys[endpoint] + tail;
                return scaleOrdinates ? exterior * ordinateScale : exterior;
            }

            // --- Step 4: Locate the segment containing x ---
            // x is strictly interior here (boundary cases returned above), so the shared O(log N)
            // bracket search applies. FindInterval returns i with xs[i] <= x < xs[i+1]; this method's
            // segment index k is 1-based (it reads xs[k-1], xs[k]), hence k = i + 1.
            int k = (normalizedCoordinates
                ? InterpolationHelper.FindInterval(originalX, originalXs)
                : InterpolationHelper.FindInterval(x, xs)) + 1;

            // --- Step 5: Integrate the forward f = F[k] + g within segment k (eq. 37) ---
            // u = relative position in [0, 1]; deltaR is the NCR increment over u. The deviation g
            // from F[k] is the basic quadratic (eq. 47) or its monotonicity/convexity replacement
            // for the sector of (g[k-1] - F[k], g[k] - F[k]) (eqs. 45-56).
            T offset;
            T u;
            var offsetExact = true;
            if (!normalizedCoordinates)
            {
                offset = x - xs[k - 1];
                u = offset / h[k];
            }
            else if (InterpolationHelper.TryLocalOffset(originalX, originalXs[k - 1], originalXs[k], out offset, out var originalWidth))
            {
                u = offset / originalWidth;
            }
            else
            {
                offsetExact = false;
                offset = T.Zero;
                u = (x - xs[k - 1]) / h[k];
            }

            var bracket = F[k] * u + IntegrateDeviation(g[k - 1] - F[k], g[k] - F[k], u);
            T deltaR;
            if (offsetExact
                && offset != T.Zero
                && g[k - 1] != T.Zero
                && InterpolationHelper.IsFirstOrder(u)
                && (InterpolationHelper.IsCoarseFraction(u) || InterpolationHelper.IsCoarseFraction(bracket))
                && InterpolationHelper.TryScaledProduct(g[k - 1], offset, coordinateScale * derivativeScale, out var linear))
            {
                // The integral of the forward over so short a stretch is the node forward times the
                // unrounded offset (the curvature terms are smaller than u times that), and either u
                // or the integrand, which the segment width scales back up, has lost its digits.
                deltaR = linear;
            }
            else
            {
                deltaR = InterpolationHelper.IntegratedDerivative(bracket, h[k], derivativeScale);
            }

            var result = ys[k - 1] + deltaR;
            return scaleOrdinates ? result * ordinateScale : result;
        }
    }

    /// <summary>
    /// Integral over <c>[0, u]</c> of the forward deviation <c>g(x) = f(x) − Fᵢ</c> on one interval,
    /// with <paramref name="left"/> = <c>gᵢ₋₁</c> and <paramref name="right"/> = <c>gᵢ</c>
    /// (Hagan &amp; West 2006, "Enforcing Monotonicity and Convexity", eqs. 45–56).
    /// </summary>
    private static T IntegrateDeviation(T left, T right, T u)
    {
        // Footnotes 7, 8 and 10: trap x = 0 and the flat curve g = 0 before any division.
        if (u == T.Zero || (left == T.Zero && right == T.Zero))
        {
            return T.Zero;
        }

        var halfLeft = left / s_two;
        var twiceLeft = s_two * left;

        // Sector (i): the basic quadratic, eq. 47, already has the required monotonicity.
        if ((left > T.Zero && right <= -halfLeft && right >= -twiceLeft)
            || (left < T.Zero && right >= -halfLeft && right <= -twiceLeft))
        {
            var u2 = u * u;
            return left * (u - s_two * u2 + u2 * u) + right * (u2 * u - u2);
        }

        // Sector (ii): flat at gᵢ₋₁ up to η, then a quadratic to gᵢ (eqs. 49-50).
        if ((left < T.Zero && right > -twiceLeft) || (left > T.Zero && right < -twiceLeft))
        {
            var eta = (right + twiceLeft) / (right - left);
            if (u <= eta)
            {
                return left * u;
            }

            // Width of the quadratic part, 1 - eta = 3*left/(left - right). The operands have opposite
            // signs, so unlike one minus the rounded eta this keeps its digits when eta is close to one.
            var width = s_three * left / (left - right);
            if (s_two * u > T.One && width > T.Zero)
            {
                return RemainingArea(left, right, u, width);
            }

            var tail = (u - eta) / (T.One - eta);
            return left * u + (right - left) * (T.One - eta) * tail * tail * tail / s_three;
        }

        // Sector (iii): a quadratic from gᵢ₋₁ to gᵢ reached at η, then flat (eqs. 51-52).
        if ((left > T.Zero && right < T.Zero && right > -halfLeft)
            || (left < T.Zero && right > T.Zero && right < -halfLeft))
        {
            var eta = s_three * right / (right - left);
            if (u >= eta)
            {
                // The deviation integrates to zero over the whole interval, so after the kink the
                // integral is minus the flat tail, right * (u - 1). Building it from the rounded eta
                // subtracts two nearly equal numbers as u approaches one.
                return right * (u - T.One);
            }

            // Up to the kink g falls from left to right on a quadratic with zero slope at eta, so with
            // w = u / eta the integral is left*u + (right - left)*u*w*(1 - w/3). Subtracting the cubed
            // head (eta - u)/eta from one instead loses the first-order term for a small u.
            var w = u / eta;
            var h = w * (T.One - (w / s_three));
            return u * ((left * (T.One - h)) + (right * h));
        }

        // Sector (iv): gᵢ₋₁ and gᵢ share a sign; two quadratics meet at their extremum
        // A = −gᵢ₋₁gᵢ/(gᵢ₋₁ + gᵢ) at η = gᵢ/(gᵢ₋₁ + gᵢ) (eqs. 53-56). A = −gᵢ₋₁·η avoids the
        // product gᵢ₋₁gᵢ, which can overflow for finite decimal forwards.
        var etaIv = right / (left + right);
        var extremum = -left * etaIv;
        if (u <= etaIv)
        {
            // Same quadratic as sector (iii) before the kink: with w = u / eta the integral is
            // left*u + (extremum - left)*u*w*(1 - w/3), without the cancellation of one minus the
            // cubed head for a small u.
            var w = u / etaIv;
            var h = w * (T.One - (w / s_three));
            return u * ((left * (T.One - h)) + (extremum * h));
        }

        // Width of the second quadratic, 1 - etaIv = left/(left + right); same-sign operands, so no cancellation.
        var widthIv = left / (left + right);
        if (s_two * u > T.One && widthIv > T.Zero)
        {
            return RemainingArea(extremum, right, u, widthIv);
        }

        var rise = (u - etaIv) / (T.One - etaIv);
        return extremum * u
            + (left - extremum) * etaIv / s_three
            + (right - extremum) * (T.One - etaIv) * rise * rise * rise / s_three;
    }

    /// <summary>
    /// Integral over <c>[0, u]</c> of a deviation whose integral over <c>[0, 1]</c> is zero, taken as minus
    /// the integral over <c>[u, 1]</c>. There the forward rises from <paramref name="plateau"/> (the flat value
    /// or the extremum) to <paramref name="right"/> on a quadratic of the given <paramref name="width"/> with
    /// zero slope at its start, so its mean over <c>[u, 1]</c> is <c>plateau + (right - plateau)·k</c> with
    /// <c>k = 1 - v + v²/3</c> and <c>v = (1 - u)/width</c>. Close to the end of the interval this keeps the
    /// relative accuracy that the sum of the two near-equal areas it replaces loses.
    /// </summary>
    private static T RemainingArea(T plateau, T right, T u, T width)
    {
        // v lies in [0, 1] while u is past the kink; the clamp only absorbs rounding in the kink position.
        var v = T.Min((T.One - u) / width, T.One);
        var k = T.One - v + (v * v / s_three);
        return -(T.One - u) * (plateau + ((right - plateau) * k));
    }
}

/// <summary>
/// Hagan-West (2006) monotone-convex interpolation. Guarantees non-negative instantaneous
/// forward rates across the full maturity range for finite, non-decreasing normalized cumulative
/// return (NCR) node arrays derived from discount factors.
/// </summary>
/// <remarks>
/// <para>
/// Domain-layer usage: callers supply an augmented node array whose first element is the virtual
/// origin (<c>xs[0] = 0, ys[0] = 0</c>), followed by the actual curve nodes expressed as year
/// fractions and NCR values (<c>ys[i] = −ln P(xs[i])</c>). The method returns the interpolated
/// NCR; convert back to a discount factor via <c>P = exp(−result)</c>. NCR values must be finite
/// and non-decreasing; the generic implementation rejects arrays that violate this condition.
/// </para>
/// <para>
/// Algorithm: Hagan, P. S. and West, G. (2006), "Interpolation Methods for Curve Construction",
/// Applied Mathematical Finance 13(2), pp. 89–129.
/// </para>
/// <para>
/// Unlike the monotone cubic Hermite method (<see cref="MonotoneCubicInterpolator"/>), which
/// preserves monotonicity of the y-values, the Hagan-West method preserves <em>non-negativity
/// of the derivative</em> for finite, non-decreasing NCR nodes — equivalently, non-negative
/// instantaneous forward rates — which is the relevant no-arbitrage constraint for
/// discount-factor curves.
/// </para>
/// <para>
/// Constant-forward extrapolation beyond the node range. Stateless; exposed as the singleton
/// <see cref="Instance"/>. Every call recomputes tangents from the node arrays.
/// For repeated evaluations on the same node set consider caching the node arrays
/// rather than this instance, as there is no stateful precomputed form of this algorithm.
/// </para>
/// </remarks>
[Obsolete(
    "Use MonotoneConvexInterpolator<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class MonotoneConvexInterpolator : IInterpolator
{
    /// <summary>
    /// Shared singleton instance.
    /// </summary>
    public static readonly MonotoneConvexInterpolator Instance = new();

    private MonotoneConvexInterpolator() { }

    /// <inheritdoc />
    public string Name => "MonotoneConvex";

    /// <inheritdoc />
    public double Interpolate(double x, ReadOnlySpan<double> xs, ReadOnlySpan<double> ys)
    {
        return MonotoneConvexInterpolator<double>.Instance.Interpolate(x, xs, ys);
    }
}
