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
/// Shared helper methods used by all interpolator implementations.
/// </summary>
internal static class InterpolationHelper
{
    // Limits of the working type, measured from its own arithmetic (halving and comparing, so
    // they stay within the Tier A operations) and giving double, float, Half and decimal each
    // the limits matching how they round.
    private static class Limits<T> where T : IFloatingPoint<T>
    {
        // Smallest power of two e with 1 + e / 2 == 1: the relative spacing of values near one.
        private static readonly T s_epsilon = MeasureEpsilon();

        // The smallest positive increment (the absolute grid every value is rounded to) and its
        // square root, to within a factor of two: halving a grid of 2^-k gives 2^-(k/2).
        private static readonly (T Quantum, T Root) s_grid = MeasureGrid();

        // Largest tolerated relative error of a stored quotient. A few units in the last place
        // where the grid is negligible beside the precision (IEEE types, whose subnormal grid is
        // astronomically finer than their relative spacing); half the digits where the grid is
        // the precision (decimal's fixed 1e-28 step), because a decimal loses a digit per decade
        // below one and only a severe loss is worth a different evaluation order.
        private static readonly T s_tolerance = T.Max(T.CreateChecked(8) * s_epsilon, s_grid.Root);

        // A stored value v is rounded to within Quantum / 2, a relative error of Quantum / (2|v|).
        // Magnitudes below this bound carry a larger error than the tolerance, which is the point
        // where a subnormal (or a decimal on a wide interval) stops resembling the exact quotient.
        private static readonly T s_coarseBelow = s_grid.Quantum / (T.CreateChecked(2) * s_tolerance);

        // A fraction below this bound leaves the quadratic and cubic terms of a local expansion
        // smaller, relative to the linear term, than the tolerance (or than the coarse bound, where
        // that is the larger), so the linear term alone is as accurate as the type allows.
        private static readonly T s_firstOrderBelow = T.Max(s_coarseBelow, s_tolerance / T.CreateChecked(4));

        internal static T CoarseBelow => s_coarseBelow;

        internal static T FirstOrderBelow => s_firstOrderBelow;

        private static T MeasureEpsilon()
        {
            var half = T.One / T.CreateChecked(2);
            var epsilon = T.One;
            while (T.One + (epsilon * half) != T.One)
            {
                epsilon *= half;
            }

            return epsilon;
        }

        private static (T Quantum, T Root) MeasureGrid()
        {
            var half = T.One / T.CreateChecked(2);
            var quantum = T.One;
            var halvings = 0;
            while (quantum * half != T.Zero)
            {
                quantum *= half;
                halvings++;
            }

            var root = T.One;
            for (var i = 0; i < halvings / 2; i++)
            {
                root *= half;
            }

            return (quantum, root);
        }
    }

    /// <summary>
    /// True when a quotient of two non-zero quantities is too small to hold its significant
    /// digits: it underflowed to a subnormal (or zero) in a binary type, or it sits within a
    /// few decades of decimal's absolute grid. Such a quotient must not be multiplied back up.
    /// </summary>
    internal static bool IsCoarseFraction<T>(T fraction) where T : IFloatingPoint<T> =>
        T.Abs(fraction) < Limits<T>.CoarseBelow;

    /// <summary>
    /// True when a local fraction is so small that the linear term of the local expansion carries
    /// it to the working type's accuracy (the higher-order terms are negligible beside it).
    /// </summary>
    internal static bool IsFirstOrder<T>(T fraction) where T : IFloatingPoint<T> =>
        T.Abs(fraction) < Limits<T>.FirstOrderBelow;

    /// <summary>
    /// Evaluates <c>a · b / divisor</c> in whichever order keeps the most significant digits.
    /// The three orders (<c>(a/d)·b</c>, <c>(b/d)·a</c>, <c>(a·b)/d</c>) agree in exact arithmetic;
    /// they differ in the magnitude of their first intermediate, and a larger intermediate is
    /// stored with a smaller relative error. An order whose intermediate or result is not
    /// finite is skipped.
    /// </summary>
    /// <returns><see langword="false"/> when no order produces a finite result.</returns>
    internal static bool TryScaledProduct<T>(T a, T b, T divisor, out T result)
        where T : IFloatingPoint<T>
    {
        result = T.Zero;
        if (divisor == T.Zero || !T.IsFinite(a) || !T.IsFinite(b) || !T.IsFinite(divisor))
        {
            return false;
        }

        var found = false;
        var largest = T.Zero;
        try
        {
            var quotient = a / divisor;
            Consider(quotient, quotient * b, ref found, ref largest, ref result);
        }
        catch (OverflowException)
        {
            // Decimal throws where IEEE types produce infinity; try the next order.
        }

        try
        {
            var quotient = b / divisor;
            Consider(quotient, quotient * a, ref found, ref largest, ref result);
        }
        catch (OverflowException)
        {
            // Decimal throws where IEEE types produce infinity; try the next order.
        }

        try
        {
            var product = a * b;
            Consider(product, product / divisor, ref found, ref largest, ref result);
        }
        catch (OverflowException)
        {
            // Decimal throws where IEEE types produce infinity.
        }

        return found;

        static void Consider(T intermediate, T value, ref bool found, ref T largest, ref T result)
        {
            if (!T.IsFinite(intermediate) || !T.IsFinite(value))
            {
                return;
            }

            var magnitude = T.Abs(intermediate);
            if (!found || magnitude > largest)
            {
                found = true;
                largest = magnitude;
                result = value;
            }
        }
    }

    /// <summary>
    /// Differences of a query against the left end of a segment and of the segment itself, both
    /// taken in the caller's own units. Dividing a small query by a grid magnitude before
    /// subtracting erases it, so the offset has to be formed before any normalization.
    /// </summary>
    /// <returns><see langword="false"/> when either difference is not representable.</returns>
    internal static bool TryLocalOffset<T>(T x, T left, T right, out T offset, out T width)
        where T : IFloatingPoint<T>
    {
        try
        {
            offset = x - left;
            width = right - left;
            return T.IsFinite(offset) && T.IsFinite(width) && width != T.Zero;
        }
        catch (OverflowException)
        {
            // Decimal throws where IEEE types produce infinity.
            offset = T.Zero;
            width = T.Zero;
            return false;
        }
    }

    /// <summary>
    /// The difference of a query against one node, taken in the caller's own units.
    /// </summary>
    /// <returns><see langword="false"/> when the difference is not representable.</returns>
    internal static bool TryOffset<T>(T x, T node, out T offset)
        where T : IFloatingPoint<T>
    {
        try
        {
            offset = x - node;
            return T.IsFinite(offset);
        }
        catch (OverflowException)
        {
            // Decimal throws where IEEE types produce infinity.
            offset = T.Zero;
            return false;
        }
    }

    /// <summary>
    /// Rejects a non-finite ordinate before the derivative-scale search, whose halving loop
    /// would otherwise never terminate on it.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="ys"/> holds NaN or an infinity.</exception>
    internal static void ValidateFiniteOrdinates<T>(ReadOnlySpan<T> ys, string paramName)
        where T : IFloatingPoint<T>
    {
        for (var i = 0; i < ys.Length; i++)
        {
            if (!T.IsFinite(ys[i]))
            {
                throw new ArgumentException($"ys[{i}] must be finite (value {ys[i]}).", paramName);
            }
        }
    }

    // DerivativeScale already provides tangent headroom. Ordinate scaling is
    // needed only when a difference itself overflows. Division by four leaves
    // rounding headroom even at decimal's odd integer maximum. Scaling by the
    // largest ordinate needlessly destroys small independent intervals.
    internal static T OrdinateScale<T>(ReadOnlySpan<T> ys) where T : IFloatingPoint<T>
    {
        for (var i = 0; i < ys.Length - 1; i++)
        {
            try
            {
                if (!T.IsFinite(ys[i + 1] - ys[i]))
                {
                    return T.CreateChecked(4);
                }
            }
            catch (OverflowException)
            {
                return T.CreateChecked(4);
            }
        }

        return T.One;
    }

    // A common derivative scale cancels out of tangent limiting, so it is a free parameter that
    // keeps every scaled secant (and everything the tangent formulas build from them) inside the
    // working type's range.
    //
    // A non-finite ordinate has no scale: every candidate secant stays non-finite, so the search
    // refuses it here, at the one place every caller reaches, rather than relying on each public
    // entry to have validated first.
    //
    // Ordinary data never needs it: when every secant has the legacy headroom (eight times the
    // secant times the widest width is finite) and none has underflowed, the scale is one and the
    // legacy tangent formulas apply, bit for bit.
    //
    // Otherwise the robust formulation applies. Its tangents are weighted means of the secants
    // (weights at most one, so they need no width headroom) and the scale is chosen on the
    // secants alone:
    //  - a secant without headroom (a steep, narrow interval) shrinks the scale just enough;
    //  - a secant that underflowed or fell onto a decimal's grid (tiny ordinates on a wide grid)
    //    grows the scale as far as the largest secant allows, because the scale is a power of
    //    two and scaling by it is exact.
    // A single scale cannot serve secants that span the type's whole exponent range; those
    // beyond the headroom stay as coarse as the type makes them.
    internal static T DerivativeScale<T>(ReadOnlySpan<T> xs, ReadOnlySpan<T> ys, out bool robust)
        where T : IFloatingPoint<T>
    {
        ValidateFiniteOrdinates(ys, nameof(ys));
        robust = false;
        if (LegacyScaleSuffices(xs, ys))
        {
            return T.One;
        }

        robust = true;
        var scale = T.One;
        if (!HasHeadroom(xs, ys, scale))
        {
            // Largest power of two below one that restores headroom; zero scales every secant away,
            // so the search always ends.
            var half = T.One / T.CreateChecked(2);
            while (!HasHeadroom(xs, ys, scale))
            {
                var next = scale * half;
                if (next == T.Zero)
                {
                    break;
                }

                scale = next;
            }

            return scale;
        }

        return HasCoarseSecant(xs, ys, scale) ? GrowWhileHeadroom(xs, ys, scale) : scale;
    }

    // The unit-scale conditions under which the legacy formulas are valid.
    private static bool LegacyScaleSuffices<T>(ReadOnlySpan<T> xs, ReadOnlySpan<T> ys)
        where T : IFloatingPoint<T>
    {
        var largestWidth = T.One;
        for (var i = 0; i < xs.Length - 1; i++)
        {
            largestWidth = T.Max(largestWidth, xs[i + 1] - xs[i]);
        }

        for (var i = 0; i < xs.Length - 1; i++)
        {
            try
            {
                var rise = ys[i + 1] - ys[i];
                var slope = rise / (xs[i + 1] - xs[i]);
                if (!T.IsFinite((T.CreateChecked(8) * slope) * largestWidth)
                    || (rise != T.Zero && IsCoarseFraction(slope)))
                {
                    return false;
                }
            }
            catch (OverflowException)
            {
                // Decimal signals missing headroom with an exception.
                return false;
            }
        }

        return true;
    }

    // Every secant, scaled by `scale`, is finite with room to multiply by eight.
    private static bool HasHeadroom<T>(ReadOnlySpan<T> xs, ReadOnlySpan<T> ys, T scale)
        where T : IFloatingPoint<T>
    {
        for (var i = 0; i < xs.Length - 1; i++)
        {
            try
            {
                if (!TryScaledSecant(ys[i + 1] - ys[i], xs[i + 1] - xs[i], scale, out var secant)
                    || !T.IsFinite(T.CreateChecked(8) * secant))
                {
                    return false;
                }
            }
            catch (OverflowException)
            {
                return false;
            }
        }

        return true;
    }

    // Some non-zero secant, scaled by `scale`, has lost digits (underflowed or sits on decimal's grid).
    private static bool HasCoarseSecant<T>(ReadOnlySpan<T> xs, ReadOnlySpan<T> ys, T scale)
        where T : IFloatingPoint<T>
    {
        for (var i = 0; i < xs.Length - 1; i++)
        {
            var rise = ys[i + 1] - ys[i];
            if (rise != T.Zero
                && TryScaledSecant(rise, xs[i + 1] - xs[i], scale, out var secant)
                && IsCoarseFraction(secant))
            {
                return true;
            }
        }

        return false;
    }

    // Largest power of two, reached from `scale` by greedy steps of 2, 4, 16, 256, ... (each step
    // tried until it no longer fits, largest first), that keeps headroom. A step of 2^(2^k) can be
    // taken at most once after the larger ones, so the search needs O(log) headroom checks.
    private static T GrowWhileHeadroom<T>(ReadOnlySpan<T> xs, ReadOnlySpan<T> ys, T scale)
        where T : IFloatingPoint<T>
    {
        var steps = new List<T>();
        var step = T.CreateChecked(2);
        while (true)
        {
            steps.Add(step);
            T squared;
            try
            {
                squared = step * step;
            }
            catch (OverflowException)
            {
                break;
            }

            if (!T.IsFinite(squared))
            {
                break;
            }

            step = squared;
        }

        for (var i = steps.Count - 1; i >= 0; i--)
        {
            while (true)
            {
                T candidate;
                try
                {
                    candidate = scale * steps[i];
                }
                catch (OverflowException)
                {
                    break;
                }

                if (!T.IsFinite(candidate) || !HasHeadroom(xs, ys, candidate))
                {
                    break;
                }

                scale = candidate;
            }
        }

        return scale;
    }

    // rise / run * scale, ordered so that no intermediate underflows or overflows when the result fits.
    private static bool TryScaledSecant<T>(T rise, T run, T scale, out T secant)
        where T : IFloatingPoint<T>
    {
        if (scale == T.One)
        {
            secant = rise / run;
            return T.IsFinite(secant);
        }

        return TryScaledProduct(rise, scale, run, out secant);
    }

    /// <summary>
    /// The secant <c>rise / run</c> times the derivative scale. At unit scale this is the plain
    /// quotient, so the legacy path is untouched; otherwise it is ordered so that the product of
    /// a tiny scale and a wide run cannot underflow before the rise multiplies it back up.
    /// </summary>
    internal static T ScaledSecant<T>(T rise, T run, T scale, bool robust)
        where T : IFloatingPoint<T>
    {
        if (!robust || scale == T.One)
        {
            return rise / run;
        }

        return TryScaledProduct(rise, scale, run, out var secant) ? secant : (rise / run) * scale;
    }

    /// <summary>
    /// Tangent at an interior node as the mean of the adjacent secants weighted by the opposite
    /// widths. The legacy form multiplies a secant by a width before dividing by the width sum;
    /// the robust form divides first, so a width as large as the type allows cannot overflow it.
    /// </summary>
    internal static T InteriorTangent<T>(T previousWidth, T nextWidth, T previousSecant, T nextSecant, bool robust)
        where T : IFloatingPoint<T>
    {
        if (!robust)
        {
            return ((nextWidth * previousSecant) + (previousWidth * nextSecant)) / (previousWidth + nextWidth);
        }

        var sum = previousWidth + nextWidth;
        return (previousSecant * (nextWidth / sum)) + (nextSecant * (previousWidth / sum));
    }

    /// <summary>
    /// One-sided three-point tangent at an end node, from the end secant and the one beside it.
    /// </summary>
    internal static T EndpointTangent<T>(T nearWidth, T farWidth, T nearSecant, T farSecant, bool robust)
        where T : IFloatingPoint<T>
    {
        var two = T.CreateChecked(2);
        if (!robust)
        {
            return (((two * nearWidth) + farWidth) * nearSecant - (nearWidth * farSecant)) / (nearWidth + farWidth);
        }

        var weight = nearWidth / (nearWidth + farWidth);
        return (nearSecant * (T.One + weight)) - (farSecant * weight);
    }

    internal static T IntegratedDerivative<T>(T derivative, T width, T derivativeScale) where T : IFloatingPoint<T>
    {
        // A finite width/scale can still round to zero (or one decimal ulp).
        // Use the shared ordering selection before the final narrowing instead.
        return TryScaledProduct(derivative, width, derivativeScale, out var result)
            ? result
            : (derivative / derivativeScale) * width;
    }

    // Leave ordinary grids unchanged. Extreme grids are divided before subtraction,
    // so both overflowing widths and unrepresentable reciprocal widths are avoided.
    internal static T CoordinateScale<T>(ReadOnlySpan<T> xs) where T : IFloatingPoint<T>
    {
        var magnitude = T.Max(T.Abs(xs[0]), T.Abs(xs[^1]));
        var headroom = T.CreateChecked(16);
        try
        {
            if (T.IsFinite(magnitude * headroom)
                && T.IsFinite(headroom / (xs[^1] - xs[0])))
            {
                return T.One;
            }
        }
        catch (OverflowException)
        {
            // Decimal throws where IEEE arithmetic produces infinity.
        }

        var previous = xs[0] / magnitude;
        for (var i = 1; i < xs.Length; i++)
        {
            var current = xs[i] / magnitude;
            if (current <= previous)
            {
                throw new InvalidOperationException("Coordinate normalization cannot preserve distinct knots in the working type.");
            }

            previous = current;
        }

        return magnitude;
    }

    internal static T LinearInterpolate<T>(T x, T x0, T y0, T x1, T y1)
        where T : IFloatingPoint<T>
    {
        var weight = LinearWeight(x, x0, x1);
        if (x != x0 && T.IsFinite(x) && IsCoarseFraction(weight)
            && TryLinearContribution(x, x0, y0, x1, y1, out var result))
        {
            // A weight that rounded to zero, to a subnormal or onto decimal's grid has lost digits
            // that the ordinate contribution still carries once the scales cancel. Rebuild the
            // contribution from the unrounded differences instead of multiplying the weight up.
            return result;
        }

        return LinearBlend(y0, y1, weight);
    }

    // y0 + (x - x0) · (y1 - y0) / (x1 - x0), ordered so that no intermediate underflows.
    // Returns false when the differences or the sum are not representable, leaving the bounded
    // blend to handle them.
    private static bool TryLinearContribution<T>(T x, T x0, T y0, T x1, T y1, out T result)
        where T : IFloatingPoint<T>
    {
        result = T.Zero;
        try
        {
            var offset = x - x0;
            var width = x1 - x0;
            var rise = y1 - y0;
            if (!T.IsFinite(offset) || !T.IsFinite(width) || !T.IsFinite(rise)
                || !TryScaledProduct(offset, rise, width, out var contribution))
            {
                return false;
            }

            var sum = y0 + contribution;
            if (!T.IsFinite(sum))
            {
                return false;
            }

            result = sum;
            return true;
        }
        catch (OverflowException)
        {
            // Retain the bounded blend when the alternate order overflows.
            return false;
        }
    }

    /// <summary>
    /// Validates that <paramref name="xs"/> is strictly ascending.
    /// </summary>
    /// <exception cref="ArgumentException">xs is not strictly increasing.</exception>
    internal static void ValidateSortedAscending<T>(ReadOnlySpan<T> xs, string paramName)
        where T : IFloatingPoint<T>
    {
        for (var i = 0; i < xs.Length; i++)
        {
            if (!T.IsFinite(xs[i]))
            {
                throw new ArgumentException($"xs[{i}] must be finite (value {xs[i]}).", paramName);
            }
        }

        for (var i = 1; i < xs.Length; i++)
        {
            if (xs[i] <= xs[i - 1])
            {
                throw new ArgumentException(
                    $"xs must be strictly ascending (xs[{i - 1}] = {xs[i - 1]}, xs[{i}] = {xs[i]}).",
                    paramName);
            }
        }
    }

    /// <summary>
    /// Rejects a non-finite target before a caller's extrapolation branches.
    /// </summary>
    internal static void ValidateFiniteTarget<T>(T x)
        where T : IFloatingPoint<T>
    {
        if (!T.IsFinite(x))
        {
            throw new ArgumentOutOfRangeException(nameof(x), "The interpolation target must be finite.");
        }
    }

    /// <summary>
    /// Computes the linear weight without overflowing a difference between finite coordinates.
    /// The ordinary path retains its original arithmetic; only an overflowing subtraction
    /// quarters all three coordinates before taking their differences.
    /// </summary>
    internal static T LinearWeight<T>(T x, T x0, T x1)
        where T : IFloatingPoint<T>
    {
        if (!T.IsFinite(x) || !T.IsFinite(x0) || !T.IsFinite(x1))
        {
            return (x - x0) / (x1 - x0);
        }

        try
        {
            var numerator = x - x0;
            var denominator = x1 - x0;
            if (T.IsFinite(numerator) && T.IsFinite(denominator))
            {
                return numerator / denominator;
            }
        }
        catch (OverflowException)
        {
            // decimal throws on overflow; IEEE types return an infinity instead.
        }

        // Quartering leaves headroom even when decimal.MaxValue / 2 rounds up.
        var quarter = T.One / T.CreateChecked(4);
        return ((x * quarter) - (x0 * quarter)) / ((x1 * quarter) - (x0 * quarter));
    }

    /// <summary>
    /// Blends finite ordinates without overflowing their difference when the result is finite.
    /// Returns the supplied second ordinate at unit weight; otherwise retains the original
    /// arithmetic whenever its intermediate result is representable.
    /// </summary>
    internal static T LinearBlend<T>(T y0, T y1, T weight)
        where T : IFloatingPoint<T>
    {
        // Subtracting a much larger y0 can erase y1 even without overflow.
        if (weight == T.One)
        {
            return y1;
        }

        try
        {
            var result = y0 + weight * (y1 - y0);
            if (T.IsFinite(result))
            {
                return result;
            }
        }
        catch (OverflowException)
        {
            // decimal throws where IEEE types produce infinity.
        }

        return (T.One - weight) * y0 + weight * y1;
    }

    /// <summary>
    /// Finds the index i such that xs[i] &lt;= x &lt; xs[i+1] using binary search.
    /// The caller must ensure that xs[0] &lt; x &lt; xs[^1] (boundary cases handled before calling).
    /// </summary>
    internal static int FindInterval<T>(T x, ReadOnlySpan<T> xs)
        where T : IFloatingPoint<T>
    {
        int lo = 0, hi = xs.Length - 2;
        while (lo < hi)
        {
            var mid = (lo + hi + 1) / 2;
            if (x < xs[mid])
            {
                hi = mid - 1;
            }
            else
            {
                lo = mid;
            }
        }

        return lo;
    }
}
