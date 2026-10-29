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

namespace Boutquin.Numerics.MonteCarlo;

/// <summary>
/// Linear-interpolation percentile. The sample at index <c>p·(N−1)</c> is
/// returned, with the fractional part interpolated between the two bracketing
/// samples. Matches NumPy's default <c>linear</c> method and R's <c>type=7</c>.
/// </summary>
/// <remarks>
/// <para>
/// The input array must contain finite values sorted ascending. Empty input returns 0;
/// single-element input returns that element. The percentile rank must be
/// finite; values outside [0, 1] are clamped to the nearest endpoint.
/// </para>
/// <para>
/// Tier A: Arithmetic order-statistic operations on floating-point types.
/// </para>
/// </remarks>
public static class Percentile
{
    /// <summary>Computes the linear-interpolation percentile of a sorted array.</summary>
    /// <param name="sorted">Finite, sorted (ascending) sample.</param>
    /// <param name="p">Finite percentile rank; values outside [0, 1] are clamped.</param>
    public static T Compute<T>(ReadOnlySpan<T> sorted, double p)
        where T : IFloatingPoint<T>
    {
        if (double.IsFinite(p))
        {
            p = Math.Clamp(p, 0.0, 1.0);
        }

        ValidateArguments(sorted, p);

        if (sorted.Length == 0)
        {
            return T.Zero;
        }

        if (sorted.Length == 1)
        {
            return sorted[0];
        }

        var index = p * (sorted.Length - 1);
        var lower = (int)Math.Floor(index);
        var upper = Math.Min(lower + 1, sorted.Length - 1);
        if (typeof(T) == typeof(Half) || typeof(T) == typeof(float))
        {
            // The probability is a double. Preserve its fractional rank until
            // after multiplication; narrowing it first can erase a finite result.
            var low = double.CreateChecked(sorted[lower]);
            var high = double.CreateChecked(sorted[upper]);
            return T.CreateChecked(low + (index - lower) * (high - low));
        }

        var rankFraction = index - lower;
        if (typeof(T) == typeof(decimal) && rankFraction > 0.0 && rankFraction < DecimalFullSignificanceFraction)
        {
            // decimal stores at most 28 fractional digits, so converting a fraction
            // this small keeps fewer than 15 significant digits, or none at all.
            // Form the interval contribution first and narrow nothing before it.
            var low = decimal.CreateChecked(sorted[lower]);
            var high = decimal.CreateChecked(sorted[upper]);
            var increment = low < 0m && high > 0m
                ? ScaleByFraction(high, rankFraction) - ScaleByFraction(low, rankFraction)
                : ScaleByFraction(high - low, rankFraction);
            return T.CreateChecked(low + increment);
        }

        var fraction = T.CreateChecked(rankFraction);

        if (fraction == T.Zero)
        {
            return sorted[lower];
        }

        if (sorted[lower] < T.Zero && sorted[upper] > T.Zero)
        {
            return (T.One - fraction) * sorted[lower] + fraction * sorted[upper];
        }

        return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
    }

    // Smallest double fraction whose decimal conversion keeps all 15 significant
    // digits within decimal's 28-digit scale limit.
    private const double DecimalFullSignificanceFraction = 1e-14;

    // value · fraction for 0 < fraction < 1. The fraction is split exactly as
    // g · 2^-shift with g in [0.5, 1), so its decimal conversion keeps 15 significant
    // digits; the exact power-of-two divisions then round only the scaled product.
    internal static decimal ScaleByFraction(decimal value, double fraction)
    {
        var shift = -(Math.ILogB(fraction) + 1);
        var scaled = value * (decimal)Math.ScaleB(fraction, shift);
        while (shift > 0 && scaled != 0m)
        {
            var step = Math.Min(shift, 62);
            scaled /= 1L << step;
            shift -= step;
        }

        return scaled;
    }

    internal static void ValidateArguments<T>(ReadOnlySpan<T> sorted, double p)
        where T : IFloatingPoint<T>
    {
        if (!double.IsFinite(p) || p < 0.0 || p > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(p), p, "Percentile must be finite and in [0, 1].");
        }

        for (var i = 0; i < sorted.Length; i++)
        {
            if (!T.IsFinite(sorted[i]) || (i > 0 && sorted[i] < sorted[i - 1]))
            {
                throw new ArgumentException("Input must contain finite values sorted in ascending order.", nameof(sorted));
            }
        }
    }
}
