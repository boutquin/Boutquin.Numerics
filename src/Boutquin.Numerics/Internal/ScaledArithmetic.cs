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

namespace Boutquin.Numerics.Internal;

/// <summary>
/// Scaled reductions that keep a sum of squares, an inner product or a norm inside the type's range when the
/// inputs are representable but their raw squares are not.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. Dispatches on <c>typeof(T)</c> at JIT time, like
/// <see cref="NumericPrecision{T}"/>. A span is first divided by a power of the type's radix chosen from its
/// largest magnitude, the reduction runs on values of magnitude at most one, and the caller restores the
/// result with <see cref="Rescale"/> (or <see cref="TryRescale"/>) using twice the exponent for a square.
/// </para>
/// <para>
/// Binary IEEE types (<see cref="double"/>, <see cref="float"/>, <see cref="Half"/>) scale by <c>2^e</c>, which
/// is exact except where a scaled value is subnormal. <see cref="decimal"/> scales by <c>10^e</c>, which is
/// exact while the result needs at most 28 decimal places; for a value with more fractional digits than that
/// the excess digits are rounded by at most <c>0.5e-28</c> per step in scaled units. Because the scaled
/// maximum lies in <c>[0.1, 1)</c>, this is at most <c>5e-28</c> relative to that maximum for one step;
/// multi-step rescaling can round more than once. Other <typeparamref name="T"/> use exponent 0
/// and run the reduction unscaled.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
internal static class ScaledArithmetic<T>
    where T : IFloatingPoint<T>
{
    // 10^0 .. 10^28 and 10^-0 .. 10^-28, each exactly representable as decimal.
    private static readonly decimal[] s_decimalPowers = BuildDecimalPowers(negative: false);
    private static readonly decimal[] s_decimalInversePowers = BuildDecimalPowers(negative: true);

    /// <summary>
    /// Exponent <c>e</c> of the type's radix such that every <c>x . radix^-e</c> of the span lies in
    /// <c>[-1, 1]</c> and the largest magnitude lies in <c>[0.5, 1)</c> (binary IEEE) or <c>[0.1, 1)</c>
    /// (<see cref="decimal"/>). An all-zero or empty span returns 0.
    /// </summary>
    internal static int ScaleExponent(ReadOnlySpan<T> values)
    {
        var largest = T.Zero;
        foreach (var value in values)
        {
            largest = T.Max(largest, T.Abs(value));
        }

        return largest == T.Zero ? 0 : ExponentOfMagnitude(largest);
    }

    /// <summary>
    /// <c>Σ (x_i . radix^-e)^2</c> in <typeparamref name="T"/>. Restore with <c>Rescale(result, 2 * e)</c>.
    /// </summary>
    internal static T SumOfSquares(ReadOnlySpan<T> values, int exponent)
    {
        var sum = T.Zero;
        foreach (var value in values)
        {
            var scaled = Rescale(value, -exponent);
            sum += scaled * scaled;
        }

        return sum;
    }

    /// <summary>
    /// <c>Σ (a_i . radix^-ea) (b_i . radix^-eb)</c> in <typeparamref name="T"/>. Restore with
    /// <c>Rescale(result, ea + eb)</c>.
    /// </summary>
    internal static T Dot(ReadOnlySpan<T> left, ReadOnlySpan<T> right, int leftExponent, int rightExponent)
    {
        if (left.Length != right.Length)
        {
            throw new ArgumentException("Spans must have the same length.", nameof(right));
        }

        var sum = T.Zero;
        for (var i = 0; i < left.Length; i++)
        {
            sum += Rescale(left[i], -leftExponent) * Rescale(right[i], -rightExponent);
        }

        return sum;
    }

    /// <summary>
    /// Euclidean norm, formed on the scaled span and restored once (LAPACK <c>dnrm2</c> semantics for
    /// <see cref="double"/>): finite whenever the norm itself is representable, including spans whose raw sum
    /// of squares overflows or underflows. An unrepresentable norm follows the type's convention
    /// (<c>+∞</c> for IEEE types, <see cref="OverflowException"/> for <see cref="decimal"/>).
    /// </summary>
    internal static T Norm2(ReadOnlySpan<T> values)
    {
        var exponent = ScaleExponent(values);
        var root = NumericPrecision<T>.Sqrt(SumOfSquares(values, exponent));
        return Rescale(root, exponent);
    }

    /// <summary>
    /// Two-pass centred sum of squares <c>Σ (x_i − mean)^2</c>, returned scaled together with its exponent:
    /// the sum equals <c>Rescale(ScaledSum, 2 * Exponent)</c>. Never forms <c>Σ x^2 − n . mean^2</c>.
    /// </summary>
    internal static (T ScaledSum, int Exponent) CentredSumOfSquares(ReadOnlySpan<T> values, T mean)
    {
        var largest = T.Zero;
        foreach (var value in values)
        {
            largest = T.Max(largest, T.Abs(value - mean));
        }

        if (largest == T.Zero)
        {
            return (T.Zero, 0);
        }

        var exponent = ExponentOfMagnitude(largest);
        var sum = T.Zero;
        foreach (var value in values)
        {
            var scaled = Rescale(value - mean, -exponent);
            sum += scaled * scaled;
        }

        return (sum, exponent);
    }

    /// <summary>
    /// <c>value . radix^exponent</c>. An unrepresentable result follows the type's convention (<c>±∞</c> for
    /// IEEE types, <see cref="OverflowException"/> for <see cref="decimal"/>); a result below the type's
    /// resolution rounds toward zero.
    /// </summary>
    internal static T Rescale(T value, int exponent)
    {
        if (exponent == 0)
        {
            return value;
        }

        if (typeof(T) == typeof(double))
        {
            return (T)(object)Math.ScaleB((double)(object)value, exponent);
        }

        if (typeof(T) == typeof(float))
        {
            return (T)(object)MathF.ScaleB((float)(object)value, exponent);
        }

        if (typeof(T) == typeof(Half))
        {
            return (T)(object)Half.ScaleB((Half)(object)value, exponent);
        }

        if (typeof(T) == typeof(decimal))
        {
            return (T)(object)ScaleDecimal((decimal)(object)value, exponent);
        }

        throw new NotSupportedException($"ScaledArithmetic has no radix scaling for {typeof(T)}; exponent must be 0.");
    }

    /// <summary>
    /// Forms <c>first * second * third / divisor * radix^exponent</c> without an overflowing
    /// or vanishing intermediate. The optional exponent keeps a geometric weight in scaled form.
    /// </summary>
    internal static T ProductQuotient(T first, T second, T third, T divisor, long exponent = 0)
    {
        if (first == T.Zero || second == T.Zero || third == T.Zero)
        {
            return T.Zero;
        }
        var e1 = ExponentOfMagnitude(T.Abs(first));
        var e2 = ExponentOfMagnitude(T.Abs(second));
        var e3 = ExponentOfMagnitude(T.Abs(third));
        var ed = ExponentOfMagnitude(T.Abs(divisor));
        var mantissa = Rescale(first, -e1) * Rescale(second, -e2)
            * Rescale(third, -e3) / Rescale(divisor, -ed);
        return Rescale(mantissa, int.CreateSaturating(exponent + e1 + e2 + e3 - ed));
    }

    /// <summary>
    /// Multiplies a radix-scaled value by a finite factor, retaining its exponent separately so
    /// a tiny geometric weight is not rounded away before a later product can amplify it.
    /// </summary>
    internal static (T Significand, long Exponent) MultiplyScaled(T significand, long exponent, T factor)
    {
        if (significand == T.Zero || factor == T.Zero)
        {
            return (T.Zero, 0);
        }

        var factorExponent = ExponentOfMagnitude(T.Abs(factor));
        var product = significand * Rescale(factor, -factorExponent);
        var productExponent = ExponentOfMagnitude(T.Abs(product));
        return (Rescale(product, -productExponent), exponent + factorExponent + productExponent);
    }

    /// <summary>Divides a radix-scaled value without narrowing the physical quotient.</summary>
    internal static (T Significand, long Exponent) DivideScaled(T significand, long exponent, T divisor)
    {
        if (significand == T.Zero)
        {
            return (T.Zero, 0);
        }
        var divisorExponent = ExponentOfMagnitude(T.Abs(divisor));
        var quotient = significand / Rescale(divisor, -divisorExponent);
        var shift = ExponentOfMagnitude(T.Abs(quotient));
        return (Rescale(quotient, -shift), exponent - divisorExponent + shift);
    }

    /// <summary>Four-factor product retaining its radix exponent separately.</summary>
    internal static (T Significand, long Exponent) ProductScaled(T first, T second, T third, T fourth)
    {
        var (significand, exponent) = MultiplyScaled(T.One, 0, first);
        (significand, exponent) = MultiplyScaled(significand, exponent, second);
        (significand, exponent) = MultiplyScaled(significand, exponent, third);
        return MultiplyScaled(significand, exponent, fourth);
    }

    /// <summary>Adds a radix-scaled term without materializing an unrepresentable physical value.</summary>
    internal static void AddScaled(ref T sum, ref long exponent, T term, long termExponent)
    {
        if (term == T.Zero)
        {
            return;
        }

        var shift = ExponentOfMagnitude(T.Abs(term));
        term = Rescale(term, -shift);
        termExponent += shift;
        var common = sum == T.Zero ? termExponent : Math.Max(exponent, termExponent);
        sum = Rescale(sum, int.CreateSaturating(exponent - common))
            + Rescale(term, int.CreateSaturating(termExponent - common));
        shift = sum == T.Zero ? 0 : ExponentOfMagnitude(T.Abs(sum));
        sum = Rescale(sum, -shift);
        exponent = common + shift;
    }

    /// <summary>Clipped positive ratio retaining its exponent through a later weighted blend.</summary>
    internal static (T Significand, long Exponent) UnitRatioScaled(
        T numerator, long numeratorExponent, T denominator, long denominatorExponent, T divisor)
    {
        if (numerator <= T.Zero)
        {
            return (T.Zero, 0);
        }

        (denominator, denominatorExponent) = MultiplyScaled(denominator, denominatorExponent, divisor);
        if (numeratorExponent > denominatorExponent
            || (numeratorExponent == denominatorExponent && numerator >= denominator))
        {
            return (T.One, 0);
        }

        var ratio = numerator / denominator;
        var shift = ExponentOfMagnitude(ratio);
        return (Rescale(ratio, -shift), numeratorExponent - denominatorExponent + shift);
    }

    /// <summary>Convex interpolation retaining both products until their sum is rounded.</summary>
    internal static T BlendScaled(T first, T second, T weight, long exponent)
    {
        if (weight == T.Zero || first == second)
        {
            return first;
        }
        if (weight == T.One && exponent == 0)
        {
            return second;
        }

        var fraction = Rescale(weight, int.CreateSaturating(exponent));
        var left = ProductScaled(first, T.One - fraction, T.One, T.One);
        var right = ProductScaled(second, weight, T.One, T.One);
        var sum = T.Zero;
        long sumExponent = 0;
        AddScaled(ref sum, ref sumExponent, left.Significand, left.Exponent);
        AddScaled(ref sum, ref sumExponent, right.Significand, right.Exponent + exponent);
        return ProductQuotient(sum, T.One, T.One, T.One, sumExponent);
    }

    /// <summary>
    /// <see cref="Rescale"/> that reports an unrepresentable (overflowing) result as <see langword="false"/>
    /// instead of <c>±∞</c> or <see cref="OverflowException"/>, so the caller applies its documented exception
    /// contract once. A finite input always gives <see langword="true"/> when the restored value fits.
    /// </summary>
    internal static bool TryRescale(T value, int exponent, out T result)
    {
        if (typeof(T) == typeof(decimal))
        {
            try
            {
                result = Rescale(value, exponent);
                return true;
            }
            catch (OverflowException)
            {
                result = T.Zero;
                return false;
            }
        }

        result = Rescale(value, exponent);
        if (T.IsFinite(result))
        {
            return true;
        }

        result = T.Zero;
        return false;
    }

    // Exponent of a positive finite magnitude: [0.5, 1) after binary scaling, [0.1, 1) after decimal scaling.
    internal static int ExponentOfMagnitude(T magnitude)
    {
        if (typeof(T) == typeof(double))
        {
            return Math.ILogB((double)(object)magnitude) + 1;
        }

        if (typeof(T) == typeof(float))
        {
            return MathF.ILogB((float)(object)magnitude) + 1;
        }

        if (typeof(T) == typeof(Half))
        {
            return Half.ILogB((Half)(object)magnitude) + 1;
        }

        if (typeof(T) == typeof(decimal))
        {
            return DecimalExponent((decimal)(object)magnitude);
        }

        return 0;
    }

    // For m = mantissa . 10^-scale with d digits in the mantissa, 10^(d - 1 - scale) <= m < 10^(d - scale).
    private static int DecimalExponent(decimal magnitude)
    {
        Span<int> bits = stackalloc int[4];
        decimal.GetBits(magnitude, bits);
        var scale = (bits[3] >> 16) & 0xFF;
        var mantissa = ((UInt128)(uint)bits[2] << 64) | ((UInt128)(uint)bits[1] << 32) | (uint)bits[0];
        var digits = 0;
        while (mantissa != 0)
        {
            mantissa /= 10;
            digits++;
        }

        return digits - scale;
    }

    private static decimal ScaleDecimal(decimal value, int exponent)
    {
        var powers = exponent > 0 ? s_decimalPowers : s_decimalInversePowers;
        var remaining = Math.Abs((long)exponent);
        while (remaining > 0 && value != 0m)
        {
            var step = (int)Math.Min(remaining, 28);
            value *= powers[step];
            remaining -= step;
        }

        return value;
    }

    private static decimal[] BuildDecimalPowers(bool negative)
    {
        var powers = new decimal[29];
        for (var k = 0; k <= 28; k++)
        {
            powers[k] = negative ? new decimal(1, 0, 0, false, (byte)k) : Pow10(k);
        }

        return powers;

        static decimal Pow10(int k)
        {
            var result = 1m;
            for (var i = 0; i < k; i++)
            {
                result *= 10m;
            }

            return result;
        }
    }
}
