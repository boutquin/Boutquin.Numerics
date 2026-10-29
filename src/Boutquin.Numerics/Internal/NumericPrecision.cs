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
using System.Runtime.CompilerServices;

namespace Boutquin.Numerics.Internal;

/// <summary>
/// Compile-time-dispatched numeric helpers that bridge the gap between
/// <see cref="IFloatingPoint{TSelf}"/> (which <see cref="decimal"/> implements)
/// and <see cref="IRootFunctions{TSelf}"/> (which <see cref="decimal"/> does not).
/// </summary>
/// <remarks>
/// <para>
/// <b>Purpose:</b> Tier A+Sqrt algorithms (Cholesky, OLS QR, Jacobi eigen, monotone cubic)
/// need <c>Sqrt</c> but should accept any <c>T : IFloatingPoint&lt;T&gt;</c> — including
/// <c>decimal</c>, which lacks <see cref="IRootFunctions{TSelf}"/>. This dispatcher
/// provides <see cref="Sqrt"/> by branching on <c>typeof(T)</c> at JIT time:
/// </para>
/// <list type="bullet">
///   <item><description>
///     <c>T = decimal</c> — uses Newton-Raphson iteration seeded from the
///     <c>double</c> approximation and refined in <c>decimal</c> arithmetic.
///   </description></item>
///   <item><description>
///     <c>T = double</c> — delegates to <see cref="Math.Sqrt(double)"/>,
///     which is the hardware <c>FSQRT</c> / <c>SQRTSD</c> instruction,
///     bit-identical to <c>double.Sqrt</c>.
///   </description></item>
///   <item><description>
///     <c>T = float</c> — delegates to <see cref="MathF.Sqrt(float)"/>,
///     which is the hardware <c>SQRTSS</c> instruction,
///     bit-identical to <c>float.Sqrt</c>.
///   </description></item>
///   <item><description>
///     <c>T = Half</c> — converts to <c>float</c>, applies <see cref="MathF.Sqrt(float)"/>,
///     and converts back, matching <c>Half.Sqrt</c> semantics (Half has no hardware sqrt;
///     the BCL promotes to float internally).
///   </description></item>
/// </list>
/// <para>
/// <b>Implementation:</b> The type dispatch is a library implementation choice.
/// Known IEEE types use the corresponding runtime square-root operation;
/// the decimal branch uses Newton iteration.
/// </para>
/// <para>
/// <b>Contract:</b> For known IEEE types (<c>double</c>, <c>float</c>, <c>Half</c>),
/// the result is bit-identical to the type's native <c>Sqrt</c>. For <c>decimal</c>,
/// the result uses Newton iteration with an absolute stopping threshold of <c>1e-28</c>;
/// correct rounding is not guaranteed. For unknown <c>T</c> implementing
/// <see cref="IFloatingPoint{TSelf}"/>, the fallback round-trips through
/// <c>double</c>, which is correct to <c>double</c> precision (~15 digits).
/// </para>
/// </remarks>
internal static class NumericPrecision<T>
    where T : IFloatingPoint<T>
{
    private static readonly T s_lnTwo = T.CreateChecked(0.6931471805599453094172321215m);

    /// <summary>
    /// Computes the square root of <paramref name="value"/> for any
    /// <typeparamref name="T"/> that implements <see cref="IFloatingPoint{TSelf}"/>.
    /// </summary>
    /// <param name="value">Non-negative value.</param>
    /// <returns>The square root of <paramref name="value"/>.</returns>
    /// <exception cref="ArgumentException">
    /// <paramref name="value"/> is negative when <c>T = decimal</c>.
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static T Sqrt(T value)
    {
        // Per-type dispatch using typeof(T) checks. The JIT treats these as
        // compile-time constants and eliminates all non-matching branches,
        // so only one path survives per T instantiation — zero branch overhead
        // at steady state.

        if (typeof(T) == typeof(decimal))
        {
            // Tier A+Sqrt decimal path: Newton-Raphson iteration in decimal.
            return (T)(object)SqrtDecimal((decimal)(object)value);
        }

        if (typeof(T) == typeof(double))
        {
            // Hardware SQRTSD — bit-identical to double.Sqrt / Math.Sqrt.
            return (T)(object)Math.Sqrt((double)(object)value);
        }

        if (typeof(T) == typeof(float))
        {
            // Hardware SQRTSS — bit-identical to float.Sqrt / MathF.Sqrt.
            return (T)(object)MathF.Sqrt((float)(object)value);
        }

        if (typeof(T) == typeof(Half))
        {
            // Half has no native hardware sqrt; the BCL promotes to float internally.
            return (T)(object)(Half)MathF.Sqrt((float)(Half)(object)value);
        }

        // Fallback for unknown IFloatingPoint<T> implementations (e.g., future
        // BigFloat or third-party types). Round-trips through double — correct
        // to double precision, which is the best we can do without IRootFunctions<T>.
        return T.CreateChecked(Math.Sqrt(double.CreateChecked(value)));
    }

    /// <summary>
    /// Computes the natural logarithm of a positive integer using range reduction
    /// and the atanh series, entirely in <typeparamref name="T"/>.
    /// </summary>
    internal static T LogPositiveInteger(int value)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value);

        var exponent = BitOperations.Log2((uint)value);
        var scale = 1u << exponent;
        var mantissa = T.CreateChecked(value) / T.CreateChecked(scale);
        var z = (mantissa - T.One) / (mantissa + T.One);
        var zSquared = z * z;
        var term = z;
        var sum = term;

        for (var k = 1; k < 256; k++)
        {
            term *= zSquared;
            var next = sum + term / T.CreateChecked(2 * k + 1);
            if (next == sum)
            {
                break;
            }

            sum = next;
        }

        return T.CreateChecked(2) * sum + T.CreateChecked(exponent) * s_lnTwo;
    }

    /// <summary>
    /// Computes <c>value^(-numerator/denominator)</c> for positive integer inputs,
    /// using a Newton integer-root iteration and integer powers in <typeparamref name="T"/>.
    /// </summary>
    internal static T NegativeRationalPowerOfPositiveInteger(
        int value,
        int numerator,
        int denominator)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value);
        ArgumentOutOfRangeException.ThrowIfNegative(numerator);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(denominator);

        if (numerator == 0)
        {
            return T.One;
        }

        var root = IntegerRoot(value, denominator);
        return T.One / IntegerPower(root, numerator);
    }

    private static T IntegerRoot(int value, int degree)
    {
        if (degree == 1 || value == 1)
        {
            return T.CreateChecked(value);
        }

        // All current callers use an observation count (Int32) and a 20th root;
        // the exact integer upper bound is therefore tiny (at most 3).
        var upper = 2;
        while (IntegerPowerBelow(upper, degree, value))
        {
            upper++;
        }

        var guess = T.CreateChecked(upper);
        var valueT = T.CreateChecked(value);
        var degreeT = T.CreateChecked(degree);
        var degreeMinusOneT = T.CreateChecked(degree - 1);
        var previous = T.Zero;
        for (var i = 0; i < 128; i++)
        {
            var next = (degreeMinusOneT * guess + valueT / IntegerPower(guess, degree - 1)) / degreeT;
            if (next == guess || next == previous)
            {
                return next;
            }

            previous = guess;
            guess = next;
        }

        return guess;
    }

    private static bool IntegerPowerBelow(int value, int exponent, int target)
    {
        ulong product = 1;
        var factor = (ulong)value;
        var targetValue = (ulong)target;
        for (var i = 0; i < exponent; i++)
        {
            if (product >= targetValue || product > ulong.MaxValue / factor)
            {
                return false;
            }

            product *= factor;
        }

        return product < targetValue;
    }

    private static T IntegerPower(T value, int exponent)
    {
        var result = T.One;
        var factor = value;
        var remaining = exponent;
        while (remaining > 0)
        {
            if ((remaining & 1) != 0)
            {
                result *= factor;
            }

            remaining >>= 1;
            if (remaining > 0)
            {
                factor *= factor;
            }
        }

        return result;
    }

    private static decimal SqrtDecimal(decimal value)
    {
        if (value < 0m)
        {
            throw new ArgumentException("Cannot compute square root of a negative number.", nameof(value));
        }

        if (value == 0m)
        {
            return 0m;
        }

        var guess = (decimal)Math.Sqrt((double)value);
        if (guess == 0m)
        {
            guess = 1m;
        }

        for (var i = 0; i < 30; i++)
        {
            var next = (guess + value / guess) * 0.5m;
            if (Math.Abs(next - guess) < 1e-28m)
            {
                return next;
            }

            guess = next;
        }

        return guess;
    }
}
