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
///     <c>double</c> approximation and polished to full 28-digit precision.
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
/// <b>Authority:</b> The <c>typeof(T) == typeof(decimal)</c> dispatch pattern follows
/// the precedent set by <c>System.Numerics.Tensors.TensorPrimitives</c> in
/// <c>dotnet/runtime#74055</c>, where compile-time-constant-typeof checks are used
/// to specialise generic numeric kernels.
/// </para>
/// <para>
/// <b>Contract:</b> For known IEEE types (<c>double</c>, <c>float</c>, <c>Half</c>),
/// the result is bit-identical to the type's native <c>Sqrt</c>. For <c>decimal</c>,
/// the result agrees with the internal Newton-Raphson decimal sqrt — full 28-digit
/// precision via Newton iteration. For unknown <c>T</c> implementing
/// <see cref="IFloatingPoint{TSelf}"/>, the fallback round-trips through
/// <c>double</c>, which is correct to <c>double</c> precision (~15 digits).
/// </para>
/// </remarks>
internal static class NumericPrecision<T>
    where T : IFloatingPoint<T>
{
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
            // Tier A+Sqrt decimal path: full 28-digit Newton-Raphson iteration.
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
