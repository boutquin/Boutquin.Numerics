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
/// Exact diagonal equilibration of the covariance of the budget-constrained box QP solved by
/// <c>ActiveSetQpSolver&lt;T&gt;</c>: <c>D = diag(√Σ_ii)</c> rounded to powers of the type's radix and the working
/// covariance <c>Σ̃ = D⁻¹ΣD⁻¹</c> (unit-scale diagonal). The scaling is of the covariance alone: the active-set core reads
/// only the exponents and <c>Σ̃</c>, uses the bounds and the means in the original units, and restores no weight.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt (one <see cref="NumericPrecision{T}.Sqrt"/> per diagonal entry, used only to
/// choose an exponent). Supported for <see cref="double"/> and <see cref="decimal"/>; <see cref="float"/> is
/// best-effort; <see cref="Half"/> is rejected with <see cref="NotSupportedException"/>.
/// </para>
/// <para>
/// <b>Exactness.</b> Each factor is <c>D_i = radix^e_i</c>, with <c>e_i</c> from
/// <see cref="ScaledArithmetic{T}.ScaleExponent"/> applied to <c>√Σ_ii</c> (so <c>Σ̃_ii</c> lies in
/// <c>[1/4, 1)</c> for binary types and <c>[1/100, 1)</c> for <see cref="decimal"/>). Every entry of <c>Σ̃</c> is a product
/// by a radix power and is formed without rounding: an exponent is moved toward zero, one step at a time, wherever an
/// entry would round (a subnormal or overflowing binary result, a decimal result needing more than 28 decimal places or
/// overflowing). Exponent 0 is always exact, so <c>D·Σ̃·D</c> equals <c>Σ</c> as a value; at <see cref="double"/> it
/// reproduces <c>Σ</c> bit for bit, while at <see cref="decimal"/> a shift beyond 28 places may drop trailing zeros of the
/// mantissa (the value is unchanged).
/// </para>
/// <para>
/// <b>Qualification.</b> The solver's conditioning guard measures <c>κ_eq</c> on <c>Σ̃</c>, and the corpus-membership
/// definition, the corpus generator and the test oracles take <c>κ_eq</c> after this same scaling, so qualification and
/// the guard's refusal judge one block.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type of the problem.</typeparam>
internal static class QpScaling<T>
    where T : IFloatingPoint<T>
{
    /// <summary>
    /// Equilibrates the covariance: the radix exponents <c>e_i</c> of <c>√Σ_ii</c>, limited only where an entry of
    /// <c>Σ̃ = D⁻¹ΣD⁻¹</c> would round, and <c>Σ̃</c>. Non-positive or non-finite diagonal entries keep factor 1.
    /// </summary>
    /// <param name="covariance">Square covariance <c>Σ</c>.</param>
    /// <returns>The radix exponents <c>e_i</c> and the exactly scaled covariance.</returns>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is <see cref="Half"/> or has no radix scaling.</exception>
    /// <exception cref="ArgumentException">The covariance is not square.</exception>
    internal static (int[] Exponents, T[,] Covariance) EquilibrateCovariance(T[,] covariance)
    {
        var exponents = InitialExponents(covariance);
        LimitForExactness(covariance, exponents);
        return (exponents, ScaleCovariance(covariance, exponents));
    }

    // Validates the type and the shape and returns e_i = ScaleExponent(√Σ_ii) (0 for a non-positive or non-finite Σ_ii).
    private static int[] InitialExponents(T[,] covariance)
    {
        ArgumentNullException.ThrowIfNull(covariance);
        if (typeof(T) != typeof(double) && typeof(T) != typeof(float) && typeof(T) != typeof(decimal))
        {
            throw new NotSupportedException($"QpScaling supports double, float and decimal, not {typeof(T)}.");
        }

        var n = covariance.GetLength(0);
        if (covariance.GetLength(1) != n)
        {
            throw new ArgumentException("The covariance must be square.", nameof(covariance));
        }

        var exponents = new int[n];
        for (var i = 0; i < n; i++)
        {
            var diagonal = covariance[i, i];
            if (T.IsFinite(diagonal) && diagonal > T.Zero)
            {
                exponents[i] = ScaledArithmetic<T>.ScaleExponent([NumericPrecision<T>.Sqrt(diagonal)]);
            }
        }

        return exponents;
    }

    private static T[,] ScaleCovariance(T[,] covariance, int[] exponents)
    {
        var n = exponents.Length;
        var scaled = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                scaled[i, j] = Shift(covariance[i, j], -(exponents[i] + exponents[j])).Value;
            }
        }

        return scaled;
    }

    // Moves exponents toward zero until every entry of the scaled covariance is exact. Each step lowers Σ|e_i| by one,
    // and at e = 0 every shift is the identity, so the loop terminates.
    private static void LimitForExactness(T[,] covariance, int[] exponents)
    {
        var n = exponents.Length;
        bool changed;
        do
        {
            changed = false;
            for (var i = 0; i < n; i++)
            {
                for (var j = i; j < n; j++)
                {
                    var k = -(exponents[i] + exponents[j]);
                    if (k != 0 && !(Shift(covariance[i, j], k).Exact && Shift(covariance[j, i], k).Exact))
                    {
                        // k < 0 needs a smaller e_i + e_j; k > 0 a larger one. Move the dominant exponent.
                        var m = k < 0 ? (exponents[i] >= exponents[j] ? i : j) : (exponents[i] <= exponents[j] ? i : j);
                        exponents[m] += k < 0 ? -1 : 1;
                        changed = true;
                    }
                }
            }
        }
        while (changed);
    }

    /// <summary><c>value · radix^k</c> and whether it is formed without rounding.</summary>
    private static (T Value, bool Exact) Shift(T value, int k)
    {
        if (k == 0)
        {
            return (value, true);
        }

        if (typeof(T) == typeof(decimal))
        {
            return ShiftDecimal((decimal)(object)value, k);
        }

        var shifted = ScaledArithmetic<T>.Rescale(value, k);
        return (shifted, T.IsFinite(shifted) && BitsEqual(ScaledArithmetic<T>.Rescale(shifted, -k), value));
    }

    // A decimal times 10^k: a pure scale-byte move when 0 <= scale - k <= 28; past zero places the mantissa grows (exact
    // unless it overflows); past 28 places only trailing zeros may be dropped.
    private static (T Value, bool Exact) ShiftDecimal(decimal value, int k)
    {
        if (value == 0m)
        {
            return ((T)(object)value, true);
        }

        Span<int> bits = stackalloc int[4];
        decimal.GetBits(value, bits);
        var negative = bits[3] < 0;
        var scale = (bits[3] >> 16) & 0xFF;
        var target = scale - k;
        if (target is >= 0 and <= 28)
        {
            return ((T)(object)new decimal(bits[0], bits[1], bits[2], negative, (byte)target), true);
        }

        if (target < 0)
        {
            return ScaledArithmetic<T>.TryRescale((T)(object)value, k, out var grown) ? (grown, true) : (T.Zero, false);
        }

        var mantissa = ((UInt128)(uint)bits[2] << 64) | ((UInt128)(uint)bits[1] << 32) | (uint)bits[0];
        while (target > 28 && mantissa % 10 == 0)
        {
            mantissa /= 10;
            target--;
        }

        if (target > 28)
        {
            return (T.Zero, false);
        }

        var shifted = new decimal((int)(uint)mantissa, (int)(uint)(mantissa >> 32), (int)(uint)(mantissa >> 64), negative, (byte)target);
        return ((T)(object)shifted, true);
    }

    private static bool BitsEqual(T a, T b) =>
        typeof(T) == typeof(double)
            ? BitConverter.DoubleToInt64Bits((double)(object)a) == BitConverter.DoubleToInt64Bits((double)(object)b)
            : BitConverter.SingleToInt32Bits((float)(object)a) == BitConverter.SingleToInt32Bits((float)(object)b);
}
