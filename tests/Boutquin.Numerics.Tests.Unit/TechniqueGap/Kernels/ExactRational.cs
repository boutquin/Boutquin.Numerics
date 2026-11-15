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

namespace Boutquin.Numerics.Tests.Unit.TechniqueGap.Kernels;

/// <summary>
/// Exact rational arithmetic for test oracles. Every finite <see cref="double"/>,
/// <see cref="float"/> and <see cref="decimal"/> converts without rounding, so a reference
/// computed here is the exact answer to the stored problem.
/// </summary>
internal readonly struct ExactRational : IComparable<ExactRational>
{
    private ExactRational(BigInteger numerator, BigInteger denominator)
    {
        if (denominator.Sign < 0)
        {
            numerator = -numerator;
            denominator = -denominator;
        }

        var divisor = BigInteger.GreatestCommonDivisor(numerator, denominator);
        Numerator = divisor.IsOne || divisor.IsZero ? numerator : numerator / divisor;
        Denominator = divisor.IsOne || divisor.IsZero ? denominator : denominator / divisor;
    }

    public static ExactRational Zero => new(BigInteger.Zero, BigInteger.One);

    public BigInteger Numerator { get; }

    public BigInteger Denominator { get; }

    public static ExactRational From(long value) => new(value, BigInteger.One);

    public static ExactRational From(double value)
    {
        if (!double.IsFinite(value))
        {
            throw new ArgumentOutOfRangeException(nameof(value), value, "Only finite values convert exactly.");
        }

        var bits = BitConverter.DoubleToInt64Bits(value);
        var biased = (int)((bits >> 52) & 0x7FF);
        var fraction = bits & 0xFFFFFFFFFFFFFL;
        var mantissa = biased == 0 ? fraction : fraction | (1L << 52);
        var exponent = (biased == 0 ? 1 : biased) - 1075;
        var magnitude = exponent >= 0
            ? new ExactRational(new BigInteger(mantissa) << exponent, BigInteger.One)
            : new ExactRational(mantissa, BigInteger.One << -exponent);
        return bits < 0 ? -magnitude : magnitude;
    }

    public static ExactRational From(decimal value)
    {
        Span<int> parts = stackalloc int[4];
        decimal.GetBits(value, parts);
        var mantissa = ((BigInteger)(uint)parts[2] << 64) | ((BigInteger)(uint)parts[1] << 32) | (uint)parts[0];
        var scale = (parts[3] >> 16) & 0xFF;
        var magnitude = new ExactRational(mantissa, BigInteger.Pow(10, scale));
        return parts[3] < 0 ? -magnitude : magnitude;
    }

    public static ExactRational From<T>(T value)
        where T : IFloatingPoint<T> => value switch
        {
            double d => From(d),
            float f => From((double)f),
            decimal m => From(m),
            _ => throw new NotSupportedException(typeof(T).Name),
        };

    public static ExactRational operator +(ExactRational left, ExactRational right) =>
        new((left.Numerator * right.Denominator) + (right.Numerator * left.Denominator), left.Denominator * right.Denominator);

    public static ExactRational operator -(ExactRational left, ExactRational right) => left + (-right);

    public static ExactRational operator -(ExactRational value) => new(-value.Numerator, value.Denominator);

    public static ExactRational operator *(ExactRational left, ExactRational right) =>
        new(left.Numerator * right.Numerator, left.Denominator * right.Denominator);

    public static ExactRational operator /(ExactRational left, ExactRational right) =>
        new(left.Numerator * right.Denominator, left.Denominator * right.Numerator);

    public static bool operator <=(ExactRational left, ExactRational right) => left.CompareTo(right) <= 0;

    public static bool operator >=(ExactRational left, ExactRational right) => left.CompareTo(right) >= 0;

    public ExactRational Abs() => new(BigInteger.Abs(Numerator), Denominator);

    public ExactRational Pow(int exponent)
    {
        var result = From(1);
        for (var i = 0; i < exponent; i++)
        {
            result *= this;
        }

        return result;
    }

    public int CompareTo(ExactRational other) => (Numerator * other.Denominator).CompareTo(other.Numerator * Denominator);

    /// <summary>The nearest double, for building inputs and reporting; not used as a reference.</summary>
    public double ToDouble()
    {
        if (Numerator.IsZero)
        {
            return 0;
        }

        // Scale the quotient to about 80 significant bits before the one rounding to double.
        var shift = 80 - (int)(BigInteger.Abs(Numerator).GetBitLength() - Denominator.GetBitLength());
        var scaled = shift >= 0 ? (Numerator << shift) / Denominator : Numerator / (Denominator << -shift);
        return Math.ScaleB((double)scaled, -shift);
    }

    public override string ToString() => ToDouble().ToString("R", System.Globalization.CultureInfo.InvariantCulture);
}
