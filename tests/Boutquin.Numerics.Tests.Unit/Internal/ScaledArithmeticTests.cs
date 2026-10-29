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

using Boutquin.Numerics.Internal;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Internal;

/// <summary>
/// Range behaviour of <see cref="ScaledArithmetic{T}"/>: norms and sums of squares stay finite and correct where
/// the raw sum of squares overflows or underflows, decimal reductions stay finite where the raw sum throws, and
/// <see cref="ScaledArithmetic{T}.TryRescale"/> reports exactly the unrepresentable restores.
/// </summary>
public sealed class ScaledArithmeticTests
{
    // Integer components times 2^shift: the exact norm is sqrt(Σ k_i^2) . 2^shift, and the integer sum of
    // squares is exact in double, so the correctly rounded norm is Math.ScaleB(Math.Sqrt(sum), shift).
    public static TheoryData<long[], int> IntegerVectors => new()
    {
        { new long[] { 3, 4 }, 0 },
        { new long[] { 3, 4 }, 600 },     // raw squares overflow (2^1200)
        { new long[] { 3, 4 }, 1000 },    // raw squares overflow; norm 5 . 2^1000 is finite
        { new long[] { 3, 4 }, -600 },    // raw squares underflow to zero
        { new long[] { 3, 4 }, -1070 },   // subnormal components
        { new long[] { 1, 1, 1, 1 }, 1020 },
        { new long[] { 1, 2, 2 }, -540 },
        { new long[] { 7, -3, 11, 0, 5, 2 }, 900 },
        { new long[] { 7, -3, 11, 0, 5, 2 }, -900 },
    };

    [Theory]
    [MemberData(nameof(IntegerVectors))]
    public void Norm2_Double_IsCorrectlyRoundedWhereTheRawSumOfSquaresLeavesTheRange(long[] components, int shift)
    {
        var values = new double[components.Length];
        long sumOfSquares = 0;
        for (var i = 0; i < components.Length; i++)
        {
            values[i] = Math.ScaleB(components[i], shift);
            sumOfSquares += components[i] * components[i];
        }

        var expected = Math.ScaleB(Math.Sqrt(sumOfSquares), shift);

        ScaledArithmetic<double>.Norm2(values).Should().Be(expected);
    }

    // Both algorithms are accurate to the summation bound of about (n + 2) units in the last place; the
    // comparison uses that bound, the single-ULP claim is pinned exactly by the integer vectors above.
    [Fact]
    public void Norm2_Double_AgreesWithDnrm2WithinTheSummationBound()
    {
        var random = new System.Random(20261008);
        for (var trial = 0; trial < 500; trial++)
        {
            var length = random.Next(1, 40);
            var shift = random.Next(-1000, 1000);
            var values = new double[length];
            for (var i = 0; i < length; i++)
            {
                values[i] = Math.ScaleB(random.NextDouble() - 0.5, shift);
            }

            var reference = Dnrm2(values);
            var actual = ScaledArithmetic<double>.Norm2(values);
            if (reference == 0.0)
            {
                actual.Should().Be(0.0);
                continue;
            }

            var ulp = Math.BitIncrement(reference) - reference;
            Math.Abs(actual - reference).Should().BeLessThanOrEqualTo(
                (length + 2) * ulp, $"trial {trial}, length {length}, shift {shift}");
        }
    }

    [Fact]
    public void SumOfSquares_Decimal_IsFiniteAtMagnitude1e14WhereTheRawSumThrows()
    {
        var values = new decimal[40];
        var exactScaledSum = 0m;
        for (var i = 0; i < values.Length; i++)
        {
            var digit = (i % 9) + 1;
            values[i] = digit * 1e13m;               // magnitudes 1e13 .. 9e13, inside the decimal domain
            exactScaledSum += (digit / 10m) * (digit / 10m);
        }

        var raw = () =>
        {
            var sum = 0m;
            foreach (var value in values)
            {
                sum += value * value;
            }

            return sum;
        };
        raw.Should().Throw<OverflowException>("the raw sum of squares exceeds decimal.MaxValue");

        var exponent = ScaledArithmetic<decimal>.ScaleExponent(values);
        exponent.Should().Be(14);
        ScaledArithmetic<decimal>.SumOfSquares(values, exponent).Should().Be(exactScaledSum);
        ScaledArithmetic<decimal>.Norm2(values).Should().Be(
            NumericPrecision<decimal>.Sqrt(exactScaledSum) * 1e14m);
    }

    [Fact]
    public void Dot_Decimal_ScalesEachSpanByItsOwnExponent()
    {
        decimal[] left = [2e-14m, -3e-14m, 5e-14m];
        decimal[] right = [4e13m, 1e13m, -2e13m];
        var leftExponent = ScaledArithmetic<decimal>.ScaleExponent(left);
        var rightExponent = ScaledArithmetic<decimal>.ScaleExponent(right);
        leftExponent.Should().Be(-13);
        rightExponent.Should().Be(14);

        var dot = ScaledArithmetic<decimal>.Dot(left, right, leftExponent, rightExponent);
        ScaledArithmetic<decimal>.Rescale(dot, leftExponent + rightExponent).Should().Be(-0.5m);
    }

    [Fact]
    public void CentredSumOfSquares_Double_KeepsDeviationsWhoseRawSquaresUnderflow()
    {
        double[] values = [1e-170, 3e-170, 2e-170];
        var (scaledSum, exponent) = ScaledArithmetic<double>.CentredSumOfSquares(values, 2e-170);
        var root = ScaledArithmetic<double>.Rescale(Math.Sqrt(scaledSum), exponent);

        var relative = Math.Abs(root - (Math.Sqrt(2) * 1e-170)) / (Math.Sqrt(2) * 1e-170);
        relative.Should().BeLessThan(1e-15);
    }

    [Theory]
    [InlineData(1.0, 1023, true)]
    [InlineData(1.0, 1024, false)]
    [InlineData(0.75, 1024, true)]
    [InlineData(-1.0, 1024, false)]
    [InlineData(1.0, -1074, true)]
    [InlineData(1.0, -1100, true)]
    public void TryRescale_Double_FailsExactlyWhenTheRestoredValueOverflows(double value, int exponent, bool representable)
    {
        ScaledArithmetic<double>.TryRescale(value, exponent, out var result).Should().Be(representable);
        if (representable)
        {
            result.Should().Be(Math.ScaleB(value, exponent));
        }
    }

    [Theory]
    [InlineData("7.9", 28, true, "79000000000000000000000000000")]
    [InlineData("8", 28, false, "0")]
    [InlineData("-8", 28, false, "0")]
    [InlineData("0.79", 29, true, "79000000000000000000000000000")]
    [InlineData("1", 29, false, "0")]
    [InlineData("1", -28, true, "0.0000000000000000000000000001")]
    [InlineData("1", -40, true, "0")]
    public void TryRescale_Decimal_FailsExactlyWhenTheRestoredValueOverflows(
        string text, int exponent, bool representable, string expectedText)
    {
        var value = decimal.Parse(text, System.Globalization.CultureInfo.InvariantCulture);
        var expected = decimal.Parse(expectedText, System.Globalization.CultureInfo.InvariantCulture);
        var attempt = () => ScaledArithmetic<decimal>.TryRescale(value, exponent, out _);
        attempt.Should().NotThrow();
        ScaledArithmetic<decimal>.TryRescale(value, exponent, out var result).Should().Be(representable);
        result.Should().Be(expected);
    }

    // LAPACK 3.x dnrm2: one pass with a running scale and scaled sum of squares (dlassq update).
    private static double Dnrm2(double[] x)
    {
        var scale = 0.0;
        var ssq = 1.0;
        foreach (var value in x)
        {
            if (value == 0.0)
            {
                continue;
            }

            var absolute = Math.Abs(value);
            if (scale < absolute)
            {
                ssq = 1.0 + ssq * (scale / absolute) * (scale / absolute);
                scale = absolute;
            }
            else
            {
                ssq += (absolute / scale) * (absolute / scale);
            }
        }

        return scale * Math.Sqrt(ssq);
    }
}
