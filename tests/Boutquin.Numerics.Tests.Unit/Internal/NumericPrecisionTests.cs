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
/// Verifies that <see cref="NumericPrecision{T}.Sqrt"/> produces results
/// that agree with each type's native square root implementation.
/// </summary>
public sealed class NumericPrecisionTests
{
    /// <summary>
    /// Representative test inputs spanning several orders of magnitude,
    /// including zero, small values, and large values.
    /// </summary>
    private static readonly double[] s_testValues =
    [
        0.0,
        0.25,
        1.0,
        2.0,
        3.0,
        4.0,
        9.0,
        16.0,
        100.0,
        12345.6789,
        1e-10,
        1e10,
        1e-20,
        1e20,
        0.5,
        Math.PI,
        Math.E,
    ];

    // ─── double ───────────────────────────────────────────────────────

    [Fact]
    public void Sqrt_Double_AgreesWithMathSqrt()
    {
        foreach (var v in s_testValues)
        {
            var expected = Math.Sqrt(v);
            var actual = NumericPrecision<double>.Sqrt(v);

            actual.Should().Be(expected,
                because: $"NumericPrecision<double>.Sqrt({v}) must be bit-identical to Math.Sqrt({v})");
        }
    }

    [Fact]
    public void Sqrt_Double_SpecialValues()
    {
        NumericPrecision<double>.Sqrt(double.PositiveInfinity)
            .Should().Be(double.PositiveInfinity);

        double.IsNaN(NumericPrecision<double>.Sqrt(double.NaN))
            .Should().BeTrue();

        NumericPrecision<double>.Sqrt(0.0).Should().Be(0.0);
    }

    // ─── float ────────────────────────────────────────────────────────

    [Fact]
    public void Sqrt_Float_AgreesWithMathFSqrt()
    {
        foreach (var v in s_testValues)
        {
            var fv = (float)v;
            var expected = MathF.Sqrt(fv);
            var actual = NumericPrecision<float>.Sqrt(fv);

            actual.Should().Be(expected,
                because: $"NumericPrecision<float>.Sqrt({fv}) must be bit-identical to MathF.Sqrt({fv})");
        }
    }

    [Fact]
    public void Sqrt_Float_SpecialValues()
    {
        NumericPrecision<float>.Sqrt(float.PositiveInfinity)
            .Should().Be(float.PositiveInfinity);

        float.IsNaN(NumericPrecision<float>.Sqrt(float.NaN))
            .Should().BeTrue();

        NumericPrecision<float>.Sqrt(0f).Should().Be(0f);
    }

    // ─── Half ─────────────────────────────────────────────────────────

    [Fact]
    public void Sqrt_Half_AgreesWithHalfSqrt()
    {
        // Half has limited range (~6.5e4 max), so filter test values.
        foreach (var v in s_testValues)
        {
            if (v > 60000.0 || v < 0.0)
            {
                continue;
            }

            var hv = (Half)v;
            var expected = Half.Sqrt(hv);
            var actual = NumericPrecision<Half>.Sqrt(hv);

            actual.Should().Be(expected,
                because: $"NumericPrecision<Half>.Sqrt({hv}) must match Half.Sqrt({hv})");
        }
    }

    [Fact]
    public void Sqrt_Half_SpecialValues()
    {
        NumericPrecision<Half>.Sqrt(Half.PositiveInfinity)
            .Should().Be(Half.PositiveInfinity);

        Half.IsNaN(NumericPrecision<Half>.Sqrt(Half.NaN))
            .Should().BeTrue();

        NumericPrecision<Half>.Sqrt((Half)0f).Should().Be((Half)0f);
    }

    // ─── decimal ──────────────────────────────────────────────────────

    [Fact]
    public void Sqrt_Decimal_ProducesCorrectResults()
    {
        NumericPrecision<decimal>.Sqrt(0m).Should().Be(0m);
        NumericPrecision<decimal>.Sqrt(1m).Should().Be(1m);
        NumericPrecision<decimal>.Sqrt(4m).Should().Be(2m);
        NumericPrecision<decimal>.Sqrt(9m).Should().Be(3m);
        NumericPrecision<decimal>.Sqrt(16m).Should().Be(4m);
        NumericPrecision<decimal>.Sqrt(100m).Should().Be(10m);
        NumericPrecision<decimal>.Sqrt(0.25m).Should().Be(0.5m);

        var sqrt2 = NumericPrecision<decimal>.Sqrt(2m);
        (sqrt2 * sqrt2).Should().BeApproximately(2m, 1e-27m,
            because: "sqrt(2)^2 must be 2 to full decimal precision");
    }

    [Fact]
    public void Sqrt_Decimal_PerfectSquaresExact()
    {
        // Perfect squares should return exact results.
        NumericPrecision<decimal>.Sqrt(0m).Should().Be(0m);
        NumericPrecision<decimal>.Sqrt(1m).Should().Be(1m);
        NumericPrecision<decimal>.Sqrt(4m).Should().Be(2m);
        NumericPrecision<decimal>.Sqrt(9m).Should().Be(3m);
        NumericPrecision<decimal>.Sqrt(16m).Should().Be(4m);
        NumericPrecision<decimal>.Sqrt(100m).Should().Be(10m);
    }

    [Fact]
    public void Sqrt_Decimal_NegativeThrows()
    {
        var act = () => NumericPrecision<decimal>.Sqrt(-1m);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Sqrt_Decimal_HighPrecision()
    {
        // Verify that decimal sqrt achieves better than double precision.
        // sqrt(2) ≈ 1.41421356237309504880168872420969807856967...
        var sqrt2 = NumericPrecision<decimal>.Sqrt(2m);

        // The result squared should be very close to 2 — within decimal's
        // precision envelope (~1e-28).
        var residual = Math.Abs(sqrt2 * sqrt2 - 2m);
        residual.Should().BeLessThan(1e-26m,
            because: "decimal Sqrt should achieve near-28-digit precision");
    }
}
