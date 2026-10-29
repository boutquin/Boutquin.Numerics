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

using Boutquin.Numerics.Interpolation;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Interpolation;

public sealed class MonotoneCubicSplineTests
{
    private static readonly double[] s_xs = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0];
    private static readonly double[] s_ys = [0.04, 0.041, 0.043, 0.045, 0.048, 0.05, 0.052];

    [Fact]
    public void Interpolate_TwoNodesWideFiniteSpan_IsLinear()
    {
        double[] xs = [-1e308, 1e308];
        double[] ys = [1, 4];
        new MonotoneCubicSpline<double>(xs, ys).Interpolate(0).Should().BeApproximately(2.5, 1e-12);
    }

    [Fact]
    public void Interpolate_TwoNodesWideFiniteOrdinates_PreservesMidpoint()
    {
        new MonotoneCubicSpline<double>([0, 1], [-1e308, 1e308])
            .Interpolate(0.5).Should().Be(0);
    }

    [Fact]
    public void Interpolate_ThreeNodesWideFiniteOrdinates_StaysInSegment()
    {
        var spline = new MonotoneCubicSpline<double>([0, 1, 2], [-1e308, 0, 1e308]);
        var result = spline.Interpolate(0.5);

        double.IsFinite(result).Should().BeTrue();
        result.Should().BeInRange(-1e308, 0);
    }

    [Fact]
    public void Interpolate_WideOrdinatesAndNarrowIntervals_StaysFinite()
    {
        var spline = new MonotoneCubicSpline<double>(
            [0, 1e-10, 2e-10], [-1e300, 0, 1e300]);
        var result = spline.Interpolate(0.5e-10);

        double.IsFinite(result).Should().BeTrue();
        result.Should().BeInRange(-1e300, 0);
        (result / -5e299).Should().BeApproximately(1, 1e-14);
    }

    [Fact]
    public void Interpolate_TwoNodesWideDecimalSpan_IsLinear()
    {
        decimal[] xs = [-decimal.MaxValue, decimal.MaxValue];
        decimal[] ys = [1m, 4m];
        decimal result = 0;
        Action interpolate = () => result = new MonotoneCubicSpline<decimal>(xs, ys).Interpolate(0m);

        interpolate.Should().NotThrow();
        result.Should().Be(2.5m);
    }

    [Fact]
    public void Constructor_DecimalSteepFiniteSlopes_DoesNotOverflowSignTests()
    {
        // Slopes of 1e15 are representable in decimal, but their products (1e30) are not.
        // Collinear data is reproduced exactly by every Fritsch-Carlson tangent choice.
        decimal[] xs = [0m, 1e-5m, 2e-5m, 3e-5m];
        decimal[] ys = [1e10m, 2e10m, 3e10m, 4e10m];
        var construct = () => new MonotoneCubicSpline<decimal>(xs, ys);

        construct.Should().NotThrow<OverflowException>();
        construct().Interpolate(1.5e-5m).Should().Be(2.5e10m);
    }

    [Theory]
    [InlineData(0.5, 0.25)]
    [InlineData(2.0, 4.0)]
    [InlineData(2.5, 6.25)]
    public void Interpolate_MonotoneQuadraticOnNonUniformGrid_UsesFritschCarlsonStepOne(double x, double expected)
    {
        // Fritsch & Carlson (1980) §4 Step 1: the three-point interior derivative at x = 1 is
        // (h1·δ0 + h0·δ1)/(h0 + h1) = (2·1 + 1·4)/3 = 2 = f'(1), and the left end value
        // ((2h0 + h1)δ0 − h0δ1)/(h0 + h1) = 0 has the wrong sign, so it becomes 0 = f'(0).
        // No tangent is clipped (φ ≤ 9), so the Hermite interpolant is x² itself.
        new MonotoneCubicSpline<double>([0.0, 1.0, 3.0], [0.0, 1.0, 9.0]).Interpolate(x)
            .Should().BeApproximately(expected, 1e-14);
    }

    [Fact]
    public void Interpolate_AtNodes_ReturnsExactValues()
    {
        var spline = new MonotoneCubicSpline(s_xs, s_ys);
        for (int i = 0; i < s_xs.Length; i++)
        {
            spline.Interpolate(s_xs[i]).Should().BeApproximately(s_ys[i], 1e-10);
        }
    }

    [Fact]
    public void Interpolate_PreservesMonotonicity()
    {
        var spline = new MonotoneCubicSpline(s_xs, s_ys);

        // Sweep 15 interior samples per segment and assert both no-overshoot and that the
        // interpolant is monotone across consecutive samples.
        const int samplesPerSegment = 15;
        for (var i = 0; i < s_xs.Length - 1; i++)
        {
            var yMin = Math.Min(s_ys[i], s_ys[i + 1]);
            var yMax = Math.Max(s_ys[i], s_ys[i + 1]);
            var ascending = s_ys[i + 1] >= s_ys[i];

            var prev = spline.Interpolate(s_xs[i]);
            for (var k = 1; k <= samplesPerSegment; k++)
            {
                var t = k / (double)(samplesPerSegment + 1);
                var x = s_xs[i] + t * (s_xs[i + 1] - s_xs[i]);
                var result = spline.Interpolate(x);

                result.Should().BeGreaterThanOrEqualTo(yMin, $"interval [{s_xs[i]}, {s_xs[i + 1]}] at x={x}");
                result.Should().BeLessThanOrEqualTo(yMax, $"interval [{s_xs[i]}, {s_xs[i + 1]}] at x={x}");

                if (ascending)
                {
                    result.Should().BeGreaterThanOrEqualTo(prev - 1e-12, $"monotone ascending at x={x}");
                }
                else
                {
                    result.Should().BeLessThanOrEqualTo(prev + 1e-12, $"monotone descending at x={x}");
                }

                prev = result;
            }
        }
    }

    [Fact]
    public void Interpolate_MatchesStatelessInterpolator()
    {
        var spline = new MonotoneCubicSpline(s_xs, s_ys);
        var stateless = MonotoneCubicInterpolator.Instance;

        for (var i = 0; i < s_xs.Length - 1; i++)
        {
            var xMid = (s_xs[i] + s_xs[i + 1]) / 2.0;
            var preComputed = spline.Interpolate(xMid);
            var onTheFly = stateless.Interpolate(xMid, s_xs, s_ys);
            preComputed.Should().BeApproximately(onTheFly, 1e-14);
        }
    }

    [Fact]
    public void Interpolate_ExtremeFiniteSlopeTransition_StaysWithinSegment()
    {
        double[] xs = [0, 1, 2, 3];
        double[] ys = [-1e300, 0, 1e-100, 2e-100];
        var spline = new MonotoneCubicSpline<double>(xs, ys);

        var result = spline.Interpolate(1.5);

        double.IsFinite(result).Should().BeTrue();
        result.Should().BeGreaterThanOrEqualTo(ys[1]);
        result.Should().BeLessThanOrEqualTo(ys[2]);
    }

    [Fact]
    public void Interpolate_DecimalSlopeRatioBeyondRange_StaysWithinSegment()
    {
        decimal[] xs = [0m, 1m, 2m, 3m];
        decimal[] ys = [-1e14m, 0m, 1e-20m, 2e-20m];
        decimal result = 0;
        Action interpolate = () => result = new MonotoneCubicSpline<decimal>(xs, ys).Interpolate(1.5m);

        interpolate.Should().NotThrow();
        result.Should().BeGreaterThanOrEqualTo(ys[1]);
        result.Should().BeLessThanOrEqualTo(ys[2]);
    }

    [Fact]
    public void Interpolate_BelowRange_ExtrapolatesFlat()
    {
        var spline = new MonotoneCubicSpline(s_xs, s_ys);
        spline.Interpolate(0.1).Should().BeApproximately(s_ys[0], 1e-10);
    }

    [Fact]
    public void Interpolate_AboveRange_ExtrapolatesFlat()
    {
        var spline = new MonotoneCubicSpline(s_xs, s_ys);
        spline.Interpolate(50.0).Should().BeApproximately(s_ys[^1], 1e-10);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Interpolate_NonFiniteQuery_IsRejected(double query)
    {
        var generic = new MonotoneCubicSpline<double>(s_xs, s_ys);
        var legacy = new MonotoneCubicSpline(s_xs, s_ys);

        Action genericCall = () => generic.Interpolate(query);
        Action legacyCall = () => legacy.Interpolate(query);

        genericCall.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("x");
        legacyCall.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("x");
    }

    [Fact]
    public void Constructor_LessThanTwoPoints_Throws()
    {
        double[] xs = [1.0];
        double[] ys = [0.05];

        var act = () => new MonotoneCubicSpline(xs, ys);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Constructor_MismatchedLengths_Throws()
    {
        double[] xs = [1.0, 2.0, 3.0];
        double[] ys = [0.05, 0.06];

        var act = () => new MonotoneCubicSpline(xs, ys);
        act.Should().Throw<ArgumentException>();
    }
}
