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

public sealed class MonotoneCubicInterpolatorTests
{
    private static readonly double[] s_xs = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0];
    private static readonly double[] s_ys = [0.04, 0.041, 0.043, 0.045, 0.048, 0.05, 0.052];

    private readonly MonotoneCubicInterpolator _sut = MonotoneCubicInterpolator.Instance;

    [Fact]
    public void Interpolate_TwoNodesWideFiniteSpan_IsLinear()
    {
        double[] xs = [-1e308, 1e308];
        double[] ys = [1, 4];
        _sut.Interpolate(0, xs, ys).Should().BeApproximately(2.5, 1e-12);
    }

    [Fact]
    public void Interpolate_TwoNodesWideFiniteOrdinates_PreservesMidpoint()
    {
        MonotoneCubicInterpolator<double>.Instance.Interpolate(
            0.5, [0, 1], [-1e308, 1e308]).Should().Be(0);
    }

    [Fact]
    public void Interpolate_ThreeNodesWideFiniteOrdinates_StaysInSegment()
    {
        var result = MonotoneCubicInterpolator<double>.Instance.Interpolate(
            0.5, [0, 1, 2], [-1e308, 0, 1e308]);

        double.IsFinite(result).Should().BeTrue();
        result.Should().BeInRange(-1e308, 0);
    }

    [Fact]
    public void Interpolate_WideOrdinatesAndNarrowIntervals_StaysFinite()
    {
        var result = MonotoneCubicInterpolator<double>.Instance.Interpolate(
            0.5e-10, [0, 1e-10, 2e-10], [-1e300, 0, 1e300]);

        double.IsFinite(result).Should().BeTrue();
        result.Should().BeInRange(-1e300, 0);
        (result / -5e299).Should().BeApproximately(1, 1e-14);
    }

    [Fact]
    public void Interpolate_DecimalSteepFiniteSlopes_DoesNotOverflowSignTests()
    {
        // Slopes of 1e15 are representable in decimal, but their products (1e30) are not.
        // Collinear data is reproduced exactly by every Fritsch-Carlson tangent choice.
        var act = () => MonotoneCubicInterpolator<decimal>.Instance.Interpolate(
            1.5e-5m, [0m, 1e-5m, 2e-5m, 3e-5m], [1e10m, 2e10m, 3e10m, 4e10m]);

        act.Should().NotThrow<OverflowException>();
        act().Should().Be(2.5e10m);
    }

    [Fact]
    public void Interpolate_DecreasingDataWithClippedTangent_StaysWithinSegment()
    {
        // On [0, 1] δ₀ = −1 and the interior tangent m₁ = −5 gives β = 5 > 3, so the S₂ circle
        // rescales it to m₁ = τ·m₁ with τ = 3/√(α² + β²) > 0 — still negative. A sign flip to +3
        // would make the Hermite cubic overshoot below −1.
        double[] xs = [0.0, 1.0, 2.0];
        double[] ys = [0.0, -1.0, -10.0];
        var spline = new MonotoneCubicSpline<double>(xs, ys);
        for (var k = 1; k < 20; k++)
        {
            var x = k / 20.0;
            MonotoneCubicInterpolator<double>.Instance.Interpolate(x, xs, ys).Should().BeInRange(-1.0, 0.0, $"stateless at x = {x}");
            spline.Interpolate(x).Should().BeInRange(-1.0, 0.0, $"spline at x = {x}");
        }
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
        MonotoneCubicInterpolator<double>.Instance.Interpolate(x, [0.0, 1.0, 3.0], [0.0, 1.0, 9.0])
            .Should().BeApproximately(expected, 1e-14);
    }

    [Fact]
    public void Interpolate_AtNode_ReturnsExactValue()
    {
        var result = _sut.Interpolate(1.0, s_xs, s_ys);
        result.Should().BeApproximately(0.043, 1e-10);
    }

    [Fact]
    public void Interpolate_BetweenNodes_PreservesMonotonicity()
    {
        // Sweep 15 interior samples per segment (not just the midpoint) and additionally assert the
        // interpolant is itself monotone across consecutive samples — no overshoot anywhere.
        const int samplesPerSegment = 15;
        for (var i = 0; i < s_xs.Length - 1; i++)
        {
            var yMin = Math.Min(s_ys[i], s_ys[i + 1]);
            var yMax = Math.Max(s_ys[i], s_ys[i + 1]);
            var ascending = s_ys[i + 1] >= s_ys[i];

            var prev = _sut.Interpolate(s_xs[i], s_xs, s_ys);
            for (var k = 1; k <= samplesPerSegment; k++)
            {
                var t = k / (double)(samplesPerSegment + 1);
                var x = s_xs[i] + t * (s_xs[i + 1] - s_xs[i]);
                var result = _sut.Interpolate(x, s_xs, s_ys);

                result.Should().BeGreaterThanOrEqualTo(yMin, $"interval [{s_xs[i]}, {s_xs[i + 1]}] at x={x}");
                result.Should().BeLessThanOrEqualTo(yMax, $"interval [{s_xs[i]}, {s_xs[i + 1]}] at x={x}");

                // No reversals within the segment (monotone, allowing flat segments).
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
    public void Interpolate_BelowFirstNode_ExtrapolatesFlat()
    {
        var result = _sut.Interpolate(0.1, s_xs, s_ys);
        result.Should().BeApproximately(0.04, 1e-10);
    }

    [Fact]
    public void Interpolate_AboveLastNode_ExtrapolatesFlat()
    {
        var result = _sut.Interpolate(50.0, s_xs, s_ys);
        result.Should().BeApproximately(0.052, 1e-10);
    }

    [Fact]
    public void Interpolate_NonMonotoneData_StillNoOvershoot()
    {
        double[] xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        double[] ys = [0.01, 0.05, 0.04, 0.02, 0.03];

        // Sweep 15 interior samples per segment — overshoot at non-midpoint locations is the
        // classic failure the midpoint-only test missed.
        const int samplesPerSegment = 15;
        for (var i = 0; i < xs.Length - 1; i++)
        {
            var yMin = Math.Min(ys[i], ys[i + 1]);
            var yMax = Math.Max(ys[i], ys[i + 1]);

            for (var k = 1; k <= samplesPerSegment; k++)
            {
                var t = k / (double)(samplesPerSegment + 1);
                var x = xs[i] + t * (xs[i + 1] - xs[i]);
                var result = _sut.Interpolate(x, xs, ys);

                result.Should().BeGreaterThanOrEqualTo(yMin, $"interval [{xs[i]}, {xs[i + 1]}] at x={x}");
                result.Should().BeLessThanOrEqualTo(yMax, $"interval [{xs[i]}, {xs[i + 1]}] at x={x}");
            }
        }
    }

    [Fact]
    public void Interpolate_ThreeNodesMinimum_ReturnsValue()
    {
        double[] xs = [1.0, 2.0, 3.0];
        double[] ys = [0.03, 0.04, 0.05];

        var result = _sut.Interpolate(1.5, xs, ys);

        result.Should().BeGreaterThanOrEqualTo(0.03);
        result.Should().BeLessThanOrEqualTo(0.04);
    }

    [Fact]
    public void Interpolate_ExtremeFiniteSlopeTransition_StaysWithinSegment()
    {
        double[] xs = [0, 1, 2, 3];
        double[] ys = [-1e300, 0, 1e-100, 2e-100];

        var result = MonotoneCubicInterpolator<double>.Instance.Interpolate(1.5, xs, ys);

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
        Action interpolate = () => result = MonotoneCubicInterpolator<decimal>.Instance.Interpolate(1.5m, xs, ys);

        interpolate.Should().NotThrow();
        result.Should().BeGreaterThanOrEqualTo(ys[1]);
        result.Should().BeLessThanOrEqualTo(ys[2]);
    }

    [Fact]
    public void Interpolate_SingleNode_ReturnsYValue()
    {
        double[] xs = [1.0];
        double[] ys = [0.05];

        var result = _sut.Interpolate(2.0, xs, ys);
        result.Should().BeApproximately(0.05, 1e-10);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Interpolate_SingleNodeNonFiniteTarget_IsRejected(double target)
    {
        double[] xs = [1.0];
        double[] ys = [0.05];

        Action generic = () => MonotoneCubicInterpolator<double>.Instance.Interpolate(target, xs, ys);
        Action legacy = () => _sut.Interpolate(target, xs, ys);

        generic.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("x");
        legacy.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("x");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Interpolate_SingleNonFiniteAbscissa_IsRejected(double abscissa)
    {
        double[] xs = [abscissa];
        double[] ys = [0.05];

        Action generic = () => MonotoneCubicInterpolator<double>.Instance.Interpolate(2.0, xs, ys);
        Action legacy = () => _sut.Interpolate(2.0, xs, ys);

        generic.Should().Throw<ArgumentException>().WithParameterName("xs");
        legacy.Should().Throw<ArgumentException>().WithParameterName("xs");
    }

    [Fact]
    public void Name_ReturnsMonotoneCubic()
    {
        _sut.Name.Should().Be("MonotoneCubic");
    }

    [Fact]
    public void Instance_IsSingleton()
    {
        var first = MonotoneCubicInterpolator.Instance;
        var second = MonotoneCubicInterpolator.Instance;
        ReferenceEquals(first, second).Should().BeTrue();
    }
}
