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
        for (var i = 0; i < s_xs.Length - 1; i++)
        {
            var xMid = (s_xs[i] + s_xs[i + 1]) / 2.0;
            var yMin = Math.Min(s_ys[i], s_ys[i + 1]);
            var yMax = Math.Max(s_ys[i], s_ys[i + 1]);
            var result = spline.Interpolate(xMid);

            result.Should().BeGreaterThanOrEqualTo(yMin);
            result.Should().BeLessThanOrEqualTo(yMax);
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
