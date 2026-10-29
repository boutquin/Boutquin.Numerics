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

public sealed class Arena20261001ScaleTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Monotone_TwoKnots_PreservesUnderflowedWeightContribution(bool cached)
    {
        double[] xs = [0, 1e300];
        double[] ys = [0, 1e300];
        var actual = cached ? new MonotoneCubicSpline<double>(xs, ys).Interpolate(1e-100)
            : MonotoneCubicInterpolator<double>.Instance.Interpolate(1e-100, xs, ys);
        actual.Should().BeApproximately(1e-100, 1e-114);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Linear_WeightUnderflow_PreservesRepresentableContribution(bool twoPoint)
    {
        var actual = twoPoint
            ? TwoPointLinearInterpolator<double>.Interpolate(0, 0, 1e300, 1e300, 1e-100)
            : LinearInterpolator<double>.Instance.Interpolate(1e-100, [0, 1e300], [0, 1e300]);
        actual.Should().BeApproximately(1e-100, 1e-114);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Cubic_OverflowingWidth_MatchesAnalyticNaturalSpline(bool asymmetric)
    {
        double[] xs = asymmetric ? [-1e308, 9e307, 1e308] : [-1e308, 1e308, 1.5e308];
        double[] ys = asymmetric ? [0, 1, 2] : [-1, 1, 1.5];
        var spline = new CubicSplineInterpolator<double>(xs, ys);
        spline.Interpolate(asymmetric ? -5.25e307 : 0).Should()
            .BeApproximately(asymmetric ? -1.75390625 : 0, 1e-13);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void Monotone_ExtremeSpacing_ReproducesLine(bool cached, bool tiny)
    {
        double[] xs = tiny ? [0, 1e-310, 2e-310] : [-1e308, 0, 1e308];
        double[] ys = tiny ? [0, .5, 1] : [0, 1, 2];
        var query = tiny ? .5e-310 : -7.5e307;
        var actual = cached ? new MonotoneCubicSpline<double>(xs, ys).Interpolate(query)
            : MonotoneCubicInterpolator<double>.Instance.Interpolate(query, xs, ys);
        actual.Should().BeApproximately(.25, 1e-12);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Monotone_OverflowingWidth_ReproducesIndependentHermite(bool cached)
    {
        double[] xs = [-1e308, 9e307, 1e308];
        double[] ys = [0, 1, 2];
        var actual = cached ? new MonotoneCubicSpline<double>(xs, ys).Interpolate(-5.25e307)
            : MonotoneCubicInterpolator<double>.Instance.Interpolate(-5.25e307, xs, ys);
        actual.Should().BeApproximately(1d / 64, 1e-13);
    }

    [Theory]
    [InlineData(1e-309)]
    [InlineData(1e-310)]
    public void Convex_TinySpacing_ReproducesConstantForward(double spacing)
    {
        MonotoneConvexInterpolator<double>.Instance.Interpolate(spacing / 2, [0, spacing, 2 * spacing], [0, .5, 1])
            .Should().BeApproximately(.25, 1e-12);
    }
}
