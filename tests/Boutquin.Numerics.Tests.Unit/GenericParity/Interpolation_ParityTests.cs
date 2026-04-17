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

namespace Boutquin.Numerics.Tests.Unit.GenericParity;

public sealed class Interpolation_ParityTests
{
    private static readonly double[] s_xs = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    private static readonly double[] s_ys = { 10.0, 20.0, 30.0, 40.0, 50.0 };
    private static readonly double[] s_discountFactors = { 1.0, 0.99, 0.97, 0.94, 0.90 };

    [Fact]
    public void LinearInterpolator_GenericMatchesLegacy_AtDouble()
    {
        var testX = 2.5;

        var legacy = LinearInterpolator.Instance.Interpolate(testX, s_xs, s_ys);
        var generic = LinearInterpolator<double>.Instance.Interpolate(testX, s_xs, s_ys);

        generic.Should().BeApproximately(legacy, 1e-12);
    }

    [Fact]
    public void LogLinearInterpolator_GenericMatchesLegacy_AtDouble()
    {
        var testX = 2.5;

        var legacy = LogLinearInterpolator.Instance.Interpolate(testX, s_xs, s_discountFactors);
        var generic = LogLinearInterpolator<double>.Instance.Interpolate(testX, s_xs, s_discountFactors);

        generic.Should().BeApproximately(legacy, 1e-12);
    }

    [Fact]
    public void FlatForwardInterpolator_GenericMatchesLegacy_AtDouble()
    {
        var testX = 2.5;

        var legacy = FlatForwardInterpolator.Instance.Interpolate(testX, s_xs, s_discountFactors);
        var generic = FlatForwardInterpolator<double>.Instance.Interpolate(testX, s_xs, s_discountFactors);

        generic.Should().BeApproximately(legacy, 1e-12);
    }

    [Fact]
    public void MonotoneCubicInterpolator_GenericMatchesLegacy_AtDouble()
    {
        var testX = 2.5;

        var legacy = MonotoneCubicInterpolator.Instance.Interpolate(testX, s_xs, s_ys);
        var generic = MonotoneCubicInterpolator<double>.Instance.Interpolate(testX, s_xs, s_ys);

        generic.Should().BeApproximately(legacy, 1e-12);
    }

    [Fact]
    public void MonotoneConvexInterpolator_GenericMatchesLegacy_AtDouble()
    {
        // NCR nodes: virtual origin (0,0) + 4 actual nodes
        double[] xs = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
        double[] ys = { 0.0, 0.04, 0.09, 0.15, 0.22, 0.30 };
        var testX = 2.5;

        var legacy = MonotoneConvexInterpolator.Instance.Interpolate(testX, xs, ys);
        var generic = MonotoneConvexInterpolator<double>.Instance.Interpolate(testX, xs, ys);

        generic.Should().BeApproximately(legacy, 1e-12);
    }

    [Fact]
    public void TwoPointLinearInterpolator_GenericMatchesLegacy_AtDouble()
    {
        var x0 = 1.0;
        var y0 = 10.0;
        var x1 = 3.0;
        var y1 = 30.0;
        var x = 2.0;

        var legacy = TwoPointLinearInterpolator.Interpolate(x0, y0, x1, y1, x);
        var generic = TwoPointLinearInterpolator<double>.Interpolate(x0, y0, x1, y1, x);

        generic.Should().BeApproximately(legacy, 1e-12);
    }

    [Fact]
    public void CubicSplineInterpolator_GenericMatchesLegacy_AtDouble()
    {
        var testX = 2.5;

        var legacy = new CubicSplineInterpolator(s_xs, s_ys);
        var generic = new CubicSplineInterpolator<double>(s_xs, s_ys);

        var legacyResult = legacy.Interpolate(testX);
        var genericResult = generic.Interpolate(testX);

        genericResult.Should().BeApproximately(legacyResult, 1e-12);
    }

    [Fact]
    public void MonotoneCubicSpline_GenericMatchesLegacy_AtDouble()
    {
        var testX = 2.5;

        var legacy = new MonotoneCubicSpline(s_xs, s_ys);
        var generic = new MonotoneCubicSpline<double>(s_xs, s_ys);

        var legacyResult = legacy.Interpolate(testX);
        var genericResult = generic.Interpolate(testX);

        genericResult.Should().BeApproximately(legacyResult, 1e-12);
    }
}
