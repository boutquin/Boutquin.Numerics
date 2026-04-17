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
    public void Interpolate_AtNode_ReturnsExactValue()
    {
        var result = _sut.Interpolate(1.0, s_xs, s_ys);
        result.Should().BeApproximately(0.043, 1e-10);
    }

    [Fact]
    public void Interpolate_BetweenNodes_PreservesMonotonicity()
    {
        for (var i = 0; i < s_xs.Length - 1; i++)
        {
            var xMid = (s_xs[i] + s_xs[i + 1]) / 2.0;
            var yMin = Math.Min(s_ys[i], s_ys[i + 1]);
            var yMax = Math.Max(s_ys[i], s_ys[i + 1]);

            var result = _sut.Interpolate(xMid, s_xs, s_ys);

            result.Should().BeGreaterThanOrEqualTo(yMin, $"interval [{s_xs[i]}, {s_xs[i + 1]}]");
            result.Should().BeLessThanOrEqualTo(yMax, $"interval [{s_xs[i]}, {s_xs[i + 1]}]");
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

        for (var i = 0; i < xs.Length - 1; i++)
        {
            var xMid = (xs[i] + xs[i + 1]) / 2.0;
            var yMin = Math.Min(ys[i], ys[i + 1]);
            var yMax = Math.Max(ys[i], ys[i + 1]);

            var result = _sut.Interpolate(xMid, xs, ys);

            result.Should().BeGreaterThanOrEqualTo(yMin, $"interval [{xs[i]}, {xs[i + 1]}]");
            result.Should().BeLessThanOrEqualTo(yMax, $"interval [{xs[i]}, {xs[i + 1]}]");
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
    public void Interpolate_SingleNode_ReturnsYValue()
    {
        double[] xs = [1.0];
        double[] ys = [0.05];

        var result = _sut.Interpolate(2.0, xs, ys);
        result.Should().BeApproximately(0.05, 1e-10);
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
