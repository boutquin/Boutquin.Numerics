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

public sealed class LogLinearInterpolatorTests
{
    private static readonly double[] s_xs = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0];
    private static readonly double[] s_ys = [0.04, 0.041, 0.043, 0.045, 0.048, 0.05, 0.052];

    private readonly LogLinearInterpolator _sut = LogLinearInterpolator.Instance;

    [Fact]
    public void Interpolate_AtNode_ReturnsExactValue()
    {
        var result = _sut.Interpolate(1.0, s_xs, s_ys);
        result.Should().BeApproximately(0.043, 1e-10);
    }

    [Fact]
    public void Interpolate_BetweenNodes_IsLogLinear()
    {
        var t = 0.5;
        var lnY0 = Math.Log(0.043);
        var lnY1 = Math.Log(0.045);
        var expected = Math.Exp(lnY0 + t * (lnY1 - lnY0));

        var result = _sut.Interpolate(1.5, s_xs, s_ys);
        result.Should().BeApproximately(expected, 1e-10);
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
    public void Interpolate_NegativeY_ReturnsNaN()
    {
        double[] xs = [1.0, 2.0];
        double[] ys = [-0.01, 0.02];
        var result = _sut.Interpolate(1.5, xs, ys);
        double.IsNaN(result).Should().BeTrue();
    }

    [Fact]
    public void Name_ReturnsLogLinear()
    {
        _sut.Name.Should().Be("LogLinear");
    }

    [Fact]
    public void Instance_IsSingleton()
    {
        var first = LogLinearInterpolator.Instance;
        var second = LogLinearInterpolator.Instance;
        ReferenceEquals(first, second).Should().BeTrue();
    }
}
