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

public sealed class FlatForwardInterpolatorTests
{
    private static readonly double[] s_xs = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0];
    private static readonly double[] s_ys = [0.04, 0.041, 0.043, 0.045, 0.048, 0.05, 0.052];

    private readonly FlatForwardInterpolator _sut = FlatForwardInterpolator.Instance;

    [Fact]
    public void Interpolate_WideFiniteOrdinateRatio_RemainsFinite()
    {
        double[] xs = [0, 1];
        double[] ys = [1e-300, 1e300];

        var result = FlatForwardInterpolator<double>.Instance.Interpolate(0.75, xs, ys);

        double.IsFinite(result).Should().BeTrue();
        result.Should().BeApproximately(1e150, 1e137);
    }

    [Fact]
    public void Interpolate_WideFiniteSpan_PreservesGeometricMidpoint()
    {
        double[] xs = [-1e308, 1e308];
        double[] ys = [1, 4];
        _sut.Interpolate(0, xs, ys).Should().BeApproximately(2, 1e-12);
    }

    [Fact]
    public void Interpolate_AtNode_ReturnsExactValue()
    {
        var result = _sut.Interpolate(1.0, s_xs, s_ys);
        result.Should().BeApproximately(0.043, 1e-10);
    }

    [Fact]
    public void Interpolate_BetweenNodes_ConstantForward()
    {
        // Independent golden (house independent-goldens discipline): x = 1.5 is
        // exactly the midpoint of the [1.0, 2.0] node interval (y1 = 0.043,
        // y2 = 0.045). At the midpoint t = (x - x1)/(x2 - x1) = 0.5, so the
        // constant-forward-rate formula DF(x) = y1 * (y2/y1)^t collapses
        // algebraically to DF(x) = y1^0.5 * y2^0.5 = sqrt(y1 * y2) — the geometric
        // mean of the two node discount factors. Math.Sqrt is a wholly different
        // function than the production path's Log/Exp pair, so a regression in
        // the log-linear forward-rate formula is not masked by recomputing it the
        // same way here. Cross-checked independently (scratch-Python, 2026-07-07):
        //   >>> import math
        //   >>> math.sqrt(0.043 * 0.045)
        //   0.04398863489584554
        var expected = Math.Sqrt(0.043 * 0.045);

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

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Interpolate_NonFiniteTarget_IsRejected(double target)
    {
        double[] xs = [0, 1, 2];
        double[] ys = [1, 0.9, 0.8];

        Action act = () => _sut.Interpolate(target, xs, ys);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("x");
    }

    [Fact]
    public void Name_ReturnsFlatForward()
    {
        _sut.Name.Should().Be("FlatForward");
    }

    [Fact]
    public void Instance_IsSingleton()
    {
        var first = FlatForwardInterpolator.Instance;
        var second = FlatForwardInterpolator.Instance;
        ReferenceEquals(first, second).Should().BeTrue();
    }
}
