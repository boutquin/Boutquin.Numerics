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

public sealed class TwoPointLinearInterpolatorTests
{
    [Fact]
    public void Interpolate_OverflowingExtrapolationWeight_PreservesRepresentableLines()
    {
        using var scope = new FluentAssertions.Execution.AssertionScope();
        var tiny = (Half)0.00001;
        TwoPointLinearInterpolator<Half>.Interpolate((Half)0, (Half)0, tiny, tiny, (Half)1)
            .Should().Be((Half)1);
        TwoPointLinearInterpolator<double>.Interpolate(0, 0, 1e-320, 1e-320, 1)
            .Should().Be(1);
        TwoPointLinearInterpolator<Half>.Interpolate((Half)0, (Half)1, tiny, (Half)1, (Half)1)
            .Should().Be((Half)1);
        TwoPointLinearInterpolator<decimal>.Interpolate(0m, 0m, 1e-28m, 1e-28m, 10m)
            .Should().Be(10m);
    }

    [Fact]
    public void Interpolate_Midpoint_ReturnsAverage()
    {
        var result = TwoPointLinearInterpolator.Interpolate(0.0, 10.0, 2.0, 20.0, 1.0);
        result.Should().BeApproximately(15.0, 1e-10);
    }

    [Fact]
    public void Interpolate_AtFirstPoint_ReturnsY0()
    {
        var result = TwoPointLinearInterpolator.Interpolate(1.0, 5.0, 3.0, 15.0, 1.0);
        result.Should().BeApproximately(5.0, 1e-10);
    }

    [Fact]
    public void Interpolate_AtSecondPoint_ReturnsY1()
    {
        var result = TwoPointLinearInterpolator.Interpolate(1.0, 5.0, 3.0, 15.0, 3.0);
        result.Should().BeApproximately(15.0, 1e-10);
    }

    [Fact]
    public void Interpolate_Extrapolates_BeyondRange()
    {
        // x=4.0 is beyond [1, 3], extrapolation should work
        var result = TwoPointLinearInterpolator.Interpolate(1.0, 5.0, 3.0, 15.0, 4.0);
        result.Should().BeApproximately(20.0, 1e-10);
    }

    [Fact]
    public void Interpolate_IdenticalX_Throws()
    {
        var act = () => TwoPointLinearInterpolator.Interpolate(1.0, 5.0, 1.0, 15.0, 1.5);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Interpolate_WideFiniteCoordinates_PreserveFractionalPosition()
    {
        TwoPointLinearInterpolator<double>.Interpolate(-1e308, 0, 1e308, 1, 0)
            .Should().BeApproximately(0.5, 1e-15);
        TwoPointLinearInterpolator<double>.Interpolate(-1e308, 0, 1e308, 1, 5e307)
            .Should().BeApproximately(0.75, 1e-15);
        TwoPointLinearInterpolator<decimal>.Interpolate(
            -decimal.MaxValue, 0m, decimal.MaxValue, 1m, 0m).Should().Be(0.5m);
    }

    [Fact]
    public void Interpolate_WideFiniteOrdinates_PreservesMidpoint()
    {
        TwoPointLinearInterpolator<double>.Interpolate(0, -1e308, 1, 1e308, 0.5)
            .Should().Be(0);
        TwoPointLinearInterpolator<decimal>.Interpolate(
            0m, -decimal.MaxValue, 1m, decimal.MaxValue, 0.5m).Should().Be(0m);
    }
}
