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

#pragma warning disable BNU0001

namespace Boutquin.Numerics.Tests.Unit.Interpolation;

/// <summary>
/// Pins the documented refusal of a clamped double slope outside the working type's finite range after scaling:
/// the non-generic <see cref="CubicSplineInterpolator{T}"/> boundary constructor throws
/// <see cref="ArgumentOutOfRangeException"/> at every <c>T</c>, including <see cref="decimal"/>, which has no
/// infinity and whose narrowing conversion would otherwise raise <see cref="OverflowException"/>. The decimal range
/// edge (<see cref="decimal.MaxValue"/> is about 7.9228e28) is the independent oracle for which slopes must construct.
/// </summary>
public sealed class CubicSplineClampedSlopeRefusalContractTests
{
    private static readonly decimal[] s_unitKnots = [0m, 1m, 2m, 3m];

    [Theory]
    [InlineData(1e30, 0.0, "leftSlope")]
    [InlineData(0.0, -1e30, "rightSlope")]
    [InlineData(8e28, 0.0, "leftSlope")]
    [InlineData(0.0, -8e28, "rightSlope")]
    [InlineData(1e300, 0.0, "leftSlope")]
    public void DecimalClampedSlopeBeyondFiniteRange_IsRefusedAsArgumentOutOfRange(double left, double right, string parameter)
    {
        var act = () => new CubicSplineInterpolator<decimal>(s_unitKnots, new decimal[4], SplineBoundary.CreateClamped(left, right));

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(parameter);
    }

    [Theory]
    [InlineData(7.9e28, 0.0)]
    [InlineData(0.0, -7.9e28)]
    [InlineData(1e14, -1e14)]
    public void DecimalClampedSlopeWithinFiniteRange_Constructs(double left, double right)
    {
        var spline = new CubicSplineInterpolator<decimal>(s_unitKnots, new decimal[4], SplineBoundary.CreateClamped(left, right));

        spline.Interpolate(1.5m).Should().NotBe(decimal.MaxValue);
    }

    [Fact]
    public void DecimalClampedSlopeBroughtIntoRangeByScaling_Constructs()
    {
        // Knots 1e-12 wide: coordinate scaling multiplies the slope down, so 1e30 is representable in final units.
        decimal[] knots = [0m, 0.000000000001m, 0.000000000002m, 0.000000000003m];
        var act = () => new CubicSplineInterpolator<decimal>(knots, new decimal[4], SplineBoundary.CreateClamped(1e30, 0.0));

        act.Should().NotThrow();
    }

    [Theory]
    [InlineData(1e39, 0.0, "leftSlope")]
    [InlineData(0.0, -1e39, "rightSlope")]
    public void FloatClampedSlopeBeyondFiniteRange_IsRefusedAsArgumentOutOfRange(double left, double right, string parameter)
    {
        var act = () => new CubicSplineInterpolator<float>([0f, 1f, 2f, 3f], new float[4], SplineBoundary.CreateClamped(left, right));

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(parameter);
    }
}
