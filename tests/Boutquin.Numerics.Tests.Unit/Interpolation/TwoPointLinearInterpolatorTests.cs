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
}
