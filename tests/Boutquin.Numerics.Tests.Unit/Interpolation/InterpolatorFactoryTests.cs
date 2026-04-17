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

public sealed class InterpolatorFactoryTests
{
    [Fact]
    public void Create_Linear_ReturnsLinearInterpolator()
    {
        var result = InterpolatorFactory.Create(InterpolatorKind.Linear);
        result.Should().BeOfType<LinearInterpolator>();
    }

    [Fact]
    public void Create_LogLinear_ReturnsLogLinearInterpolator()
    {
        var result = InterpolatorFactory.Create(InterpolatorKind.LogLinear);
        result.Should().BeOfType<LogLinearInterpolator>();
    }

    [Fact]
    public void Create_FlatForward_ReturnsFlatForwardInterpolator()
    {
        var result = InterpolatorFactory.Create(InterpolatorKind.FlatForward);
        result.Should().BeOfType<FlatForwardInterpolator>();
    }

    [Fact]
    public void Create_MonotoneCubic_ReturnsMonotoneCubicInterpolator()
    {
        var result = InterpolatorFactory.Create(InterpolatorKind.MonotoneCubic);
        result.Should().BeOfType<MonotoneCubicInterpolator>();
    }

    [Fact]
    public void Create_MonotoneConvex_ReturnsMonotoneConvexInterpolator()
    {
        var result = InterpolatorFactory.Create(InterpolatorKind.MonotoneConvex);
        result.Should().BeOfType<MonotoneConvexInterpolator>();
    }

    [Fact]
    public void Create_CubicSpline_ThrowsNotSupportedException()
    {
        var act = () => InterpolatorFactory.Create(InterpolatorKind.CubicSpline);
        act.Should().Throw<NotSupportedException>();
    }

    [Fact]
    public void Create_ReturnsSingletonInstances()
    {
        var first = InterpolatorFactory.Create(InterpolatorKind.Linear);
        var second = InterpolatorFactory.Create(InterpolatorKind.Linear);
        ReferenceEquals(first, second).Should().BeTrue();
    }

    [Fact]
    public void Create_UnknownKind_ThrowsArgumentOutOfRangeException()
    {
        var act = () => InterpolatorFactory.Create((InterpolatorKind)999);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
