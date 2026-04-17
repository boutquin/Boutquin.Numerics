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

using Boutquin.Numerics.Distributions;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Distributions;

public sealed class InverseNormalTests
{
    [Fact]
    public void Evaluate_AtHalf_ReturnsZero()
    {
        InverseNormal.Evaluate(0.5).Should().BeApproximately(0.0, 1e-14);
    }

    [Theory]
    [InlineData(0.84134474606854, 1.0)]
    [InlineData(0.15865525393146, -1.0)]
    [InlineData(0.97500210485178, 1.96)]
    [InlineData(0.99, 2.32634787404084)]
    [InlineData(0.025, -1.95996398454005)]
    public void Evaluate_KnownQuantiles(double p, double expected)
    {
        InverseNormal.Evaluate(p).Should().BeApproximately(expected, 1e-10);
    }

    [Theory]
    [InlineData(0.1)]
    [InlineData(0.3)]
    [InlineData(0.7)]
    [InlineData(0.9)]
    public void Evaluate_Symmetry(double p)
    {
        var sum = InverseNormal.Evaluate(p) + InverseNormal.Evaluate(1.0 - p);
        sum.Should().BeApproximately(0.0, 1e-12);
    }

    [Theory]
    [InlineData(-3.0)]
    [InlineData(-1.0)]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(3.0)]
    public void Evaluate_RoundTrip_WithCdf(double x)
    {
        var p = CumulativeNormal.Evaluate(x);
        InverseNormal.Evaluate(p).Should().BeApproximately(x, 1e-10);
    }

    [Fact]
    public void Evaluate_ExtremeLowTail()
    {
        var result = InverseNormal.Evaluate(1e-10);
        result.Should().BeNegative();
        result.Should().BeLessThan(-6.0);
    }

    [Fact]
    public void Evaluate_ExtremeHighTail()
    {
        var result = InverseNormal.Evaluate(1.0 - 1e-10);
        result.Should().BeGreaterThan(6.0);
    }

    [Fact]
    public void Evaluate_ThrowsForZero()
    {
        var act = () => InverseNormal.Evaluate(0.0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Evaluate_ThrowsForOne()
    {
        var act = () => InverseNormal.Evaluate(1.0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Evaluate_ThrowsForNegative()
    {
        var act = () => InverseNormal.Evaluate(-0.1);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void NormalDistribution_InverseCdf_DelegatesToInverseNormal()
    {
        NormalDistribution.InverseCdf(0.5).Should().BeApproximately(0.0, 1e-14);
    }
}
