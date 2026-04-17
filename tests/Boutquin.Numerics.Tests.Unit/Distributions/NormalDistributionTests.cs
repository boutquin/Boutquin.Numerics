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

public sealed class NormalDistributionTests
{
    [Fact]
    public void Pdf_AtZero_ReturnsMaxDensity()
    {
        var result = NormalDistribution.Pdf(0.0);
        var expected = 1.0 / Math.Sqrt(2.0 * Math.PI);
        result.Should().BeApproximately(expected, 1e-12);
    }

    [Fact]
    public void Pdf_IsSymmetric()
    {
        NormalDistribution.Pdf(1.5).Should().BeApproximately(NormalDistribution.Pdf(-1.5), 1e-12);
    }

    [Fact]
    public void Pdf_AtLargeValue_IsNearZero()
    {
        NormalDistribution.Pdf(10.0).Should().BeLessThan(1e-20);
    }

    [Fact]
    public void Cdf_AtZero_ReturnsHalf()
    {
        // Laikov's erf(0) = 0 exactly → N(0) = 0.5 exactly
        CumulativeNormal.Evaluate(0.0).Should().BeApproximately(0.5, 1e-14);
    }

    [Fact]
    public void Cdf_AtLargePositive_ReturnsNearOne()
    {
        CumulativeNormal.Evaluate(6.0).Should().BeApproximately(1.0, 1e-7);
    }

    [Fact]
    public void Cdf_AtLargeNegative_ReturnsNearZero()
    {
        CumulativeNormal.Evaluate(-6.0).Should().BeApproximately(0.0, 1e-7);
    }

    [Theory]
    [InlineData(0.0, 0.5)]
    [InlineData(1.0, 0.84134474606854)]
    [InlineData(-1.0, 0.15865525393146)]
    [InlineData(1.96, 0.97500210485178)]
    [InlineData(2.326, 0.98999072465913)]
    public void Cdf_KnownValues_HighPrecision(double x, double expected)
    {
        // Laikov's approximation provides ~14 decimal digits.
        // Use 1e-10 tolerance to leave margin for the erf→N(x) composition.
        CumulativeNormal.Evaluate(x).Should().BeApproximately(expected, 1e-10);
    }

    [Fact]
    public void Cdf_Symmetry_NxPlusNNegX_EqualsOne()
    {
        var x = 1.5;
        var sum = CumulativeNormal.Evaluate(x) + CumulativeNormal.Evaluate(-x);
        sum.Should().BeApproximately(1.0, 1e-12);
    }

    [Fact]
    public void Cdf_ExtremeTail_IsNonZero()
    {
        // N(-8) is tiny but must not be zero — important for risk/VaR tail computations
        var result = CumulativeNormal.Evaluate(-8.0);
        result.Should().BeGreaterThan(0.0);
        result.Should().BeLessThan(1e-10);
    }
}
