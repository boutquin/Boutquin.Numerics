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

public sealed class C2IndependentDistributionCenterTests
{
    [Theory]
    [InlineData(.499, -.002506630899571764005358657067)]
    [InlineData(.501, .002506630899571764005358657067)]
    public void NearMedian_UsesCentralApproximationWithIndependentIntegralInverse(double p, double expected)
    {
        // 90-digit Gaussian-integral inverse from mpmath erfinv; no production
        // coefficients or output enters the expected value.
        InverseNormal<double>.Evaluate(p).Should().BeApproximately(expected, 1e-12);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ExtremeDecimalTails_HaveIndependentGaussianInverse(bool upper)
    {
        const decimal Tail = .00000000000000000001m;
        const decimal Expected = 9.262340089798407573717356977m;
        var p = upper ? 1m - Tail : Tail;
        decimal actual = 0;
        FluentActions.Invoking(() => actual = InverseNormal<decimal>.Evaluate(p)).Should().NotThrow();
        actual.Should().BeApproximately(upper ? Expected : -Expected, 1e-12m);
    }

    [Fact]
    public void GaussianSymmetry_FixesCdfAndQuantileAtCenter()
    {
        CumulativeNormal<double>.Evaluate(0).Should().BeApproximately(.5, 1e-15);
        InverseNormal<double>.Evaluate(.5).Should().BeApproximately(0, 1e-14);
    }
}
