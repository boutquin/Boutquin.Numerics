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

public sealed class C2IndependentNormalFacadeTests
{
    [Theory]
    [InlineData(0, 0.398942280401432677939946059934)]
    [InlineData(2, 0.0539909665131880519505642004107)]
    [InlineData(-2, 0.0539909665131880519505642004107)]
    public void Pdf_HasIndependentGaussianDensityValues(double x, double expected)
    {
        // exp(-x²/2)/sqrt(2*pi), independently evaluated to 80 decimal digits.
        NormalDistribution<double>.Pdf(x).Should().BeApproximately(expected, 2e-16);
    }

    [Fact]
    public void Cdf_HasIndependentTailIntegralValue()
    {
        NormalDistribution<double>.Cdf(-3.5).Should().BeApproximately(0.000232629079035525036349925886728, 7e-16);
    }

    [Fact]
    public void InverseCdf_HasIndependentGaussianQuantileValue()
    {
        NormalDistribution<double>.InverseCdf(.1).Should().BeApproximately(-1.28155156554460046696510332945, 1e-12);
    }
}
