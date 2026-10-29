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

public sealed class C2IndependentDistributionTests
{
    // Independent 90-decimal-digit mpmath erfc/erfinv values, evaluated directly
    // from the Gaussian integral; no Acklam/Laikov coefficients are in the oracle.
    [Fact]
    public void Cdf_InteriorTail_MatchesIndependentGaussianIntegral()
    {
        CumulativeNormal<double>.Evaluate(-3.5)
            .Should().BeApproximately(0.000232629079035525036349925886728, 7e-16);
    }

    [Fact]
    public void Quantile_Central_MatchesIndependentGaussianInverse()
    {
        InverseNormal<double>.Evaluate(.1)
            .Should().BeApproximately(-1.28155156554460046696510332945, 1e-12);
    }

    [Fact]
    public void Quantile_LowerTail_MatchesIndependentGaussianInverse()
    {
        InverseNormal<double>.Evaluate(.01)
            .Should().BeApproximately(-2.32634787404084110088560616335, 1e-12);
    }

    [Theory]
    [InlineData(.02425, -1.97296105131188485026979885948)]
    [InlineData(.97575, 1.97296105131188485026979885948)]
    [InlineData(.0000512, -3.88483359678135854948245536395)]
    public void Quantile_BreakpointsAndCdfLimitedRegion_HaveIndependentValues(double p, double expected)
    {
        InverseNormal<double>.Evaluate(p).Should().BeApproximately(expected, 1e-12);
    }

    [Fact]
    public void Quantile_InvalidFiniteAndNonfiniteInputs_RejectBeforeEvaluation()
    {
        foreach (var p in new[] { -1d, 0d, 1d, 2d, double.NaN, double.NegativeInfinity, double.PositiveInfinity })
        {
            FluentActions.Invoking(() => InverseNormal<double>.Evaluate(p))
                .Should().Throw<ArgumentOutOfRangeException>();
        }
    }
}
