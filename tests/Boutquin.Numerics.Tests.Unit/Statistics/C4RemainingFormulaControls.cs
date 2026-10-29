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

using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

[Trait("Tier", "Unit")]
public sealed class C4RemainingFormulaControls
{
    [Fact]
    public void Dsr_AsymmetricPopulationMoments_HaveExactNormalArgument()
    {
        // BLdP raw-moment definitions: [0,0,0,4] has mean 1, variance 3,
        // skew 2/sqrt(3), Pearson kurtosis 7/3. With one trial, SR0=0,
        // A=1-2/3+1/9=4/9, so z=(1/sqrt(3))*sqrt(3)/(2/3)=3/2.
        // Upper normal tail at 1.5 is 0.066807201268858066 (independent erf).
        var result = DeflatedSharpeRatio<double>.ComputeFromReturns([0, 0, 0, 4], 1, 1);
        result.PValue.Should().BeApproximately(0.066807201268858066, 1e-12);
        result.DeflatedSharpe.Should().BeApproximately(1 / Math.Sqrt(3), 1e-14);
    }

    [Fact]
    public void Dsr_DecimalAsymmetricPopulationMoments_HaveExactNormalArgument()
    {
        var result = DeflatedSharpeRatio<decimal>.ComputeFromReturns([0m, 0m, 0m, 4m], 1, 1);
        result.PValue.Should().BeApproximately(0.066807201268858066m, 1e-12m);
        result.DeflatedSharpe.Should().BeApproximately(0.5773502691896257645m, 1e-15m);
    }

    [Fact]
    public void Ewma_DefaultDailyDecay_HasExactThreePointWeightedVariance()
    {
        // Lambda=47/50 gives weights [2209,2350,2500]/7059 on [-1,0,1].
        // E[X]=291/7059; Var(X)=4709/7059-(291/7059)^2.
        var result = new ExponentiallyWeightedCovarianceEstimator<decimal>().Estimate(new decimal[,] { { -1 }, { 0 }, { 1 } });
        result[0, 0].Should().BeApproximately((4709m * 7059m - 291m * 291m) / (7059m * 7059m), 1e-24m);
    }
}
