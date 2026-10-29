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

using Boutquin.Numerics.MonteCarlo;
using Boutquin.Numerics.Random;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class PolitisWhiteCorrectedTableTests
{
    [Theory]
    [InlineData(0.7, 200, 11.47, 13.12)]
    [InlineData(0.7, 800, 18.20, 20.83)]
    [InlineData(0.1, 200, 2.01, 2.30)]
    [InlineData(0.1, 800, 3.20, 3.66)]
    [InlineData(-0.4, 200, 5.66, 6.48)]
    [InlineData(-0.4, 800, 8.99, 10.29)]
    public void StationaryOptimum_CorrectedTable1_MatchesPopulationAutocovariances(
        double phi, int count, double printedStationary, double reconciledCircular)
    {
        // Unit innovation variance: R(k)=phi^|k|/(1-phi²). Sum the
        // two-sided signed covariance series independently of the AR(1)
        // closed-form optimum. At 512 lags the omitted tail is < 1e-73.
        var variance = 1.0 / (1.0 - phi * phi);
        var spectralSum = variance;
        var covarianceMoment = 0.0;
        var covariance = variance;
        for (var lag = 1; lag <= 512; lag++)
        {
            covariance *= phi;
            spectralSum += 2.0 * covariance;
            covarianceMoment += 2.0 * lag * covariance;
        }

        var actual = PolitisWhiteBlockLength.StationaryOptimum(covarianceMoment, spectralSum, count);
        var closedForm = Math.Pow(2.0 * Math.Abs(phi) / (1.0 - phi * phi), 2.0 / 3.0) * Math.Cbrt(count);

        actual.Should().BeApproximately(closedForm, 1e-12);
        actual.Should().BeApproximately(printedStationary, 0.005,
            "Table 1 prints the population stationary optimum to two decimal places");

        // Circular selection is not public API. This is a theoretical ratio
        // check: D_CB=4g(0)²/3, so b_CB=(3/2)^(1/3)*b_SB.
        // Two printed cells require reconciliation: 2.31 -> 2.30 and
        // 10.23 -> 10.29. Do not force production math to match those cells.
        var circular = actual * Math.Cbrt(1.5);
        circular.Should().BeApproximately(reconciledCircular, 0.005);
    }

    [Fact]
    public void EstimateDouble_NegativeAr1_ShortSampleUsesCorrectedCompanionWindow()
    {
        // Held-out seed, independent of the six Table 2 oracle streams.
        // For phi=-.4 and N=200 the first significant correlation must
        // participate in the window; skipping it selects M=2 rather than 4
        // on many samples and approximately doubles the mean estimate.
        var sampler = new GaussianSampler(new Xoshiro256StarStarRandomSource(20090101UL));
        const int count = 200;
        const int replications = 1000;
        var series = new double[count];
        var total = 0.0;
        for (var trial = 0; trial < replications; trial++)
        {
            var x = sampler.Next() / Math.Sqrt(0.84);
            for (var i = 0; i < count; i++)
            {
                x = -0.4 * x + sampler.Next();
                series[i] = x;
            }

            total += PolitisWhiteBlockLength.EstimateDouble(series);
        }

        var optimum = Math.Cbrt(count * Math.Pow(0.8 / 0.84, 2));
        (total / replications / optimum).Should().BeInRange(0.8, 1.4,
            "the corrected selector must not reproduce the former roughly 2.26 mean ratio");
    }
}
