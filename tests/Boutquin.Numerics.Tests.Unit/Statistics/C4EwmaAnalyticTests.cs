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

public sealed class C4EwmaAnalyticTests
{
    [Fact]
    public void FiniteGeometricWeights_MatchExactWeightedCenteredCovariance()
    {
        // Chronological weights(1,2,4)/7; mean=(1,9/7).
        // E[x^2]-mu_x^2=10/7,E[y^2]-mu_y^2=234/49,E[xy]-mu_xmu_y=2.
        decimal[,] observations = { { -1, 1 }, { 0, -2 }, { 2, 3 } };
        var actual = new ExponentiallyWeightedCovarianceEstimator<decimal>(0.5m).Estimate(observations);
        actual[0, 0].Should().BeApproximately(10m / 7, 1e-20m);
        actual[1, 1].Should().BeApproximately(234m / 49, 1e-20m);
        actual[0, 1].Should().BeApproximately(2, 1e-20m);
        actual[1, 0].Should().BeApproximately(2, 1e-20m);
    }

    [Fact]
    public void DefaultDailyDecay_MatchesExactNormalizedWeights()
    {
        // lambda47/50 gives exact weights(2209,2350,2500)/7059.
        decimal[,] observations = { { -1, 1 }, { 0, -2 }, { 2, 3 } };
        var actual = new ExponentiallyWeightedCovarianceEstimator<decimal>().Estimate(observations);
        actual[0, 0].Should().BeApproximately(78393650m / 49829481, 1e-20m);
        actual[1, 1].Should().BeApproximately(215685350m / 49829481, 1e-20m);
        actual[0, 1].Should().BeApproximately(76311550m / 49829481, 1e-20m);
        actual[1, 0].Should().BeApproximately(76311550m / 49829481, 1e-20m);
    }
}
