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
public sealed class C4SuppliedGarchLimitControls
{
    [Theory]
    [InlineData(.000002, .13, .86, .0002, 1.4142135623730950488)]
    [InlineData(.000003981826267827113, .2237925232529377, .7475767721285745,
        .000139075385006623801255, 1.1793022725604483514)]
    public void SuppliedHullParameters_ReachIndependentLongRunVariance(
        double omega, double alpha, double beta, double expectedVariance, double expectedDailyPercent)
    {
        // Hull's illustrative tuple and authenticated author-workbook tuple.
        // Independent50-digit Decimal calculation: V=omega/(1-alpha-beta),
        // daily percent volatility=100*sqrt(V). This supplies parameters; it does
        // not assert that Fit recovers them, or adopt the author's presample seed.
        var supplied = new GarchFitResult<double>(0, 0, omega, alpha, beta, [.000256], [-.625], 0, true);
        var actual = supplied.ForecastVariance(10_001);
        actual.Should().BeApproximately(expectedVariance, 2e-17);
        (100 * Math.Sqrt(actual)).Should().BeApproximately(expectedDailyPercent, 1e-11);
    }
}
