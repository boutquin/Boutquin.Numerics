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

public sealed class C4GarchForecastTests
{
    [Fact]
    public void Forecast_BothFeedbackCoefficients_UsesExactConditionalRecurrence()
    {
        // Hull supplied parameters: h1=.000002+.13*.0001+.86*.000256.
        // Future innovations have conditional second moment equal to forecast h,
        // so h2=.000002+.99*h1 and h3=.000002+.99*h2.
        var fit = new GarchFitResult<double>(0, 0, .000002, .13, .86, [.000256], [-.625], 0, true);
        fit.ForecastVariance(1).Should().BeApproximately(.00023516, 1e-15);
        fit.ForecastVariance(2).Should().BeApproximately(.0002348084, 1e-15);
        fit.ForecastVariance(3).Should().BeApproximately(.000234460316, 1e-15);
        fit.ForecastVariance(10_001).Should().BeApproximately(.0002, 1e-15);
    }
}
