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

public sealed class C4HullAuthorTableTests
{
    [Theory]
    [InlineData(4, 0.0001224574514573928, 0.004453158965322246, 9.996611810649313e-05)]
    [InlineData(5, 9.996611810649313e-05, -0.0007349861476157641, 7.883506792293213e-05)]
    [InlineData(6, 7.883506792293213e-05, 0.008014680547327778, 7.729242794346732e-05)]
    [InlineData(1258, 0.0001834132295929838, -0.010818532233573217, 0.00016729011649173562)]
    [InlineData(1259, 0.00016729011649173562, 0.007827461901406754, 0.000142755611436984)]
    public void EachPrintedDay_MatchesAuthenticatedAuthorVariance(
        int day, double previousVariance, double previousReturn, double expected)
    {
        // Hull OFOD11e Table23.1: exact cached Basic GARCH author cells,
        // independently audited against all1259 original S&P observations.
        // This supplied-state forecast checks the variance column only.
        // It does not assert Fit initialization, likelihood, mean or solver parity.
        var fit = new GarchFitResult<double>(0, 0,
            3.981826267827113e-6, 0.2237925232529377, 0.7475767721285745,
            [previousVariance], [previousReturn / Math.Sqrt(previousVariance)], 0, true);
        fit.ForecastVariance(1).Should().BeApproximately(expected, 2e-17,
            "the authenticated author variance for day{0} must match", day);
    }
}
