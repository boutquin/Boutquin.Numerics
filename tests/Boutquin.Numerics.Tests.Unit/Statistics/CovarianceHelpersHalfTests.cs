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

/// <summary>
/// Half-precision reductions in <c>CovarianceHelpers</c>. Every input and the exact
/// result below are representable in <see cref="Half"/>; only an intermediate sum is
/// not, so a correct reduction must not manufacture a NaN or depend on row order.
/// </summary>
public sealed class CovarianceHelpersHalfTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(699)]
    public void SampleCovariance_Half_OutlierRowDoesNotOverflowMean(int outlierRow)
    {
        // 700 observations, one of them 100, the rest 0. Exact mean 1/7 and exact
        // Bessel variance (10000 - 700/49) / 699 = 14.2857…. With the outlier in row 0
        // an anchored deviation sum reaches 699 * -100, beyond Half.MaxValue (65504).
        const int t = 700;
        var returns = new Half[t, 1];
        for (var i = 0; i < t; i++)
        {
            returns[i, 0] = Half.Zero;
        }

        returns[outlierRow, 0] = (Half)100;

        var exactMean = 100.0 / t;
        var exactVariance = (10000.0 - (t * exactMean * exactMean)) / (t - 1);

        var mean = (double)CovarianceHelpers<Half>.ComputeMeans(returns)[0];
        var variance = (double)new SampleCovarianceEstimator<Half>().Estimate(returns)[0, 0];

        mean.Should().BeApproximately(exactMean, 1e-3);
        variance.Should().BeApproximately(exactVariance, 0.05);
    }
}
