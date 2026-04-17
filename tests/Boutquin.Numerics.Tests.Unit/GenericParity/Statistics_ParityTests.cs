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

namespace Boutquin.Numerics.Tests.Unit.GenericParity;

/// <summary>
/// Parity tests verifying that the generic Statistics types instantiated at
/// <c>T = decimal</c> produce identical results to the pre-migration legacy
/// concrete-typed implementations.
/// </summary>
public sealed class Statistics_ParityTests
{
    private static readonly decimal[] s_series = { 1.5m, 2.3m, 3.1m, 4.7m, 5.9m, 6.2m, 7.8m, 8.4m, 9.1m, 10.0m };

    [Fact]
    public void WelfordMoments_GenericMatchesLegacy_AtDecimal()
    {
        var legacy = new WelfordMoments();
        var generic = new WelfordMoments<decimal>();

        foreach (var v in s_series)
        {
            legacy.Add(v);
            generic.Add(v);
        }

        generic.Mean.Should().Be(legacy.Mean, because: "generic WelfordMoments mean must match legacy");
        generic.Variance.Should().Be(legacy.Variance, because: "generic WelfordMoments variance must match legacy");
        generic.StdDev.Should().Be(legacy.StdDev, because: "generic WelfordMoments stddev must match legacy");
        generic.Count.Should().Be(legacy.Count);
    }

    [Fact]
    public void SampleCovarianceEstimator_GenericMatchesLegacy_AtDecimal()
    {
        // 5 observations, 2 assets
        var returns = new decimal[,]
        {
            { 0.01m, 0.02m },
            { -0.01m, 0.01m },
            { 0.02m, -0.01m },
            { 0.005m, 0.015m },
            { -0.005m, 0.005m },
        };

        var legacyCov = new SampleCovarianceEstimator().Estimate(returns);
        var genericCov = new SampleCovarianceEstimator<decimal>().Estimate(returns);

        var n = legacyCov.GetLength(0);
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                genericCov[i, j].Should().Be(legacyCov[i, j],
                    because: $"generic SampleCov[{i},{j}] must match legacy");
            }
        }
    }

    [Fact]
    public void PearsonCorrelation_GenericMatchesLegacy_AtDecimal()
    {
        var x = new decimal[] { 1m, 2m, 3m, 4m, 5m };
        var y = new decimal[] { 2m, 4m, 5m, 4m, 5m };

        var legacy = PearsonCorrelation.Compute(x.AsSpan(), y.AsSpan());
        var generic = PearsonCorrelation<decimal>.Compute(x.AsSpan(), y.AsSpan());

        generic.Should().Be(legacy, because: "generic Pearson must match legacy");
    }

    [Fact]
    public void RankCorrelation_Spearman_GenericMatchesLegacy_AtDecimal()
    {
        var x = new decimal[] { 1m, 2m, 3m, 4m, 5m };
        var y = new decimal[] { 5m, 6m, 7m, 8m, 7m };

        var legacy = RankCorrelation.Spearman(x.AsSpan(), y.AsSpan());
        var generic = RankCorrelation<decimal>.Spearman(x.AsSpan(), y.AsSpan());

        generic.Should().Be(legacy, because: "generic Spearman must match legacy");
    }
}
