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

namespace Boutquin.Numerics.Tests.Verification;

/// <summary>
/// Independent replay of all six corrected Table 2 configurations. The paper
/// publishes neither sample seeds nor observations; its rounded simulation
/// statistics cannot serve as exact golden outputs. Exact controls come from
/// the separate NumPy covariance/kernel implementation on our published seeds.
/// </summary>
public sealed class PolitisWhite2009VerificationTests : CrossLanguageVerificationBase
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void EstimateDouble_CorrectedTable2_SeededExperimentMatchesIndependentOracle(int rowIndex)
    {
        using var doc = LoadVector("ppw2009");
        var row = doc.RootElement.GetProperty("rows")[rowIndex];
        var phi = row.GetProperty("phi").GetDouble();
        var count = row.GetProperty("length").GetInt32();
        var replications = row.GetProperty("replications").GetInt32();
        var sampler = new GaussianSampler(new Xoshiro256StarStarRandomSource(row.GetProperty("seed").GetUInt64()));
        var optimum = row.GetProperty("stationary_optimum").GetDouble();
        var ratios = new double[replications];
        var series = new double[count];
        var checkpoints = row.GetProperty("checkpoints").EnumerateArray()
            .ToDictionary(item => item.GetProperty("trial").GetInt32(), item => item.GetProperty("estimate").GetDouble());
        for (var trial = 0; trial < replications; trial++)
        {
            // Exact stationary initialization, rather than an undocumented
            // burn-in. Innovations are independent N(0,1).
            var x = sampler.Next() / Math.Sqrt(1.0 - phi * phi);
            for (var i = 0; i < count; i++)
            {
                x = phi * x + sampler.Next();
                series[i] = x;
            }

            var estimate = PolitisWhiteBlockLength.EstimateDouble(series);
            ratios[trial] = estimate / optimum;
            if (checkpoints.TryGetValue(trial, out var expected))
            {
                estimate.Should().BeApproximately(expected, 1e-9,
                    "NumPy independently evaluates the same published observation stream");
            }
        }

        var mean = ratios.Average();
        var sd = Math.Sqrt(ratios.Sum(value => Math.Pow(value - mean, 2)) / (replications - 1));
        var rmse = Math.Sqrt(ratios.Average(value => Math.Pow(value - 1.0, 2)));

        mean.Should().BeApproximately(row.GetProperty("mean").GetDouble(), 1e-9);
        sd.Should().BeApproximately(row.GetProperty("standard_deviation").GetDouble(), 1e-9);
        rmse.Should().BeApproximately(row.GetProperty("rmse").GetDouble(), 1e-9);

        // Six combined MC standard errors + printed rounding (0.0005),
        // with printed rounding. Both oracle and production follow the corrected
        // author common-row covariance and upper-only cap.
        // SD/RMSE SEs use companion fourth/influence moments, not a Gaussian
        // assumption about these heavy-tailed finite-sample estimates.
        mean.Should().BeApproximately(row.GetProperty("printed_mean").GetDouble(), row.GetProperty("mean_paper_tolerance").GetDouble());
        sd.Should().BeApproximately(row.GetProperty("printed_sd").GetDouble(), row.GetProperty("sd_paper_tolerance").GetDouble());
        rmse.Should().BeApproximately(row.GetProperty("printed_rmse").GetDouble(), row.GetProperty("rmse_paper_tolerance").GetDouble());
    }
}
