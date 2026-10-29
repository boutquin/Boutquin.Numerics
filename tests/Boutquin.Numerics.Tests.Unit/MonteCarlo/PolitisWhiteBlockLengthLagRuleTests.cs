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

/// <summary>
/// Pins the lag rule and the block-length cap of <see cref="PolitisWhiteBlockLength.EstimateDouble"/>
/// to Politis &amp; White (2004) and Patton's reference code
/// (<c>opt_block_length_REV_dec07.m</c>):
/// <list type="bullet">
/// <item><description><c>m̂</c> is the smallest <c>m</c> with <c>|ρ̂(m+k)| &lt; 2·√(log₁₀N/N)</c>
/// for <c>k = 0..K_N−1</c> in the corrected companion selector, where
/// <c>K_N = max(5, √log₁₀N)</c> — five lags for any practical N;</description></item>
/// <item><description>the block length is capped at <c>B_max = ⌈min(3√N, N/3)⌉</c>.</description></item>
/// </list>
/// </summary>
public sealed class PolitisWhiteBlockLengthLagRuleTests
{
    [Fact]
    public void EstimateDouble_WindowIsFiveLags_EvenWhenLog10NExceedsFive()
    {
        // x_t = e_t + 0.5·e_{t−1} + 0.3·(f_t + f_{t−7}) with independent Gaussian e, f:
        // ρ(1) ≈ 0.35, ρ(7) ≈ 0.063, ρ(2..6) = 0. At N = 200 000 the band is ≈ 0.0103, so
        // lags 2..6 are insignificant: K_N = 5 gives m̂ = 2 and M = 4. A six-lag window
        // (⌈log₁₀N⌉ = 6) would reach lag 7 and push m̂ to 8.
        const int n = 200_000;
        var sampler = new GaussianSampler(new Xoshiro256StarStarRandomSource(20260930UL));
        var e = new double[n + 7];
        var f = new double[n + 7];
        for (var i = 0; i < n + 7; i++)
        {
            e[i] = sampler.Next();
            f[i] = sampler.Next();
        }

        var x = new double[n];
        for (var t = 0; t < n; t++)
        {
            x[t] = e[t + 7] + (0.5 * e[t + 6]) + (0.3 * (f[t + 7] + f[t]));
        }

        // Fixed M4: evaluate the author's truncated-column sample covariances.
        var response = x.Skip(4).ToArray();
        var leftMean = response.Average();
        var covariances = new double[4];
        for (var lag = 0; lag <= 3; lag++)
        {
            var shifted = x.Skip(4 - lag).Take(n - 4).ToArray();
            var rightMean = shifted.Average();
            covariances[lag] = response.Zip(shifted,
                (left, right) => (left - leftMean) * (right - rightMean)).Sum() / (n - 5);
        }

        var g = 2.0 * covariances[1] + 4.0 * covariances[2] + 3.0 * covariances[3];
        var spectrum = covariances[0] + 2.0 * covariances[1] + 2.0 * covariances[2] + covariances[3];
        var expected = Math.Cbrt(n * g * g / (spectrum * spectrum));

        var actual = PolitisWhiteBlockLength.EstimateDouble(x);

        actual.Should().BeApproximately(expected, 1e-9 * expected,
            "K_N = max(5, √log₁₀N) is five lags, so the first insignificant window is lags 2..6 and M = 4");
    }

    [Fact]
    public void EstimateDouble_NearUnitRootSeries_IsCappedAtPattonBmax()
    {
        // A Gaussian random walk is significantly autocorrelated at every lag, so M
        // reaches the search horizon and the raw estimate exceeds 3√N at this length.
        const int n = 100_000;
        var sampler = new GaussianSampler(new Xoshiro256StarStarRandomSource(20261001UL));
        var x = new double[n];
        var level = 0.0;
        for (var t = 0; t < n; t++)
        {
            level += sampler.Next();
            x[t] = level;
        }

        var bMax = Math.Ceiling(Math.Min(3.0 * Math.Sqrt(n), n / 3.0));

        // The uncapped estimate here is ≈ 1156 (> B_max = 949), so the cap must bind exactly.
        PolitisWhiteBlockLength.EstimateDouble(x).Should().Be(bMax,
            "Patton's code caps the block length at ⌈min(3√N, N/3)⌉");
    }
}
