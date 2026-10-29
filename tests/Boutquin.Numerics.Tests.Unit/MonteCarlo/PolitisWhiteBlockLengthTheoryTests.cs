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
/// Checks <see cref="PolitisWhiteBlockLength.EstimateDouble"/> against the
/// theoretical stationary-bootstrap optimum for an AR(1) process, the oracle
/// Patton, Politis &amp; White (2009) tabulate. For
/// <c>x_t = φ·x_{t−1} + ε_t</c> with autocovariances <c>R(k) = σ²·φ^|k|</c>:
/// <list type="bullet">
/// <item><description><c>G = Σ_k |k|·R(k) = 2σ²·φ/(1 − φ)²</c> (signed autocovariances, PW 2004 eq. 8);</description></item>
/// <item><description><c>g(0) = Σ_k R(k) = σ²·(1 + φ)/(1 − φ)</c>;</description></item>
/// <item><description><c>D_SB = 2·g(0)²</c> (PPW 2009);</description></item>
/// <item><description><c>b_opt = (2G²/D_SB · N)^{1/3} = (2|φ|/(1 − φ²))^{2/3}·N^{1/3}</c>.</description></item>
/// </list>
/// The flat-top estimator truncates the kernel at a data-driven lag, so the
/// finite-sample mean sits a few percent from the optimum; the bands below allow
/// for that while still rejecting an unsigned <c>G</c> (which inflates the
/// negative-φ estimate about threefold) and a missing factor of 2 in
/// <c>D_SB</c> (which inflates every estimate by <c>2^{1/3} ≈ 1.26</c>).
/// </summary>
public sealed class PolitisWhiteBlockLengthTheoryTests
{
    private const int Length = 5000;
    private const int Replications = 40;

    [Fact]
    public void EstimateDouble_PositiveAr1_MeanTracksStationaryBootstrapOptimum()
    {
        // φ = 0.5: every autocorrelation is positive, so this case isolates D_SB.
        var ratio = MeanEstimate(0.5, seed: 20260927UL) / TheoreticalOptimum(0.5, Length);

        ratio.Should().BeInRange(0.85, 1.10,
            "the mean block length must track (2|φ|/(1−φ²))^(2/3)·N^(1/3) with D_SB = 2·g(0)²");
    }

    [Fact]
    public void EstimateDouble_NegativeAr1_MeanTracksStationaryBootstrapOptimum()
    {
        // φ = −0.4: odd-lag autocorrelations are negative, so G must use signed
        // autocovariances for the lag contributions to partly cancel.
        var ratio = MeanEstimate(-0.4, seed: 20260928UL) / TheoreticalOptimum(-0.4, Length);

        ratio.Should().BeInRange(0.85, 1.15,
            "the mean block length must track the AR(1) optimum, which depends on |φ| only");
    }

    [Fact]
    public void EstimateDouble_PersistentAr1_LagWindowIsNotTruncatedBelowPaperHorizon()
    {
        // φ = 0.95: autocorrelations stay significant past lag 19, so the lag window
        // reaches the search horizon. Patton's horizon ⌈√N⌉ + K_N (76 at N = 5000)
        // lets M track the optimum; a ⌈5·log₁₀N⌉ = 19 cap halves the estimate.
        var ratio = MeanEstimate(0.95, seed: 20260929UL) / TheoreticalOptimum(0.95, Length);

        ratio.Should().BeInRange(0.75, 1.10,
            "the search horizon ⌈√N⌉ + K_N must not truncate the lag window of a persistent series");
    }

    private static double TheoreticalOptimum(double phi, int n)
        => Math.Pow(2.0 * Math.Abs(phi) / (1.0 - phi * phi), 2.0 / 3.0) * Math.Cbrt(n);

    private static double MeanEstimate(double phi, ulong seed)
    {
        var sampler = new GaussianSampler(new Xoshiro256StarStarRandomSource(seed));
        const int burnIn = 200;
        var series = new double[Length];
        var sum = 0.0;
        for (var r = 0; r < Replications; r++)
        {
            var x = 0.0;
            for (var t = 0; t < burnIn + Length; t++)
            {
                x = phi * x + sampler.Next();
                if (t >= burnIn)
                {
                    series[t - burnIn] = x;
                }
            }

            sum += PolitisWhiteBlockLength.EstimateDouble(series);
        }

        return sum / Replications;
    }
}
