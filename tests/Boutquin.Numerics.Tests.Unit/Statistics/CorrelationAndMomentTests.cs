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

public sealed class CorrelationAndMomentTests
{
    [Fact]
    public void DistanceCorrelation_TinyFiniteSelfSeriesRemainsOne()
    {
        double[] values = [0.0, 1e-10, 2e-10, 3e-10];

        DistanceCorrelation<double>.Compute(values, values).Should().BeApproximately(1.0, 1e-12);
    }

    [Fact]
    public void DistanceCorrelation_LargeFiniteSelfSeriesRemainsOne()
    {
        double[] values = [1e100, 2e100, 3e100];

        DistanceCorrelation<double>.Compute(values, values).Should().BeApproximately(1.0, 1e-12);
    }

    [Fact]
    public void DistanceCorrelation_HalfSelfSeriesAtSquaredCountBoundaryRemainsOne()
    {
        var values = Enumerable.Range(0, 256).Select(i => (Half)(i % 7)).ToArray();

        ((float)DistanceCorrelation<Half>.Compute(values, values)).Should().BeApproximately(1f, 0.02f);
    }

    [Fact]
    public void Spearman_PerfectMonotoneNonLinear_EqualsOne()
    {
        // y = x³ — strictly monotonic, but Pearson < 1.
        decimal[] x = [-3m, -2m, -1m, 0m, 1m, 2m, 3m];
        decimal[] y = [-27m, -8m, -1m, 0m, 1m, 8m, 27m];
        RankCorrelation.Spearman(x, y).Should().Be(1m);
    }

    [Fact]
    public void Spearman_PerfectMonotone_IsExactlyPlusOne()
    {
        decimal[] x = [1m, 2m, 3m, 4m, 5m];
        decimal[] y = [10m, 20m, 30m, 40m, 50m];
        Math.Abs(RankCorrelation.Spearman(x, y) - 1m).Should().BeLessThan(1e-14m);
    }

    [Fact]
    public void Spearman_PerfectAntiMonotone_IsExactlyMinusOne()
    {
        decimal[] x = [1m, 2m, 3m, 4m, 5m];
        decimal[] y = [50m, 40m, 30m, 20m, 10m];
        Math.Abs(RankCorrelation.Spearman(x, y) - (-1m)).Should().BeLessThan(1e-14m);
    }

    [Fact]
    public void Spearman_HalfLongTiedSeriesReturnsZero()
    {
        var values = new Half[65_520];

        RankCorrelation<Half>.Spearman(values, values).Should().Be((Half)0);
    }

    [Fact]
    public void KendallTauB_PerfectMonotone_IsExactlyPlusOne()
    {
        decimal[] x = [1m, 2m, 3m, 4m, 5m];
        decimal[] y = [2m, 4m, 6m, 8m, 10m];
        Math.Abs(RankCorrelation.KendallTauB(x, y) - 1m).Should().BeLessThan(1e-14m);
    }

    [Fact]
    public void KendallTauB_PerfectAntiMonotone_IsExactlyMinusOne()
    {
        decimal[] x = [1m, 2m, 3m, 4m, 5m];
        decimal[] y = [10m, 8m, 6m, 4m, 2m];
        Math.Abs(RankCorrelation.KendallTauB(x, y) - (-1m)).Should().BeLessThan(1e-14m);
    }

    [Fact]
    public void KendallTauB_HandlesTies()
    {
        decimal[] x = [1m, 2m, 2m, 3m, 4m];
        decimal[] y = [1m, 2m, 3m, 3m, 5m];
        // Both have ties; τ-b should still produce a valid value in [−1, 1].
        var tau = RankCorrelation.KendallTauB(x, y);
        tau.Should().BeGreaterThan(0m);
        tau.Should().BeLessThanOrEqualTo(1m);
    }

    [Fact]
    public void KendallTauB_ExtremeFiniteDecimalOrderDoesNotOverflow()
    {
        decimal[] x = [decimal.MinValue, 0m, decimal.MaxValue];
        decimal[] y = [decimal.MaxValue, 0m, decimal.MinValue];

        decimal genericResult = 0m;
        var genericFailure = Record.Exception(() => genericResult = RankCorrelation<decimal>.KendallTauB(x, y));
        genericFailure.Should().BeNull();
        genericResult.Should().Be(-1m);
    }

    [Fact]
    public void KendallTauB_HalfTwentyFourConcordantPairsRemainOne()
    {
        var values = Enumerable.Range(0, 24).Select(i => (Half)i).ToArray();

        RankCorrelation<Half>.KendallTauB(values, values).Should().Be((Half)1);
    }

    [Fact]
    public void KendallTauB_GuaranteedTypesLargeConcordantSeriesRemainOne()
    {
        var values = Enumerable.Range(0, 400).Select(i => (double)i).ToArray();
        var decimals = Array.ConvertAll(values, static value => (decimal)value);

        RankCorrelation<double>.KendallTauB(values, values).Should().Be(1d);
        RankCorrelation<decimal>.KendallTauB(decimals, decimals).Should().Be(1m);
    }

    [Fact]
    public void DistanceCorrelation_QuadraticDependence_MatchesReferenceValue()
    {
        // y = x² on a fixed, deterministic grid — Pearson = 0, dCor > 0.
        // Externally computed (reference Székely 2007 implementation): 0.9869160440537484.
        double[] x = [1.0, 2.0, 3.0, 4.0, 5.0];
        double[] y = [1.0, 4.0, 9.0, 16.0, 25.0];
        DistanceCorrelation.Compute(x, y).Should().BeApproximately(0.9869160440537484, 1e-6);
    }

    [Fact]
    public void Welford_MatchesNumericallyStableMeanAndVariance()
    {
        decimal[] values = [1.0000001m, 1.0000002m, 1.0000003m, 1.0000004m, 1.0000005m];
        var (mean, var) = WelfordMoments.Compute(values);

        mean.Should().BeApproximately(1.0000003m, 1e-15m);
        // Sample variance (N-1 divisor) of arithmetic progression with step 1e-7.
        // Deviations from mean: -2e-7, -1e-7, 0, 1e-7, 2e-7 → sum sq = 10e-14 → var = 10e-14 / 4 = 2.5e-14.
        ((double)var).Should().BeApproximately(2.5e-14, 1e-20);
    }

    [Fact]
    public void WelfordPearson_MatchesBatchPearson()
    {
        var rng = new System.Random(7);
        var n = 50;
        var x = new decimal[n];
        var y = new decimal[n];
        var pearson = new WelfordMoments.Pearson();
        for (var i = 0; i < n; i++)
        {
            x[i] = (decimal)rng.NextDouble();
            y[i] = (decimal)rng.NextDouble();
            pearson.Add(x[i], y[i]);
        }

        var batch = PearsonCorrelation.Compute(x, y);
        ((double)Math.Abs(pearson.Correlation - batch)).Should().BeLessThan(1e-10);
    }

    [Fact]
    public void FisherZ_ConfidenceIntervalContainsR()
    {
        // r = 0.5 with n = 30 → reasonable CI should contain 0.5.
        var (lo, hi) = FisherZTransform.ConfidenceInterval(0.5, 30, 0.95);
        lo.Should().BeLessThan(0.5);
        hi.Should().BeGreaterThan(0.5);
        lo.Should().BeGreaterThan(0.0);
        hi.Should().BeLessThan(1.0);
    }
}
