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
    public void Spearman_PerfectMonotoneNonLinear_EqualsOne()
    {
        // y = x³ — strictly monotonic, but Pearson < 1.
        decimal[] x = [-3m, -2m, -1m, 0m, 1m, 2m, 3m];
        decimal[] y = [-27m, -8m, -1m, 0m, 1m, 8m, 27m];
        RankCorrelation.Spearman(x, y).Should().Be(1m);
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
    public void DistanceCorrelation_DetectsQuadraticDependence()
    {
        // y = x² with x ~ U(-1, 1) — Pearson = 0 but dCor > 0.
        var rng = new System.Random(42);
        var n = 400;
        var x = new double[n];
        var y = new double[n];
        for (var i = 0; i < n; i++)
        {
            x[i] = rng.NextDouble() * 2.0 - 1.0;
            y[i] = x[i] * x[i];
        }

        var dcor = DistanceCorrelation.Compute(x, y);
        dcor.Should().BeGreaterThan(0.3, "distance correlation should detect quadratic dependence");
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
