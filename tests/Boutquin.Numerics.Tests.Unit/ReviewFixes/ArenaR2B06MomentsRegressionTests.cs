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

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Round-2 code-arena fixes (batch B06-moments): the Welford overflow fallbacks must keep
/// each observation's (and each partition's) second-moment contribution, the online Pearson
/// accumulator must normalize x and y by their own magnitudes, and SampleExcessKurtosis
/// must be as scale-invariant as its SampleSkewness sibling.
/// </summary>
public sealed class ArenaR2B06MomentsRegressionTests
{
    private static readonly double[] s_kurtosisBase = [1, 2, 3, 4, 10, 2.5, 7];
    private static readonly double[] s_pearsonA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    private static readonly double[] s_pearsonB = [2, 1, 4, 3, 7, 5, 8, 6, 10, 9];

    // F-L1-02: the univariate fallback must add the observation's squared deviation. The
    // true sample variance of {-1e308, 1e308} is 2e616, which is not representable: the
    // correct double result is +Infinity, never 0 (which reads as a constant series).
    [Fact]
    public void Add_OppositeSignExtremeDoublePair_VarianceIsInfiniteNotZero()
    {
        var moments = new WelfordMoments<double>();
        moments.Add(-1e308);
        moments.Add(1e308);

        moments.Mean.Should().Be(0.0);
        moments.Variance.Should().Be(double.PositiveInfinity);
        moments.PopulationVariance.Should().Be(double.PositiveInfinity);
    }

    // F-L1-02 (decimal): the same unrepresentable variance must surface as an
    // OverflowException, as the scaled non-fallback path already does, not as 0.
    [Fact]
    public void Add_OppositeSignExtremeDecimalPair_VarianceOverflowsNotZero()
    {
        var moments = new WelfordMoments<decimal>();
        moments.Add(-7e28m);
        moments.Add(7e28m);

        moments.Mean.Should().Be(0m);
        var act = () => moments.Variance;
        act.Should().Throw<OverflowException>();
    }

    // F-L1-03: when (otherMean - mean) overflows, the merge must still add the
    // Chan-Golub-LeVeque cross term. Both partitions are constant, so the whole variance
    // (true value 4/3 * 1e616) lives in that term.
    [Fact]
    public void Merge_OppositeSignExtremePartitionMeans_VarianceIsInfiniteNotZero()
    {
        var left = new WelfordMoments<double>();
        left.Add(1e308);
        left.Add(1e308);
        var right = new WelfordMoments<double>();
        right.Add(-1e308);
        right.Add(-1e308);

        left.Merge(right);

        left.Mean.Should().Be(0.0);
        left.Variance.Should().Be(double.PositiveInfinity);
    }

    [Fact]
    public void Merge_OppositeSignExtremeDecimalPartitionMeans_VarianceOverflowsNotZero()
    {
        var left = new WelfordMoments<decimal>();
        left.Add(7e28m);
        left.Add(7e28m);
        var right = new WelfordMoments<decimal>();
        right.Add(-7e28m);
        right.Add(-7e28m);

        left.Merge(right);

        left.Mean.Should().Be(0m);
        var act = () => left.Variance;
        act.Should().Throw<OverflowException>();
    }

    // F-L1-01: the Pearson fallback must add the pair's M2 and co-moment contribution.
    // Correlation is scale-invariant and representable; the batch sibling is the oracle.
    [Fact]
    public void PearsonAdd_MixedSignExtremeDoublePairs_MatchesBatchCorrelation()
    {
        // Not collinear: dropping the overflowing pair changes the correlation, so the
        // assertion cannot pass by coincidence on the remaining points alone.
        double[] xs = [-1e308, 1e308, 0, 5e307, -5e307];
        double[] ys = [-1, 1, 0, 0.2, -0.7];
        var expected = PearsonCorrelation<double>.Compute(xs, ys);

        var online = new WelfordMoments<double>.Pearson();
        for (var i = 0; i < xs.Length; i++)
        {
            online.Add(xs[i], ys[i]);
        }

        online.Correlation.Should().BeApproximately(expected, 1e-12);
    }

    [Fact]
    public void PearsonAdd_MixedSignExtremeDecimalPairs_MatchesBatchCorrelation()
    {
        decimal[] xs = [-7e28m, 7e28m, 0m, 3e28m, -3e28m];
        decimal[] ys = [-1m, 1m, 0m, 0.5m, -0.5m];
        var expected = PearsonCorrelation<decimal>.Compute(xs, ys);

        var online = new WelfordMoments<decimal>.Pearson();
        for (var i = 0; i < xs.Length; i++)
        {
            online.Add(xs[i], ys[i]);
        }

        Math.Abs(online.Correlation - expected).Should().BeLessThan(1e-12m);
    }

    // F-L5-01: once the co-moment path switches to scaled form, x and y must each be
    // normalized by their own magnitude; a shared scale set by the larger series
    // underflows the smaller series' M2 and collapses the correlation.
    [Fact]
    public void PearsonAdd_HeterogeneousMagnitudeDoubleSeries_MatchesBatchCorrelation()
    {
        var xs = s_pearsonA.Select(a => a * 1e200).ToArray();
        var ys = s_pearsonB.ToArray();
        var expected = PearsonCorrelation<double>.Compute(xs, ys);

        var online = new WelfordMoments<double>.Pearson();
        for (var i = 0; i < xs.Length; i++)
        {
            online.Add(xs[i], ys[i]);
        }

        online.Correlation.Should().BeApproximately(expected, 1e-12);
    }

    [Fact]
    public void PearsonAdd_HeterogeneousMagnitudeDecimalSeries_MatchesBatchCorrelation()
    {
        var xs = s_pearsonA.Select(a => (decimal)a * 1e15m).ToArray();
        var ys = s_pearsonB.Select(b => (decimal)b * 0.001m).ToArray();
        var expected = PearsonCorrelation<decimal>.Compute(xs, ys);

        var online = new WelfordMoments<decimal>.Pearson();
        for (var i = 0; i < xs.Length; i++)
        {
            online.Add(xs[i], ys[i]);
        }

        Math.Abs(online.Correlation - expected).Should().BeLessThan(1e-20m);
        Math.Abs(online.Covariance - TwoPassSampleCovariance(xs, ys)).Should().BeLessThan(1e-6m);
    }

    // F-L5-02: excess kurtosis is scale-invariant; at 1e200 the unnormalized Welford
    // variance overflows to +Infinity, every z collapses to 0, and the method returned -5.4.
    [Fact]
    public void Kurtosis_LargeMagnitudeDouble_IsScaleInvariant()
    {
        var expected = SampleExcessKurtosis<double>.Compute(s_kurtosisBase);
        var scaled = s_kurtosisBase.Select(v => v * 1e200).ToArray();

        SampleExcessKurtosis<double>.Compute(scaled).Should().BeApproximately(expected, 1e-12);
    }

    [Fact]
    public void Kurtosis_LargeMagnitudeDecimal_DoesNotThrowAndIsScaleInvariant()
    {
        var baseValues = s_kurtosisBase.Select(v => (decimal)v).ToArray();
        var expected = SampleExcessKurtosis<decimal>.Compute(baseValues);
        var scaled = baseValues.Select(v => v * 1e15m).ToArray();

        var act = () => SampleExcessKurtosis<decimal>.Compute(scaled);

        act.Should().NotThrow();
        Math.Abs(act() - expected).Should().BeLessThan(1e-20m);
    }

    // F-L23-01: decimal observations clustered at ~1e20 (±5% jitter) — the sibling
    // SampleSkewness handles this magnitude; kurtosis threw OverflowException.
    [Fact]
    public void Kurtosis_DecimalClusteredNear1e20_MatchesUnitMagnitudeResult()
    {
        decimal[] jitter = [0.01m, -0.03m, 0.05m, -0.02m, 0.04m, -0.05m, 0.00m, 0.03m, -0.01m, 0.02m];
        var unit = jitter.Select(j => 1m + j).ToArray();
        var large = unit.Select(u => u * 1e20m).ToArray();
        var expected = SampleExcessKurtosis<decimal>.Compute(unit);

        var act = () => SampleExcessKurtosis<decimal>.Compute(large);

        act.Should().NotThrow();
        Math.Abs(act() - expected).Should().BeLessThan(1e-20m);
    }

    // Independent two-pass sample covariance (N-1 divisor) in decimal: the textbook
    // definition, not the Welford recurrence under test.
    private static decimal TwoPassSampleCovariance(decimal[] xs, decimal[] ys)
    {
        var mx = xs.Sum() / xs.Length;
        var my = ys.Sum() / ys.Length;
        var sum = 0m;
        for (var i = 0; i < xs.Length; i++)
        {
            sum += (xs[i] - mx) * (ys[i] - my);
        }

        return sum / (xs.Length - 1);
    }
}
