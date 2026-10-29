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

using System.Numerics;
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

public sealed class ReleaseQualificationStatisticsTests
{
    [Fact]
    public void Dsr_ZeroMomentVarianceUsesSameGaussianFallbackAsNegativeVariance()
    {
        AssertDsr<double>();
        AssertDsr<decimal>();
    }

    private static void AssertDsr<T>() where T : IFloatingPoint<T>
    {
        var result = DeflatedSharpeRatio<T>.Compute(T.CreateChecked(2), 1, T.CreateChecked(100), T.One, T.CreateChecked(2), 1);
        result.PValue.Should().Be(T.Zero);
        var above = DeflatedSharpeRatio<T>.ComputeWithTrialMoments(T.CreateChecked(2), 1,
            T.CreateChecked(3), T.Zero, T.CreateChecked(100), T.One, T.CreateChecked(2), 1);
        above.PValue.Should().Be(T.One);
    }

    [Fact]
    public void NeweyWestAutomatic_DividesBeforeRestoringUnrepresentableLongRunVariance()
    {
        var series = LargeSmoothSeries();
        var scaled = series.Select(value => value / 1e7m).ToArray();
        var expected = NeweyWestVariance<decimal>.MeanVarianceAutomatic(scaled) * 1e14m;
        expected.Should().BeInRange(1e28m, 2e28m);
        NeweyWestVariance<decimal>.MeanVarianceAutomatic(series)
            .Should().BeApproximately(expected, expected * 1e-14m);
    }

    [Fact]
    public void NeweyWestLongRun_UnrepresentableCovarianceUsesDocumentedRefusal()
    {
        var series = LargeSmoothSeries();
        var instruments = new decimal[series.Length, 1];
        for (var i = 0; i < series.Length; i++)
        {
            instruments[i, 0] = 1m;
        }
        var act = () => NeweyWestVariance<decimal>.LongRunCovariance(instruments, series, [1m]);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void NeweyWestLongRun_DummyVariableTrapIsRefusedAtBothGuaranteedTypes()
    {
        AssertRankRefusal<double>();
        AssertRankRefusal<decimal>();
    }

    private static void AssertRankRefusal<T>() where T : IFloatingPoint<T>
    {
        var instruments = new T[80, 3];
        var residuals = new T[80];
        for (var i = 0; i < residuals.Length; i++)
        {
            instruments[i, 0] = T.One;
            instruments[i, 1] = T.CreateChecked(i % 2);
            instruments[i, 2] = T.One - instruments[i, 1];
            residuals[i] = T.CreateChecked((i % 7 - 3) / 10m + (i % 3) / 20m);
        }
        var act = () => NeweyWestVariance<T>.LongRunCovariance(instruments, residuals, [T.One, T.One, T.One]);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Pbo_DecimalUnrepresentableSharpeStillRanksAndReturnsFiniteResult()
    {
        var panel = new decimal[2, 512];
        for (var i = 0; i < 512; i++)
        {
            // Every value <=1e14. The first strategy's Sharpe exceeds decimal.MaxValue
            // but it is only a ranking key, and its rank is unambiguously best.
            panel[0, i] = 9e13m + (i % 256 == 0 ? 1e-14m : 0m);
            panel[1, i] = i % 2 + 1;
        }
        var result = ProbabilityOfBacktestOverfitting<decimal>.Compute(panel, 2);
        result.Pbo.Should().Be(0m);
        ((double)result.LogitMedian).Should().BeApproximately(Math.Log(2), 1e-12);
    }

    private static decimal[] LargeSmoothSeries() => Enumerable.Range(0, 200)
        .Select(i => decimal.Round((decimal)(1e14 * (.9 * Math.Sin(.15 * i) + .1 * Math.Cos(1.7 * i)))))
        .ToArray();
}
