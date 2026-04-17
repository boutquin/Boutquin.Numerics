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

public sealed class StatisticsPhase3_ParityTests
{
    private static readonly decimal[] s_seriesDecimal = { 1.5m, 2.3m, 3.1m, 4.7m, 5.9m, 6.2m, 7.8m, 8.4m, 9.1m, 10.0m };
    private static readonly double[] s_seriesDouble = { 1.5, 2.3, 3.1, 4.7, 5.9, 6.2, 7.8, 8.4, 9.1, 10.0 };
    private static readonly double[] s_x = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    private static readonly double[] s_y = { 2.1, 4.3, 5.8, 8.2, 10.5, 12.1, 14.7, 16.9, 19.2, 21.5 };

    [Fact]
    public void NeweyWestVariance_GenericMatchesLegacy_AtDecimal()
    {
        var maxLag = 3;

        var legacy = NeweyWestVariance.MeanVariance(s_seriesDecimal, maxLag);
        var generic = NeweyWestVariance<decimal>.MeanVariance(s_seriesDecimal, maxLag);

        generic.Should().Be(legacy);
    }

    [Fact]
    public void FisherZTransform_Forward_GenericMatchesLegacy_AtDouble()
    {
        var r = 0.5;

        var legacy = FisherZTransform.Forward(r);
        var generic = FisherZTransform<double>.Forward(r);

        generic.Should().BeApproximately(legacy, 1e-12);
    }

    [Fact]
    public void FisherZTransform_Inverse_GenericMatchesLegacy_AtDouble()
    {
        var z = 0.5493061443340548;

        var legacy = FisherZTransform.Inverse(z);
        var generic = FisherZTransform<double>.Inverse(z);

        generic.Should().BeApproximately(legacy, 1e-12);
    }

    [Fact]
    public void DistanceCorrelation_GenericMatchesLegacy_AtDouble()
    {
        var legacy = DistanceCorrelation.Compute(s_x.AsSpan(), s_y.AsSpan());
        var generic = DistanceCorrelation<double>.Compute(s_x.AsSpan(), s_y.AsSpan());

        generic.Should().BeApproximately(legacy, 1e-12);
    }

    [Fact]
    public void MinimumTrackRecordLength_GenericMatchesLegacy_AtDouble()
    {
        var observedSharpe = 1.5;
        var targetSharpe = 1.0;
        var skewness = 0.0;
        var kurtosis = 3.0;
        var alpha = 0.05;
        var tradingDaysPerYear = 252;

        var legacy = MinimumTrackRecordLength.Compute(observedSharpe, targetSharpe, skewness, kurtosis, alpha, tradingDaysPerYear);
        var generic = MinimumTrackRecordLength<double>.Compute(observedSharpe, targetSharpe, skewness, kurtosis, alpha, tradingDaysPerYear);

        generic.Should().BeApproximately(legacy, 1e-12);
    }

    [Fact]
    public void DeflatedSharpeRatio_GenericMatchesLegacy_AtDecimal()
    {
        var observedSharpe = 1.5m;
        var numTrials = 10;
        var backTestYears = 5.0m;
        var skewness = 0.0m;
        var kurtosis = 3.0m;
        var tradingDaysPerYear = 252;

        var legacy = DeflatedSharpeRatio.Compute(observedSharpe, numTrials, backTestYears, skewness, kurtosis, tradingDaysPerYear);
        var generic = DeflatedSharpeRatio<decimal>.Compute(observedSharpe, numTrials, backTestYears, skewness, kurtosis, tradingDaysPerYear);

        generic.DeflatedSharpe.Should().Be(legacy.DeflatedSharpe);
        generic.PValue.Should().Be(legacy.PValue);
        generic.ExpectedMaxSharpe.Should().Be(legacy.ExpectedMaxSharpe);
    }

    [Fact]
    public void HaircutSharpe_GenericMatchesLegacy_AtDecimal()
    {
        var observedSharpe = 1.5m;
        var numTrials = 10;
        var backTestYears = 5.0m;
        var tradingDaysPerYear = 252;

        var legacy = HaircutSharpe.Compute(observedSharpe, numTrials, backTestYears, tradingDaysPerYear);
        var generic = HaircutSharpe<decimal>.Compute(observedSharpe, numTrials, backTestYears, tradingDaysPerYear);

        generic.HaircutSharpe.Should().Be(legacy.HaircutSharpe);
        generic.ObservedSharpe.Should().Be(legacy.ObservedSharpe);
        generic.HaircutAmount.Should().Be(legacy.HaircutAmount);
        generic.BonferroniPValue.Should().Be(legacy.BonferroniPValue);
        generic.HolmPValue.Should().Be(legacy.HolmPValue);
        generic.BhyPValue.Should().Be(legacy.BhyPValue);
    }

    [Fact]
    public void GeneralizationScore_GenericMatchesLegacy_AtDouble()
    {
        var subPeriodCount = 2;
        var tradingDaysPerYear = 252;
        var pW = 0.25;
        var sW = 0.25;
        var cW = 0.25;
        var dW = 0.25;

        var legacy = GeneralizationScore.Compute(s_seriesDouble.AsSpan(), subPeriodCount, tradingDaysPerYear, pW, sW, cW, dW);
        var generic = GeneralizationScore<double>.Compute(s_seriesDouble.AsSpan(), subPeriodCount, tradingDaysPerYear, pW, sW, cW, dW);

        generic.Score.Should().BeApproximately(legacy.Score, 1e-12);
        generic.PerformanceComponent.Should().BeApproximately(legacy.PerformanceComponent, 1e-12);
        generic.SignificanceComponent.Should().BeApproximately(legacy.SignificanceComponent, 1e-12);
        generic.ConsistencyComponent.Should().BeApproximately(legacy.ConsistencyComponent, 1e-12);
        generic.DownsideRiskComponent.Should().BeApproximately(legacy.DownsideRiskComponent, 1e-12);
    }
}
