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

using System.Reflection;

using Boutquin.Numerics.MonteCarlo;
using Boutquin.Numerics.Random;
using Boutquin.Numerics.Statistics;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

public sealed class ArenaR1C3MathRegressionTests
{
    [Fact]
    public void Halton_CounterExhaustion_IsSticky()
    {
        var sequence = new HaltonSequence<double>(dimension: 1);
        SetPrivateField(sequence, "_index", long.MaxValue - 2);

        _ = sequence.Next();
        var firstExhaustedCall = () => sequence.Next();
        var secondExhaustedCall = () => sequence.Next();

        firstExhaustedCall.Should().Throw<InvalidOperationException>();
        secondExhaustedCall.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public async Task Sobol_SkipBeyondFinalAuthorPoint_IsRejectedImmediately()
    {
        var construction = Task.Run(
            () => Record.Exception(() => new SobolSequence<double>(dimension: 1, skip: 1L << 30)));

        var completed = await Task.WhenAny(construction, Task.Delay(TimeSpan.FromMilliseconds(250)));

        completed.Should().BeSameAs(construction,
            "an unusable maximum skip must be rejected before serial advancement");
        var exception = await construction;
        exception.Should().BeOfType<ArgumentOutOfRangeException>()
            .Which.ParamName.Should().Be("skip");
    }

    [Fact]
    public void FastDoubleBootstrap_InvalidTail_HasNoCallerOrRandomSideEffects()
    {
        const ulong seed = 734_821UL;
        var usedRandom = new Pcg64RandomSource(seed);
        var pristineRandom = new Pcg64RandomSource(seed);
        var bootstrap = new FastDoubleBootstrap(outerCount: 50, usedRandom);
        var calls = 0;

        var act = () => bootstrap.PValue(
            new decimal[] { 1m, 2m, 3m },
            observed: 2m,
            sample =>
            {
                calls++;
                return sample[0];
            },
            (PValueTail)99);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("tail");
        calls.Should().Be(0);
        usedRandom.NextULong().Should().Be(pristineRandom.NextULong());
    }

    [Fact]
    public void HaircutSharpe_NonFiniteInputs_AreRejected()
    {
        Action nonFiniteSharpe = () => HaircutSharpe<double>.Compute(
            double.PositiveInfinity, numTrials: 10, backTestYears: 1.0);
        Action nonFiniteYears = () => HaircutSharpe<double>.Compute(
            observedSharpe: 1.0, numTrials: 10, double.PositiveInfinity);

        nonFiniteSharpe.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("observedSharpe");
        nonFiniteYears.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("backTestYears");
    }

    [Fact]
    public void DeflatedSharpeRatio_NonFiniteDirectAndSeriesInputs_AreRejected()
    {
        Action direct = () => DeflatedSharpeRatio<double>.Compute(
            observedSharpe: double.NaN,
            numTrials: 10,
            backTestYears: 1.0,
            skewness: 0.0,
            kurtosis: 3.0);
        Action fromReturns = () => DeflatedSharpeRatio<double>.ComputeFromReturns(
            new double[] { 0.01, 0.02, double.NaN, 0.04 },
            numTrials: 10);

        direct.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("observedSharpe");
        fromReturns.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("returns");
    }

    [Fact]
    public void MinimumTrackRecordLength_NonFiniteFormulaInput_IsRejected()
    {
        var act = () => MinimumTrackRecordLength<double>.Compute(
            observedSharpe: double.NaN,
            benchmarkSharpe: 0.0,
            skewness: 0.0,
            kurtosis: 3.0,
            significanceLevel: 0.05,
            tradingDaysPerYear: 252);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("observedSharpe");
    }

    [Fact]
    public void NeweyWestVariance_NonFiniteObservation_IsRejected()
    {
        var act = () => NeweyWestVariance<double>.MeanVariance(
            new double[] { 1.0, double.NaN, 3.0 },
            lags: 0);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("series");
    }

    [Fact]
    public void PearsonCorrelation_NonFiniteBatchAndRollingInputs_AreRejected()
    {
        Action batch = () => PearsonCorrelation<double>.Compute(
            new double[] { 1.0, double.NaN, 3.0 },
            new double[] { 1.0, 2.0, 3.0 });
        Action rolling = () => PearsonCorrelation<double>.Rolling(
            new double[] { 1.0, 2.0, 3.0, 4.0 },
            new double[] { 1.0, double.PositiveInfinity, 3.0, 4.0 },
            windowSize: 3);

        batch.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("x");
        rolling.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("returnsB");
    }

    [Fact]
    public void RankCorrelation_SpearmanRejectsUnequalFiniteSeries()
    {
        var act = () => RankCorrelation<double>.Spearman(
            new double[] { 1.0, 2.0, 3.0 },
            new double[] { 1.0, 2.0, 3.0, 4.0 });

        act.Should().Throw<ArgumentException>().WithParameterName("y");
    }

    [Fact]
    public void Halton_CounterStateThatWouldRoundToOne_IsRejected()
    {
        var sequence = new HaltonSequence<double>(dimension: 1);
        SetPrivateField(sequence, "_index", long.MaxValue - 1);

        var act = () => sequence.Next();

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Halton_LargeLegalSkip_StaysBelowOneAcrossSupportedTypesAndFacades()
    {
        const long skip = (1L << 54) - 2;

        AssertHalfOpen(new HaltonSequence<double>(1, skip).Next()[0]);
        AssertHalfOpen(new HaltonSequence<float>(1, skip).Next()[0]);
        AssertHalfOpen(new HaltonSequence<Half>(1, skip).Next()[0]);
        AssertHalfOpen(new HaltonSequence<decimal>(1, skip).Next()[0]);
        AssertHalfOpen(new HaltonSequence(1, skip).Next()[0]);

        var destination = new double[1];
        new HaltonSequence<double>(1, skip).Next(destination);
        AssertHalfOpen(destination[0]);
    }

    [Fact]
    public void RankCorrelation_KendallRejectsUnequalFiniteSeries()
    {
        var act = () => RankCorrelation<double>.KendallTauB(
            new double[] { 1.0, 2.0, 3.0, 4.0 },
            new double[] { 1.0, 2.0, 3.0 });

        act.Should().Throw<ArgumentException>().WithParameterName("y");
    }

    [Fact]
    public void HaircutSharpe_TrialCountBeyondDocumentedCeiling_IsRejected()
    {
        var act = () => HaircutSharpe<double>.Compute(
            observedSharpe: 1.0,
            numTrials: 1_000_001,
            backTestYears: 1.0);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("numTrials");
    }

    [Fact]
    public void PearsonRolling_TwoObservationWindows_FollowBatchDegeneratePolicy()
    {
        var result = PearsonCorrelation<double>.Rolling(
            new double[] { 1.0, 2.0, 3.0 },
            new double[] { 2.0, 4.0, 6.0 },
            windowSize: 2);

        result.Should().Equal(0.0, 0.0);
    }

    private static void SetPrivateField<TValue>(object target, string fieldName, TValue value)
    {
        var field = target.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        field.Should().NotBeNull();
        field!.SetValue(target, value);
    }

    private static void AssertHalfOpen<TValue>(TValue value)
        where TValue : System.Numerics.IFloatingPoint<TValue>
    {
        Assert.True(value >= TValue.Zero && value < TValue.One,
            $"Expected [0, 1), got {value} for {typeof(TValue).Name}.");
    }
}
