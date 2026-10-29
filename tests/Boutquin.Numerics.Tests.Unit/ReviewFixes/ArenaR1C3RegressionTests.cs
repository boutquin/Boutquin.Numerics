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
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

public sealed class ArenaR1C3RegressionTests
{
    [Fact]
    public void ConditionalTailMatrix_PreservesBothDirectionalEstimates()
    {
        double[] x = [1.0, 1.0, 1.0, 2.0];
        double[] y = [1.0, 2.0, 3.0, 4.0];
        var options = new TailOptions<double>(Quantiles: [0.5]);

        var matrix = LowerTailDependence<double>.EstimateMatrix([x, y], options);

        matrix[0, 1].Should().BeApproximately(2.0 / 3.0, 1e-12);
        matrix[1, 0].Should().Be(1.0);
    }

    [Theory]
    [InlineData(-1.2)]
    [InlineData(1.2)]
    public void FisherConfidenceInterval_RejectsFiniteCorrelationsOutsideUnitRange(double r)
    {
        var act = () => FisherZTransform<double>.ConfidenceInterval(r, 20);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(r));
    }

    [Theory]
    [InlineData(-2.0)]
    [InlineData(2.0)]
    [InlineData(double.NaN)]
    public void FisherForward_RejectsInvalidCorrelations(double r)
    {
        var act = () => FisherZTransform<double>.Forward(r);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(r));
    }

    [Fact]
    public void NeweyWestAutomaticLags_ComposesWithOnePointEstimator()
    {
        var lags = NeweyWestVariance<decimal>.AutomaticLags(1);

        lags.Should().Be(0);
        NeweyWestVariance<decimal>.MeanVariance([0.1m], lags).Should().Be(0m);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void NeweyWestAutomaticLags_RejectsNonPositiveSampleLengths(int t)
    {
        var act = () => NeweyWestVariance<decimal>.AutomaticLags(t);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(t));
    }

    [Fact]
    public void PolitisWhite_NondegenerateSmallScalePreservesBlockLength()
    {
        var series = new double[256];
        var rng = new System.Random(54921);
        var previous = 0.0;
        for (var i = 0; i < series.Length; i++)
        {
            previous = 0.5 * previous + 2.0 * rng.NextDouble() - 1.0;
            series[i] = previous;
        }

        var expected = PolitisWhiteBlockLength.EstimateDouble(series);
        var scaled = series.Select(value => value * 1e-6).ToArray();
        var actual = PolitisWhiteBlockLength.EstimateDouble(scaled);

        expected.Should().BeGreaterThan(1.0);
        actual.Should().BeApproximately(expected, 1e-9);
    }

    [Fact]
    public void PolitisWhite_NondegenerateLargeScalePreservesBlockLength()
    {
        var series = new double[256];
        var rng = new System.Random(54921);
        var previous = 0.0;
        for (var i = 0; i < series.Length; i++)
        {
            previous = 0.5 * previous + 2.0 * rng.NextDouble() - 1.0;
            series[i] = previous;
        }

        var expected = PolitisWhiteBlockLength.EstimateDouble(series);
        var scaled = series.Select(value => value * 1e160).ToArray();

        expected.Should().BeGreaterThan(1.0);
        PolitisWhiteBlockLength.EstimateDouble(scaled).Should().BeApproximately(expected, 1e-9);
    }

    [Fact]
    public void PboResult_SnapshotsEveryInputAndDoesNotExposeAnArray()
    {
        double[] genericInput = [-2.0, 0.0, 2.0];
        var generic = new PboResult<double>(0.25, 0.0, genericInput);
        genericInput[0] = 99.0;

        generic.LogitValues.Should().Equal(-2.0, 0.0, 2.0);
        generic.LogitValues.Should().NotBeOfType<double[]>();

        double[] replacement = [-3.0, 3.0];
        var replaced = generic with { LogitValues = replacement };
        replacement[0] = 99.0;

        replaced.LogitValues.Should().Equal(-3.0, 3.0);
        replaced.LogitValues.Should().NotBeOfType<double[]>();

        decimal[] facadeInput = [-1m, 1m];
        var facade = new PboResult(0.5m, 0m, facadeInput);
        facadeInput[0] = 99m;

        facade.LogitValues.Should().Equal(-1m, 1m);
        facade.LogitValues.Should().NotBeOfType<decimal[]>();
    }

    [Fact]
    public void TailResult_SnapshotsEveryInputAndReturnsFreshArrays()
    {
        double[] thresholds = [0.05, 0.10];
        int[] jointCounts = [2, 4];
        var result = new TailResult<double>(0.4, thresholds, jointCounts);

        thresholds[0] = 99.0;
        jointCounts[0] = 99;
        result.Threshold.Should().Equal(0.05, 0.10);
        result.JointCount.Should().Equal(2, 4);

        var exposedThresholds = result.Threshold;
        var exposedCounts = result.JointCount;
        exposedThresholds[0] = 88.0;
        exposedCounts[0] = 88;
        result.Threshold.Should().Equal(0.05, 0.10);
        result.JointCount.Should().Equal(2, 4);

        double[] replacementThresholds = [0.15, 0.20];
        int[] replacementCounts = [6, 8];
        var replaced = result with
        {
            Threshold = replacementThresholds,
            JointCount = replacementCounts,
        };
        replacementThresholds[0] = 77.0;
        replacementCounts[0] = 77;

        replaced.Threshold.Should().Equal(0.15, 0.20);
        replaced.JointCount.Should().Equal(6, 8);
    }
}
