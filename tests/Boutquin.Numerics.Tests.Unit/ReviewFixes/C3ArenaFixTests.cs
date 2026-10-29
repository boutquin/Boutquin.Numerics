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
using Boutquin.Numerics.Statistics;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

public sealed class C3ArenaFixTests
{
    [Fact]
    public void Halton_MaximumSkip_IsRejectedBeforeCounterOverflow()
    {
        var act = () => new HaltonSequence<double>(dimension: 1, skip: long.MaxValue);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("skip");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void FisherConfidenceInterval_NonFiniteCorrelation_IsRejected(double r)
    {
        var act = () => FisherZTransform<double>.ConfidenceInterval(r, 20);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(r));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void FisherConfidenceInterval_NonFiniteConfidenceLevel_IsRejected(double confidenceLevel)
    {
        var act = () => FisherZTransform<double>.ConfidenceInterval(0.5, 20, confidenceLevel);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(confidenceLevel));
    }

    [Theory]
    [InlineData(true, true)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(false, false)]
    public void RankCorrelation_NonFiniteObservations_AreRejected(bool spearman, bool invalidInX)
    {
        double[] x = [1.0, 2.0, 3.0, 4.0];
        double[] y = [1.0, 2.0, 3.0, 4.0];
        (invalidInX ? x : y)[2] = double.NaN;

        Action act = spearman
            ? () => RankCorrelation<double>.Spearman(x, y)
            : () => RankCorrelation<double>.KendallTauB(x, y);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(invalidInX ? "x" : "y");
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void RankCorrelation_NonFiniteTrailingObservation_IsRejected(bool spearman)
    {
        double[] x = [1.0, 2.0, 3.0, double.NaN];
        double[] y = [1.0, 2.0, 3.0];

        Action act = spearman
            ? () => RankCorrelation<double>.Spearman(x, y)
            : () => RankCorrelation<double>.KendallTauB(x, y);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("x");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void Pbo_NonFinitePanelEntry_IsRejected(double invalid)
    {
        double[,] returns =
        {
            { 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08 },
            { 0.02, 0.01, 0.04, 0.03, 0.06, 0.05, invalid, 0.07 },
        };

        var act = () => ProbabilityOfBacktestOverfitting<double>.Compute(returns, splitCount: 4);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("returns");
    }

    [Fact]
    public void DsrDirectCompute_NonPositiveTradingDays_IsRejected()
    {
        var act = () => DeflatedSharpeRatio<double>.Compute(
            observedSharpe: 1.0,
            numTrials: 10,
            backTestYears: 1.0,
            skewness: 0.0,
            kurtosis: 3.0,
            tradingDaysPerYear: 0);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("tradingDaysPerYear");
    }

    [Fact]
    public void DsrDirectCompute_UsesAuthoritativeObservationCountDenominator()
    {
        var result = DeflatedSharpeRatio<double>.Compute(
            observedSharpe: 2.0,
            numTrials: 10,
            backTestYears: 1.0,
            skewness: 0.0,
            kurtosis: 3.0,
            tradingDaysPerYear: 252);

        // BLdP (2014) eqs. 1–2 in SciPy, per-period SR = 2/√252, T = 252,
        // null trial variance 1/(T−1): SE uses √(T−1), as the paper's eq. 2 does.
        result.PValue.Should().BeApproximately(0.3373278190124802, 1e-10);
        result.PValue.Should().NotBeApproximately(0.33702271755098334, 1e-10, "that value divides the SR variance by T instead of the paper's T-1");
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void DistanceCorrelation_NonFiniteObservation_IsRejected(bool invalidInX)
    {
        double[] x = [1.0, 2.0, 3.0];
        double[] y = [2.0, 4.0, 6.0];
        (invalidInX ? x : y)[1] = double.NaN;

        var act = () => DistanceCorrelation<double>.Compute(x, y);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName(invalidInX ? "x" : "y");
    }

    [Fact]
    public void PolitisWhite_NonFiniteObservation_IsRejected()
    {
        var series = Enumerable.Range(0, 32).Select(static i => (double)i).ToArray();
        series[17] = double.NaN;

        var act = () => PolitisWhiteBlockLength.EstimateDouble(series);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("series");
    }

    [Fact]
    public void RunMulti_ResultOwnsNamesAndEveryExposedCollection()
    {
        var names = new List<string> { "mean" };
        var result = BootstrapMonteCarloEngine<double>.FromSeed(simulationCount: 10, seed: 17)
            .RunMulti(
                new double[] { 1.0, 2.0, 3.0 },
                statisticCount: 1,
                static (sample, output) => output[0] = sample[0],
                names);

        var originalName = result.Names[0];
        var originalStatistic = result.Statistics[0][0];
        var originalMean = result.Means[0];
        names[0] = "mutated";
        result.Statistics[0][0] = -999.0;
        var mutateMean = () => ((IList<double>)result.Means)[0] = -999.0;

        result.Names[0].Should().Be(originalName);
        result.Statistics[0][0].Should().Be(originalStatistic);
        mutateMean.Should().Throw<NotSupportedException>();
        result.Means[0].Should().Be(originalMean);
    }

    [Fact]
    public void PearsonCompute_MismatchedLengths_IsRejected()
    {
        var act = () => PearsonCorrelation<double>.Compute(
            new double[] { 1.0, 2.0, 3.0 },
            new double[] { 2.0, 4.0, 6.0, 8.0 });

        act.Should().Throw<ArgumentException>().WithParameterName("y");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void MinimumTrackRecordLength_NonFiniteSignificance_IsRejected(double significanceLevel)
    {
        var act = () => MinimumTrackRecordLength<double>.Compute(
            observedSharpe: 1.0,
            benchmarkSharpe: 0.0,
            skewness: 0.0,
            kurtosis: 3.0,
            significanceLevel,
            tradingDaysPerYear: 252);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(significanceLevel));
    }

    [Fact]
    public void Sobol_ExhaustionIsSticky()
    {
        var sequence = new SobolSequence<double>(dimension: 1, skip: (1L << 30) - 1);
        sequence.Next(); // Emit the final valid author point before exhaustion.

        var first = () => sequence.Next();
        var second = () => sequence.Next();

        first.Should().Throw<InvalidOperationException>();
        second.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void PSquare_LongCountBoundary_DoesNotOverflowMarkerPositions()
    {
        var estimator = new PSquareEstimator<double>(0.5);
        foreach (var value in new[] { 1.0, 2.0, 3.0, 4.0, 5.0 })
        {
            estimator.Add(value);
        }

        var markerField = typeof(PSquareEstimator<double>).GetField("_n", BindingFlags.Instance | BindingFlags.NonPublic)!;
        markerField.FieldType.Should().Be(typeof(long[]), "marker positions share the estimator's long stream lifetime");
        markerField.SetValue(estimator, new long[]
        {
            int.MaxValue - 4L,
            int.MaxValue - 3L,
            int.MaxValue - 2L,
            int.MaxValue - 1L,
            int.MaxValue,
        });
        SetPrivateField(estimator, "_count", (long)int.MaxValue);

        estimator.Add(6.0);

        estimator.Count.Should().Be((long)int.MaxValue + 1L);
        ((long[])markerField.GetValue(estimator)!).Should().OnlyContain(static position => position > 0L);
    }

    [Fact]
    public void Pbo_ExcessiveFoldCount_IsRejectedBeforeMaterialization()
    {
        var returns = new double[2, 48];
        for (var strategy = 0; strategy < 2; strategy++)
        {
            for (var observation = 0; observation < 48; observation++)
            {
                returns[strategy, observation] = (strategy + 1) * (observation + 1) * 0.001;
            }
        }

        var act = () => ProbabilityOfBacktestOverfitting<double>.Compute(returns, splitCount: 24);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("splitCount");
    }

    [Fact]
    public void CiWorkflow_SolutionScopedDotnetCommands_NameTheSolutionExplicitly()
    {
        var repositoryRoot = FindRepositoryRoot();
        foreach (var relativePath in new[] { ".github/workflows/pr-verify.yml", ".github/workflows/publish.yml" })
        {
            var workflowPath = Path.Combine(repositoryRoot, relativePath);
            var solutionCommands = File.ReadLines(workflowPath)
                .Select(static (line, index) => (Text: line.Trim(), Number: index + 1))
                .Where(static line =>
                    (line.Text.StartsWith("run: dotnet ", StringComparison.Ordinal) ||
                     line.Text.StartsWith("if ! dotnet ", StringComparison.Ordinal)) &&
                    (line.Text.Contains("dotnet restore", StringComparison.Ordinal) ||
                     line.Text.Contains("dotnet build", StringComparison.Ordinal) ||
                     line.Text.Contains("dotnet test", StringComparison.Ordinal) ||
                     line.Text.Contains("dotnet format", StringComparison.Ordinal) ||
                     line.Text.Contains("dotnet pack", StringComparison.Ordinal)));

            foreach (var command in solutionCommands)
            {
                command.Text.Should().Contain(
                    "Boutquin.Numerics.slnx",
                    $"{relativePath}:{command.Number} must not rely on implicit solution discovery");
            }
        }
    }

    private static void SetPrivateField<TTarget, TValue>(TTarget target, string name, TValue value)
        where TTarget : class
        => typeof(TTarget).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(target, value);

    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "Boutquin.Numerics.slnx")))
        {
            directory = directory.Parent;
        }

        return directory?.FullName ?? throw new InvalidOperationException("Could not locate the repository root.");
    }
}
