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

using Boutquin.Numerics.LinearAlgebra;
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

public sealed class C2IndependentPcaTests
{
    [Fact]
    public void StandardizedResult_PublicStateRebuildPreservesProjection()
    {
        decimal[,] data = { { 1, 8 }, { 2, 3 }, { 3, 7 }, { 4, 2 }, { 5, 9 } };
        var fit = PrincipalComponentAnalysis.FromReturns(new ReturnsMatrix(data), standardize: true);
        var scalesProperty = typeof(PcaResult).GetProperty("StandardDeviations");
        var scales = scalesProperty?.GetValue(fit) as decimal[] ?? [];
        var rebuilt = new PcaResult
        {
            Eigenvalues = fit.Eigenvalues,
            Eigenvectors = fit.Eigenvectors,
            ExplainedVarianceRatio = fit.ExplainedVarianceRatio,
            CumulativeExplainedVariance = fit.CumulativeExplainedVariance,
            Mean = fit.Mean,
            StandardDeviations = scales,
        };
        var expected = fit.Project(data, 2);
        scales[0] = 0m;
        var detached = fit.StandardDeviations;
        detached[0] = 0m;
        var actual = rebuilt.Project(data, 2);
        for (var i = 0; i < data.GetLength(0); i++)
        {
            for (var j = 0; j < 2; j++) { actual[i, j].Should().BeApproximately(expected[i, j], 1e-25m); }
        }
    }

    [Fact]
    public void StandardizedResult_RejectsRetainedScalesWithTheWrongShape()
    {
        var result = PrincipalComponentAnalysis.Decompose(new decimal[,] { { 2, 0 }, { 0, 1 } })
            with
        { StandardDeviations = [1m] };
        FluentActions.Invoking(() => result.Project(new decimal[,] { { 1, 2 } }, 1))
            .Should().Throw<InvalidOperationException>().WithMessage("*StandardDeviations*");
    }

    [Fact]
    public void TranslatedOrthogonalPanel_HasExactMeanSpectrumAndScores()
    {
        // Deviations are ±3e1, ±2e2, ±e3; X'X=diag(18,8,2), n-1=5.
        decimal[,] data = { { 13, 20, 30 }, { 7, 20, 30 }, { 10, 22, 30 },
            { 10, 18, 30 }, { 10, 20, 31 }, { 10, 20, 29 } };
        var result = PrincipalComponentAnalysis.FromReturns((ReturnsMatrix)data);
        result.Mean.Should().Equal(10m, 20m, 30m);
        result.Eigenvalues.Should().Equal(3.6m, 1.6m, .4m);
        var scores = result.Project(data, 3);
        for (var i = 0; i < 6; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                scores[i, j].Should().Be(data[i, j] - (j + 1) * 10m);
            }
        }
    }

    [Fact]
    public void NonDiagonalCovariance_HasAnalyticEigenpairs()
    {
        double[,] matrix = { { 2, 1 }, { 1, 2 } };
        var result = PrincipalComponentAnalysis<double>.Decompose(matrix);
        result.Eigenvalues[0].Should().BeApproximately(3, 1e-14);
        result.Eigenvalues[1].Should().BeApproximately(1, 1e-14);
        var vectors = result.Eigenvectors;
        var rootHalf = 1 / Math.Sqrt(2);
        Math.Abs(vectors[0, 0]).Should().BeApproximately(rootHalf, 1e-14);
        Math.Abs(vectors[1, 0]).Should().BeApproximately(rootHalf, 1e-14);
        (vectors[0, 0] * vectors[1, 0]).Should().BeApproximately(.5, 1e-14);
        (vectors[0, 1] * vectors[1, 1]).Should().BeApproximately(-.5, 1e-14);
        for (var k = 0; k < 2; k++)
        {
            (vectors[0, k] * vectors[0, k] + vectors[1, k] * vectors[1, k]).Should().BeApproximately(1, 1e-14);
        }
        (vectors[0, 0] * vectors[0, 1] + vectors[1, 0] * vectors[1, 1]).Should().BeApproximately(0, 1e-14);
        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                (3 * vectors[i, 0] * vectors[j, 0] + vectors[i, 1] * vectors[j, 1]).Should().BeApproximately(matrix[i, j], 1e-14);
            }
        }
    }

    [Fact]
    public void SignConvention_FirstLargestMagnitudeWinsTie()
    {
        decimal[,] vectors = { { -.8m, -.6m }, { .6m, .6m } };
        PrincipalComponentAnalysis<decimal>.ApplySignConvention(vectors);
        vectors[0, 0].Should().Be(.8m);
        vectors[1, 0].Should().Be(-.6m);
        vectors[0, 1].Should().Be(.6m);
        vectors[1, 1].Should().Be(-.6m);
    }

    [Fact]
    public void ExplainedVariance_ClipsNegativeSpectrumAndHandlesZeroTotal()
    {
        var positive = PrincipalComponentAnalysis<decimal>.ComputeExplainedVariance([3m, -1m, 1m]);
        positive.Ratio.Should().Equal(.75m, 0m, .25m);
        positive.Cumulative.Should().Equal(.75m, .75m, 1m);
        (decimal[] Ratio, decimal[] Cumulative) zero = default;
        FluentActions.Invoking(() => zero = PrincipalComponentAnalysis<decimal>.ComputeExplainedVariance([0m, -1m]))
            .Should().NotThrow();
        zero.Ratio.Should().Equal(0m, 0m);
        zero.Cumulative.Should().Equal(0m, 0m);
    }

    [Fact]
    public void Threshold_ExactHalfIncludesEquality()
    {
        var result = PrincipalComponentAnalysis<double>.Decompose(new double[,] { { 1, 0 }, { 0, 1 } });
        result.NumComponentsForExplainedVariance(.5).Should().Be(1);
        result.NumComponentsForExplainedVariance(Math.BitIncrement(.5)).Should().Be(2);
    }

    [Theory]
    [InlineData("0.6128382599273879", 0.6128382599273875, 1)]
    [InlineData("0.7212284390680621", 0.7212284390680626, 2)]
    [InlineData("0.50000000000000005", 0.5000000000000001, 2)]
    public void DecimalThreshold_UsesTheRequestedDoubleBoundary(string firstText, double threshold, int expected)
    {
        var first = decimal.Parse(firstText, System.Globalization.CultureInfo.InvariantCulture);
        decimal[,] covariance = { { first, 0 }, { 0, 1m - first } };
        PrincipalComponentAnalysis<decimal>.Decompose(covariance)
            .NumComponentsForExplainedVariance(threshold).Should().Be(expected);
        PrincipalComponentAnalysis.Decompose(covariance)
            .NumComponentsForExplainedVariance(threshold).Should().Be(expected);
    }
}
