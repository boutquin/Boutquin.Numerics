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
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

/// <summary>
/// Regression pins for fitted PCA centering at a large decimal offset: the absolute mean of
/// rows translated by 1e28 is not representable, so projection must center with the exact
/// fitted (origin, shifted mean) pair instead of the rounded absolute mean.
/// </summary>
public sealed class Arena414074R1PcaCenteringTests
{
    [Fact]
    public void FromReturns_TranslatedDecimalRows_ProjectsTrainingRowsToCenteredScores()
    {
        // The rows differ by exactly one, so the exact mean is the half-integer 1e28 + 0.5, which
        // needs 30 digits and is not a decimal. The centered training rows are -0.5 and +0.5.
        var rows = new[,] { { 1e28m }, { 1e28m + 1m } };

        var fit = PrincipalComponentAnalysis.FromReturns(rows);

        fit.Eigenvalues[0].Should().Be(0.5m);
        fit.Mean[0].Should().Be(1e28m, "the public mean is the nearest representable absolute mean");
        var scores = fit.Project(rows, 1);
        scores[0, 0].Should().Be(-0.5m);
        scores[1, 0].Should().Be(0.5m);
    }

    [Fact]
    public void FromReturns_TranslatedStandardizedRows_ProjectsExactlyAsTheTranslationFreeFit()
    {
        var plain = new[,] { { 0m }, { 1m } };
        var translated = new[,] { { 1e28m }, { 1e28m + 1m } };

        var expected = PrincipalComponentAnalysis.FromReturns(plain, standardize: true).Project(plain, 1);
        var actual = PrincipalComponentAnalysis.FromReturns(translated, standardize: true).Project(translated, 1);

        // The translation-free pair is the correct correlation-PCA score +/- 1/sqrt(2).
        expected[1, 0].Should().BeApproximately(0.7071067811865475244008443621m, 1e-27m);
        expected[0, 0].Should().Be(-expected[1, 0]);
        actual[0, 0].Should().Be(expected[0, 0]);
        actual[1, 0].Should().Be(expected[1, 0]);
    }

    [Fact]
    public void FromReturns_TranslatedPanel_ProjectsExactlyAsTheTranslationFreePanelAndCentersEveryColumn()
    {
        // Integer panel with fractional column means (2.5, 0.5, 4.75), each column shifted by a
        // different multiple of 1e27 so no absolute mean is representable.
        decimal[,] plain =
        {
            { 0m, 2m, 7m },
            { 3m, -5m, 1m },
            { 1m, 4m, 9m },
            { 6m, 1m, 2m },
        };
        decimal[] offsets = [1e28m, -2e28m, 5e27m];
        var translated = new decimal[4, 3];
        for (var i = 0; i < 4; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                translated[i, j] = plain[i, j] + offsets[j];
            }
        }

        var expected = PrincipalComponentAnalysis.FromReturns(plain).Project(plain, 3);
        var actual = PrincipalComponentAnalysis.FromReturns(translated).Project(translated, 3);

        for (var k = 0; k < 3; k++)
        {
            var columnSum = 0m;
            for (var i = 0; i < 4; i++)
            {
                actual[i, k].Should().Be(expected[i, k], $"row {i}, component {k}");
                columnSum += actual[i, k];
            }

            // Centered training rows have zero total score on every component.
            columnSum.Should().BeApproximately(0m, 1e-24m, $"component {k}");
        }
    }

    [Fact]
    public void Project_OppositeExtremeObservation_KeepsTheShiftedMeanThroughTheScaledFallback()
    {
        // Two perfectly correlated columns at opposite extremes of decimal's range, 2e14 apart
        // between the rows. A new observation at the other extreme makes x - origin unrepresentable
        // (1.4e29), so projection takes the scaled fallback, where the +/-1.4e29 terms cancel
        // against loadings of equal weight and only the shifted mean 1e14 survives.
        var rows = new[,] { { -7e28m, 7e28m }, { -7e28m + 2e14m, 7e28m + 2e14m } };
        var fit = PrincipalComponentAnalysis.FromReturns(rows);
        var loadings = fit.Eigenvectors;

        var scores = fit.Project(new[,] { { 7e28m, -7e28m } }, 1);

        // score = l0 (1.4e29 - 1e14) + l1 (-1.4e29 - 1e14) = -(l0 + l1) 1e14 + (l0 - l1) 1.4e29,
        // and the loadings of a rank-one block of identical columns are equal.
        var expected = -(loadings[0, 0] + loadings[1, 0]) * 1e14m;
        scores[0, 0].Should().BeApproximately(expected, 1e5m);
        scores[0, 0].Should().BeLessThan(-1e14m);
    }

    [Fact]
    public void Project_StandardizedOppositeExtremeObservation_KeepsTheShiftedMeanThroughTheScaledFallback()
    {
        var rows = new[,] { { -7e28m, 7e28m }, { -7e28m + 2e14m, 7e28m + 2e14m } };
        var fit = PrincipalComponentAnalysis.FromReturns(rows, standardize: true);

        var scores = fit.Project(new[,] { { 7e28m, -7e28m } }, 1);

        // Each column has sample standard deviation sqrt(2e28) = 1.414e14, so the standardized
        // centered values are 9.9e14 - 0.7071 and -9.9e14 - 0.7071 against equal loadings
        // 1/sqrt(2): the score is -2 (1/sqrt(2)) (1e14 / 1.414e14) = -1.
        scores[0, 0].Should().BeApproximately(-1m, 1e-9m);
    }

    [Fact]
    public void Project_ReplacedMean_CentersWithTheSuppliedMeanNotTheRetainedFit()
    {
        var rows = new[,] { { 1e28m }, { 1e28m + 1m } };
        var fit = PrincipalComponentAnalysis.FromReturns(rows);

        // A caller who replaces Mean has chosen a different centering; the exact fitted pair no
        // longer describes it and must not be used.
        var replaced = fit with { Mean = [1e28m + 1m] };

        var scores = replaced.Project(rows, 1);
        scores[0, 0].Should().Be(-1m);
        scores[1, 0].Should().Be(0m);
    }
}
