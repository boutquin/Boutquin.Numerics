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

public sealed class C4MomentTranslationTests
{
    [Fact]
    public void ConstantCorrelation_UnequalVariancesAndNonzeroMeans_MatchesRationalCrossMoments()
    {
        // Honey Appendix B evaluated exactly after column scales(1,2,3):
        // pi=3626,rho=23024/15,gamma=1370/9,delta=47049/54800 atT16.
        // Nonzero means(10,20,30) do not change centered moments.
        decimal[,] rows = { { 7, 14, 27 }, { 9, 22, 39 }, { 11, 26, 21 }, { 13, 18, 33 } };
        var panel = new decimal[16, 3];
        for (var i = 0; i < 16; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                panel[i, j] = rows[i % 4, j];
            }
        }

        decimal[,] expected =
        {
            { 16m / 3, 4158m / 3425, 15683m / 17125 },
            { 4158m / 3425, 64m / 3, 15864m / 17125 },
            { 15683m / 17125, 15864m / 17125, 48m },
        };
        AssertMatrix(new LedoitWolfConstantCorrelationEstimator<decimal>().Estimate(panel), expected);
    }

    [Fact]
    public void ScaledIdentity_NonzeroMeans_MatchesExactCenteredPanel()
    {
        decimal[,] panel = { { 7, 19 }, { 9, 21 }, { 11, 19 }, { 13, 21 } };
        decimal[,] expected = { { 76m / 15, 8m / 15 }, { 8m / 15, 44m / 15 } };
        AssertMatrix(new LedoitWolfShrinkageEstimator<decimal>().Estimate(panel), expected);
    }

    [Fact]
    public void SingleFactor_NonzeroMeans_MatchesExactCenteredMarket()
    {
        decimal[,] panel =
        {
            { 13, 21, 32 }, { 7, 19, 28 }, { 11, 19, 30 },
            { 9, 21, 30 }, { 12, 23, 29 }, { 8, 17, 31 },
        };
        decimal[,] expected =
        {
            { 28m / 5, 5095m / 1542, 8m / 5 },
            { 5095m / 1542, 22m / 5, -149m / 20046 },
            { 8m / 5, -149m / 20046, 2m },
        };
        AssertMatrix(new LedoitWolfSingleFactorEstimator<decimal>().Estimate(panel), expected);
    }

    private static void AssertMatrix(decimal[,] actual, decimal[,] expected)
    {
        for (var i = 0; i < expected.GetLength(0); i++)
        {
            for (var j = 0; j < expected.GetLength(1); j++)
            {
                actual[i, j].Should().BeApproximately(expected[i, j], 1e-20m);
            }
        }
    }
}
