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

/// <summary>
/// Scale-conditioning regressions for the shrinkage and covariance estimators that share
/// <c>CovarianceHelpers</c>. Every expected matrix below comes from an independent 120-digit
/// evaluation of the estimator's documented formula in a multiple-precision library on the exact
/// input values; no arithmetic is shared with the code under test.
/// </summary>
public sealed class Arena414074R1ShrinkageTests
{
    // 14 observations x 4 assets of small integers (100 x a fixed pseudo-random factor panel).
    // Scaling by a power of two (IEEE) or of ten (decimal) is exact, so the expected matrix at any
    // scale c is c^2 times the scale-1 expectation below.
    private static readonly int[,] s_panel14 =
    {
        { -38, -182, 24, -108 }, { -140, -169, 17, -76 }, { -181, -131, -220, -75 }, { -161, -167, 11, -149 },
        { 121, 101, 76, 116 }, { 123, 97, -16, 17 }, { -190, -60, -170, -47 }, { 161, 7, 30, 106 },
        { -43, 63, -60, 0 }, { 101, 60, -58, -69 }, { 179, 284, 176, 78 }, { -360, -301, -184, -10 },
        { -272, -145, -139, -191 }, { -36, 77, 114, -51 },
    };

    private static readonly int[,] s_panel4X2 = { { -2, -1 }, { -1, 2 }, { 1, -2 }, { 2, 1 } };

    // Scaled-identity Ledoit-Wolf estimate of s_panel14 (delta = 0.172356527809...).
    private static readonly double[,] s_lwiPanel14 =
    {
        { 27579.919219197652, 18626.416476097402, 11861.340589259613, 8574.4955116940627 },
        { 18626.416476097402, 23662.455290738851, 9552.0243072001637, 7235.3138037902635 },
        { 11861.340589259613, 9552.0243072001637, 14602.728782269168, 3687.1198361682788 },
        { 8574.4955116940627, 7235.3138037902635, 3687.1198361682788, 10112.753850651472 },
    };

    // Single-factor Ledoit-Wolf estimate of s_panel14 (delta = 0.727017988533...).
    private static readonly double[,] s_lwsfPanel14 =
    {
        { 29368.879120879121, 23216.676311902544, 15058.141132081355, 10885.903601650192 },
        { 23216.676311902544, 24635.604395604396, 12965.693348200893, 9482.1434656522975 },
        { 15058.141132081355, 12965.693348200893, 13689.192307692308, 5848.4488992973349 },
        { 10885.903601650192, 9482.1434656522975, 5848.4488992973349, 8264.1813186813187 },
    };

    // Single-factor Ledoit-Wolf estimate of s_panel4X2 (delta = 0.32).
    private static readonly double[,] s_lwsfPanel4X2 =
    {
        { 3.3333333333333333, 0.53333333333333333 },
        { 0.53333333333333333, 3.3333333333333333 },
    };

    private static T[,] Scale<T>(int[,] panel, T factor)
        where T : IFloatingPoint<T>
    {
        var result = new T[panel.GetLength(0), panel.GetLength(1)];
        for (var i = 0; i < result.GetLength(0); i++)
        {
            for (var j = 0; j < result.GetLength(1); j++)
            {
                result[i, j] = T.CreateChecked(panel[i, j]) * factor;
            }
        }

        return result;
    }

    private static void AssertEquivariant<T>(T[,] actual, double[,] scaleOneExpectation, double factor, double relativeTolerance)
        where T : IFloatingPoint<T>
    {
        var n = scaleOneExpectation.GetLength(0);
        var reference = 0.0;
        foreach (var value in scaleOneExpectation)
        {
            reference = Math.Max(reference, Math.Abs(value));
        }

        // factor is a power of two or ten; its square is formed in double, whose range covers every
        // case below. The deviation is measured against the largest expected entry.
        var worst = 0.0;
        var worstAt = (0, 0);
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var expected = scaleOneExpectation[i, j] * factor * factor;
                var deviation = Math.Abs(double.CreateChecked(actual[i, j]) - expected) / (reference * factor * factor);
                if (deviation > worst)
                {
                    worst = deviation;
                    worstAt = (i, j);
                }
            }
        }

        worst.Should().BeLessThan(
            relativeTolerance,
            $"entry [{worstAt.Item1},{worstAt.Item2}] must equal the scale-1 estimate times the squared scale");
    }

    private static void AssertComponentwise<T>(T[,] actual, double[,] expected, double relativeTolerance, double absoluteFloor = 0.0)
        where T : IFloatingPoint<T>
    {
        for (var i = 0; i < expected.GetLength(0); i++)
        {
            for (var j = 0; j < expected.GetLength(1); j++)
            {
                var observed = double.CreateChecked(actual[i, j]);
                if (expected[i, j] == 0.0)
                {
                    observed.Should().Be(0.0, $"entry [{i},{j}] is exactly zero");
                }
                else
                {
                    // A decimal entry sits on a 1e-28 grid, so tiny entries carry an absolute floor.
                    var allowed = Math.Max(relativeTolerance * Math.Abs(expected[i, j]), absoluteFloor);
                    Math.Abs(observed - expected[i, j]).Should().BeLessThanOrEqualTo(
                        allowed,
                        $"entry [{i},{j}] expected {expected[i, j]:R}, actual {observed:R}");
                }
            }
        }
    }

    // F-L1-13 panel: C = 2^245 Q and D = 2^245 (Q + R) dominate (maximum 2^247, so the unnormalized
    // path runs); A and B carry 2^-540 of C plus an idiosyncratic 2^-297 part. beta_A * beta_B is about
    // 2^-1078 and underflows, although the target entry beta_A * beta_B * Var(market) is normal.
    private static double[,] SingleFactorProductOrderPanel()
    {
        int[] q = [2, -3, 1, 4, -2, -1, 3, -4, 0, 2, -2, 1, -1, 3, -2, 0, 2, -3, 1, -1, 2, 0, -2, 1];
        int[] r = [1, 0, -1, 1, -1, 0, 1, -1, 0, 1, -1, 1, 0, -1, 1, 0, -1, 1, 1, 0, -1, -1, 0, 1];
        int[] ea = [1, -2, 0, 1, 2, -1, 0, -1, 1, 2, -2, 0, 1, -1, 0, 2, -1, 1, 0, -2, 1, 0, 1, -1];
        int[] eb = [0, 1, -1, 2, -1, 1, -2, 0, 1, -1, 0, 2, -1, 1, 1, -2, 0, 1, -1, 0, 2, -1, 0, 1];
        var c2 = Math.ScaleB(1.0, 245);
        var eps = Math.ScaleB(1.0, -540);
        var a2 = Math.ScaleB(1.0, -297);
        var x = new double[24, 4];
        for (var k = 0; k < 24; k++)
        {
            var cv = c2 * q[k];
            x[k, 2] = cv;
            x[k, 3] = c2 * (q[k] + r[k]);
            x[k, 0] = eps * cv + a2 * ea[k];
            x[k, 1] = eps * cv + a2 * eb[k];
        }

        return x;
    }

    // Single-factor estimate of SingleFactorProductOrderPanel (delta = 0.3203536077...).
    private static readonly double[,] s_productOrderExpectation =
    {
        { 1.2409201281037599e-177, 1.1737904065034258e-177, 4.2160160633549293e-15, 4.6971440017335557e-15 },
        { 1.1737904065034258e-177, 1.1795335316558638e-177, 4.1075342362723436e-15, 4.5630662332186887e-15 },
        { 4.2160160633549293e-15, 4.1075342362723436e-15, 1.4865676111139528e+148, 1.6271353333128204e+148 },
        { 4.6971440017335557e-15, 4.5630662332186887e-15, 1.6271353333128204e+148, 1.9712801512395386e+148 },
    };

    [Fact]
    public void LedoitWolfSingleFactor_NormalTargetEntryWithUnderflowingBetaProduct_MatchesExactEstimate()
    {
        var estimate = new LedoitWolfSingleFactorEstimator<double>().Estimate(SingleFactorProductOrderPanel());
        AssertComponentwise(estimate, s_productOrderExpectation, 1e-12);
    }

    // F-L1-12: an equal-weight market that nearly cancels gives betas of about +/-5e224. Their product
    // overflows although every entry of the target (beta_i * beta_j * Var(market)) and of the estimate
    // is representable.
    [Fact]
    public void LedoitWolfSingleFactor_NearCancellingMarketDouble_MatchesExactEstimate()
    {
        const double big = 1e75;
        const double tiny = 1e-140;
        const double delta = 1e-150;
        var x = new[,]
        {
            { big, -big }, { big, -big }, { big, -big }, { -big, big }, { tiny, -tiny + 2 * delta },
        };
        double[,] expected =
        {
            { 7.9999999999999988e+149, -5.6999999999999992e+149 },
            { -5.6999999999999992e+149, 7.9999999999999988e+149 },
        };

        var estimate = new LedoitWolfSingleFactorEstimator<double>().Estimate(x);

        AssertComponentwise(estimate, expected, 1e-12);
    }

    [Fact]
    public void LedoitWolfSingleFactor_NearCancellingMarketDecimal_MatchesExactEstimate()
    {
        var x = new[,]
        {
            { 100000m, -100000m }, { 100000m, -100000m }, { 100000m, -100000m }, { -100000m, 100000m }, { 1m, -0.9999999999998m },
        };
        double[,] expected =
        {
            { 7999980000.2, -5699967000.23 },
            { -5699967000.23, 7999980000.2 },
        };

        decimal[,]? estimate = null;
        var thrown = Record.Exception(() => estimate = new LedoitWolfSingleFactorEstimator<decimal>().Estimate(x));

        thrown.Should().BeNull("the exact estimate is representable in decimal");
        AssertComponentwise(estimate!, expected, 1e-12);
    }

    // Same near-cancelling equal-weight market at 1e81, above the 2^251 normalization threshold: after the common
    // division by the maximum the market's variance (1e-451) underflows to zero, the betas are forced to zero and
    // the intensity comes from an unrelated target. The market scale cancels from every term of the estimator, so
    // the estimate does not depend on it.
    [Fact]
    public void LedoitWolfSingleFactor_NearCancellingMarketAboveNormalizationThreshold_MatchesExactEstimate()
    {
        const double big = 1e81;
        const double tiny = 1e-136;
        var x = new[,]
        {
            { big, -big }, { big, -big }, { big, -big }, { big, -big }, { big, -big }, { -big, big }, { tiny, -tiny + 2 * tiny * 1e-8 },
        };
        double[,] expected =
        {
            { 6.1904761904761895e+161, -3.5941043083900221e+161 },
            { -3.5941043083900221e+161, 6.1904761904761895e+161 },
        };

        var estimate = new LedoitWolfSingleFactorEstimator<double>().Estimate(x);

        AssertComponentwise(estimate, expected, 1e-12);
    }

    // Decimal analogue with a nonzero intensity (delta = 0.097): A and B at 1e-12, C and D at 1e8. Dividing A by
    // the common maximum (4e8 before rescaling) leaves it 2.5e-21 and rounds seven of its digits away on the
    // 1e-28 grid, so the cross entries between the small and the large columns must be formed from the original
    // columns, not from the normalized ones.
    [Fact]
    public void LedoitWolfSingleFactor_DecimalSmallColumnsBesideLargeColumns_KeepTheirCrossEntries()
    {
        int[,] counts =
        {
            { 0, 0, 0, 0 }, { -2, -4, 0, 0 }, { -3, -3, 0, 0 }, { -2, -2, 0, 0 }, { 3, 4, 0, 0 }, { -3, -2, 0, 0 },
            { -5, 3, 0, 0 }, { 0, -1, 0, 0 }, { 0, 5, 0, 0 }, { -2, 1, 0, 0 }, { 3, 5, 0, 0 }, { -2, 2, 0, 0 },
            { 0, 0, 0, 0 }, { 0, 0, 5, 4 }, { 0, 0, 4, -2 }, { 0, 0, -2, 1 }, { 0, 0, -5, 1 }, { 0, 0, 4, -1 },
            { 0, 0, 5, -1 }, { 0, 0, -1, -2 }, { 0, 0, 0, -2 }, { 0, 0, -2, -4 }, { 0, 0, 5, 3 }, { 0, 0, -3, -2 },
        };
        var x = new decimal[24, 4];
        for (var k = 0; k < 24; k++)
        {
            x[k, 0] = counts[k, 0] * 1e-12m;
            x[k, 1] = counts[k, 1] * 1e-12m;
            x[k, 2] = counts[k, 2] * 1e8m;
            x[k, 3] = counts[k, 3] * 1e8m;
        }

        double[,] expected =
        {
            { 3.0416666666666667e-24, 1.465600246247121e-24, 2.2022562720273429e-5, -1.0247200401432849e-5 },
            { 1.465600246247121e-24, 4.8405797101449275e-24, -1.3552346289399033e-5, 6.3059694778048303e-6 },
            { 2.2022562720273429e-5, -1.3552346289399033e-5, 63405797101449275.0, 13907377761083943.0 },
            { -1.0247200401432849e-5, 6.3059694778048303e-6, 13907377761083943.0, 26068840579710145.0 },
        };

        var estimate = new LedoitWolfSingleFactorEstimator<decimal>().Estimate(x);

        AssertComponentwise(estimate, expected, 1e-9, 2e-28);
    }

    // F-L1-11 (single-factor site): C and D at 2^330 force the normalized path, where A and B (2^-240) are
    // formed at 2^-570 of the common maximum and their cross covariance 2^-480 underflows to zero.
    private static T[,] DecoupledHeterogeneousPanel<T>(T small, T large)
        where T : IFloatingPoint<T>
    {
        int[] u = [3, -1, 2, -4, 1, 0, -2, 5, -3, 1, -2];
        int[] w = [1, 2, -3, 1, -2, 3, -1, 0, 2, -1, -2];
        int[] v = [2, -3, 1, 4, -2, -1, 3, -4, 0, 2, -2];
        int[] s = [1, 0, -1, 1, -1, 0, 1, -1, 0, 1, -1];
        var x = new T[24, 4];
        for (var k = 0; k < 11; k++)
        {
            x[1 + k, 0] = small * T.CreateChecked(u[k]);
            x[1 + k, 1] = small * T.CreateChecked(w[k]);
            x[13 + k, 2] = large * T.CreateChecked(v[k]);
            x[13 + k, 3] = large * T.CreateChecked(v[k] + s[k]);
        }

        return x;
    }

    [Fact]
    public void LedoitWolfSingleFactor_SmallColumnsBesideNormalizingColumns_KeepTheirCrossCovariance()
    {
        double[,] expected =
        {
            { 1.0306375585638224e-144, -1.6713041490224147e-145, 0.0, 0.0 },
            { -1.6713041490224147e-145, 5.2924631385709798e-145, 0.0, 0.0 },
            { 0.0, 0.0, 1.4144194341232137e+199, 1.7888245784499467e+199 },
            { 0.0, 0.0, 1.7888245784499467e+199, 2.3296320091441166e+199 },
        };

        var estimate = new LedoitWolfSingleFactorEstimator<double>()
            .Estimate(DecoupledHeterogeneousPanel(Math.ScaleB(1.0, -240), Math.ScaleB(1.0, 330)));

        AssertComponentwise(estimate, expected, 1e-12);
    }

    // F-L1-10: OAS divided every observation by one global maximum before forming the second moments, so a
    // column 2^570 below the largest had its cross moment (2^-480) formed at 2^-1140 and lost to underflow.
    [Fact]
    public void OracleApproximatingShrinkage_DecoupledSmallColumns_KeepCrossCovariance()
    {
        double[,] expected =
        {
            { 7.4238254239962871e+197, -1.469109854841989e-145, 0.0, 0.0 },
            { -1.469109854841989e-145, 7.4238254239962871e+197, 0.0, 0.0 },
            { 0.0, 0.0, 1.3175412944232099e+199, 1.5724126684670477e+199 },
            { 0.0, 0.0, 1.5724126684670477e+199, 2.1220314968947226e+199 },
        };

        var estimate = new OracleApproximatingShrinkageEstimator<double>()
            .Estimate(DecoupledHeterogeneousPanel(Math.ScaleB(1.0, -240), Math.ScaleB(1.0, 330)));

        AssertComponentwise(estimate, expected, 1e-12);
    }

    [Fact]
    public void OracleApproximatingShrinkage_DecimalDecoupledSmallColumns_KeepCrossCovariance()
    {
        double[,] expected =
        {
            { 1551781651470311.511, -4.5861915596079169304e-25, 0.0, 0.0 },
            { -4.5861915596079169304e-25, 1551781651470311.511, 0.0, 0.0 },
            { 0.0, 0.0, 27540200489248507.45, 32867706177190071.334 },
            { 0.0, 0.0, 32867706177190071.334, 44356236207810869.528 },
        };

        var estimate = new OracleApproximatingShrinkageEstimator<decimal>()
            .Estimate(DecoupledHeterogeneousPanel(1e-12m, 1e8m));

        AssertComponentwise(estimate, expected, 1e-12, 2e-28);
    }

    // Coupled panel: every column is nonzero in every row, columns scaled 2^-240, 2^-235, 2^330, 2^328.
    [Fact]
    public void OracleApproximatingShrinkage_CoupledSmallColumns_KeepEveryEntry()
    {
        int[,] counts =
        {
            { 0, -3, 1, 5 }, { -5, -4, 3, -4 }, { 0, 4, -5, 3 }, { -2, -5, -4, 1 }, { 1, -4, -2, -4 },
            { 3, 1, -5, 4 }, { -4, -2, 5, 5 }, { 4, -5, 4, 4 }, { 1, -5, -2, -5 }, { 3, -3, -1, 1 },
            { -3, 3, -4, 4 }, { -1, 3, 5, -3 }, { -4, 4, 4, 5 }, { -2, 0, -4, 3 }, { -4, 4, -5, 4 },
            { -2, 2, 5, 3 }, { 1, 0, 2, 4 }, { 2, 0, -1, -2 }, { -3, -2, -4, 4 }, { -1, 3, 2, 0 },
        };
        int[] exponents = [-240, -235, 330, 328];
        var x = new double[counts.GetLength(0), 4];
        for (var k = 0; k < counts.GetLength(0); k++)
        {
            for (var j = 0; j < 4; j++)
            {
                x[k, j] = Math.ScaleB(counts[k, j], exponents[j]);
            }
        }

        double[,] expected =
        {
            { 1.9124929129911255e+198, -1.9159618798564645e-143, -7.7128193514333242e+26, -6.7487169325041587e+26 },
            { -1.9159618798564645e-143, 1.9124929129911255e+198, -1.0577580824822845e+28, 2.5121754458954256e+28 },
            { -7.7128193514333242e+26, -1.0577580824822845e+28, 6.0248031934217941e+199, -1.1709688489662317e+198 },
            { -6.7487169325041587e+26, 2.5121754458954256e+28, -1.1709688489662317e+198, 5.5052382450466091e+198 },
        };

        var estimate = new OracleApproximatingShrinkageEstimator<double>().Estimate(x);

        AssertComponentwise(estimate, expected, 1e-12);
    }

    // F-L1-07: NERCOME divided every observation by one global maximum. A and B (2^-240) share no row with C
    // and D (2^330), so the second-moment block of A and B is independent of the rest, yet at 2^-570 of the
    // common maximum its entries (2^-1140) underflow, the eigenvectors of the block are lost and the whole
    // A,B block of the estimate is zero. Expected matrices: 120-digit evaluation of the documented
    // estimator on the first subsets the public Xoshiro256** stream yields (seed 7: {0, 8, 10, 13, 15, 21}).
    [Fact]
    public void Nercome_DecoupledSmallColumns_KeepTheirBlock()
    {
        double[,] expected =
        {
            { 8.6916082224377121e-145, -9.0254655695235357e-147, 0.0, 0.0 },
            { -9.0254655695235357e-147, 6.4352418300568282e-145, 0.0, 0.0 },
            { 0.0, 0.0, 1.5149541488035401e+199, 2.0730951509943181e+199 },
            { 0.0, 0.0, 2.0730951509943181e+199, 2.8970175827997522e+199 },
        };

        var estimate = new NercomeCovarianceEstimator<double>(0.25, 1, 7UL)
            .Estimate(DecoupledHeterogeneousPanel(Math.ScaleB(1.0, -240), Math.ScaleB(1.0, 330)));

        AssertComponentwise(estimate, expected, 1e-10);
    }

    [Fact]
    public void Nercome_DefaultSplitSelection_DecoupledSmallColumns_KeepTheirBlock()
    {
        double[,] expected =
        {
            { 7.9076416726201905e-145, -6.7347016744361833e-146, 0.0, 0.0 },
            { -6.7347016744361833e-146, 6.8148765761182605e-145, 0.0, 0.0 },
            { 0.0, 0.0, 1.4759930616850521e+199, 1.8621686333671534e+199 },
            { 0.0, 0.0, 1.8621686333671534e+199, 2.4316318290814687e+199 },
        };

        var estimate = new NercomeCovarianceEstimator<double>()
            .Estimate(DecoupledHeterogeneousPanel(Math.ScaleB(1.0, -240), Math.ScaleB(1.0, 330)));

        AssertComponentwise(estimate, expected, 1e-10);
    }

    [Fact]
    public void Nercome_DecimalDecoupledSmallColumns_KeepTheirBlock()
    {
        double[,] expected =
        {
            { 2.7133015368309486e-24, -2.8175234057586999e-26, 0.0, 0.0 },
            { -2.8175234057586999e-26, 2.0089206853912736e-24, 0.0, 0.0 },
            { 0.0, 0.0, 31666666666666667.0, 43333333333333333.0 },
            { 0.0, 0.0, 43333333333333333.0, 60555555555555556.0 },
        };

        var estimate = new NercomeCovarianceEstimator<decimal>(0.25m, 1, 7UL)
            .Estimate(DecoupledHeterogeneousPanel(1e-12m, 1e8m));

        // The decimal grid is 1e-28 absolute, so the 1e-24-scale block carries four to five digits.
        AssertComponentwise(estimate, expected, 1e-10, 2e-28);
    }

    // The split size is chosen by the Frobenius norm of the averaged residual in the original units. A group
    // of columns conditioned by its own, much smaller, maximum enters that norm at its true weight (2^-1140
    // relative here, which is zero): counting its block at the weight of its own normalized units would pick
    // size 19 from the candidate grid {9, 4, 14, 19, 11, 16}; the true objective picks 4.
    [Fact]
    public void Nercome_DefaultSplitSelection_ScoresGroupsByTheirTrueWeight()
    {
        int[,] smallGroup =
        {
            { 1, 1 }, { 3, 2 }, { 2, 0 }, { 3, -3 }, { -4, 4 }, { 4, -3 }, { -4, 5 }, { 3, -4 }, { 0, 4 }, { 3, 2 }, { -1, -5 },
        };
        int[,] largeGroup =
        {
            { -5, -1 }, { 1, -1 }, { 4, -2 }, { -1, -3 }, { -1, 3 }, { -1, -4 }, { 0, 2 }, { 0, 1 }, { 5, -2 }, { 2, 3 }, { 3, -5 },
        };
        var x = new double[24, 4];
        for (var k = 0; k < 11; k++)
        {
            x[1 + k, 0] = Math.ScaleB(smallGroup[k, 0], -240);
            x[1 + k, 1] = Math.ScaleB(smallGroup[k, 1], -240);
            x[13 + k, 2] = Math.ScaleB(largeGroup[k, 0], 330);
            x[13 + k, 3] = Math.ScaleB(largeGroup[k, 1], 330);
        }

        double[,] expected =
        {
            { 1.3191593655712715e-144, -3.9292173677581377e-145, 0.0, 0.0 },
            { -3.9292173677581377e-145, 1.4853586341612163e-144, 0.0, 0.0 },
            { 0.0, 0.0, 1.6309021688280154e+199, 2.6109837544071203e+196 },
            { 0.0, 0.0, 2.6109837544071203e+196, 1.6892394499182694e+199 },
        };

        var estimate = new NercomeCovarianceEstimator<double>(null, 10, 0UL).Estimate(x);

        AssertComponentwise(estimate, expected, 1e-10);
    }

    // F-L7-C4-a2-02: the 1000-point Marcenko-Pastur / Gaussian-KDE objective of the noise-variance fit has
    // minima narrower than the 1/256 scan step when T << N: its smallest feature in the variance is
    // bandwidth / upper-edge = 0.01 / 31.6 = 3e-4 here. The selected variance must be the minimizer of the
    // declared objective over [1e-5, 1 - 1e-5], not that of whichever coarse basin the scan lands in.
    // Spectra: n = 128 assets, t = 6 or 5 observations, nonzero correlation eigenvalues as below, the rest
    // zero. Expected minima come from a 40001-point scan with bounded refinement of the 30 best candidates
    // in an independent array-language implementation of the same objective.
    private static double MarcenkoPasturObjective(double[] eigenvalues, double q, double variance)
    {
        const int points = 1000;
        const double bandwidth = 0.01;
        var root = Math.Sqrt(1 / q);
        var lower = (1 - root) * (1 - root);
        var upper = (1 + root) * (1 + root);
        var error = 0.0;
        for (var i = 0; i < points; i++)
        {
            var abscissa = lower + (upper - lower) * i / (points - 1);
            var density = q * Math.Sqrt(Math.Max(0, (upper - abscissa) * (abscissa - lower))) / (2 * Math.PI * abscissa);
            var empirical = 0.0;
            foreach (var eigenvalue in eigenvalues)
            {
                var z = (variance * abscissa - eigenvalue) / bandwidth;
                empirical += Math.Exp(-0.5 * z * z);
            }

            empirical /= eigenvalues.Length * bandwidth * Math.Sqrt(2 * Math.PI);
            var difference = density / variance - empirical;
            error += difference * difference;
        }

        return error;
    }

    private static double[] PaddedSpectrum(double[] nonzero, int length)
    {
        var spectrum = new double[length];
        nonzero.CopyTo(spectrum, 0);
        return spectrum;
    }

    [Theory]
    [InlineData(6, 0.8974855467641698, 0.10354794679863352)]
    [InlineData(5, 0.9997029674424543, 0.07194921901389875)]
    public void NoiseVarianceFit_NarrowGlobalMinimum_IsLocated(int observations, double expectedVariance, double expectedObjective)
    {
        double[] nonzero = observations == 6
            ? [30.64653142037455, 28.365553226224346, 14.151610879890523, 46.56159874957671, 8.274705723933874]
            : [6.45519199381458, 25.964257791892265, 82.24553370505065, 13.335016509242514];
        var spectrum = PaddedSpectrum(nonzero, 128);
        var q = observations / 128.0;

        var selected = CovarianceHelpers<double>.FitNoiseVariance(spectrum, q);

        var objective = MarcenkoPasturObjective(spectrum, q, selected);
        objective.Should().BeLessThanOrEqualTo(
            expectedObjective * (1 + 1e-9),
            "the selected variance must reach the global minimum of the declared objective");
        selected.Should().BeApproximately(expectedVariance, 1e-6);
    }

    // The same spectrum through the public estimator: the lower objective moves the Marcenko-Pastur edge from
    // 8.2 to 28.3 and with it the signal count from 5 to 3, which changes the returned covariance by 0.18.
    [Fact]
    public void DenoisedCovariance_NarrowGlobalMinimum_UsesTheGlobalSignalCount()
    {
        const int n = 128;
        const int t = 6;
        double[] lambda = [30.64653142037455, 28.365553226224346, 14.151610879890523, 46.56159874957671, 8.274705723933874];
        const int r = t - 1;
        static double Hadamard(int i, int j) => BitOperations.PopCount((uint)(i & j)) % 2 == 0 ? 1.0 : -1.0;
        var helmert = new double[t, r];
        for (var k = 1; k <= r; k++)
        {
            var norm = Math.Sqrt(k * (k + 1.0));
            for (var i = 0; i < k; i++)
            {
                helmert[i, k - 1] = 1.0 / norm;
            }

            helmert[k, k - 1] = -k / norm;
        }

        var x = new double[t, n];
        for (var i = 0; i < t; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var sum = 0.0;
                for (var k = 0; k < r; k++)
                {
                    sum += helmert[i, k] * Math.Sqrt(lambda[k] * (t - 1)) * Hadamard(j, k) / Math.Sqrt(n);
                }

                x[i, j] = sum;
            }
        }

        // Expected: signal eigenpairs lambda_k > edge keep their eigenvalue, the n - 3 remaining directions share
        // their mean (n - sum of the kept eigenvalues) / (n - 3), the result rescaled to a unit diagonal.
        // The global minimizer v = 0.89749 puts the edge at v * 31.5709 = 28.334: three eigenvalues exceed it.
        int[] signal = [0, 1, 3];
        var kept = signal.Sum(k => lambda[k]);
        var noiseMean = (n - kept) / (n - signal.Length);
        var covariance = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var projector = 0.0;
                var spectral = 0.0;
                foreach (var k in signal)
                {
                    var w = Hadamard(i, k) * Hadamard(j, k) / n;
                    projector += w;
                    spectral += lambda[k] * w;
                }

                covariance[i, j] = spectral + noiseMean * ((i == j ? 1.0 : 0.0) - projector);
            }
        }

        var estimate = new DenoisedCovarianceEstimator<double>().Estimate(x);

        var worst = 0.0;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var expected = covariance[i, j] / Math.Sqrt(covariance[i, i] * covariance[j, j]);
                worst = Math.Max(worst, Math.Abs(estimate[i, j] - expected));
            }
        }

        worst.Should().BeLessThan(1e-9);
    }

    // F-L23-C4-07: the unnormalized path was kept while the fourth-order bound was positive but at or
    // below the type's resolution, so the fourth-moment sums that set the shrinkage intensity were lost.
    [Theory]
    [InlineData("1e-8")]
    [InlineData("1e-9")]
    [InlineData("5e-10")]
    public void LedoitWolfIdentity_DecimalTinyScale_IsScaleEquivariant(string scaleText)
    {
        var scale = decimal.Parse(scaleText, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture);
        var estimate = new LedoitWolfShrinkageEstimator<decimal>().Estimate(Scale(s_panel14, scale));
        AssertEquivariant(estimate, s_lwiPanel14, (double)scale, 1e-12);
    }

    [Theory]
    [InlineData("1e-9")]
    [InlineData("5e-10")]
    public void LedoitWolfSingleFactor_DecimalTinyScale_IsScaleEquivariant(string scaleText)
    {
        var scale = decimal.Parse(scaleText, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture);
        var estimate = new LedoitWolfSingleFactorEstimator<decimal>().Estimate(Scale(s_panel14, scale));
        AssertEquivariant(estimate, s_lwsfPanel14, (double)scale, 1e-12);
    }

    [Fact]
    public void LedoitWolfSingleFactor_DecimalFourByTwoTinyScale_IsScaleEquivariant()
    {
        var estimate = new LedoitWolfSingleFactorEstimator<decimal>().Estimate(Scale(s_panel4X2, 5e-8m));
        AssertEquivariant(estimate, s_lwsfPanel4X2, 5e-8, 1e-12);
    }

    [Fact]
    public void LedoitWolfIdentity_FloatSubnormalFourthPower_IsScaleEquivariant()
    {
        var factor = MathF.ScaleB(1f, -45);
        var estimate = new LedoitWolfShrinkageEstimator<float>().Estimate(Scale(s_panel14, factor));
        AssertEquivariant(estimate, s_lwiPanel14, factor, 1e-5);
    }

    [Fact]
    public void LedoitWolfSingleFactor_FloatSubnormalFourthPower_IsScaleEquivariant()
    {
        var factor = MathF.ScaleB(1f, -45);
        var estimate = new LedoitWolfSingleFactorEstimator<float>().Estimate(Scale(s_panel14, factor));
        AssertEquivariant(estimate, s_lwsfPanel14, factor, 1e-5);
    }

    [Fact]
    public void LedoitWolfIdentity_DoubleSubnormalFourthPower_IsScaleEquivariant()
    {
        var factor = Math.ScaleB(1.0, -277);
        var estimate = new LedoitWolfShrinkageEstimator<double>().Estimate(Scale(s_panel14, factor));
        AssertEquivariant(estimate, s_lwiPanel14, factor, 1e-12);
    }

    [Fact]
    public void LedoitWolfSingleFactor_DoubleSubnormalFourthPower_IsScaleEquivariant()
    {
        var factor = Math.ScaleB(1.0, -277);
        var estimate = new LedoitWolfSingleFactorEstimator<double>().Estimate(Scale(s_panel14, factor));
        AssertEquivariant(estimate, s_lwsfPanel14, factor, 1e-12);
    }
}
