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
/// Small columns beside columns that force the common shrinkage normalization, for the scaled-identity and
/// constant-correlation Ledoit-Wolf estimators. Dividing every observation by one common maximum puts a column
/// far below it beyond the type's range (double) or rounds its digits away (decimal's 1e-28 grid), so its
/// covariances, and through r-bar every constant-correlation target entry, were lost. Every expected matrix is
/// an independent 120-digit evaluation of the estimator's documented formula on the exact inputs; no
/// arithmetic is shared with the code under test.
/// </summary>
public sealed class Arena414074R2LedoitWolfTests
{
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

    // A and B are nonzero on rows 1-11 only, C and D on rows 13-23 only (exact multiples of the two scales).
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

    // 20 observations x 4 assets, every column nonzero in every row.
    private static readonly int[,] s_counts =
    {
        { 0, -3, 1, 5 }, { -5, -4, 3, -4 }, { 0, 4, -5, 3 }, { -2, -5, -4, 1 }, { 1, -4, -2, -4 },
        { 3, 1, -5, 4 }, { -4, -2, 5, 5 }, { 4, -5, 4, 4 }, { 1, -5, -2, -5 }, { 3, -3, -1, 1 },
        { -3, 3, -4, 4 }, { -1, 3, 5, -3 }, { -4, 4, 4, 5 }, { -2, 0, -4, 3 }, { -4, 4, -5, 4 },
        { -2, 2, 5, 3 }, { 1, 0, 2, 4 }, { 2, 0, -1, -2 }, { -3, -2, -4, 4 }, { -1, 3, 2, 0 },
    };

    private static double[,] ScaledCounts(params int[] exponents)
    {
        var x = new double[s_counts.GetLength(0), exponents.Length];
        for (var k = 0; k < s_counts.GetLength(0); k++)
        {
            for (var j = 0; j < exponents.Length; j++)
            {
                x[k, j] = Math.ScaleB(s_counts[k, j], exponents[j]);
            }
        }

        return x;
    }

    // Every column nonzero in every row, columns scaled 2^-240, 2^-235, 2^330, 2^328.
    private static double[,] CoupledHeterogeneousPanel() => ScaledCounts(-240, -235, 330, 328);

    // Scaled identity, F-L1-07 double panel (delta = 0.1534087734). The cross entry (A,B) is (1 - delta) S_AB;
    // with every observation divided by the common maximum 2^332 it is formed at 2^-1144 and underflows.
    [Fact]
    public void LedoitWolfIdentity_SmallColumnsBesideNormalizingColumns_KeepTheirCrossCovariance()
    {
        double[,] expected =
        {
            { 1.4359258486490135e+198, -1.4149114295419015e-145, 0.0, 0.0 },
            { -1.4149114295419015e-145, 1.4359258486490135e+198, 0.0, 0.0 },
            { 0.0, 0.0, 1.3410276685255974e+199, 1.5144031940414686e+199 },
            { 0.0, 0.0, 1.5144031940414686e+199, 2.1158386050119302e+199 },
        };

        var estimate = new LedoitWolfShrinkageEstimator<double>()
            .Estimate(DecoupledHeterogeneousPanel(Math.ScaleB(1.0, -240), Math.ScaleB(1.0, 330)));

        AssertComponentwise(estimate, expected, 1e-12);
    }

    // Decimal analogue, A and B at 1e-12 beside C and D at 1e8: after the common division A is 2.5e-21 and its
    // products round to zero on the 1e-28 grid.
    [Fact]
    public void LedoitWolfIdentity_DecimalSmallColumnsBesideLargeColumns_KeepTheirCrossCovariance()
    {
        double[,] expected =
        {
            { 3.0014760013120012e+15, -4.4169977039979591e-25, 0.0, 0.0 },
            { -4.4169977039979591e-25, 3.0014760013120012e+15, 0.0, 0.0 },
            { 0.0, 0.0, 2.8031129657300436e+16, 3.1655150211985374e+16 },
            { 0.0, 0.0, 3.1655150211985374e+16, 4.4226787905292953e+16 },
        };
        var x = DecoupledHeterogeneousPanel(1e-12m, 1e8m);

        AssertComponentwise(new LedoitWolfShrinkageEstimator<decimal>().Estimate(x), expected, 1e-12, 2e-28);
    }

    // Coupled columns (every column nonzero in every row): every entry involving A or B, including the cross
    // entries to the large columns, comes from products formed beyond the range at the common scale.
    [Fact]
    public void LedoitWolfIdentity_CoupledSmallColumns_KeepEveryEntry()
    {
        double[,] expected =
        {
            { 6.899271392290938e+197, -2.5526158199808441e-143, -1.1779470072502074e+27, -3.6654202087306986e+26 },
            { -2.5526158199808441e-143, 6.899271392290938e+197, -1.7443640788215837e+28, 3.578953885858077e+28 },
            { -1.1779470072502074e+27, -1.7443640788215837e+28, 6.6600267325674886e+199, -7.5063208882432754e+197 },
            { -3.6654202087306986e+26, 3.578953885858077e+28, -7.5063208882432754e+197, 4.0011832084783451e+198 },
        };

        var estimate = new LedoitWolfShrinkageEstimator<double>().Estimate(CoupledHeterogeneousPanel());

        AssertComponentwise(estimate, expected, 1e-12);
    }

    // Two assets, 2^-830 and 2^330: at the common scale the small column is 2^-1160 and vanishes entirely. The
    // denoised estimator forwards fewer than three assets to the scaled-identity estimator when shrinkage is on.
    [Fact]
    public void LedoitWolfIdentity_TwoAssetsAcrossTheRange_KeepTheirCrossCovarianceThroughTheDenoisedRoute()
    {
        int[] p = [2, -3, 1, 4, -2, -1, 3, -4, 0, 2, -2, 1, -3, 2, 0, -1];
        int[] q = [1, -2, 2, 3, -1, -2, 2, -3, 1, 1, -2, 2, -2, 1, 1, -1];
        var x = new double[p.Length, 2];
        for (var k = 0; k < p.Length; k++)
        {
            x[k, 0] = Math.ScaleB(p[k], -830);
            x[k, 1] = Math.ScaleB(q[k], 330);
        }

        double[,] expected =
        {
            { 6.5274326621322607e+197, 1.1474549789092862e-150 },
            { 1.1474549789092862e-150, 1.6231022050057807e+199 },
        };

        AssertComponentwise(new LedoitWolfShrinkageEstimator<double>().Estimate(x), expected, 1e-12);
        AssertComponentwise(new DenoisedCovarianceEstimator<double>(true).Estimate(x), expected, 1e-12);
    }

    // Constant correlation, F-L1-07 double panel (delta = 0.1339753214, r-bar = 0.1265260997). The A,B standard
    // deviations underflow at the common scale, so r-bar kept only the C,D pair, the intensity collapsed and
    // every entry involving A or B, and the dominant C,D entry, moved.
    [Fact]
    public void LedoitWolfConstantCorrelation_SmallColumnsBesideNormalizingColumns_KeepEveryEntry()
    {
        double[,] expected =
        {
            { 1.0306375585638224e-144, -1.3221956825907383e-145, 6.4721268142302402e+25, 8.3061844924531851e+25 },
            { -1.3221956825907383e-145, 5.2924631385709798e-145, 4.6379168334270549e+25, 5.9521999467621911e+25 },
            { 6.4721268142302402e+25, 4.6379168334270549e+25, 1.4144194341232137e+199, 1.5799369534881487e+199 },
            { 8.3061844924531851e+25, 5.9521999467621911e+25, 1.5799369534881487e+199, 2.3296320091441166e+199 },
        };

        var estimate = new LedoitWolfConstantCorrelationEstimator<double>()
            .Estimate(DecoupledHeterogeneousPanel(Math.ScaleB(1.0, -240), Math.ScaleB(1.0, 330)));

        AssertComponentwise(estimate, expected, 1e-12);
    }

    [Fact]
    public void LedoitWolfConstantCorrelation_DecimalSmallColumnsBesideLargeColumns_KeepEveryEntry()
    {
        double[,] expected =
        {
            { 3.2173913043478261e-24, -4.1275624553617011e-25, 5.228142404995943e-6, 6.7096823988729345e-6 },
            { -4.1275624553617011e-25, 1.6521739130434783e-24, 3.746479382074404e-6, 4.8081488261727031e-6 },
            { 5.228142404995943e-6, 3.746479382074404e-6, 2.9565217391304348e+16, 3.3024984221450603e+16 },
            { 6.7096823988729345e-6, 4.8081488261727031e-6, 3.3024984221450603e+16, 4.8695652173913043e+16 },
        };
        var x = DecoupledHeterogeneousPanel(1e-12m, 1e8m);

        AssertComponentwise(new LedoitWolfConstantCorrelationEstimator<decimal>().Estimate(x), expected, 1e-12, 2e-28);
    }

    // Coupled panel (delta clamps to 1, so the estimate is the target): r-bar must average all six pairwise
    // correlations, not only the C,D pair whose standard deviations survive the common scale.
    [Fact]
    public void LedoitWolfConstantCorrelation_CoupledSmallColumns_KeepEveryEntry()
    {
        double[,] expected =
        {
            { 2.2457049960285393e-144, -4.2614234353898953e-144, -5.8738805798483224e+26, -1.3165723191297123e+26 },
            { -4.2614234353898953e-144, 3.6073710235303844e-141, -2.3542043964353407e+28, -5.2767166437698961e+27 },
            { -5.8738805798483224e+26, -2.3542043964353407e+28, 6.8538036449472071e+199, -7.2733451366976321e+197 },
            { -1.3165723191297123e+26, -5.2767166437698961e+27, -7.2733451366976321e+197, 3.4432683631393482e+198 },
        };

        var estimate = new LedoitWolfConstantCorrelationEstimator<double>().Estimate(CoupledHeterogeneousPanel());

        AssertComponentwise(estimate, expected, 1e-12);
    }

    // One dominant column (1e7) beside three moderate ones (0.03, 0.002, 0.01), decimal: the common normalization
    // fires. The target shares the sample diagonal, so the dominant diagonal cancels from pi-hat - rho-hat and
    // vanishes from gamma-hat, and the intensity (delta = 0.8798278757) rests on the pairs. At the common scale those
    // pairs' fourth-order terms sit at 1e-27 to 1e-36, on or below decimal's 1e-28 grid.
    [Fact]
    public void LedoitWolfConstantCorrelation_DecimalOneDominantColumnUnderNormalization_KeepsTheIntensity()
    {
        decimal[] scales = [1e7m, 0.03m, 0.002m, 0.01m];
        var x = new decimal[s_counts.GetLength(0), 4];
        for (var k = 0; k < s_counts.GetLength(0); k++)
        {
            for (var j = 0; j < 4; j++)
            {
                x[k, j] = s_counts[k, j] * scales[j];
            }
        }

        double[,] expected =
        {
            { 7.0105263157894737e+14, -2.0308381187293357e+5, -1.0727503656261552e+4, -5.2228680013969391e+4 },
            { -2.0308381187293357e+5, 9.8976315789473684e-3, -3.467366779388411e-5, -5.1566550260954882e-6 },
            { -1.0727503656261552e+4, -3.467366779388411e-5, 5.7305263157894737e-5, -1.2269572288285864e-5 },
            { -5.2228680013969391e+4, -5.1566550260954882e-6, -1.2269572288285864e-5, 1.1515789473684211e-3 },
        };

        var estimate = new LedoitWolfConstantCorrelationEstimator<decimal>().Estimate(x);

        AssertComponentwise(estimate, expected, 1e-12, 2e-28);
    }

    // Surfaced (no normalization): one dominant column (2^10) beside columns 2^-25..2^-35 of it. pi-hat and rho-hat
    // both carry the dominant pi_11 term, which cancels exactly between them; the pair terms that set the intensity
    // are 2^-70 of it and were lost to the rounding of that difference (oracle delta clamps at 1).
    [Fact]
    public void LedoitWolfConstantCorrelation_OneDominantColumn_KeepsTheOffDiagonalIntensity()
    {
        double[,] expected =
        {
            { 7.3510696421052632e+6, -3.9646304940316084e-7, -1.414085312310568e-8, -1.2982405393219896e-5 },
            { -3.9646304940316084e-7, 9.5386965869829959e-18, -1.6108120143708866e-20, -1.4788509859183916e-17 },
            { -1.414085312310568e-8, -1.6108120143708866e-20, 1.2134861486453713e-20, -5.2746944801825656e-19 },
            { -1.2982405393219896e-5, -1.4788509859183916e-17, -5.2746944801825656e-19, 1.0228075696336179e-14 },
        };

        var estimate = new LedoitWolfConstantCorrelationEstimator<double>().Estimate(ScaledCounts(10, -30, -35, -25));

        AssertComponentwise(estimate, expected, 1e-12);
    }

    // Surfaced (no normalization): two unit-scale columns beside columns at 2^-560 and 2^-565. The small pair's
    // covariance (about 1e-336) is below double's range, so its standard deviations came out of a subnormal or zero
    // variance and its correlation was lost from r-bar, although every correlation is an ordinary number. The small
    // block itself is not representable (expected entries near 1e-336), hence the absolute floor.
    [Fact]
    public void LedoitWolfConstantCorrelation_TinyColumnsBesideUnitColumns_KeepTheirCorrelationsInRBar()
    {
        double[,] expected =
        {
            { 7.0105263157894737, -8.6866894905659124e-1, -1.5416062745350195e-169, -4.9135224883784623e-171 },
            { -8.6866894905659124e-1, 1.0997368421052632e+1, -1.4993922251828459e-169, 2.9912817230934234e-171 },
            { -1.5416062745350195e-169, -1.4993922251828459e-169, 1.0058642577422409e-336, -1.3546402957676596e-339 },
            { -4.9135224883784623e-171, 2.9912817230934234e-171, -1.3546402957676596e-339, 7.8958450384742058e-340 },
        };

        var estimate = new LedoitWolfConstantCorrelationEstimator<double>().Estimate(ScaledCounts(0, 0, -560, -565));

        AssertComponentwise(estimate, expected, 1e-12, 1e-320);
    }
}
