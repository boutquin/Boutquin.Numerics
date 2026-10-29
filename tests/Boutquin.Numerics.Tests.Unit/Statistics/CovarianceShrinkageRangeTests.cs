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
/// Range behaviour of the single-factor Ledoit-Wolf estimator. Every expected matrix is an independent
/// 200-digit mpmath evaluation of the estimator's documented formula (Ledoit and Wolf 2003, Bessel blend) on
/// the exact inputs, re-checked at 400 digits; no arithmetic is shared with the code under test.
/// </summary>
public sealed class CovarianceShrinkageRangeTests
{
    private static void AssertComponentwise<T>(T[,] actual, double[,] expected, double relativeTolerance, double absoluteFloor = 0.0)
        where T : IFloatingPoint<T>
    {
        for (var i = 0; i < expected.GetLength(0); i++)
        {
            for (var j = 0; j < expected.GetLength(1); j++)
            {
                var observed = double.CreateChecked(actual[i, j]);
                var allowed = Math.Max(relativeTolerance * Math.Abs(expected[i, j]), absoluteFloor);
                double.IsFinite(observed).Should().BeTrue($"entry [{i},{j}] expected {expected[i, j]:R}, actual {observed:R}");
                Math.Abs(observed - expected[i, j]).Should().BeLessThanOrEqualTo(
                    allowed,
                    $"entry [{i},{j}] expected {expected[i, j]:R}, actual {observed:R}");
            }
        }
    }

    // T observations of two columns; row 0 carries the market (+a, +a), every other row is zero.
    private static T[,] IdenticalSpikeColumns<T>(T a, int t)
        where T : IFloatingPoint<T>
    {
        var x = new T[t, 2];
        x[0, 0] = a;
        x[0, 1] = a;
        return x;
    }

    // F-L23-R2C4-04. Two identical columns [a, 0, ..., 0], T = 8: the single-factor model fits exactly, so the
    // estimate is the sample covariance a^2 / 8 for any intensity. The unit market has Var(u) = 8/49, so the
    // loading product loading_i . loading_j = a^2 / 8 / Var(u) leaves the type's range from 0.163 . Max although
    // the target and the result (0.28 . Max, 0.25 . Max) are representable.
    [Fact]
    public void SingleFactor_IdenticalColumnsNearTheTopOfTheRange_ReturnTheSampleCovariance()
    {
        var aDouble = Math.ScaleB(3.0, 511);
        var expectedDouble = 9.0 * Math.ScaleB(1.0, 1019); // 5.0560119418002635e+307
        AssertComponentwise(
            new LedoitWolfSingleFactorEstimator<double>().Estimate(IdenticalSpikeColumns(aDouble, 8)),
            new[,] { { expectedDouble, expectedDouble }, { expectedDouble, expectedDouble } },
            1e-14);

        var aFloat = MathF.ScaleB(3.0f, 63);
        var expectedFloat = 9.0 * Math.ScaleB(1.0, 123); // 9.5704415696513943e+37
        AssertComponentwise(
            new LedoitWolfSingleFactorEstimator<float>().Estimate(IdenticalSpikeColumns(aFloat, 8)),
            new[,] { { expectedFloat, expectedFloat }, { expectedFloat, expectedFloat } },
            1e-6);

        var expectedDecimal = new[,] { { 2e28, 2e28 }, { 2e28, 2e28 } };
        var x = IdenticalSpikeColumns(400000000000000m, 8);
        AssertComponentwise(new LedoitWolfSingleFactorEstimator<decimal>().Estimate(x), expectedDecimal, 1e-14);
    }

    // T = 100, row 0 = (c, c), rows 1..99 = (c s, -c s) with s = 2^-20: the market is the first row alone, so
    // Var(u) is about 1/(T - 1) and the original-units loadings reach sd . sqrt(T - 1).
    private static T[,] SpikeMarket<T>(T c, T cs)
        where T : IFloatingPoint<T>
    {
        var x = new T[100, 2];
        x[0, 0] = c;
        x[0, 1] = c;
        for (var k = 1; k < 100; k++)
        {
            x[k, 0] = cs;
            x[k, 1] = -cs;
        }

        return x;
    }

    // F-L1-05. The spike panel scaled near the top of each type: every input and every returned entry is
    // representable, but loading_i . loading_j (about (T - 1) times the target) is not. Double at 2^513 and
    // float at 2^65 (oracle delta 0: the off-diagonal is the sample covariance), decimal at 3e14 (delta 1).
    [Fact]
    public void SingleFactor_SpikeMarketNearTheTopOfTheRange_KeepsTheOffDiagonal()
    {
        AssertComponentwise(
            new LedoitWolfSingleFactorEstimator<double>().Estimate(SpikeMarket(Math.ScaleB(1.0, 513), Math.ScaleB(1.0, 493))),
            new[,]
            {
                { 7.1907588241456316e+306, 7.1907725394427237e+306 },
                { 7.1907725394427237e+306, 7.1907862547659756e+306 },
            },
            1e-12);

        AssertComponentwise(
            new LedoitWolfSingleFactorEstimator<float>().Estimate(SpikeMarket(MathF.ScaleB(1.0f, 65), MathF.ScaleB(1.0f, 45))),
            new[,]
            {
                { 1.3611268715365625e+37, 1.3611294676825159e+37 },
                { 1.3611294676825159e+37, 1.3611320638334211e+37 },
            },
            2e-5);

        var expectedDecimal = new[,]
        {
            { 8.9999828338704901e+26, 8.9999999999918145e+26 },
            { 8.9999999999918145e+26, 9.0000171661458808e+26 },
        };
        var x = SpikeMarket(300000000000000m, 286102294.921875m); // 3e14 . 2^-20 exactly
        AssertComponentwise(new LedoitWolfSingleFactorEstimator<decimal>().Estimate(x), expectedDecimal, 1e-12);
    }

    private static readonly int[,] s_dominantCounts =
    {
        { -8, -4, -6, 2 }, { 6, -2, 3, 8 }, { -6, 9, -2, -9 }, { -3, 4, -1, -4 }, { 3, -4, -7, -5 },
        { 5, -5, -5, -9 }, { -9, -3, -3, -4 }, { -4, 0, 1, -3 }, { 8, -3, -4, -3 }, { 3, 0, -9, 2 },
        { 4, -4, -5, -1 }, { -7, 1, 0, 9 }, { -9, 1, -7, 0 }, { 2, 0, 6, 1 }, { -4, 6, 6, -4 },
        { -8, -1, -9, 2 }, { 3, -9, 8, 4 },
    };

    // Surfaced S3 (FX-R2-LW sibling census). One column far above the rest, unnormalized double (columns at
    // 2^15, 2^-10, 2^-24, 2^-17). p-hat and r-hat both carry p_ii = r_ii of the dominant column, of order 2^60
    // above the pair terms that decide the intensity; their difference rounded those away. Oracle delta 1.
    [Fact]
    public void SingleFactor_OneDominantColumn_KeepsThePairIntensity()
    {
        int[] exponents = [15, -10, -24, -17];
        var x = new double[s_dominantCounts.GetLength(0), exponents.Length];
        for (var k = 0; k < x.GetLength(0); k++)
        {
            for (var j = 0; j < exponents.Length; j++)
            {
                x[k, j] = Math.ScaleB(s_dominantCounts[k, j], exponents[j]);
            }
        }

        double[,] expected =
        {
            { 3.7186205816470588e+10, -3.495293971856355e+2, 9.5214845690353871e-3, -7.4448573078349093e-2 },
            { -3.495293971856355e+2, 1.7909442677217371e-5, -8.9496593929279256e-11, 6.9977466907582166e-10 },
            { 9.5214845690353871e-3, -8.9496593929279256e-11, 1.0080825063596421e-13, -1.9062470187217498e-14 },
            { -7.4448573078349093e-2, 6.9977466907582166e-10, -1.9062470187217498e-14, 1.5151111737770193e-9 },
        };

        AssertComponentwise(new LedoitWolfSingleFactorEstimator<double>().Estimate(x), expected, 1e-12);
    }

    private static readonly int[,] s_decimalDominantCounts =
    {
        { -7, 1, -5 }, { 2, -6, 9 }, { 5, 1, -4 }, { 8, 9, -6 }, { 2, -6, -7 }, { -3, -3, 1 }, { 2, -4, 5 },
        { -7, -2, 3 }, { 8, 5, 3 }, { -9, -6, 2 }, { -5, -9, 6 }, { 3, 1, -7 }, { 6, 0, -9 }, { -6, 6, 2 },
        { -3, 1, 3 }, { -5, 0, 6 }, { -7, 9, -6 }, { 0, 4, 1 }, { -7, 7, 4 },
    };

    // Surfaced S3, decimal route: one column at 2e7 beside columns at 0.004 and 0.3 is normalized by the common
    // maximum, where the small columns' fourth-order pair terms sit near decimal's 1e-28 grid. Oracle delta 0, so
    // the estimate is the sample covariance.
    [Fact]
    public void SingleFactor_DecimalOneDominantColumnUnderNormalization_KeepsThePairIntensity()
    {
        decimal[] scales = [20000000m, 0.004m, 0.3m];
        var x = new decimal[s_decimalDominantCounts.GetLength(0), scales.Length];
        for (var k = 0; k < x.GetLength(0); k++)
        {
            for (var j = 0; j < scales.Length; j++)
            {
                x[k, j] = s_decimalDominantCounts[k, j] * scales[j];
            }
        }

        double[,] expected =
        {
            { 1.2514619883040936e+16, 2.7415204678362573e+5, -5.9263157894736842e+7 },
            { 2.7415204678362573e+5, 4.5033918128654971e-4, -1.1294736842105263e-2 },
            { -5.9263157894736842e+7, -1.1294736842105263e-2, 2.6147368421052632 },
        };

        AssertComponentwise(new LedoitWolfSingleFactorEstimator<decimal>().Estimate(x), expected, 1e-12, 2e-28);
    }
}
