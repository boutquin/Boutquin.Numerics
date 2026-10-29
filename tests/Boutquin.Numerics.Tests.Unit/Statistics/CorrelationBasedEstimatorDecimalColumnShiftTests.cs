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
/// Translation invariance of the decimal sample correlation behind the correlation-based estimators. Adding a
/// constant level to a column leaves its centred covariance, its standard deviation and every correlation
/// unchanged, so a column that sits far above its own spread (for example <c>1e13 + 0.01 z</c>) must keep the
/// variance and correlations it has at level zero. The expected values come from an exact rational evaluation
/// of the centred covariance of the stored decimal inputs, not from the code path under test; the public
/// estimators are compared against the same estimator on the unshifted panel.
/// </summary>
public sealed class CorrelationBasedEstimatorDecimalColumnShiftTests
{
    private const int Seed = 20261008;
    private const int OracleDigits = 40;
    private const int SigmaDigits = 60;

    private static readonly decimal[] s_levels = [0m, 1e4m, 1e7m, 1e13m];
    private static readonly decimal[] s_spreads = [0.00000001m, 0.000001m, 0.01m, 1m];

    // Per-column multipliers of the spread: mixed positive rescales within one panel.
    private static readonly decimal[] s_multipliers = [1m, 0.0001m, 10000m, 0.01m, 100m];

    public static TheoryData<decimal, decimal, int> OracleCases
    {
        get
        {
            var data = new TheoryData<decimal, decimal, int>();
            foreach (var level in s_levels)
            {
                foreach (var spread in s_spreads)
                {
                    foreach (var observations in new[] { 2, 3, 40 })
                    {
                        data.Add(level, spread, observations);
                    }
                }
            }

            return data;
        }
    }

    public static TheoryData<string, decimal, decimal> EstimatorCases
    {
        get
        {
            var data = new TheoryData<string, decimal, decimal>();
            foreach (var name in new[]
                     {
                         "Denoised", "Detoned", "TracyWidom", "DoublySparse",
                     })
            {
                foreach (var level in s_levels.Where(l => l != 0m))
                {
                    foreach (var spread in s_spreads)
                    {
                        data.Add(name, level, spread);
                    }
                }
            }

            return data;
        }
    }

    private static Func<decimal[,], decimal[,]> Create(string name) => name switch
    {
        "Denoised" => new DenoisedCovarianceEstimator<decimal>().Estimate,
        "Detoned" => new DetonedCovarianceEstimator<decimal>().Estimate,
        "TracyWidom" => new TracyWidomDenoisedCovarianceEstimator<decimal>().Estimate,
        "DoublySparse" => new DoublySparseEstimator<decimal>().Estimate,
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, null),
    };

    // Integer factor scores: one common factor plus idiosyncratic noise, |z| <= 700, so every stored value
    // level + multiplier * spread * z is an exact decimal (at most 26 significant digits at level 1e13).
    private static int[,] FactorScores(int observations, int assets, int seed)
    {
        var random = new System.Random(seed);
        var z = new int[observations, assets];
        for (var k = 0; k < observations; k++)
        {
            var factor = random.Next(-400, 401);
            for (var j = 0; j < assets; j++)
            {
                z[k, j] = factor + random.Next(-300, 301);
            }
        }

        return z;
    }

    // Columns 0-4: factor columns at level (column 1 at -level) with spread * multiplier; column 5: constant at
    // the level; column 6: near-constant (one row moved by the spread); column 7: absolute scale 1e-12;
    // column 8: absolute scale 1e12 (scores divided by 14 to stay inside the decimal domain bound 1e14).
    private static decimal[,] ShiftedPanel(decimal level, decimal spread, int observations)
    {
        const int assets = 9;
        var z = FactorScores(observations, assets, Seed + observations);
        var x = new decimal[observations, assets];
        for (var k = 0; k < observations; k++)
        {
            for (var j = 0; j < s_multipliers.Length; j++)
            {
                var columnLevel = j == 1 ? -level : level;
                x[k, j] = columnLevel + (s_multipliers[j] * spread * z[k, j]);
            }

            x[k, 5] = level;
            x[k, 6] = k == observations - 1 ? level + spread : level;
            x[k, 7] = level + (0.000000000001m * z[k, 7]);
            x[k, 8] = level + (1000000000000m * (z[k, 8] / 14));
        }

        return x;
    }

    [Theory]
    [MemberData(nameof(OracleCases))]
    public void Decimal_SampleCorrelation_ShiftedAndRescaledColumns_MatchExactRationalOracle(
        decimal level, decimal spread, int observations)
    {
        var panel = ShiftedPanel(level, spread, observations);
        var assets = panel.GetLength(1);
        var (correlation, stdDevs) = CovarianceHelpers<decimal>.SampleCorrelation(panel);

        var centred = ExactCentredCrossProducts(panel);
        var n = new BigInteger(observations);
        var oracleUnit = BigInteger.Pow(10, OracleDigits);
        var rhoTolerance = BigInteger.Pow(10, OracleDigits - 20);
        var nonVacuous = 0;
        for (var i = 0; i < assets; i++)
        {
            // sigma_i = sqrt(C_ii / (n (n - 1))) / 10^28, in fixed point at 10^SigmaDigits. Tolerance: 1e-20
            // relative plus one unit of decimal's 1e-28 grid, the resolution at which sigma_i is returned.
            var (sigmaMantissa, sigmaScale) = Exact(stdDevs[i]);
            if (centred[i, i].IsZero)
            {
                stdDevs[i].Should().Be(0m, $"column {i} is constant (level {level}, spread {spread}, n={observations})");
            }
            else
            {
                var expectedSigma = IntegerSqrt(centred[i, i] * BigInteger.Pow(10, (2 * SigmaDigits) - 56) / (n * (n - 1)));
                var actualSigma = sigmaMantissa * BigInteger.Pow(10, SigmaDigits - sigmaScale);
                var sigmaTolerance = (expectedSigma / BigInteger.Pow(10, 20)) + BigInteger.Pow(10, SigmaDigits - 28);
                BigInteger.Abs(actualSigma - expectedSigma).Should().BeLessThanOrEqualTo(
                    sigmaTolerance,
                    $"sigma[{i}] at level {level}, spread {spread}, n={observations}: actual {stdDevs[i]}, " +
                    $"exact {expectedSigma} * 1e-{SigmaDigits}");
                nonVacuous++;
            }

            for (var j = 0; j < assets; j++)
            {
                var because = $"rho[{i},{j}] at level {level}, spread {spread}, n={observations}: actual {correlation[i, j]}";
                if (centred[i, i].IsZero || centred[j, j].IsZero)
                {
                    correlation[i, j].Should().Be(i == j ? 1m : 0m, because + " (constant-column convention)");
                    continue;
                }

                var squared = centred[i, j] * centred[i, j] * BigInteger.Pow(oracleUnit, 2) / (centred[i, i] * centred[j, j]);
                var expected = IntegerSqrt(squared) * centred[i, j].Sign;
                var (rhoMantissa, rhoScale) = Exact(correlation[i, j]);
                var actual = rhoMantissa * BigInteger.Pow(10, OracleDigits - rhoScale);
                BigInteger.Abs(actual - expected).Should().BeLessThanOrEqualTo(
                    rhoTolerance, because + $", exact {expected} * 1e-{OracleDigits}");
            }
        }

        // Every column except the constant one has a positive exact variance and was compared.
        nonVacuous.Should().Be(assets - 1);
    }

    [Theory]
    [MemberData(nameof(EstimatorCases))]
    public void Decimal_Estimator_ColumnsShiftedByALevel_MatchTheUnshiftedEstimate(string name, decimal level, decimal spread)
    {
        const int observations = 40;
        var shifted = ShiftedPanel(level, spread, observations);
        var unshifted = ShiftedPanel(0m, spread, observations);
        var factorColumns = s_multipliers.Length;
        var shiftedFactors = new decimal[observations, factorColumns];
        var unshiftedFactors = new decimal[observations, factorColumns];
        for (var k = 0; k < observations; k++)
        {
            for (var j = 0; j < factorColumns; j++)
            {
                shiftedFactors[k, j] = shifted[k, j];
                unshiftedFactors[k, j] = unshifted[k, j];
                (shifted[k, j] - (j == 1 ? -level : level)).Should().Be(unshifted[k, j], "the shift is exact");
            }
        }

        var estimate = Create(name);
        var expected = estimate(unshiftedFactors);
        var actual = estimate(shiftedFactors);
        for (var i = 0; i < factorColumns; i++)
        {
            for (var j = 0; j < factorColumns; j++)
            {
                var scale = Math.Sqrt(Math.Abs((double)expected[i, i] * (double)expected[j, j]));
                var error = Math.Abs((double)(actual[i, j] - expected[i, j]));
                error.Should().BeLessThanOrEqualTo(
                    1e-12 * scale,
                    $"{name} entry [{i},{j}] with columns shifted by {level} (spread {spread}): expected {expected[i, j]}, actual {actual[i, j]}");
            }

            expected[i, i].Should().BePositive($"{name} unshifted variance [{i},{i}] is non-vacuous");
        }
    }

    // C_ij = n * sum(x_i x_j) - sum(x_i) * sum(x_j) on integers x = value * 10^28: n (n - 1) 10^56 times the
    // unbiased centred covariance of the stored decimals, exactly.
    private static BigInteger[,] ExactCentredCrossProducts(decimal[,] panel)
    {
        var t = panel.GetLength(0);
        var assets = panel.GetLength(1);
        var x = new BigInteger[t, assets];
        var sums = new BigInteger[assets];
        for (var k = 0; k < t; k++)
        {
            for (var j = 0; j < assets; j++)
            {
                var (mantissa, scale) = Exact(panel[k, j]);
                x[k, j] = mantissa * BigInteger.Pow(10, 28 - scale);
                sums[j] += x[k, j];
            }
        }

        var c = new BigInteger[assets, assets];
        for (var i = 0; i < assets; i++)
        {
            for (var j = 0; j < assets; j++)
            {
                var cross = BigInteger.Zero;
                for (var k = 0; k < t; k++)
                {
                    cross += x[k, i] * x[k, j];
                }

                c[i, j] = (t * cross) - (sums[i] * sums[j]);
            }
        }

        return c;
    }

    // value = mantissa / 10^scale exactly.
    private static (BigInteger Mantissa, int Scale) Exact(decimal value)
    {
        var bits = decimal.GetBits(value);
        var magnitude = (new BigInteger((uint)bits[2]) << 64) | (new BigInteger((uint)bits[1]) << 32) | new BigInteger((uint)bits[0]);
        var scale = (bits[3] >> 16) & 0xFF;
        return (bits[3] < 0 ? -magnitude : magnitude, scale);
    }

    private static BigInteger IntegerSqrt(BigInteger value)
    {
        if (value.IsZero)
        {
            return BigInteger.Zero;
        }

        var root = BigInteger.One << (int)((value.GetBitLength() + 1) / 2);
        while (true)
        {
            var next = (root + (value / root)) >> 1;
            if (next >= root)
            {
                return root;
            }

            root = next;
        }
    }
}
