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

/// <summary>
/// Column-scale equivariance of the correlation-based estimators at <c>T = decimal</c>. Scaling one column by an
/// exact power of ten <c>d</c> multiplies that column's row and column of the estimate by <c>d</c>
/// (<c>E(X D) = D E(X) D</c>). The expected matrix is the estimate of the per-column-normalised (unscaled) panel
/// times exact powers of ten, and that unscaled estimate is itself anchored to the <c>double</c> instantiation
/// of the same estimator, a separate arithmetic path. A column of magnitude about <c>1e-14</c> has a variance
/// below decimal's <c>1e-28</c> grid, but its correlations and its off-diagonal covariances are representable
/// and must survive.
/// </summary>
public sealed class CorrelationBasedEstimatorDecimalColumnScaleTests
{
    private const int Observations = 40;
    private const int Assets = 5;

    public static TheoryData<string, int> Cases
    {
        get
        {
            var data = new TheoryData<string, int>();
            foreach (var name in new[]
                     {
                         "Denoised", "Detoned", "TracyWidom", "DoublySparse",
                     })
            {
                foreach (var exponent in new[] { 0, 10, 12, 13 })
                {
                    data.Add(name, exponent);
                }
            }

            return data;
        }
    }

    private static Func<decimal[,], decimal[,]> CreateDecimal(string name) => name switch
    {
        "Denoised" => new DenoisedCovarianceEstimator<decimal>().Estimate,
        "Detoned" => new DetonedCovarianceEstimator<decimal>().Estimate,
        "TracyWidom" => new TracyWidomDenoisedCovarianceEstimator<decimal>().Estimate,
        "DoublySparse" => new DoublySparseEstimator<decimal>().Estimate,
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, null),
    };

    private static Func<double[,], double[,]> CreateDouble(string name) => name switch
    {
        "Denoised" => new DenoisedCovarianceEstimator<double>().Estimate,
        "Detoned" => new DetonedCovarianceEstimator<double>().Estimate,
        "TracyWidom" => new TracyWidomDenoisedCovarianceEstimator<double>().Estimate,
        "DoublySparse" => new DoublySparseEstimator<double>().Estimate,
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, null),
    };

    // One strong common factor plus idiosyncratic noise, rounded to ten decimal places so the panel and its
    // power-of-ten scalings are exact decimals.
    private static decimal[,] FactorPanel(int seed)
    {
        var random = new System.Random(seed);
        var x = new decimal[Observations, Assets];
        for (var k = 0; k < Observations; k++)
        {
            var factor = random.NextDouble() - 0.5;
            for (var j = 0; j < Assets; j++)
            {
                var value = 0.01 * ((1.0 + 0.3 * j) * factor + 0.4 * (random.NextDouble() - 0.5));
                x[k, j] = Math.Round((decimal)value, 10);
            }
        }

        return x;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Decimal_ColumnScaledByPowerOfTen_KeepsItsRepresentableCovariances(string name, int exponent)
    {
        var panel = FactorPanel(20261008);
        var asDouble = new double[Observations, Assets];
        var scaled = new decimal[Observations, Assets];
        var d = 1m;
        for (var e = 0; e < exponent; e++)
        {
            d /= 10m;
        }

        for (var k = 0; k < Observations; k++)
        {
            for (var j = 0; j < Assets; j++)
            {
                asDouble[k, j] = (double)panel[k, j];
                scaled[k, j] = j == 0 ? panel[k, j] * d : panel[k, j];
                if (j == 0)
                {
                    (scaled[k, j] / d).Should().Be(panel[k, j], "the power-of-ten scaling of the panel is exact");
                }
            }
        }

        var estimate = CreateDecimal(name);
        var unit = estimate(panel);

        // Anchor the unscaled decimal estimate to the double instantiation of the same estimator.
        var doubleUnit = CreateDouble(name)(asDouble);
        for (var i = 0; i < Assets; i++)
        {
            for (var j = 0; j < Assets; j++)
            {
                var relative = Math.Abs((double)unit[i, j] - doubleUnit[i, j]) / Math.Abs(doubleUnit[i, j]);
                relative.Should().BeLessThan(
                    1e-8, $"{name} unscaled [{i},{j}]: decimal {unit[i, j]}, double {doubleUnit[i, j]}");
            }
        }

        var actual = estimate(scaled);
        var compared = 0;
        for (var i = 0; i < Assets; i++)
        {
            for (var j = 0; j < Assets; j++)
            {
                var expected = unit[i, j] * (i == 0 ? d : 1m) * (j == 0 ? d : 1m);
                var error = Math.Abs(actual[i, j] - expected);
                var bound = 1e-9m * Math.Abs(expected) + 1e-26m;
                error.Should().BeLessThanOrEqualTo(
                    bound,
                    $"{name} entry [{i},{j}] with column 0 scaled by 1e-{exponent}: expected {expected}, actual {actual[i, j]}");
                if ((i == 0) != (j == 0) && Math.Abs(expected) > 1e-20m)
                {
                    compared++;
                }
            }
        }

        // Every off-diagonal entry of column 0 (both triangles) is far above the decimal grid.
        compared.Should().Be(2 * (Assets - 1));
    }
}
