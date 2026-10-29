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
/// Column-scale equivariance of the correlation-based estimators. Scaling one column of the returns by an exact
/// power of two <c>d</c> multiplies that column's row and column of the estimate by <c>d</c>
/// (<c>E(X D) = D E(X) D</c>), so the expected matrix comes from the same estimator on the unscaled panel and
/// an exact scaling, not from the code path under test. Only entries whose scaled value is a normal number are
/// compared; the scaled variance itself can leave the type's range while the correlations and the
/// off-diagonal covariances do not.
/// </summary>
public sealed class CorrelationBasedEstimatorColumnScaleTests
{
    public static TheoryData<string, int> DoubleCases => new()
    {
        { "Denoised", -530 }, { "Denoised", -560 }, { "Denoised", -1000 },
        { "Detoned", -530 }, { "Detoned", -560 }, { "Detoned", -1000 },
        { "TracyWidom", -530 }, { "TracyWidom", -560 }, { "TracyWidom", -1000 },
        { "DoublySparse", -530 }, { "DoublySparse", -560 }, { "DoublySparse", -1000 },
    };

    public static TheoryData<string, int> SingleCases => new()
    {
        { "Denoised", -60 }, { "Denoised", -70 },
        { "Detoned", -60 }, { "Detoned", -70 },
        { "TracyWidom", -60 }, { "TracyWidom", -70 },
        { "DoublySparse", -60 }, { "DoublySparse", -70 },
    };

    private static ICovarianceEstimator<T> Create<T>(string name)
        where T : struct, IFloatingPoint<T> => name switch
        {
            "Denoised" => new DenoisedCovarianceEstimator<T>(),
            "Detoned" => new DetonedCovarianceEstimator<T>(),
            "TracyWidom" => new TracyWidomDenoisedCovarianceEstimator<T>(),
            "DoublySparse" => new DoublySparseEstimator<T>(),
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, null),
        };

    // One strong common factor plus idiosyncratic noise, so every estimator keeps a signal component.
    private static double[,] FactorPanel(int seed, int t, int n)
    {
        var random = new System.Random(seed);
        var x = new double[t, n];
        for (var k = 0; k < t; k++)
        {
            var factor = random.NextDouble() - 0.5;
            for (var j = 0; j < n; j++)
            {
                x[k, j] = 0.01 * ((1.0 + 0.3 * j) * factor + 0.4 * (random.NextDouble() - 0.5));
            }
        }

        return x;
    }

    private static void AssertColumnEquivariant<T>(string name, int exponent, double relativeTolerance)
        where T : struct, IFloatingPointIeee754<T>
    {
        var panel = FactorPanel(20261008, 40, 5);
        var t = panel.GetLength(0);
        var n = panel.GetLength(1);
        var x = new T[t, n];
        var scaled = new T[t, n];
        var d = T.ScaleB(T.One, exponent);
        for (var k = 0; k < t; k++)
        {
            for (var j = 0; j < n; j++)
            {
                x[k, j] = T.CreateChecked(panel[k, j]);
                scaled[k, j] = j == 0 ? x[k, j] * d : x[k, j];
            }
        }

        var reference = Create<T>(name).Estimate(x);
        var actual = Create<T>(name).Estimate(scaled);

        var compared = 0;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var factor = (i == 0 ? d : T.One) * (j == 0 ? d : T.One);
                var expected = reference[i, j] * factor;
                if (!T.IsNormal(expected))
                {
                    continue;
                }

                compared++;
                var relativeError = double.CreateChecked(T.Abs(actual[i, j] - expected) / T.Abs(expected));
                relativeError.Should().BeLessThanOrEqualTo(
                    relativeTolerance,
                    $"{name} entry [{i},{j}] with column 0 scaled by 2^{exponent}: expected {expected}, actual {actual[i, j]}");
            }
        }

        // Every off-diagonal entry of column 0 and the whole unscaled block are normal.
        compared.Should().Be(n * n - 1);
    }

    [Theory]
    [MemberData(nameof(DoubleCases))]
    public void Double_ColumnScaledBelowTheVarianceRange_KeepsItsCorrelations(string name, int exponent)
        => AssertColumnEquivariant<double>(name, exponent, 1e-12);

    [Theory]
    [MemberData(nameof(SingleCases))]
    public void Single_ColumnScaledBelowTheVarianceRange_KeepsItsCorrelations(string name, int exponent)
        => AssertColumnEquivariant<float>(name, exponent, 1e-5);
}
