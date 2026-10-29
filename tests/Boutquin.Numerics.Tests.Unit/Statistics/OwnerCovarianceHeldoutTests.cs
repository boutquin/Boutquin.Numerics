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

using System.Text.Json;
using Boutquin.Numerics.Statistics;
using FluentAssertions;
namespace Boutquin.Numerics.Tests.Unit.Statistics;

public sealed class OwnerCovarianceHeldoutTests
{
    [Fact]
    public void FreshExternalPanel_MatchesIndependentRationalOracles()
    {
        var path = Environment.GetEnvironmentVariable("COVARIANCE_HELDOUT_JSON");
        VerifyFixture(string.IsNullOrEmpty(path) ? DefaultJson : File.ReadAllText(path));
    }

    internal const string DefaultJson = """
        {
          "seed": 731,
          "nercomeRows": [
            -6.0,
            -5.0,
            -8.0,
            4.0,
            6.0,
            -5.0,
            8.0,
            4.0
          ],
          "nercomeExpected": [
            41.93690790814989,
            0.0,
            0.0,
            28.56309209185011
          ],
          "nercomeSplit": 2,
          "nercomeFixedExpected": [
            41.93690790814989,
            0.0,
            0.0,
            28.56309209185011
          ],
          "nercomeRationalScores": [
            "961729140081792128/1145497630730401",
            "38919347122688/77882855625",
            "5088762849606291968/39136228965891369"
          ],
          "oasRows": [
            2.0,
            -3.0,
            1.0,
            7.0,
            1.0,
            -3.0,
            -3.0,
            5.0,
            7.0,
            -6.0,
            3.0,
            -1.0,
            -7.0,
            7.0,
            0.0,
            6.0,
            7.0,
            6.0,
            -4.0,
            -3.0,
            6.0,
            -2.0,
            0.0,
            6.0,
            3.0,
            5.0,
            7.0
          ],
          "oasExpected": [
            22.40740740740741,
            0.0,
            0.0,
            0.0,
            22.40740740740741,
            0.0,
            0.0,
            0.0,
            22.40740740740741
          ],
          "glassoSample": [
            6.0,
            -1.5,
            -1.5,
            5.0
          ],
          "glassoLambda": 0.5,
          "glassoCovariance": [
            6.5,
            -1.0,
            -1.0,
            5.5
          ],
          "glassoPrecision": [
            0.15827338129496402,
            0.02877697841726619,
            0.02877697841726619,
            0.18705035971223022
          ]
        }
        """;

    internal static void VerifyFixture(string json)
    {
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        static double[,] Matrix(JsonElement value, int width)
        {
            var flat = value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
            var result = new double[flat.Length / width, width];
            for (var i = 0; i < flat.Length; i++)
            {
                result[i / width, i % width] = flat[i];
            }

            return result;
        }

        static void Check(string family, double[,] actual, double[,] expected)
        {
            // An emptied or row-truncated expected family would otherwise loop zero times and pass vacuously.
            Assert.True(
                expected.GetLength(0) == actual.GetLength(0) && expected.GetLength(1) == actual.GetLength(1),
                $"held-out family '{family}' is {expected.GetLength(0)}x{expected.GetLength(1)} but the estimator output is {actual.GetLength(0)}x{actual.GetLength(1)}; refusing a vacuous comparison");
            for (var i = 0; i < expected.GetLength(0); i++)
            {
                for (var j = 0; j < expected.GetLength(1); j++)
                {
                    actual[i, j].Should().BeApproximately(expected[i, j], 1e-9);
                }
            }
        }

        var rows = Matrix(root.GetProperty("nercomeRows"), 2);
        Check("nercomeExpected", new NercomeCovarianceEstimator<double>().Estimate(rows), Matrix(root.GetProperty("nercomeExpected"), 2));
        Check("nercomeFixedExpected", new NercomeCovarianceEstimator<double>(.5).Estimate(rows), Matrix(root.GetProperty("nercomeFixedExpected"), 2));
        Check("oasExpected", new OracleApproximatingShrinkageEstimator<double>().Estimate(Matrix(root.GetProperty("oasRows"), 3)), Matrix(root.GetProperty("oasExpected"), 3));
        var glasso = GraphicalLassoSolver<double>.Solve(Matrix(root.GetProperty("glassoSample"), 2), root.GetProperty("glassoLambda").GetDouble());
        Check("glassoCovariance", glasso.Covariance, Matrix(root.GetProperty("glassoCovariance"), 2));
        Check("glassoPrecision", glasso.Precision, Matrix(root.GetProperty("glassoPrecision"), 2));
        glasso.Converged.Should().BeTrue();
    }
}
