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

using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

public sealed class Arena20260928NelderMeadVariantTests
{
    [Theory]
    [InlineData(3, "expansion")]
    [InlineData(5, "expansion")]
    [InlineData(10, "expansion")]
    [InlineData(3, "contraction")]
    [InlineData(5, "contraction")]
    [InlineData(10, "contraction")]
    [InlineData(3, "shrink")]
    [InlineData(5, "shrink")]
    [InlineData(10, "shrink")]
    public void CallerSuppliedGaoHanCoefficients_ProducePublishedGeometricSteps(int dimension, string branch)
    {
        var expansion = 1 + 2.0 / dimension;
        var contraction = 0.75 - 0.5 / dimension;
        var shrink = 1 - 1.0 / dimension;
        var visited = new List<double[]>();
        double Objective(double[] x)
        {
            visited.Add((double[])x.Clone());
            return branch switch
            {
                "expansion" => -x.Sum(),
                "contraction" => x.Sum(v => v * v),
                _ => x.All(v => v == 0) ? 0 : 1,
            };
        }

        _ = NelderMeadMinimizer<double>.Minimize(Objective, new double[dimension],
            new SimplexOptions<double>(MaxIterations: 1, XTolerance: 0, FTolerance: 0,
                InitialStep: 1, Expansion: expansion, Contraction: contraction, Shrink: shrink));
        if (branch == "expansion")
        {
            foreach (var value in visited[dimension + 2])
            {
                Assert.InRange(Math.Abs(value - (1 + expansion) / dimension), 0, 1e-14);
            }
        }
        else if (branch == "contraction")
        {
            for (var i = 0; i < dimension; i++)
            {
                var expected = i == dimension - 1 ? contraction : (1 - contraction) / dimension;
                Assert.InRange(Math.Abs(visited[dimension + 2][i] - expected), 0, 1e-14);
            }
        }
        else
        {
            for (var i = 0; i < dimension; i++)
            {
                for (var j = 0; j < dimension; j++)
                {
                    Assert.InRange(Math.Abs(visited[dimension + 3 + i][j] - (i == j ? shrink : 0)), 0, 1e-14);
                }
            }
        }
    }

    [Fact]
    public void DefaultInitializationAndOrStop_DifferFromGaoHanMatlabProtocol()
    {
        var visited = new List<double[]>();
        var result = NelderMeadMinimizer<double>.Minimize(x =>
        {
            visited.Add((double[])x.Clone());
            return 1;
        }, [0.1, 0.2], new SimplexOptions<double>(MaxIterations: 1, XTolerance: 1e-12, FTolerance: 1e-12));
        Assert.Equal(3, visited.Count);
        Assert.InRange(Math.Abs(visited[1][0] - 0.15), 0, 1e-15);
        Assert.InRange(Math.Abs(visited[2][1] - 0.25), 0, 1e-15);
        Assert.True(result.Converged); // Zero value spread suffices despite diameter .05.
        Assert.Equal(new[] { 0.1, 0.2 }, result.Argmin);
    }
}
