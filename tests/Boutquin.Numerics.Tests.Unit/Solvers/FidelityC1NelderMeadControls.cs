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

[Trait("Tier", "Unit")]
public sealed class FidelityC1NelderMeadControls
{
    [Fact]
    public void OutsideContractionWorseThanReflectionShrinksBothNonbestVertices()
    {
        var visited = new List<double[]>();
        double Objective(double[] x)
        {
            visited.Add((double[])x.Clone());
            return (x[0], x[1]) switch
            {
                (0, 0) => 0,
                (1, 0) => 1,
                (0, 1) => 2,
                (1, -1) => 1.5,
                (0.75, -0.5) => 1.75,
                _ => 0.5,
            };
        }
        var result = NelderMeadMinimizer<double>.Minimize(Objective, [0, 0],
            new SimplexOptions<double>(MaxIterations: 1, InitialStep: 1, XTolerance: 0, FTolerance: 0));
        Assert.Equal(11, visited.Count);
        Assert.Equal(new[] { 1.0, -1.0 }, visited[3]);
        Assert.Equal(new[] { 0.75, -0.5 }, visited[4]);
        Assert.Equal(new[] { 0.5, 0.0 }, visited[5]);
        Assert.Equal(new[] { 0.0, 0.5 }, visited[6]);
        Assert.Equal(new[] { 0.0, 0.0 }, result.Argmin);
    }

    [Fact]
    public void AcceptedReflectionReplacesOnlyWorstAndKeepsOldBest()
    {
        static double Objective(double[] x) => (x[0], x[1]) switch
        {
            (0, 0) => 0,
            (1, 0) => 1,
            (0, 1) => 2,
            (1, -1) => 0.5,
            _ => 10,
        };
        var result = NelderMeadMinimizer<double>.Minimize(Objective, [0, 0],
            new SimplexOptions<double>(MaxIterations: 1, InitialStep: 1, XTolerance: 0, FTolerance: 0));
        Assert.Equal(new[] { 0.0, 0.0 }, result.Argmin);
        Assert.Equal(0, result.MinValue);
    }

    [Fact]
    public void FunctionSpreadEqualityStopsAtInitialSimplex()
    {
        var calls = 0;
        var result = NelderMeadMinimizer<double>.Minimize(x => { calls++; return x[0] + 2 * x[1]; }, [0, 0],
            new SimplexOptions<double>(MaxIterations: 1, InitialStep: 1, XTolerance: 0, FTolerance: 2));
        Assert.Equal(3, calls);
        Assert.True(result.Converged);
        Assert.Equal(new[] { 0.0, 0.0 }, result.Argmin);
    }

    [Fact]
    public void CoordinateInfinitySpreadEqualityStopsAtInitialSimplex()
    {
        var calls = 0;
        var result = NelderMeadMinimizer<double>.Minimize(x => { calls++; return 1 - x[0] + x[1]; }, [0, 0],
            new SimplexOptions<double>(MaxIterations: 1, InitialStep: 1, XTolerance: 1, FTolerance: 0));
        // Best(1,0), worst(0,1): infinity distance1; Euclidean distance sqrt2.
        Assert.Equal(3, calls);
        Assert.True(result.Converged);
        Assert.Equal(new[] { 1.0, 0.0 }, result.Argmin);
    }
}
