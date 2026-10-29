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

public sealed class FidelityC1SimplexBranchBoundaryControls
{
    [Fact]
    public void ReflectionRangeIncludesBestAndExcludesSecondWorst()
    {
        foreach (var reflectedValue in new[] { 0d, 1d })
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
                    (1, -1) => reflectedValue,
                    (0.75, -0.5) => 0.5,
                    _ => 10,
                };
            }
            NelderMeadMinimizer<double>.Minimize(Objective, [0, 0], new SimplexOptions<double>(MaxIterations: 1, InitialStep: 1, XTolerance: 0, FTolerance: 1));
            // Centroid=(1/2,0), reflection=(1,-1). At fbest=0 reflection
            // belongs to [fbest,fsecond); at fsecond=1 it contracts outside.
            Assert.Equal(reflectedValue == 0 ? 4 : 5, visited.Count);
            Assert.Equal(new[] { 1d, -1d }, visited[3]);
            if (reflectedValue == 1) { Assert.Equal(new[] { 0.75, -0.5 }, visited[4]); }
        }
    }

    [Fact]
    public void ExpansionEqualityRetainsReflectedPoint()
    {
        static double Objective(double[] x) => (x[0], x[1]) switch
        {
            (0, 0) => 0,
            (1, 0) => 1,
            (0, 1) => 2,
            (1, -1) => -1,
            (1.5, -2) => -1,
            _ => 10,
        };
        var result = NelderMeadMinimizer<double>.Minimize(Objective, [0, 0], new SimplexOptions<double>(MaxIterations: 1, InitialStep: 1, XTolerance: 0, FTolerance: 0));
        Assert.Equal(new[] { 1d, -1d }, result.Argmin);
        Assert.Equal(-1, result.MinValue);
    }
}
