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
public sealed class FidelityC1SimplexTieControls
{
    [Fact]
    public void ReflectionTiedWithBestAndSecondWorstContracts()
    {
        var visits = new List<double[]>();
        double Objective(double[] x)
        {
            visits.Add((double[])x.Clone());
            return (x[0], x[1]) == (0, 1) ? 2 : 0;
        }
        _ = NelderMeadMinimizer<double>.Minimize(Objective, [0, 0],
            new SimplexOptions<double>(MaxIterations: 1, InitialStep: 1, XTolerance: 0, FTolerance: 0));
        Assert.Equal(5, visits.Count);
        Assert.Equal(new double[] { 1, -1 }, visits[3]);
        Assert.Equal(new double[] { 0.75, -0.5 }, visits[4]);
    }

    [Fact]
    public void NewEqualNonbestVertexFollowsBothRetainedEquals()
    {
        var visits = new List<double[]>();
        double Objective(double[] x)
        {
            visits.Add((double[])x.Clone());
            if (x.All(value => value == 0)) { return 0; }
            return x[2] == 1 ? 2 : 1;
        }
        _ = NelderMeadMinimizer<double>.Minimize(Objective, [0, 0, 0],
            new SimplexOptions<double>(MaxIterations: 2, InitialStep: 1, XTolerance: 0, FTolerance: 0));
        // First outside contraction(.5,.5,-.5) ties both retained unit vertices.
        // Stable insertion makes it next worst. Its reflection about(1/3,1/3,0)
        // is(1/6,1/6,.5), independently fixing all nonbest tie ordering.
        Assert.True(visits.Count >= 7);
        Assert.InRange(Math.Abs(visits[6][0] - 1.0 / 6), 0, 1e-15);
        Assert.InRange(Math.Abs(visits[6][1] - 1.0 / 6), 0, 1e-15);
        Assert.Equal(0.5, visits[6][2]);
    }
}
