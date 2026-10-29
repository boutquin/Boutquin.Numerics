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
public sealed class FidelityC1NewtonTraceControls
{
    [Theory]
    [InlineData(1.0)]
    [InlineData(-1.0)]
    public void InitialMidpointAndFullWidthHistoryPermitTheIndependentTangent(double sign)
    {
        // On[0,4], x³-27 has midpoint2, f=-19 and derivative12.
        // The Newton step19/12 is below halfwidth2 but above quarterwidth1:
        // full initial dxOld permits it; incorrectly halved history forces bisection.
        // New point43/12 is above root3, leaving bracket[0,43/12].
        var visits = new List<double>();
        var solver = new NewtonRaphsonSolver<double>(1e-30, 1e-30, 0, 1,
            x => sign * 3 * x * x);
        var result = solver.Solve(x =>
        {
            visits.Add(x);
            return sign * (x * x * x - 27);
        }, 0, 4);
        Assert.Equal(4, visits.Count);
        Assert.Equal(new double[] { 0, 4, 2 }, visits.Take(3));
        Assert.InRange(Math.Abs(visits[3] - 43.0 / 12), 0, 1e-14);
        Assert.InRange(Math.Abs(result.Root - 43.0 / 12), 0, 1e-14);
        Assert.InRange(Math.Abs(result.EstimatedError - 43.0 / 24), 0, 1e-14);
        Assert.False(result.Converged);
    }

    [Theory]
    [InlineData(1.0)]
    [InlineData(-1.0)]
    public void ZeroDerivativeFallsBackToTheBracketMidpoint(double sign)
    {
        // (x-2)³-1 has root3, but derivative zero at initial midpoint2.
        // Its safe first step is midpoint2; retaining the sign bracket leaves[2,4].
        var solver = new NewtonRaphsonSolver<double>(1e-30, 1e-30, 0, 1,
            x => sign * 3 * (x - 2) * (x - 2));
        var result = solver.Solve(x => sign * ((x - 2) * (x - 2) * (x - 2) - 1), 0, 4);
        Assert.Equal(2, result.Root);
        Assert.Equal(-sign, result.FinalResidual);
        Assert.Equal(1, result.EstimatedError);
        Assert.False(result.Converged);
    }
}
