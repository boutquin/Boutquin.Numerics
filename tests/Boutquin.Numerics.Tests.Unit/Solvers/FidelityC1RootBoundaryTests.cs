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
public sealed class FidelityC1RootBoundaryTests
{
    [Fact]
    public void Secant_EqualResidualMagnitudesKeepSecondSeedCurrent()
    {
        var result = new SecantSolver<double>(1e-12, 1).Solve(x => x * x + 1, -1, 1);
        Assert.Equal(1, result.Root);
        Assert.Equal(2, result.FinalResidual);
        Assert.Equal(2, result.EstimatedError);
        Assert.False(result.Converged);
    }

    [Fact]
    public void Brent_HalfWidthEqualityStopsBeforeAnotherEvaluation()
    {
        var calls = 0;
        var result = new BrentSolver<double>(2, 1).Solve(x => { calls++; return 2 * x - 1; }, 0, 2);
        // At the best endpoint b=0, tol=2*eps*|b|+2/2=1 and half-width=1.
        Assert.Equal(2, calls);
        Assert.Equal(0, result.Root);
        Assert.Equal(1, result.EstimatedError);
        Assert.True(result.Converged);
    }

    [Fact]
    public void Brent_MachineEpsilonTermControlsRepresentableWidthBoundary()
    {
        var calls = 0;
        var width = Math.ScaleB(1, -50);
        var result = new BrentSolver<double>(double.Epsilon, 1).Solve(
            x => { calls++; return 4 * ((x - 1) / width) - 1; }, 1, 1 + width);
        // Half-width=2^-51=2*machineEpsilon*|b| at b=1; the input tolerance
        // is too small to alter this value. A one-epsilon term fails this boundary.
        Assert.Equal(2, calls);
        Assert.Equal(1, result.Root);
        Assert.Equal(Math.ScaleB(1, -51), result.EstimatedError);
        Assert.True(result.Converged);
    }

    [Fact]
    public void Brent_SubToleranceInterpolationMovesByToleranceTowardBracket()
    {
        var visited = new List<double>();
        var result = new BrentSolver<double>(0.5, 1).Solve(
            x => { visited.Add(x); return x - 0.125; }, 0, 2);
        // Secant interpolation proposes the exact root1/8, but tol at b=0 is1/4.
        Assert.Equal(3, visited.Count);
        Assert.Equal(0.25, visited[2]);
        Assert.Equal(0.25, result.Root);
        Assert.Equal(0.125, result.FinalResidual);
    }
}
