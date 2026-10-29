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
public sealed class FidelityC1NewtonEndpointControls
{
    [Fact]
    public void LeftExactRootHasPriorityWhenBothEndpointsAreRoots()
    {
        // x(x-1) vanishes at both endpoints. The specified first check selects zero.
        var result = new NewtonRaphsonSolver<double>(1e-30, 1e-30, 0, 1,
            x => 2 * x - 1).Solve(x => x * (x - 1), 0, 1);
        Assert.True(result.Converged);
        Assert.Equal(0, result.Root);
        Assert.Equal(0, result.Iterations);
        Assert.Equal(0, result.FinalResidual);
        Assert.Equal(0, result.EstimatedError);
    }

    [Fact]
    public void RightExactRootReturnsBeforeAnIteration()
    {
        var result = new NewtonRaphsonSolver<double>(1e-30, 1e-30, 0, 1,
            _ => 1).Solve(x => x - 1, 0, 1);
        Assert.True(result.Converged);
        Assert.Equal(1, result.Root);
        Assert.Equal(0, result.Iterations);
        Assert.Equal(0, result.FinalResidual);
        Assert.Equal(0, result.EstimatedError);
    }
}
