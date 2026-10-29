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
public sealed class FidelityC1BisectionInitialControls
{
    [Theory]
    [InlineData(1.0)]
    [InlineData(-1.0)]
    public void ExactLeftEndpointIsReturnedWithoutAnIteration(double sign)
    {
        var calls = 0;
        RootSolverResult<double>? result = null;
        var error = Record.Exception(() => result = new BisectionSolver<double>(1e-12, 1e-12, 2)
            .Solve(x => { calls++; return sign * (x - 1); }, 1, 3));
        Assert.Null(error);
        Assert.NotNull(result);
        Assert.True(result.Converged);
        Assert.Equal(1, result.Root);
        Assert.Equal(0, result.FinalResidual);
        Assert.Equal(0, result.Iterations);
        Assert.Equal(2, calls);
    }

    [Theory]
    [InlineData(1.0)]
    [InlineData(-1.0)]
    public void ExactRightEndpointIsReturnedWithoutAnIteration(double sign)
    {
        var calls = 0;
        RootSolverResult<double>? result = null;
        var error = Record.Exception(() => result = new BisectionSolver<double>(1e-12, 1e-12, 2)
            .Solve(x => { calls++; return sign * (x - 3); }, 1, 3));
        Assert.Null(error);
        Assert.NotNull(result);
        Assert.True(result.Converged);
        Assert.Equal(3, result.Root);
        Assert.Equal(0, result.FinalResidual);
        Assert.Equal(0, result.Iterations);
        Assert.Equal(2, calls);
    }

    [Theory]
    [InlineData(1.0)]
    [InlineData(-1.0)]
    public void StrictSameSignEndpointsAreRejectedBeforeIteration(double sign)
    {
        var calls = 0;
        var error = Record.Exception(() => new BisectionSolver<double>(1e-12, 1e-12, 2)
            .Solve(x => { calls++; return sign * (x * x + 1); }, 1, 3));
        Assert.IsType<InvalidOperationException>(error);
        Assert.Equal(2, calls);
    }
}
