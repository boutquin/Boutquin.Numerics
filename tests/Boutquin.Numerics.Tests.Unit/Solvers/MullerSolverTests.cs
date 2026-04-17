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
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

public sealed class MullerSolverTests
{
    [Fact]
    public void MullerSolver_FindsRoot_OfQuadratic()
    {
        var solver = new MullerSolver();
        var result = solver.Solve(x => x * x - 2.0, 1.0, 1.5, 2.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(Math.Sqrt(2.0), 1e-10);
        result.Iterations.Should().BeGreaterThan(0);
    }

    [Fact]
    public void MullerSolver_FindsRoot_OfCubic()
    {
        var solver = new MullerSolver();
        var result = solver.Solve(x => x * x * x - x - 1.0, 0.5, 1.0, 1.5);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(1.3247179572, 1e-8);
    }

    [Fact]
    public void MullerSolver_SingleGuess_FindsRoot()
    {
        IUnbracketedRootSolver solver = new MullerSolver();
        var result = solver.Solve(x => x * x - 2.0, 1.5);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(Math.Sqrt(2.0), 1e-10);
    }

    [Fact]
    public void MullerSolver_FindsDiscountFactor_ForOisPv()
    {
        var solver = new MullerSolver();
        double rate = 0.05;
        double tau = 2.0;
        double expectedDf = 1.0 / (1.0 + rate * tau);

        var result = solver.Solve(df => rate * tau * df + df - 1.0, 0.8, 0.9, 1.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(expectedDf, 1e-10);
    }

    [Fact]
    public void MullerSolver_HandlesDegenerateQuadratic()
    {
        // Linear function: a (second divided difference) will be zero.
        var solver = new MullerSolver();
        var result = solver.Solve(x => x - 1.0, 0.0, 0.5, 2.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(1.0, 1e-10);
    }

    [Fact]
    public void MullerSolver_ReportsNonConvergence()
    {
        var solver = new MullerSolver(tolerance: 1e-20, maxIterations: 2);
        var result = solver.Solve(x => x * x - 2.0, 1.0, 1.5, 2.0);

        result.Converged.Should().BeFalse();
        result.Iterations.Should().Be(2);
    }
}
