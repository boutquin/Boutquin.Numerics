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
public sealed class FidelityC1ExhaustionControls
{
    [Fact]
    public void BrentBudgetReportsItsNonrootTrialAndRemainingBracket()
    {
        var result = new BrentSolver<double>(1e-30, 1).Solve(x => x * x - 2, 1, 2);
        CheckExhaustion(result, 4.0 / 3, 1.0 / 3);
    }

    [Fact]
    public void NewtonBudgetReportsItsNonrootTangentAndRemainingBracket()
    {
        // Midpoint3/2 has f=1/4, derivative3; Newton tangent reaches17/12.
        var result = new NewtonRaphsonSolver<double>(1e-30, 1e-30, 0, 1,
            x => 2 * x).Solve(x => x * x - 2, 1, 2);
        CheckExhaustion(result, 17.0 / 12, 5.0 / 24);
    }

    [Fact]
    public void SecantBudgetReportsItsNonrootTrialAndLastStep()
    {
        var result = new SecantSolver<double>(1e-30, 1).Solve(x => x * x - 2, 1, 2);
        CheckExhaustion(result, 4.0 / 3, 1.0 / 3);
    }

    [Fact]
    public void BisectionBudgetReportsMidpointOfTheRemainingBracket()
    {
        var result = new BisectionSolver<double>(1e-30, 1e-30, 1).Solve(x => x * x - 2, 1, 2);
        CheckExhaustion(result, 5.0 / 4, 1.0 / 4);
    }

    [Fact]
    public void SecantExactNewRootStopsEvenWhenStepExceedsTolerance()
    {
        var result = new SecantSolver<double>(1e-30, 1).Solve(x => 2 * x - 1, 0, 2);
        Assert.True(result.Converged);
        Assert.Equal(0.5, result.Root);
        Assert.Equal(0, result.FinalResidual);
        Assert.Equal(1, result.Iterations);
    }

    [Fact]
    public void SecantMayLeaveSameSignSeedInterval()
    {
        // Secant line through(2,2),(3,7) crosses zero at8/5, outside[2,3].
        var result = new SecantSolver<double>(1e-30, 1).Solve(x => x * x - 2, 2, 3);
        Assert.InRange(Math.Abs(result.Root - 8.0 / 5), 0, 1e-15);
        Assert.True(result.Root < 2);
        Assert.False(result.Converged);
    }

    private static void CheckExhaustion(RootSolverResult<double> result, double root, double error)
    {
        Assert.False(result.Converged);
        Assert.Equal(1, result.Iterations);
        Assert.InRange(Math.Abs(result.Root - root), 0, 1e-14);
        Assert.InRange(Math.Abs(result.FinalResidual - (root * root - 2)), 0, 1e-14);
        Assert.InRange(Math.Abs(result.EstimatedError - error), 0, 1e-14);
    }
}
