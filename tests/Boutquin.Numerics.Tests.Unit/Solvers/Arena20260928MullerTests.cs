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

[Trait("Tier", "Unit")]
public sealed class Arena20260928MullerTests
{
    [Fact]
    public void OverflowingTrialResidual_BacktracksWithoutReturningNonFiniteState()
    {
        static double Objective(double x) => 3 - x + (double.MaxValue / 2) * (x * (x - 1) * (x - 2));
        RootSolverResult<double>? result = null;
        Action solve = () => result = new MullerSolver<double>(1e-12, 1).Solve(Objective, 0, 1, 2);

        solve.Should().NotThrow("finite seed residuals permit halving an overflowing trial");
        result!.Converged.Should().BeFalse();
        double.IsFinite(result.FinalResidual).Should().BeTrue();
    }

    [Theory]
    [InlineData(1)]
    [InlineData(-1)]
    public void QuadraticChoosesNearerRootForEitherDenominatorSign(double sign)
    {
        var result = new MullerSolver<double>(1e-12, 1)
            .Solve(x => sign * (x * x - 1), 0, 2, 3);

        // Muller1956 eq5 chooses larger denominator magnitude, hence root+1
        // nearer currentx3. The other algebraic branch would return-1.
        result.Root.Should().BeApproximately(1, 1e-14);
        result.Converged.Should().BeTrue();
    }

    [Fact]
    public void SimpleCubicRoot_FollowsThreeErrorProductLaw()
    {
        var visited = new List<double>();
        double Objective(double x) { visited.Add(x); return x * x * x - 2; }
        new MullerSolver<double>(1e-28, 12).Solve(Objective, 0, 0.5, 1);
        var root = Math.Cbrt(2);
        var errors = visited.Select(x => Math.Abs(x - root)).ToArray();

        // Muller1956 eq17: |e_next/(e_i*e_i-1*e_i-2)| approaches
        // |f‴/(6f′)|=1/(3*cbrt(2)^2). Equation21 gives order1.839286...
        var ratio = errors[6] / (errors[5] * errors[4] * errors[3]);
        ratio.Should().BeApproximately(1 / (3 * root * root), 0.004);
        var observedOrder = Math.Log(errors[6] / errors[5]) / Math.Log(errors[5] / errors[4]);
        observedOrder.Should().BeInRange(1.7, 2.0, "finite iterates approach the positive root of p^3=p^2+p+1");
    }

    [Fact]
    public void EqualResiduals_ContinueWithUnitLambda()
    {
        var visited = new List<double>();
        double Objective(double x) { visited.Add(x); return 1; }
        var result = new MullerSolver<double>(1e-12, 1).Solve(Objective, 0, 1, 2);

        // Muller1956 p214 sets lambda=1, hence h_next=h_previous=1.
        visited.Should().Equal(0, 1, 2, 3);
        result.Converged.Should().BeFalse();
    }

    [Fact]
    public void ExcessResidual_RetriesRepeatedHalfSteps()
    {
        var visited = new List<double>();
        double Objective(double x) { visited.Add(x); return 3 - x + 20 * x * (x - 1) * (x - 2); }
        var result = new MullerSolver<double>(1e-12, 1).Solve(Objective, 0, 1, 2);

        // Initial residuals3,2,1 interpolate a line with nextx3. Its residual120
        // exceeds10*1, as do residual38 at2.5 and14.8125 at2.25. At2.125 it is6.8515625.
        visited.Should().Equal(0, 1, 2, 3, 2.5, 2.25, 2.125);
        result.Root.Should().Be(2.125);
    }
}
