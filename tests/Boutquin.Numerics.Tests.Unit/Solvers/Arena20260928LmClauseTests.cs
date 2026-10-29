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
public sealed class Arena20260928LmClauseTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UnequalDiagonalDamping_MatchesIndependentTwoByTwoSolution(bool writer)
    {
        // NR15.5.13: J'J=[[10,14],[14,20]], lambda=.1 gives[[11,14],[14,22]].
        // RHS[4,6], determinant46: h=[(88-84)/46,(66-56)/46].
        var result = Solve(writer, x => [x[0] + 2 * x[1] - 1, 3 * x[0] + 4 * x[1] - 1],
            [0, 0], _ => new double[,] { { 1, 2 }, { 3, 4 } }, damping: 0.1, residualCount: 2);
        result.Parameters[0].Should().BeApproximately(2.0 / 23, 1e-14);
        result.Parameters[1].Should().BeApproximately(5.0 / 23, 1e-14);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NielsenSchedule_PublicTrialsFollowAcceptedAndRejectedRules(bool writer)
    {
        static double R(double x) => x * x * x - 2 * x + 2;
        static double J(double x) => 3 * x * x - 2;
        var expected = new List<double> { 1 };
        var x = 1.0;
        var lambda = 0.001;
        var nu = 2.0;
        var accepted = 0;
        var rejected = 0;
        // MNT2.21/Alg3.16, scalar diagonal form: h=-r/(J*(1+lambda)).
        // Model reduction is expanded directly, not the production damped identity.
        for (var i = 0; i < 12; i++)
        {
            var r = R(x);
            var j = J(x);
            var h = -r / (j * (1 + lambda));
            var trial = x + h;
            expected.Add(trial);
            var gain = (r * r - R(trial) * R(trial)) / (-2 * r * j * h - (j * h) * (j * h));
            if (gain > 0)
            {
                accepted++;
                x = trial;
                lambda *= Math.Max(1.0 / 3, 1 - Math.Pow(2 * gain - 1, 3));
                nu = 2;
            }
            else { rejected++; lambda *= nu; nu *= 2; }
        }
        accepted.Should().BeGreaterThan(2);
        rejected.Should().BeGreaterThan(2);
        var visited = new List<double>();
        var result = Solve(writer, p => { visited.Add(p[0]); return [R(p[0])]; }, [1],
            p => new double[,] { { J(p[0]) } }, maxIterations: 12, damping: 0.001);
        visited.Should().HaveCount(expected.Count);
        for (var i = 0; i < expected.Count; i++) { visited[i].Should().BeApproximately(expected[i], 2e-11); }
        result.Parameters[0].Should().BeApproximately(x, 2e-11);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void GradientBoundary_IsInclusiveAndIndependentOfOtherTolerances(bool writer, bool atBoundary)
    {
        var result = Solve(writer, x => [x[0] - 1], [0], _ => new double[,] { { 1 } },
            gradientTolerance: atBoundary ? 1 : Math.BitDecrement(1));
        result.Parameters[0].Should().BeApproximately(atBoundary ? 0 : 0.5, 2e-15);
        result.TerminationReason.Should().Be(LmTerminationReason.GradientToleranceReached);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ParameterBoundary_UsesNormPlusAbsoluteFloor(bool writer, bool atBoundary)
    {
        // Exact step1/2 with lambda1. At x0=.5, epsilon=.5 gives epsilon*(|x0|+epsilon)=.5.
        var result = Solve(writer, x => [x[0] - 1.5], [0.5], _ => new double[,] { { 1 } },
            parameterTolerance: atBoundary ? 0.5 : Math.BitDecrement(0.5), damping: 1);
        result.Parameters[0].Should().Be(1);
        result.TerminationReason.Should().Be(atBoundary ? LmTerminationReason.ParameterToleranceReached : LmTerminationReason.MaxIterationsReached);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void FunctionBoundary_RequiresActualAndModelRelativeReductions(bool writer, bool atBoundary)
    {
        // Exact linear step1/2: both actual and model decrease3/8, divided by cost1/2.
        var result = Solve(writer, x => [x[0] - 1], [0], _ => new double[,] { { 1 } },
            functionTolerance: atBoundary ? 0.75 : Math.BitDecrement(0.75), damping: 1);
        result.TerminationReason.Should().Be(atBoundary ? LmTerminationReason.FunctionToleranceReached : LmTerminationReason.MaxIterationsReached);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RejectedRoundedTrial_CanSatisfyParameterToleranceWithoutAcceptance(bool writer)
    {
        // MNT3.15b concerns a small step, even when cost cannot decrease representably.
        // Actual cost is unchanged; predicted relative gain exceeds the tiny ftol.
        var result = Solve(writer, x => [1, x[0]], [1e-8], _ => new double[,] { { 0 }, { 1 } },
            parameterTolerance: 1e-4, residualCount: 2);
        result.Converged.Should().BeTrue();
        result.TerminationReason.Should().Be(LmTerminationReason.ParameterToleranceReached);
        result.Parameters.Should().Equal(1e-8);
    }

    private static MultivariateSolverResult<double> Solve(bool writer, Func<double[], double[]> residual,
        double[] start, Func<double[], double[,]> jacobian, int maxIterations = 1,
        double functionTolerance = 1e-30, double parameterTolerance = 1e-30,
        double gradientTolerance = 0, double damping = 1, int residualCount = 1)
    {
        var solver = new LevenbergMarquardtSolver<double>(maxIterations, functionTolerance, parameterTolerance, gradientTolerance, damping);
        if (!writer) { return solver.Solve(residual, start, jacobian); }
        var count = residualCount;
        LmResidualWriter<double> write = (x, r) => residual(x.ToArray()).CopyTo(r);
        LmJacobianWriter<double> writeJac = (x, j) =>
        {
            var matrix = jacobian(x.ToArray());
            for (var row = 0; row < count; row++)
            {
                for (var col = 0; col < start.Length; col++) { j[row * start.Length + col] = matrix[row, col]; }
            }
        };
        return solver.Solve(write, start, count, writeJac);
    }
}
