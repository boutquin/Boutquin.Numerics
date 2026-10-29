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

namespace Boutquin.Numerics.Tests.Unit;

[Trait("Tier", "Unit")]
public sealed class ArenaSuccessorC1Tests
{
    [Theory]
    [InlineData(false, false, false)]
    [InlineData(false, true, false)]
    [InlineData(true, false, false)]
    [InlineData(true, true, false)]
    [InlineData(false, false, true)]
    [InlineData(false, true, true)]
    [InlineData(true, false, true)]
    [InlineData(true, true, true)]
    public void Lm_CoupledBoundsAllowFeasibleDescentWithoutPrematureConvergence(bool writer, bool analytic, bool alternate)
    {
        var q = Math.Sqrt(0.19);
        var a = alternate ? 1 : -1;
        var b = alternate ? 0.8 / q : 9.1 / q;
        double[] Residual(double[] x) => [a + x[0] - 0.9 * x[1], b + q * x[1]];
        double[,] Jacobian(double[] _) => new[,] { { 1.0, -0.9 }, { 0.0, q } };
        void WriteResidual(ReadOnlySpan<double> x, Span<double> r)
        {
            r[0] = a + x[0] - 0.9 * x[1];
            r[1] = b + q * x[1];
        }
        void WriteJacobian(ReadOnlySpan<double> _, Span<double> j)
        {
            j[0] = 1;
            j[1] = -0.9;
            j[2] = 0;
            j[3] = q;
        }

        // At zero, gradient is (-1,10) or (1,-.1). A feasible coordinate
        // direction decreases half-SSE in each case, despite a fully clipped step.
        var solver = new LevenbergMarquardtSolver<double>(1, 1e-10, 1e-10, 1e-10, 0.001);
        var result = writer
            ? solver.Solve(WriteResidual, [0, 0], 2, analytic ? WriteJacobian : null, lowerBounds: [0, 0])
            : solver.Solve(Residual, [0, 0], analytic ? Jacobian : null, lowerBounds: [0, 0]);

        result.Parameters[alternate ? 0 : 1].Should().Be(0);
        result.Parameters[alternate ? 1 : 0].Should().BeGreaterThan(0);
        result.FinalCost.Should().BeLessThan((a * a + b * b) / 2);
        result.Converged.Should().BeFalse("clipping cannot certify a point with a feasible descent direction");
        result.TerminationReason.Should().Be(LmTerminationReason.MaxIterationsReached);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void Lm_TinyAcceptedProjectionStillRequiresBoundStationarity(bool writer, bool functionStop)
    {
        var q = Math.Sqrt(0.19);
        double[] Residual(double[] x) => [x[0] - 0.9 * x[1] - 1, q * x[1] + 9.1 / q];
        double[,] Jacobian(double[] _) => new[,] { { 1.0, -0.9 }, { 0.0, q } };
        void WriteResidual(ReadOnlySpan<double> x, Span<double> r)
        {
            r[0] = x[0] - 0.9 * x[1] - 1;
            r[1] = q * x[1] + 9.1 / q;
        }
        void WriteJacobian(ReadOnlySpan<double> _, Span<double> j)
        {
            j[0] = 1;
            j[1] = -0.9;
            j[2] = 0;
            j[3] = q;
        }

        var solver = new LevenbergMarquardtSolver<double>(1, functionStop ? 1e-5 : 1e-20,
            functionStop ? 1e-12 : 1e-3, 1e-10, 0.001);
        var result = writer
            ? solver.Solve(WriteResidual, [0, 0], 2, WriteJacobian, lowerBounds: [0, -1e-8])
            : solver.Solve(Residual, [0, 0], Jacobian, lowerBounds: [0, -1e-8]);

        result.Iterations.Should().Be(1);
        result.Parameters.Should().Equal(0, -1e-8);
        result.Converged.Should().BeFalse("the remaining x0 gradient is approximately -1 at its lower bound");
    }

    [Theory]
    [InlineData(1, 2)]
    [InlineData(2, 1)]
    public void Secant_InitialCurrentPointHasTheSmallerResidual(double first, double second)
    {
        var visited = new List<double>();
        double Objective(double x) { visited.Add(x); return x * x - 2; }
        new SecantSolver<double>(1e-28, 2).Solve(Objective, first, second);
        visited[2].Should().BeApproximately(4.0 / 3, 5e-16);
        // NR rtsec keeps seed 1 after the first line intersection, so the
        // next line through (1,-1) and (4/3,-2/9) intersects at 10/7.
        visited[3].Should().BeApproximately(10.0 / 7, 5e-16);
    }
}
