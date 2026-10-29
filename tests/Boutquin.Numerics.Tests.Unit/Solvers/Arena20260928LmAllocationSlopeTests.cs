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
public sealed class Arena20260928LmAllocationSlopeTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PooledSolve_AllocationDoesNotGrowWithActualIterationCount(bool writer)
    {
        var shortSolver = new LevenbergMarquardtSolver<double>(2, 1e-30, 1e-30, 0, 1e-3);
        var longSolver = new LevenbergMarquardtSolver<double>(7, 1e-30, 1e-30, 0, 1e-3);
        var buffers = new LevenbergMarquardtBuffers<double>(1, 1);
        double[] start = [0];
        double[] residual = [0];
        double[,] jacobian = { { -1 } };
        var calls = 0;
        // Deliberately opposite analytic slope makes every trial strictly worse.
        // There are exactly cap+1 residual calls and zero accepted steps.
        Func<double[], double[]> func = x => { calls++; residual[0] = 1 + x[0]; return residual; };
        Func<double[], double[,]> jac = _ => jacobian;
        LmResidualWriter<double> write = (x, r) => { calls++; r[0] = 1 + x[0]; };
        LmJacobianWriter<double> writeJac = (_, j) => j[0] = -1;
        MultivariateSolverResult<double> Run(LevenbergMarquardtSolver<double> solver) => writer
            ? solver.Solve(write, start.AsSpan(), buffers, writeJac)
            : solver.Solve(func, start, buffers, jac);

        for (var i = 0; i < 32; i++) { Run(shortSolver); Run(longSolver); }
        calls = 0;
        var shortResult = Run(shortSolver);
        calls.Should().Be(3);
        calls = 0;
        var longResult = Run(longSolver);
        calls.Should().Be(8);
        shortResult.TerminationReason.Should().Be(LmTerminationReason.MaxIterationsReached);
        longResult.TerminationReason.Should().Be(LmTerminationReason.MaxIterationsReached);
        shortResult.Parameters.Should().Equal(0);
        longResult.Parameters.Should().Equal(0);

        long Measure(LevenbergMarquardtSolver<double> solver)
        {
            var before = GC.GetAllocatedBytesForCurrentThread();
            for (var i = 0; i < 64; i++) { GC.KeepAlive(Run(solver)); }
            return GC.GetAllocatedBytesForCurrentThread() - before;
        }

        var shortBytes = Measure(shortSolver);
        var longBytes = Measure(longSolver);
        longBytes.Should().Be(shortBytes, "five extra rejected iterations must allocate zero additional bytes");
        // Reverse order to distinguish an iteration slope from first-window warmup.
        var longAgain = Measure(longSolver);
        var shortAgain = Measure(shortSolver);
        longAgain.Should().Be(shortAgain);
    }
}
