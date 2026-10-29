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
public sealed class FidelityC1AcceptedStateControls
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AcceptedQuarterStepUpdatesParametersResidualCostAndCounterTogether(bool writerPath)
    {
        // r=x-1, J=1, lambda3 gives exact h=1/4 from zero. The accepted
        // state is x=1/4,r=-3/4,F=9/32,count1. None meets tolerance1e-30.
        static double[] Residual(double[] x) => [x[0] - 1];
        static double[,] Jacobian(double[] _) => new double[,] { { 1 } };
        static void WriteResidual(ReadOnlySpan<double> x, Span<double> r) => r[0] = x[0] - 1;
        static void WriteJacobian(ReadOnlySpan<double> _, Span<double> j) => j[0] = 1;
        var solver = new LevenbergMarquardtSolver<double>(1, 1e-30, 1e-30, 1e-30, 3);
        var result = writerPath
            ? solver.Solve(WriteResidual, [0], 1, WriteJacobian)
            : solver.Solve(Residual, [0], Jacobian);
        Assert.Equal(new double[] { 0.25 }, result.Parameters);
        Assert.Equal(new double[] { -0.75 }, result.FinalResiduals);
        Assert.Equal(9.0 / 32, result.FinalCost);
        Assert.Equal(1, result.Iterations);
        Assert.False(result.Converged);
        Assert.Equal(LmTerminationReason.MaxIterationsReached, result.TerminationReason);
    }
}
