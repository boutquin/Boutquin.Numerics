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

public sealed class Arena20260928LmVariantBoundaryTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FunctionTolerance_RequiresMinpackRatioClause(bool writer)
    {
        const double scale = 0.0005;
        const double step = -0.1; // -(J' r) / ((1 + lambda) J' J), lambda = 9.
        var trialValue = 1 + step - 40 * step * step;
        var actualReduction = 0.5 * scale * scale * (1 - trialValue * trialValue);
        var predictedReduction = -scale * scale * step - 0.5 * scale * scale * step * step;
        Assert.True(actualReduction / predictedReduction > 2); // MINPACK's third clause is false.

        var solver = new LevenbergMarquardtSolver<double>(1, 1e-6, 1e-30, 0, 9);
        double[] Residual(double[] x) => [1, scale * (1 + x[0] - 40 * x[0] * x[0])];
        double[,] Jacobian(double[] x) => new[,] { { 0.0 }, { scale * (1 - 80 * x[0]) } };
        void ResidualWriter(ReadOnlySpan<double> x, Span<double> r)
        {
            r[0] = 1;
            r[1] = scale * (1 + x[0] - 40 * x[0] * x[0]);
        }

        void JacobianWriter(ReadOnlySpan<double> x, Span<double> j)
        {
            j[0] = 0;
            j[1] = scale * (1 - 80 * x[0]);
        }

        var result = writer
            ? solver.Solve(ResidualWriter, new[] { 0.0 }, 2, JacobianWriter)
            : solver.Solve(Residual, new[] { 0.0 }, Jacobian);
        Assert.InRange(Math.Abs(result.Parameters[0] - step), 0, 1e-15);
        Assert.False(result.Converged);
        Assert.Equal(LmTerminationReason.MaxIterationsReached, result.TerminationReason);
    }
}
