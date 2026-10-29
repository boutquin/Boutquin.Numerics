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
public sealed class FidelityC1NonlinearModelControls
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void QuadraticResidualUsesGaussNewtonRatherThanFullObjectiveHessian(bool writerPath)
    {
        // At x=1, r=x^2-2=-1, J=2 and g=-2. The selected JtJ is4,
        // while the exact objective Hessian JtJ+r*r'' is2. With lambda1,
        // the scaled Gauss-Newton system gives h=1/4, not the full-Hessian h=1/2.
        static double[] Residual(double[] x) => [x[0] * x[0] - 2];
        static double[,] Jacobian(double[] x) => new double[,] { { 2 * x[0] } };
        static void Writer(ReadOnlySpan<double> x, Span<double> r) => r[0] = x[0] * x[0] - 2;
        static void JacobianWriter(ReadOnlySpan<double> x, Span<double> j) => j[0] = 2 * x[0];
        var solver = new LevenbergMarquardtSolver<double>(1, 1e-30, 1e-30, 1e-30, 1);
        var result = writerPath ? solver.Solve(Writer, [1], 1, JacobianWriter) : solver.Solve(Residual, [1], Jacobian);
        Assert.InRange(Math.Abs(result.Parameters[0] - 1.25), 0, 1e-14);
        Assert.InRange(Math.Abs(result.FinalResiduals[0] + 7.0 / 16), 0, 1e-14);
        Assert.InRange(Math.Abs(result.FinalCost - 49.0 / 512), 0, 1e-14);
        Assert.Equal(1, result.Iterations);
        Assert.False(result.Converged);
    }
}
