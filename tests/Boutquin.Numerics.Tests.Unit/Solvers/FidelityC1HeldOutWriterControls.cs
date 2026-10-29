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
public sealed class FidelityC1HeldOutWriterControls
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NonlinearWriterAndArrayPathsPreserveAllResultBits(bool analytic)
    {
        // Modified Rosenbrock plus constant residual1 has unique optimum(1,1),
        // cost1/2, residual(0,0,1). At that point JtJ=[[401,-200],[-200,100]],
        // whose inverse is[[1,2],[2,4.01]]; RSS/(m-n)=1.
        static double[] Residual(double[] x) => [10 * (x[1] - x[0] * x[0]), 1 - x[0], 1];
        static double[,] Jacobian(double[] x) => new double[,] { { -20 * x[0], 10 }, { -1, 0 }, { 0, 0 } };
        static void WriteResidual(ReadOnlySpan<double> x, Span<double> r)
        {
            r[0] = 10 * (x[1] - x[0] * x[0]);
            r[1] = 1 - x[0];
            r[2] = 1;
        }
        static void WriteJacobian(ReadOnlySpan<double> x, Span<double> j)
        {
            j[0] = -20 * x[0];
            j[1] = 10;
            j[2] = -1;
            j[3] = 0;
            j[4] = 0;
            j[5] = 0;
        }
        var solver = new LevenbergMarquardtSolver<double>(400, 1e-12, 1e-12, 1e-10, 0.001);
        double[] start = [-1.2, 1];
        var array = solver.Solve(Residual, start, analytic ? Jacobian : null);
        var pool = new LevenbergMarquardtBuffers<double>(5, 8);
        pool.Reset(2, 3);
        var writer = solver.Solve(WriteResidual, start, pool, analytic ? WriteJacobian : null);
        Assert.True(array.Converged);
        Assert.InRange(Math.Abs(array.Parameters[0] - 1), 0, 1e-5);
        Assert.InRange(Math.Abs(array.Parameters[1] - 1), 0, 1e-5);
        Assert.InRange(Math.Abs(array.FinalCost - 0.5), 0, 1e-12);
        Assert.False(array.BoundsActive);
        var arrayCovariance = array.ParameterCovariance;
        Assert.NotNull(arrayCovariance);
        double[,] expectedCovariance = { { 1, 2 }, { 2, 4.01 } };
        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                Assert.InRange(Math.Abs(arrayCovariance[i, j] - expectedCovariance[i, j]), 0, 1e-3);
            }
        }
        Assert.Equal(array.Converged, writer.Converged);
        Assert.Equal(array.Iterations, writer.Iterations);
        Assert.Equal(array.TerminationReason, writer.TerminationReason);
        Assert.Equal(array.BoundsActive, writer.BoundsActive);
        Assert.Equal(BitConverter.DoubleToInt64Bits(array.FinalCost), BitConverter.DoubleToInt64Bits(writer.FinalCost));
        Assert.Equal(array.Parameters.Select(BitConverter.DoubleToInt64Bits), writer.Parameters.Select(BitConverter.DoubleToInt64Bits));
        Assert.Equal(array.FinalResiduals.Select(BitConverter.DoubleToInt64Bits), writer.FinalResiduals.Select(BitConverter.DoubleToInt64Bits));
        var writerCovariance = writer.ParameterCovariance;
        Assert.NotNull(writerCovariance);
        Assert.Equal(arrayCovariance.Cast<double>().Select(BitConverter.DoubleToInt64Bits),
            writerCovariance.Cast<double>().Select(BitConverter.DoubleToInt64Bits));
    }
}
