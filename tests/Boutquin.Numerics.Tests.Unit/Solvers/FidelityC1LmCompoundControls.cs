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

using System.Reflection;
using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

public sealed class FidelityC1LmCompoundControls
{
    private delegate double ProjectedGradient(double[,] jacobian, double[] residual, double[] gradient,
        double[] theta, int residualCount, int parameterCount, ReadOnlySpan<double> lower, ReadOnlySpan<double> upper);

    [Fact]
    public void GradientUsesEveryRowAndColumnAndInfinityNorm()
    {
        var method = typeof(LevenbergMarquardtSolver<double>).GetMethod("ProjectedGradientInfinityNormInto", BindingFlags.Static | BindingFlags.NonPublic)!;
        var evaluate = method.CreateDelegate<ProjectedGradient>();
        double[,] jacobian = { { 1, 2 }, { 3, -1 }, { 2, 1 } };
        double[] residual = [5, 10, -4];
        var gradient = new double[2];
        // Exact dot products:5+30-8=27 and10-10-4=-4; infinity norm27.
        Assert.Equal(27, evaluate(jacobian, residual, gradient, [0, 0], 3, 2, [], []));
        Assert.Equal(new[] { 27d, -4d }, gradient);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void StationaryVectorCostIncludesEverySquaredResidual(bool writer)
    {
        static double[] Residual(double[] _) => [3, 4, 12];
        static double[,] Jacobian(double[] _) => new double[,] { { 0 }, { 0 }, { 0 } };
        static void WriteResidual(ReadOnlySpan<double> _, Span<double> r) { r[0] = 3; r[1] = 4; r[2] = 12; }
        static void WriteJacobian(ReadOnlySpan<double> _, Span<double> j) => j.Clear();
        var solver = new LevenbergMarquardtSolver<double>(3, 1e-12, 1e-12, 1e-12, .001);
        var result = writer ? solver.Solve(WriteResidual, [0], 3, WriteJacobian) : solver.Solve(Residual, [0], Jacobian);
        Assert.Equal(84.5, result.FinalCost); // (9+16+144)/2
        Assert.Equal(new[] { 3d, 4d, 12d }, result.FinalResiduals);
        Assert.Equal(0, result.Iterations);
        Assert.Equal(LmTerminationReason.GradientToleranceReached, result.TerminationReason);
    }
}
