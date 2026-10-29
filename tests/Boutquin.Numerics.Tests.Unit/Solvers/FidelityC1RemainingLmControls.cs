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

public sealed class FidelityC1RemainingLmControls
{
    private delegate double ProjectedGradient(double[,] jacobian, double[] residual, double[] gradient,
        double[] theta, int residualCount, int parameterCount, ReadOnlySpan<double> lower, ReadOnlySpan<double> upper);

    [Fact]
    public void BoundGradientKeepsOnlyFeasibleDescentDirections()
    {
        var method = typeof(LevenbergMarquardtSolver<double>).GetMethod("ProjectedGradientInfinityNormInto", BindingFlags.Static | BindingFlags.NonPublic)!;
        var evaluate = method.CreateDelegate<ProjectedGradient>();
        // F=(x-target)^2/2: feasible descent is -gradient. At the lower bound,
        // positive gradient is blocked; at the upper bound, negative is blocked.
        foreach (var (x, residual, expected) in new[] { (0d, 3d, 0d), (0d, -3d, 3d), (1d, 3d, 3d), (1d, -3d, 0d), (.5, 3d, 3d), (.5, -3d, 3d), (0d, 0d, 0d), (1d, 0d, 0d) })
        {
            var gradient = new double[1];
            Assert.Equal(expected, evaluate(new double[,] { { 1 } }, [residual], gradient, [x], 1, 1, [0], [1]));
            Assert.Equal(residual, gradient[0]);
        }
    }

    [Fact]
    public void WriterRowMajorJacobianAndResidualBuffersReachIndependentNormalSolution()
    {
        // A columns(1,3,2),(2,-1,1) have cross product(5,3,-7).
        // y=A*(2,-1)+(5,3,-7)=(5,10,-4). Thus A'r=0 at beta=(2,-1),
        // A'A=[[14,1],[1,6]] is positive definite, RSS=83 and half-SSE=41.5.
        static void Residual(ReadOnlySpan<double> x, Span<double> r)
        {
            r[0] = x[0] + 2 * x[1] - 5;
            r[1] = 3 * x[0] - x[1] - 10;
            r[2] = 2 * x[0] + x[1] + 4;
        }
        static void Jacobian(ReadOnlySpan<double> _, Span<double> j)
        {
            j[0] = 1; j[1] = 2; j[2] = 3; j[3] = -1; j[4] = 2; j[5] = 1;
        }
        var solver = new LevenbergMarquardtSolver<double>(100, 1e-14, 1e-14, 1e-12, 1e-3);
        var pool = new LevenbergMarquardtBuffers<double>(3, 3);
        pool.Reset(2, 3);
        var result = solver.Solve(Residual, [0, 0], pool, Jacobian);
        Assert.InRange(Math.Abs(result.Parameters[0] - 2), 0, 1e-8);
        Assert.InRange(Math.Abs(result.Parameters[1] + 1), 0, 1e-8);
        Assert.InRange(Math.Abs(result.FinalCost - 41.5), 0, 1e-12);
    }
}
