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
public sealed class FidelityC1ProjectedGainControls
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NegativeProjectedModelReductionCannotBecomePositiveGain(bool writerPath)
    {
        // J=[[1,3/5],[0,4/5]], starting just inside the lower x bound. The
        // unconstrained step crosses that bound and moves y upwards. Clipping x
        // leaves a step with negative predicted reduction, equal to actual reduction
        // for these affine residuals. Dividing two negative reductions would
        // incorrectly yield gain1 and accept a strictly worse objective.
        var visits = new List<double[]>();
        double[] Residual(double[] x)
        {
            visits.Add((double[])x.Clone());
            return [x[0] + 0.6 * x[1] + 1, 0.8 * x[1] - 0.125];
        }
        void WriteResidual(ReadOnlySpan<double> x, Span<double> r)
        {
            visits.Add(x.ToArray());
            r[0] = x[0] + 0.6 * x[1] + 1;
            r[1] = 0.8 * x[1] - 0.125;
        }
        static double[,] Jacobian(double[] _) => new double[,] { { 1, 0.6 }, { 0, 0.8 } };
        static void WriteJacobian(ReadOnlySpan<double> _, Span<double> j)
        {
            j[0] = 1;
            j[1] = 0.6;
            j[2] = 0;
            j[3] = 0.8;
        }
        var solver = new LevenbergMarquardtSolver<double>(1, 1e-30, 1e-30, 1e-30, 0.01);
        double[] start = [0.001, 0];
        double[] lower = [0, -10];
        double[] upper = [10, 10];
        var result = writerPath
            ? solver.Solve(WriteResidual, start, 2, WriteJacobian, lower, upper)
            : solver.Solve(Residual, start, Jacobian, lower, upper);
        Assert.Equal(2, visits.Count);
        Assert.Equal(0, visits[1][0]);
        Assert.True(visits[1][1] > 0);
        Assert.False(result.Converged);
        Assert.Equal(0, result.Iterations);
        Assert.Equal(start, result.Parameters);
        Assert.Equal((1.001 * 1.001 + 0.125 * 0.125) / 2, result.FinalCost);
        Assert.Equal(LmTerminationReason.MaxIterationsReached, result.TerminationReason);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PositiveClippedModelReductionUsesTheUnreducedQuadraticModel(bool writerPath)
    {
        // r = (x−1, y−y²/2−1), J = [[1,0],[0,1−y]], θ₀ = (0,0), upper x bound 1/10, λ₀ = 1.
        // The first damped step (1/2,1/2) is clipped to t = (1/10,1/2). The unreduced model
        // drop −gᵀt − ½‖Jt‖² is 47/100; the actual drop is 1 − 1921/3200 = 1279/3200, so
        // ρ = 1279/1504 and the Nielsen update gives λ₁ = 1 − (2ρ−1)³. At x₁ the y residual
        // is −5/8 with slope 1/2, so the second y trial is 1/2 + (5/4)/(1+λ₁). The shortcut
        // ½δᵀ(λ·diag(JᵀJ)·δ − Jᵀr), valid only for the unclipped solved step, yields a
        // different ρ and therefore a different second trial.
        var visits = new List<double[]>();
        double[] Residual(double[] x)
        {
            visits.Add((double[])x.Clone());
            return [x[0] - 1, x[1] - (0.5 * x[1] * x[1]) - 1];
        }
        void WriteResidual(ReadOnlySpan<double> x, Span<double> r)
        {
            visits.Add(x.ToArray());
            r[0] = x[0] - 1;
            r[1] = x[1] - (0.5 * x[1] * x[1]) - 1;
        }
        static double[,] Jacobian(double[] x) => new double[,] { { 1, 0 }, { 0, 1 - x[1] } };
        static void WriteJacobian(ReadOnlySpan<double> x, Span<double> j)
        {
            j[0] = 1;
            j[1] = 0;
            j[2] = 0;
            j[3] = 1 - x[1];
        }
        var solver = new LevenbergMarquardtSolver<double>(2, 1e-30, 1e-30, 0, 1);
        double[] upper = [0.1, 10];
        _ = writerPath
            ? solver.Solve(WriteResidual, [0, 0], 2, WriteJacobian, upperBounds: upper)
            : solver.Solve(Residual, [0, 0], Jacobian, upperBounds: upper);
        var rho = 1279.0 / 1504;
        var lambda = 1 - Math.Pow((2 * rho) - 1, 3);
        var expectedSecondY = 0.5 + (1.25 / (1 + lambda));
        Assert.Equal(3, visits.Count);
        Assert.Equal(0.1, visits[1][0]);
        Assert.InRange(Math.Abs(0.5 - visits[1][1]), 0, 1e-15);
        Assert.InRange(Math.Abs(visits[2][1] - expectedSecondY), 0, 2e-14);
    }
}
