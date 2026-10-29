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
using Boutquin.Numerics.Solvers.Internal;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// Range regressions for the Levenberg–Marquardt step, model and stopping arithmetic. Each
/// problem has finite inputs and a representable exact step, model reduction or stopping
/// decision; the expected values below come from exact algebra of the affine residuals,
/// not from solver output.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class Arena414074R1LmTests
{
    private static MultivariateSolverResult<double> SolveRepeated(
        bool writer,
        LevenbergMarquardtSolver<double> solver,
        int residualCount,
        Func<double, double> residual,
        double slope,
        double start,
        List<double> visits,
        double[]? upperBounds = null)
    {
        if (writer)
        {
            return solver.Solve(
                (ReadOnlySpan<double> x, Span<double> r) =>
                {
                    visits.Add(x[0]);
                    r.Fill(residual(x[0]));
                },
                [start],
                residualCount,
                (ReadOnlySpan<double> _, Span<double> j) => j.Fill(slope),
                default,
                upperBounds);
        }

        return solver.Solve(
            x =>
            {
                visits.Add(x[0]);
                return Enumerable.Repeat(residual(x[0]), residualCount).ToArray();
            },
            [start],
            _ =>
            {
                var j = new double[residualCount, 1];
                for (var i = 0; i < residualCount; i++)
                {
                    j[i, 0] = slope;
                }

                return j;
            },
            default,
            upperBounds);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TinyGramAggregate_TakesTheDampedStep(bool writer)
    {
        // Sixteen residuals 1e-162·(θ−2) with J = 1e-162. Each square 1e-324 rounds to zero,
        // but JᵀJ = 1.6e-323 is representable; the exact damped step from θ = 1 is 1/1.001.
        var visits = new List<double>();
        var solver = new LevenbergMarquardtSolver<double>(50, 1e-14, 1e-14, 0, 1e-3);
        var result = SolveRepeated(writer, solver, 16, x => 1e-162 * (x - 2), 1e-162, 1.0, visits);

        Assert.True(visits.Count >= 2, $"no trial was evaluated; reason {result.TerminationReason}");
        Assert.InRange(Math.Abs(visits[1] - (1 + (1 / 1.001))), 0, 1e-15);
        Assert.NotEqual(LmTerminationReason.DampingOverflow, result.TerminationReason);
        Assert.InRange(Math.Abs(result.Parameters[0] - 2), 0, 1e-3);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HugeGramAggregate_TakesTheDampedStep(bool writer)
    {
        // Four residuals 1e155·(θ−1e-160) with J = 1e155. JᵀJ = 4e310 overflows, but the
        // exact damped step 1e-160/1.001 and both costs are representable.
        var visits = new List<double>();
        var solver = new LevenbergMarquardtSolver<double>(50, 1e-14, 1e-14, 0, 1e-3);
        var result = SolveRepeated(writer, solver, 4, x => 1e155 * (x - 1e-160), 1e155, 0.0, visits);

        var expectedStep = 1e-160 / 1.001;
        Assert.True(visits.Count >= 2, $"no trial was evaluated; reason {result.TerminationReason}");
        Assert.InRange(Math.Abs(visits[1] - expectedStep), 0, 1e-14 * expectedStep);
        Assert.NotEqual(LmTerminationReason.DampingOverflow, result.TerminationReason);
        Assert.InRange(Math.Abs(result.Parameters[0] - 1e-160), 0, 1e-3 * 1e-160);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UnclippedPredictedReductionNearMaxValue_AcceptsTheImprovingStep(bool writer)
    {
        // r = (θ−1e154, θ−1e154), J = (1, 1), θ₀ = 0. The step 1e154/1.001 reduces the cost
        // from 1e308 to about 1e302; the exact predicted reduction is about 9.99999e307.
        var visits = new List<double>();
        var solver = new LevenbergMarquardtSolver<double>(1, 1e-14, 1e-14, 0, 1e-3);
        var result = SolveRepeated(writer, solver, 2, x => x - 1e154, 1.0, 0.0, visits);

        var expected = 1e154 / 1.001;
        Assert.Equal(1, result.Iterations);
        Assert.InRange(Math.Abs(result.Parameters[0] - expected), 0, 1e-15 * expected);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ClippedPredictedReductionNearMaxValue_AcceptsTheImprovingStep(bool writer)
    {
        // r = θ−1.8e154, J = 1, upper bound 1e154, θ₀ = 0. The clipped step 1e154 has exact
        // projected model reduction 1.8e308−0.5e308 = 1.3e308, equal to the actual reduction.
        var visits = new List<double>();
        var solver = new LevenbergMarquardtSolver<double>(1, 1e-14, 1e-14, 0, 1e-3);
        var result = SolveRepeated(writer, solver, 1, x => x - 1.8e154, 1.0, 0.0, visits, [1e154]);

        Assert.Equal(1, result.Iterations);
        Assert.Equal(1e154, result.Parameters[0]);
        Assert.True(result.BoundsActive);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ParameterToleranceWithUnrepresentableThetaNorm_DoesNotCertifyALargeStep(bool writer)
    {
        // θ₀ = (C+1e300, C+1e300) with C = 1.5e308, r = 1e-150·(θ−C), J = 1e-150·I.
        // After the first step ‖δ‖ ≈ 1.41e300 while tol·(‖θ‖+tol) ≈ 2.12e294, so the
        // documented parameter-tolerance predicate is false although ‖θ‖ > MaxValue.
        const double c = 1.5e308;
        const double th0 = c + 1e300;
        var solver = new LevenbergMarquardtSolver<double>(50, 1e-14, 1e-14, 0, 1e-3);
        var result = writer
            ? solver.Solve(
                (ReadOnlySpan<double> x, Span<double> r) =>
                {
                    r[0] = 1e-150 * (x[0] - c);
                    r[1] = 1e-150 * (x[1] - c);
                },
                [th0, th0],
                2,
                (ReadOnlySpan<double> _, Span<double> j) =>
                {
                    j[0] = 1e-150;
                    j[1] = 0;
                    j[2] = 0;
                    j[3] = 1e-150;
                })
            : solver.Solve(
                x => [1e-150 * (x[0] - c), 1e-150 * (x[1] - c)],
                [th0, th0],
                _ => new double[,] { { 1e-150, 0 }, { 0, 1e-150 } });

        Assert.InRange(Math.Abs(result.Parameters[0] - c), 0, 1e294);
        Assert.InRange(Math.Abs(result.Parameters[1] - c), 0, 1e294);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FunctionToleranceWithUnderflowedCost_DoesNotStopOnRelativeReductionNearOne(bool writer)
    {
        // Sixteen residuals 1e-160·(θ−2), J = 1e-160, θ₀ = 1. After the first step the
        // residuals are about −1e-163 (normal) while ½Σr² ≈ 8e-326 rounds to zero. Every
        // trial on this affine problem reduces the cost by a factor close to one, so the
        // relative function-tolerance predicate is false at every iterate.
        var visits = new List<double>();
        var solver = new LevenbergMarquardtSolver<double>(50, 1e-14, 1e-14, 0, 1e-3);
        var result = SolveRepeated(writer, solver, 16, x => 1e-160 * (x - 2), 1e-160, 1.0, visits);

        Assert.NotEqual(LmTerminationReason.FunctionToleranceReached, result.TerminationReason);
        Assert.InRange(Math.Abs(result.Parameters[0] - 2), 0, 1e-6);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GradientProductsThatOverflow_CancelToTheRepresentableStationaryGradient(bool writer)
    {
        // J = (1e160, 1e160), r = (1e150, −1e150): each product overflows, Jᵀr = 0 exactly.
        var solver = new LevenbergMarquardtSolver<double>(10, 1e-14, 1e-14, 0, 1e-3);
        MultivariateSolverResult<double>? result = null;
        var error = Record.Exception(() => result = writer
            ? solver.Solve(
                (ReadOnlySpan<double> _, Span<double> r) =>
                {
                    r[0] = 1e150;
                    r[1] = -1e150;
                },
                [0.0],
                2,
                (ReadOnlySpan<double> _, Span<double> j) => j.Fill(1e160))
            : solver.Solve(
                _ => [1e150, -1e150],
                [0.0],
                _ => new double[,] { { 1e160 }, { 1e160 } }));

        Assert.Null(error);
        Assert.NotNull(result);
        Assert.True(result.Converged);
        Assert.Equal(LmTerminationReason.GradientToleranceReached, result.TerminationReason);
        Assert.Equal(0, result.Iterations);
    }

    [Fact]
    public void DampedSolve_RejectsANonFiniteStepAndKeepsTheFiniteControl()
    {
        // r = −1e154 with J = slope: the step 1e154/(slope·1.001) is finite for 1e-154 and
        // exceeds MaxValue for 1e-160.
        var control = new LevenbergMarquardtBuffers<double>(1, 1);
        control.Jacobian[0, 0] = 1e-154;
        Assert.True(DampedLinearSolve<double>.TrySolve([-1e154], 1e-3, control));
        Assert.InRange(Math.Abs(control.Delta[0] - (1e308 / 1.001)), 0, 1e-15 * 1e308);

        var witness = new LevenbergMarquardtBuffers<double>(1, 1);
        witness.Jacobian[0, 0] = 1e-160;
        Assert.False(DampedLinearSolve<double>.TrySolve([-1e154], 1e-3, witness));

        // Coupled companion: J = diag(1, 1e-160), r = (−1, −1e154). The second coordinate's
        // step is not representable, so no step may be reported.
        var coupled = new LevenbergMarquardtBuffers<double>(2, 2);
        coupled.Jacobian[0, 0] = 1;
        coupled.Jacobian[1, 1] = 1e-160;
        Assert.False(DampedLinearSolve<double>.TrySolve([-1, -1e154], 1e-3, coupled));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void NonFiniteDampedStep_NeverReachesPublicCallbacks(bool writer, bool pooled)
    {
        // r(x) = 1e-160·x − 1e154 with its exact Jacobian; the undamped minimizer 1e314 is not
        // representable, so early damped steps are infinite and must be rejected, not tried.
        var inputs = new List<double>();
        var solver = new LevenbergMarquardtSolver<double>(20, 1e-30, 1e-30, 0, 1e-3);
        MultivariateSolverResult<double>? result = null;
        Exception? error;
        if (writer)
        {
            LmResidualWriter<double> residual = (x, r) =>
            {
                inputs.Add(x[0]);
                r[0] = (1e-160 * x[0]) - 1e154;
            };
            LmJacobianWriter<double> jacobian = (x, j) =>
            {
                inputs.Add(x[0]);
                j[0] = 1e-160;
            };
            error = Record.Exception(() => result = pooled
                ? solver.Solve(residual, [0.0], new LevenbergMarquardtBuffers<double>(1, 1), jacobian)
                : solver.Solve(residual, [0.0], 1, jacobian));
        }
        else
        {
            double[] Residual(double[] x)
            {
                inputs.Add(x[0]);
                return [(1e-160 * x[0]) - 1e154];
            }

            double[,] Jacobian(double[] x)
            {
                inputs.Add(x[0]);
                return new double[,] { { 1e-160 } };
            }

            error = Record.Exception(() => result = pooled
                ? solver.Solve(Residual, [0.0], new LevenbergMarquardtBuffers<double>(1, 1), Jacobian)
                : solver.Solve(Residual, [0.0], Jacobian));
        }

        Assert.All(inputs, x => Assert.True(double.IsFinite(x), $"callback received {x}"));
        Assert.Null(error);
        Assert.NotNull(result);
        Assert.True(double.IsFinite(result.Parameters[0]));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FiniteStepWhoseTrialPointOverflows_NeverReachesPublicCallbacks(bool writer)
    {
        // r(x) = 1e-160·x − 2e148 from θ₀ = 1e308: the damped step 1e308/1.001 is finite, but
        // θ₀ + δ ≈ 1.999e308 exceeds MaxValue. That trial must be rejected, not evaluated.
        var inputs = new List<double>();
        var solver = new LevenbergMarquardtSolver<double>(20, 1e-30, 1e-30, 0, 1e-3);
        MultivariateSolverResult<double>? result = null;
        var error = Record.Exception(() => result = SolveRepeated(
            writer, solver, 1, x => (1e-160 * x) - 2e148, 1e-160, 1e308, inputs));

        Assert.All(inputs, x => Assert.True(double.IsFinite(x), $"callback received {x}"));
        Assert.Null(error);
        Assert.NotNull(result);
        Assert.True(double.IsFinite(result.Parameters[0]));
    }
}
