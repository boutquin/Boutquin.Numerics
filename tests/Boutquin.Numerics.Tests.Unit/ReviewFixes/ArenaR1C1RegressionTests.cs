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
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

public sealed class ArenaR1C1RegressionTests
{
    private static LevenbergMarquardtSolver<double> NewLm() =>
        new(maxIterations: 100, functionTolerance: 1e-12, parameterTolerance: 1e-12,
            gradientTolerance: 1e-12, initialDamping: 1e-3);

    [Fact]
    public void FL1C1002_ProjectedStepAtUpperBound_Converges()
    {
        var result = NewLm().Solve(
            p => [p[0] - 5.0],
            [0.0],
            lowerBounds: [0.0],
            upperBounds: [3.0]);

        result.Converged.Should().BeTrue();
        result.Parameters[0].Should().Be(3.0);
        result.BoundsActive.Should().BeTrue();
    }

    [Fact]
    public void FL1C1003_InvertedBounds_AreRejected()
    {
        var act = () => NewLm().Solve(
            p => [p[0] - 1.0],
            [0.0],
            lowerBounds: [2.0],
            upperBounds: [1.0]);

        act.Should().Throw<ArgumentException>().WithParameterName("lowerBounds");
    }

    [Fact]
    public void FL1C1004_NonFiniteResidualAndJacobian_AreRejected()
    {
        var residualAct = () => NewLm().Solve(_ => [double.NaN], [0.0]);
        var jacobianAct = () => NewLm().Solve(
            p => [p[0] - 1.0],
            [0.0],
            _ => new[,] { { double.NaN } });
        static void BadResidualWriter(ReadOnlySpan<double> _, Span<double> residual) =>
            residual[0] = double.NaN;
        static void FiniteResidualWriter(ReadOnlySpan<double> p, Span<double> residual) =>
            residual[0] = p[0] - 1.0;
        static void BadJacobianWriter(ReadOnlySpan<double> _, Span<double> jacobian) =>
            jacobian[0] = double.NaN;
        var writerResidualAct = () => NewLm().Solve(BadResidualWriter, [0.0], residualCount: 1);
        var writerJacobianAct = () => NewLm().Solve(
            FiniteResidualWriter, [0.0], residualCount: 1, BadJacobianWriter);

        residualAct.Should().Throw<InvalidOperationException>().WithMessage("*residual*non-finite*index 0*");
        jacobianAct.Should().Throw<InvalidOperationException>().WithMessage("*Jacobian*non-finite*row 0*column 0*");
        writerResidualAct.Should().Throw<InvalidOperationException>().WithMessage("*residual*non-finite*index 0*");
        writerJacobianAct.Should().Throw<InvalidOperationException>().WithMessage("*Jacobian*non-finite*row 0*column 0*");
    }

    [Fact]
    public void FL1C1006_NonFiniteMean_IsRejected()
    {
        var act = () => ActiveSetQpSolver<double>.SolveMeanVariance(
            new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } },
            new[] { double.NaN, 0.0 },
            riskAversion: 1.0,
            minWeight: 0.0,
            maxWeight: 1.0);

        act.Should().Throw<ArgumentException>().WithParameterName("means");
    }

    [Fact]
    public void FL1C1007_SingleAssetPureLp_RejectsInfeasibleMaximum()
    {
        var act = () => ActiveSetQpSolver<double>.SolveMeanVariance(
            new[,] { { 1.0 } },
            new[] { 1.0 },
            riskAversion: 0.0,
            minWeight: 0.0,
            maxWeight: 0.5);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void FL23C1001_NaNBounds_AreRejected()
    {
        var lowerAct = () => NewLm().Solve(
            p => [p[0] - 1.0], [0.0], lowerBounds: [double.NaN]);
        var upperAct = () => NewLm().Solve(
            p => [p[0] - 1.0], [0.0], upperBounds: [double.NaN]);

        lowerAct.Should().Throw<ArgumentException>().WithParameterName("lowerBounds");
        upperAct.Should().Throw<ArgumentException>().WithParameterName("upperBounds");
    }

    [Fact]
    public void FL5C1001_RootSolverFamily_RejectsNonFiniteObjectiveValues()
    {
        Action[] calls =
        [
            () => new BisectionSolver<double>(1e-12, 1e-12).Solve(_ => double.NaN, -1.0, 1.0),
            () => new BrentSolver<double>(1e-12).Solve(_ => double.NaN, -1.0, 1.0),
            () => new NewtonRaphsonSolver<double>(1e-12, 1e-12, 1e-12).Solve(_ => double.NaN, -1.0, 1.0),
            () => new SecantSolver<double>(1e-12).Solve(_ => double.NaN, 0.0, 1.0),
            () => new MullerSolver<double>(1e-12).Solve(_ => double.NaN, 0.0, 1.0, 2.0),
        ];

        foreach (var call in calls)
        {
            call.Should().Throw<InvalidOperationException>().WithMessage("*non-finite*");
        }
    }

    [Fact]
    public void FL5C1002_WrongSidedInfiniteBounds_AreRejected()
    {
        var lowerAct = () => NewLm().Solve(
            p => [p[0] - 1.0], [0.0], lowerBounds: [double.PositiveInfinity]);
        var upperAct = () => NewLm().Solve(
            p => [p[0] - 1.0], [0.0], upperBounds: [double.NegativeInfinity]);

        lowerAct.Should().Throw<ArgumentException>().WithParameterName("lowerBounds");
        upperAct.Should().Throw<ArgumentException>().WithParameterName("upperBounds");
    }

    [Fact]
    public void FL5C1003_ChangingResidualLength_IsRejected()
    {
        var calls = 0;
        double[] Residuals(double[] p) => ++calls == 1 ? [p[0] - 1.0] : [p[0] - 1.0, 1000.0];

        var act = () => NewLm().Solve(Residuals, [0.0]);

        act.Should().Throw<InvalidOperationException>().WithMessage("*returned 2 residuals*expected 1*");
    }

    [Fact]
    public void FL5C1004_ShrunkFuncPool_DoesNotAllocateLogicalArrayPerCallback()
    {
        var solver = NewLm();
        var residual = new double[1];
        double[] Residuals(double[] p)
        {
            residual[0] = p[0] - 1.0;
            return residual;
        }

        double Measure(LevenbergMarquardtBuffers<double> buffers)
        {
            for (var i = 0; i < 20; i++)
            {
                _ = solver.Solve(Residuals, [0.0], buffers);
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            var before = GC.GetAllocatedBytesForCurrentThread();
            for (var i = 0; i < 100; i++)
            {
                _ = solver.Solve(Residuals, [0.0], buffers);
            }

            return (GC.GetAllocatedBytesForCurrentThread() - before) / 100.0;
        }

        var fresh = Measure(new LevenbergMarquardtBuffers<double>(1, 1));
        var shrunkBuffers = new LevenbergMarquardtBuffers<double>(4, 8);
        shrunkBuffers.Reset(1, 1);
        var shrunk = Measure(shrunkBuffers);

        (shrunk - fresh).Should().BeLessThan(32.0,
            "shrinking a pool must not add a callback-array allocation on every solve");
    }

    [Fact]
    public void FL5C1005_NonFiniteQpCovarianceAndBounds_AreRejected()
    {
        var covarianceAct = () => ActiveSetQpSolver<double>.SolveMinVariance(
            new[,] { { double.NaN, 0.0 }, { 0.0, 1.0 } }, 0.0, 1.0);
        var boundAct = () => ActiveSetQpSolver<double>.SolveMinVariance(
            new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }, double.NaN, 1.0);

        covarianceAct.Should().Throw<ArgumentException>().WithParameterName("covariance");
        boundAct.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("minWeight");
    }

    [Fact]
    public void FL7C1001_WriterBoundOptimum_UsesProjectedGradient()
    {
        static void Residual(ReadOnlySpan<double> p, Span<double> r) => r[0] = p[0] - 2.0;
        static void Jacobian(ReadOnlySpan<double> _, Span<double> j) => j[0] = 1.0;

        var result = NewLm().Solve(
            Residual,
            [0.0],
            residualCount: 1,
            Jacobian,
            lowerBounds: [0.0],
            upperBounds: [1.0]);

        result.Converged.Should().BeTrue();
        result.Parameters[0].Should().Be(1.0);
    }

    [Fact]
    public void FL7C1002_ZeroJacobianColumn_IsNotIdentityDamped()
    {
        var buffers = new LevenbergMarquardtBuffers<double>(2, 1);
        buffers.Jacobian[0, 0] = 1.0;
        buffers.Jacobian[0, 1] = 0.0;

        DampedLinearSolve<double>.TrySolve([1.0], 1e-3, buffers).Should().BeTrue();

        buffers.NormalEquations[1, 1].Should().Be(0.0,
            "a structural zero column must remain undamped in the Marquardt system");
        buffers.Delta[1].Should().Be(0.0);
    }

    [Fact]
    public void FL7C1003_EitherNelderMeadTolerance_DeclaresConvergence()
    {
        var result = NelderMeadMinimizer<double>.Minimize(
            _ => 0.0,
            [0.0],
            new SimplexOptions<double>
            {
                MaxIterations = 1,
                XTolerance = 1e-12,
                FTolerance = 1e-12,
            });

        result.Converged.Should().BeTrue();
    }
}
