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

namespace Boutquin.Numerics.Tests.Unit;

[Trait("Tier", "Unit")]
public sealed class Arena20261003C1RegressionTests
{
    [Fact]
    public void MullerLargeFiniteAffineCoefficientsPreserveUnitCorrection()
    {
        var scale = Math.ScaleB(1, 600);
        var result = new MullerSolver<double>(1e-12, 100).Solve(x => scale * (x - 2), 0, 0.5, 1);
        Assert.Equal(2.0, result.Root);
        Assert.True(result.Converged);
        Assert.Equal(0.0, result.FinalResidual);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MutatingArrayCallbacksPreserveFiniteDifferenceSolve(bool pooled)
    {
        static double[] Residual(double[] x)
        {
            double[] residual = [x[0] - 1, x[1] - 2];
            Array.Fill(x, 100);
            return residual;
        }
        var solver = new LevenbergMarquardtSolver<double>(100, 1e-12, 1e-12, 1e-12, 1e-3);
        var result = pooled
            ? solver.Solve(Residual, [0, 0], new LevenbergMarquardtBuffers<double>(2, 2))
            : solver.Solve(Residual, [0, 0]);
        Assert.InRange(result.FinalCost, 0, 1e-18);
        Assert.Equal(1.0, result.Parameters[0], 9);
        Assert.Equal(2.0, result.Parameters[1], 9);
    }

    [Theory]
    [InlineData(false, -1)]
    [InlineData(false, 0)]
    [InlineData(true, -1)]
    [InlineData(true, 0)]
    public void ScalarQpRejectsNonpositiveCovariance(bool meanVariance, double variance)
    {
        Assert.Throws<InvalidOperationException>(() => meanVariance
            ? ActiveSetQpSolver<double>.SolveMeanVariance(new[,] { { variance } }, [0], 1, 0, 1)
            : ActiveSetQpSolver<double>.SolveMinVariance(new[,] { { variance } }, 0, 1));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void QpReleaseRetainsScaleInvariantOptimum(bool meanVariance)
    {
        double[,] covariance = { { 127.25, 39, 129, 120 }, { 39, 62.25, 12, 29 },
            { 129, 12, 164.25, 134 }, { 120, 29, 134, 130.25 } };
        var scale = 1.7e308 / 164.25;
        var scaled = new double[4, 4];
        for (var i = 0; i < 4; i++)
        {
            for (var j = 0; j < 4; j++)
            {
                scaled[i, j] = covariance[i, j] * scale;
            }
        }
        var actual = meanVariance
            ? ActiveSetQpSolver<double>.SolveMeanVariance(scaled, new double[4], 1, 0, 0.4)
            : ActiveSetQpSolver<double>.SolveMinVariance(scaled, 0, 0.4);
        // Solve the three-free-variable KKT system on the small matrix, with w1 = 0.4.
        double[] expected = [0.07585613984811573, 0.4, 0.15667001003009054, 0.36747385012179357];
        for (var i = 0; i < 4; i++)
        {
            Assert.Equal(expected[i], actual[i], 10);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void QpReleaseRejectsOverflowingGradient(bool meanVariance)
    {
        double[,] covariance = { { 7.8e307, 9.75e307 }, { 9.75e307, 1.3e308 } };
        var error = Assert.Throws<InvalidOperationException>(() => meanVariance
            ? ActiveSetQpSolver<double>.SolveMeanVariance(covariance, [0, 0], 1, -2, 3)
            : ActiveSetQpSolver<double>.SolveMinVariance(covariance, -2, 3));
        Assert.Equal("QP release gradient must be finite.", error.Message);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NarrowCentralDifferencePreservesRepresentableIdentityDerivative(bool writer)
    {
        var buffers = new LevenbergMarquardtBuffers<double>(1, 1);
        double[] Residual(double[] x) => [x[0]];
        void Write(ReadOnlySpan<double> x, Span<double> r) => r[0] = x[0];
        if (writer)
        {
            FiniteDifferenceJacobian<double>.EvaluateInto(Write, [1e-309], buffers, [0], [2e-309]);
        }
        else
        {
            FiniteDifferenceJacobian<double>.EvaluateInto(Residual, [1e-309], buffers, [0], [2e-309]);
        }
        Assert.Equal(1.0, buffers.Jacobian[0, 0]);
    }

    [Fact]
    public void InitialCallbackCannotMutateConvenienceSolveState()
    {
        var solver = new LevenbergMarquardtSolver<double>(100, 1e-12, 1e-12, 1e-12, 1e-3);
        double[] Residual(double[] x)
        {
            var value = x[0] - 1;
            x[0] = 100;
            return [value];
        }
        double[,] Jacobian(double[] _) => new double[,] { { 1 } };
        var result = solver.Solve(Residual, [1], Jacobian, [0], [2]);
        var pooled = solver.Solve(Residual, [1], new LevenbergMarquardtBuffers<double>(1, 1), Jacobian, [0], [2]);

        Assert.Equal(1.0, result.Parameters[0]);
        Assert.Equal(pooled.Parameters, result.Parameters);
        Assert.Equal(0.0, result.FinalCost);
        Assert.True(result.Converged);
    }

    [Theory]
    [InlineData(1e-200)]
    [InlineData(1e200)]
    public void OlsExtremeResidualScalePreservesDispersionAndRatio(double scale)
    {
        var result = OrdinaryLeastSquares<double>.Fit(new double[2, 0], [-scale, scale], false);
        Assert.InRange(result.ResidualStandardDeviation / scale, 1 - 1e-14, 1 + 1e-14);
        Assert.InRange(Math.Abs(result.RSquared), 0, 1e-14);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FiniteDifferenceAvoidsOverflowBeforeDivision(bool writer)
    {
        const double center = 1e308;
        var h = center * Math.Cbrt(double.BitIncrement(1) - 1);
        double F(double x) => 1.5e308 * Math.Tanh((x - center) / h);
        var buffers = new LevenbergMarquardtBuffers<double>(1, 1);
        if (writer)
        {
            FiniteDifferenceJacobian<double>.EvaluateInto((p, r) => r[0] = F(p[0]), [center], buffers);
        }
        else
        {
            FiniteDifferenceJacobian<double>.EvaluateInto(p => [F(p[0])], [center], buffers);
        }
        var interval = (center + h) - (center - h);
        var expected = F(center + h) / interval - F(center - h) / interval;
        Assert.InRange(buffers.Jacobian[0, 0] / expected, 1 - 1e-14, 1 + 1e-14);
    }

    [Fact]
    public void NewtonFiniteDerivativeDoesNotManufactureStagnation()
    {
        const double center = 1e308;
        var h = center * Math.Cbrt(double.BitIncrement(1) - 1);
        var root = center + 0.1 * h;
        var result = new NewtonRaphsonSolver<double>(1, 1e290, 0, 200)
            .Solve(x => 1.5e308 * Math.Tanh((x - root) / h), center - 2 * h, center + 2 * h);
        Assert.InRange(Math.Abs((result.Root - root) / h), 0, 1e-9);
    }

    [Fact]
    public void MullerTinyQuadraticRetainsDiscriminant()
    {
        var result = new MullerSolver<double>(1e-212, 50).Solve(x => 1e-200 * (x * x - 2), 0, 1, 2);
        Assert.InRange(Math.Abs(result.Root - Math.Sqrt(2)), 0, 1e-12);
        Assert.True(result.Converged);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LmCostPreservesSubnormalAggregate(bool writer)
    {
        var s = Math.ScaleB(1, -538);
        var solver = new LevenbergMarquardtSolver<double>(10, 1e-12, 1e-12, 0, 1e-3);
        var result = writer
            ? solver.Solve((ReadOnlySpan<double> _, Span<double> r) => r.Fill(s), [0.0], 8,
                (ReadOnlySpan<double> _, Span<double> j) => j.Clear())
            : solver.Solve(_ => Enumerable.Repeat(s, 8).ToArray(), [0.0], _ => new double[8, 1]);
        Assert.Equal(double.Epsilon, result.FinalCost);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LmSubnormalGradientIsNotInitialStationarity(bool writer)
    {
        var s = Math.ScaleB(1, -538);
        var j = new double[8, 1];
        for (var i = 0; i < 8; i++) { j[i, 0] = s; }
        var solver = new LevenbergMarquardtSolver<double>(1, 1e-12, 1e-12, 0, 1e-3);
        var result = writer
            ? solver.Solve((ReadOnlySpan<double> p, Span<double> r) => r.Fill(s * (p[0] - 1)), [0.0], 8,
                (ReadOnlySpan<double> _, Span<double> v) => v.Fill(s))
            : solver.Solve(p => Enumerable.Repeat(s * (p[0] - 1), 8).ToArray(), [0.0], _ => j);
        Assert.False(result.Converged && result.Parameters[0] == 0);
    }

    [Fact]
    public void OlsCenteringRetainsExactlyRepresentableTranslation()
    {
        var shift = Math.ScaleB(1, 50);
        var result = OrdinaryLeastSquares<double>.Fit(new double[4, 0], [shift, shift + 0.25, shift + 0.25, shift + 0.5]);
        Assert.InRange(Math.Abs(result.RSquared), 0, 1e-10);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LmLargeJacobianRetainsRepresentableCovariance(bool writer)
    {
        const double s = 1e160;
        var solver = new LevenbergMarquardtSolver<double>(10, 1e-12, 1e-12, 0, 1e-3);
        var result = writer
            ? solver.Solve((ReadOnlySpan<double> p, Span<double> r) => { r[0] = s * p[0] - 1; r[1] = s * p[0] + 1; }, [0.0], 2,
                (ReadOnlySpan<double> _, Span<double> j) => j.Fill(s))
            : solver.Solve(p => [s * p[0] - 1, s * p[0] + 1], [0.0], _ => new[,] { { s }, { s } });
        Assert.NotNull(result.ParameterCovariance);
        Assert.InRange(result.ParameterCovariance[0, 0] / 1e-320, 0.999, 1.001);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LmTinyResidualScaleRetainsRepresentableCovariance(bool writer)
    {
        const double s = 1e-200;
        var solver = new LevenbergMarquardtSolver<double>(10, 1e-12, 1e-12, 0, 1e-3);
        var result = writer
            ? solver.Solve((ReadOnlySpan<double> p, Span<double> r) => { r[0] = s * (p[0] - 1); r[1] = s * (p[0] + 1); }, [0.0], 2,
                (ReadOnlySpan<double> _, Span<double> j) => j.Fill(s))
            : solver.Solve(p => [s * (p[0] - 1), s * (p[0] + 1)], [0.0], _ => new[,] { { s }, { s } });
        Assert.NotNull(result.ParameterCovariance);
        Assert.InRange(result.ParameterCovariance[0, 0], 1 - 1e-14, 1 + 1e-14);
    }

    [Theory]
    [InlineData(1e-200)]
    [InlineData(1e160)]
    public void ScaledNormalInverseRetainsUnitRatio(double scale)
    {
        var result = DampedLinearSolve<double>.TryInvertNormalEquations(new[,] { { scale } }, 1, 1, scale);
        Assert.NotNull(result);
        Assert.Equal(1, result[0, 0]);
    }
}
