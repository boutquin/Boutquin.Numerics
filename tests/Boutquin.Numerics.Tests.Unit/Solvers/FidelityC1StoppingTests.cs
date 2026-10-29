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
public sealed class FidelityC1StoppingTests
{
    [Fact]
    public void Muller_RelativeStepDoesNotRequireResidualSignChange()
    {
        // The interpolating polynomial in t=x-1000 is 3t²-2t-1/8.
        // Its nearer root is (2+sqrt(11/2))/6; the cubic stays positive there.
        var expected = 1000 + (2 + Math.Sqrt(5.5)) / 6;
        Assert.True((1002 - expected) / expected < 0.002);
        var solver = new MullerSolver<double>(0.002, 1);
        var result = solver.Solve(x => 100 * (Math.Pow(x - 1000, 3) - 0.125), 1000, 1001, 1002);
        Assert.InRange(Math.Abs(result.Root - expected), 0, 2e-12);
        Assert.True(result.FinalResidual > 0.002);
        Assert.True(result.Converged);
        Assert.Equal(1, result.Iterations);
    }

    [Fact]
    public void Muller_SubunitIterateUsesRelativeStepWithoutAbsoluteFloor()
    {
        // Divided differences give a=23/100, b=43/500, c=63/8000.
        // The positive scaling of the objective cancels from the quadratic root.
        const double a = 0.23;
        const double b = 0.086;
        const double c = 0.007875;
        var expected = 0.2 - 2 * c / (b + Math.Sqrt(b * b - 4 * a * c));
        Assert.True(0.2 - expected < 0.2);
        Assert.True((0.2 - expected) / expected > 0.2);
        var solver = new MullerSolver<double>(0.2, 1);
        var result = solver.Solve(x => 100000 * (x * x * x - 0.000125), 0, 0.03, 0.2);
        Assert.InRange(Math.Abs(result.Root - expected), 0, 2e-15);
        Assert.True(result.FinalResidual < -0.2);
        Assert.False(result.Converged);
        Assert.Equal(1, result.Iterations);
    }

    [Fact]
    public void Muller_NegativeDiscriminantContinuationCannotCertifyRelativeConvergence()
    {
        // The exact quadratic (x-1000)²+1 has discriminant -4 and no real root.
        // sqrt(abs(discriminant)) visits1000+1/3, a relatively small heuristic step.
        var solver = new MullerSolver<double>(0.002, 1);
        var result = solver.Solve(x => (x - 1000) * (x - 1000) + 1, 1000, 1001, 1002);
        Assert.InRange(Math.Abs(result.Root - (1000 + 1.0 / 3)), 0, 2e-12);
        Assert.True(result.FinalResidual >= 1);
        Assert.False(result.Converged);
    }

    [Fact]
    public void Muller_RepeatedNonzeroRealQuadraticIterateSatisfiesRelativeCriterion()
    {
        // Linear polynomial root 1e16+1 lies halfway between adjacent doubles.
        // The +1 update rounds to the same nonzero iterate, making relative change zero.
        const double anchor = 1e16;
        var solver = new MullerSolver<double>(1e-12, 1);
        var result = solver.Solve(x => (x - anchor) - 1, anchor - 4, anchor - 2, anchor);
        Assert.Equal(anchor, result.Root);
        Assert.Equal(-1, result.FinalResidual);
        Assert.Equal(0, result.EstimatedError);
        Assert.Equal(1, result.Iterations);
        Assert.True(result.Converged);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Lm_RejectedTrialUsesSignedGainRatioForFunctionTolerance(bool writer)
    {
        // At x=0, r=[1,s], J=[0,s], lambda=1: h=-1/2.
        // Trial second residual is 2s, actual reduction=-1.5s²,
        // predicted reduction=.375s², ratio=-4. Both relative magnitudes
        // are below 1e-6, and the signed half-ratio <=1 despite |ratio|>2.
        const double scale = 0.0005;
        var solver = new LevenbergMarquardtSolver<double>(1, 1e-6, 1e-30, 0, 1);
        double[] Residual(double[] x) => [1, scale * (1 + x[0] + 6 * x[0] * x[0])];
        double[,] Jacobian(double[] x) => new[,] { { 0.0 }, { scale * (1 + 12 * x[0]) } };
        void ResidualWriter(ReadOnlySpan<double> x, Span<double> r)
        {
            r[0] = 1;
            r[1] = scale * (1 + x[0] + 6 * x[0] * x[0]);
        }
        void JacobianWriter(ReadOnlySpan<double> x, Span<double> j)
        {
            j[0] = 0;
            j[1] = scale * (1 + 12 * x[0]);
        }
        var result = writer
            ? solver.Solve(ResidualWriter, new[] { 0.0 }, 2, JacobianWriter)
            : solver.Solve(Residual, new[] { 0.0 }, Jacobian);
        Assert.Equal(0, result.Parameters[0]);
        Assert.Equal(0, result.Iterations);
        Assert.True(result.Converged);
        Assert.Equal(LmTerminationReason.FunctionToleranceReached, result.TerminationReason);
    }
}
