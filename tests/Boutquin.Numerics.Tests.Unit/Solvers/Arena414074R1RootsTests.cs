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
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// Range and scale regressions for the unbracketed root solvers (Muller, Newton–Raphson, Secant).
/// Each case has an exactly representable root that the solver must reach: the failure under test is
/// an intermediate quantity leaving the floating-point range, or an absolute threshold applied to a
/// scale-invariant iteration, not a property of the function.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class Arena414074R1RootsTests
{
    // ---- Muller: coefficients built from finite residuals whose differences overflow ----

    [Fact]
    public void Muller_ResidualDifferenceOverflow_ThreePoint_FindsExactAffineRoot()
    {
        // f(x)=1e308·x at seeds -1,1,1.5 has residuals -1e308, 1e308, 1.5e308 (all finite) but
        // f1-f0 = 2e308 overflows. The interpolating line is f itself, so its root 0 is exact.
        var result = new MullerSolver<double>(1e-10).Solve(x => 1e308 * x, -1, 1, 1.5);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(0, 1e-300);
    }

    [Fact]
    public void Muller_ResidualDifferenceOverflow_OnePoint_ReachesTheQuadraticRoot()
    {
        // The one-point overload seeds (19998, 20000, 20002). Residuals -1.54e308, 8.4e307, 4.2e307
        // are finite; f1-f0 = 2.38e308 overflows. f is an exact quadratic, so one Muller step
        // lands on its root 20002.4 (the unit-scale control does exactly that).
        var result = new MullerSolver<double>(1e-10)
            .Solve(x => 3.5e307 * (x - 19999) * (20002.4 - x), 20000.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(20002.4, 1e-9);
    }

    [Fact]
    public void Muller_SlopeDifferenceOverflow_FindsExactQuadraticRoot()
    {
        // f(x)=1e308·(x²-1/4) at -1,0,1: slopes -1e308 and 1e308 are finite, their difference
        // (curvature numerator) 2e308 and the end slope 2e308 are not. Exact roots: ±1/2.
        var result = new MullerSolver<double>(1e-10).Solve(x => 1e308 * (x * x - 0.25), -1, 0, 1);

        result.Converged.Should().BeTrue();
        Math.Abs(result.Root).Should().BeApproximately(0.5, 1e-15);
    }

    // ---- Newton (unbracketed) and Secant: a small nonzero |f'| or |Δf| is a scale, not a zero ----

    [Fact]
    public void NewtonUnbracketed_TinyDerivativeScale_AnalyticDerivative_ReachesRoot()
    {
        // f=1e-40·(x-2): the exact Newton iterate from 0 is 2. The derivative 1e-40 is a scale,
        // not a vanishing slope.
        var result = new NewtonRaphsonSolver<double>(1e-60, 1e-60, 0, 50, x => 1e-40)
            .Solve(x => 1e-40 * (x - 2), 0.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().Be(2);
        result.FinalResidual.Should().Be(0);
    }

    [Fact]
    public void NewtonUnbracketed_TinyDerivativeScale_NumericalDerivative_ReachesRoot()
    {
        var result = new NewtonRaphsonSolver<double>(1e-60, 1e-60, 0, 50)
            .Solve(x => 1e-40 * (x - 2), 0.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(2, 1e-12);
    }

    [Fact]
    public void Secant_TinyResidualScale_TwoPoint_ReachesRoot()
    {
        // Seeds 0 and 1: residual difference 1e-40, exact secant weight -1, exact iterate 2.
        var result = new SecantSolver<double>(1e-60).Solve(x => 1e-40 * (x - 2), 0.0, 1.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().Be(2);
        result.FinalResidual.Should().Be(0);
    }

    [Fact]
    public void Secant_TinyResidualScale_OnePoint_ReachesRoot()
    {
        var result = new SecantSolver<double>(1e-60).Solve(x => 1e-40 * (x - 2), 0.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(2, 1e-12);
    }

    [Fact]
    public void NewtonUnbracketed_ExactlyZeroDerivative_StillFailsExplicitly()
    {
        // The refusal that remains: a derivative that is exactly zero carries no direction.
        var result = new NewtonRaphsonSolver<double>(1e-12, 1e-12, 0, 50, x => 0.0)
            .Solve(x => x - 2, 0.0);

        result.Converged.Should().BeFalse();
        result.Root.Should().Be(0);
        double.IsNaN(result.EstimatedError).Should().BeTrue();
    }

    [Fact]
    public void Secant_ExactlyEqualResiduals_StillFailsExplicitly()
    {
        // f1 == f0 exactly: the secant line is horizontal and has no root to extrapolate to. The
        // explicit refusal reports the seed spacing |p1 - p0| as its error estimate; an
        // unrepresentable extrapolation (the only other refusal) would report NaN.
        var result = new SecantSolver<double>(1e-12).Solve(x => 5.0, 0.0, 1.0);

        result.Converged.Should().BeFalse();
        result.Iterations.Should().Be(1);
        result.Root.Should().Be(1.0);
        result.EstimatedError.Should().Be(1.0);
    }

    // ---- Newton (unbracketed): the step quotient overflows but the next iterate is representable ----

    [Fact]
    public void NewtonUnbracketed_StepQuotientOverflow_AnalyticDerivative_ReachesRepresentableRoot()
    {
        // f=0.5x+0.5e308 from 1e308: f=1e308, f'=0.5, so fx/dfx = 2e308 overflows, yet
        // x - fx/dfx = -1e308 is the exact root.
        var result = new NewtonRaphsonSolver<double>(1e-10, 1e-10, 0, 50, x => 0.5)
            .Solve(x => 0.5 * x + 0.5e308, 1e308);

        result.Converged.Should().BeTrue();
        result.Root.Should().Be(-1e308);
        result.FinalResidual.Should().Be(0);
    }

    [Fact]
    public void NewtonUnbracketed_StepQuotientOverflow_NumericalDerivative_ReachesRepresentableRoot()
    {
        var result = new NewtonRaphsonSolver<double>(1e-10, 1e-10, 0, 50)
            .Solve(x => 0.5 * x + 0.5e308, 1e308);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(-1e308, 1e296);
    }

    [Fact]
    public void NewtonUnbracketed_StepQuotientOverflow_Single_ReachesRepresentableRoot()
    {
        // The same overflow in the narrower working type: float.MaxValue is 3.4e38.
        var result = new NewtonRaphsonSolver<float>(1e-3f, 1e-3f, 0f, 50, x => 0.5f)
            .Solve(x => 0.5f * x + 1.5e38f, 3e38f);

        result.Converged.Should().BeTrue();
        result.Root.Should().Be(-3e38f);
    }

    [Fact]
    public void NewtonUnbracketed_UnrepresentableNextIterate_FailsWithoutEvaluatingNonFinitePoint()
    {
        // f=0.25x+0.9e308 from 1e308: f=1.15e308, fx/dfx = 4.6e308 and the next iterate -3.6e308
        // is itself unrepresentable, so the solve must stop rather than evaluate there.
        var evaluatedNonFinite = false;
        var result = new NewtonRaphsonSolver<double>(1e-10, 1e-10, 0, 50, x => 0.25)
            .Solve(x =>
            {
                evaluatedNonFinite |= !double.IsFinite(x);
                return 0.25 * x + 0.9e308;
            }, 1e308);

        evaluatedNonFinite.Should().BeFalse();
        result.Converged.Should().BeFalse();
        result.Root.Should().Be(1e308);
    }

    // ---- Secant: the step product weight·width overflows but the next iterate is representable ----

    [Fact]
    public void Secant_StepProductOverflow_TwoPoint_ReachesRepresentableRoot()
    {
        // f=0.5x+0.5e308 through 1e308 and 1e308-1e300 is exactly affine: weight·width = 2e308
        // overflows while p1 - weight·width = -1e308 is the exact root.
        var result = new SecantSolver<double>(1e-10)
            .Solve(x => 0.5 * x + 0.5e308, 1e308, 1e308 - 1e300);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(-1e308, 1e296);
        result.FinalResidual.Should().BeApproximately(0, 1e297);
    }

    [Fact]
    public void Secant_StepProductOverflow_OnePoint_ReachesRepresentableRoot()
    {
        var result = new SecantSolver<double>(1e-10).Solve(x => 0.5 * x + 0.5e308, 1e308);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(-1e308, 1e296);
    }

    [Fact]
    public void Secant_UnrepresentableNextIterate_FailsWithoutEvaluatingNonFinitePoint()
    {
        var evaluatedNonFinite = false;
        var result = new SecantSolver<double>(1e-10)
            .Solve(x =>
            {
                evaluatedNonFinite |= !double.IsFinite(x);
                return 0.25 * x + 0.9e308;
            }, 1e308, 1e308 - 1e300);

        evaluatedNonFinite.Should().BeFalse();
        result.Converged.Should().BeFalse();
    }
}
