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
/// Muller's interpolating parabola is invariant under an affine change of abscissa, so a problem
/// written in a rescaled variable <c>x = s * u</c> must behave exactly like the unit-scale problem.
/// These tests pin that invariance where the curvature divided difference would otherwise overflow
/// (small spacing) or underflow (large spacing).
/// </summary>
[Trait("Tier", "Unit")]
public sealed class MullerSolverScaledAbscissaTests
{
    private const double Tolerance = 1e-10;

    [Theory]
    [InlineData(-532)] // 2^-532, about 7e-161: the curvature of Δf/h² exceeds double.MaxValue
    [InlineData(-664)] // about 1e-200
    [InlineData(-996)] // about 1e-300
    public void ThreePointQuadratic_AtSmallAbscissaScale_ConvergesToScaledRoot(int exponent)
    {
        var scale = Math.ScaleB(1.0, exponent);

        var result = new MullerSolver<double>(Tolerance)
            .Solve(x => (x / scale) * (x / scale) - 2, scale, 1.5 * scale, 2 * scale);

        result.Converged.Should().BeTrue("a quadratic is recovered exactly by the first Muller parabola");
        (result.Root / scale).Should().BeApproximately(Math.Sqrt(2), 1e-12);
        result.Iterations.Should().BeLessThanOrEqualTo(2);
    }

    [Theory]
    [InlineData(200)]
    [InlineData(300)]
    public void ThreePointQuadratic_AtLargeAbscissaScale_ConvergesToScaledRoot(int decimalExponent)
    {
        var scale = Math.Pow(10, decimalExponent);

        var result = new MullerSolver<double>(Tolerance)
            .Solve(x => (x / scale) * (x / scale) - 2, scale, 1.5 * scale, 2 * scale);

        result.Converged.Should().BeTrue("the curvature Δf/h² underflows at this scale but the parabola is exact");
        (result.Root / scale).Should().BeApproximately(Math.Sqrt(2), 1e-12);
        result.Iterations.Should().BeLessThanOrEqualTo(2);
    }

    [Fact]
    public void OnePointQuadratic_AtLargeAbscissaScale_ConvergesToScaledRoot()
    {
        const double Scale = 1e300;

        var result = new MullerSolver<double>(Tolerance)
            .Solve(x => (x / Scale) * (x / Scale) - 2, 1.3 * Scale);

        result.Converged.Should().BeTrue();

        // The one-point entry seeds a 1e-4-relative stencil, so rounding in the residual differences
        // limits the root to the residual tolerance (|f| < 1e-10 allows about 3.5e-11), not to 1e-12.
        (result.Root / Scale).Should().BeApproximately(Math.Sqrt(2), 1e-9);
    }

    [Fact]
    public void CoefficientNormalization_DoesNotFlushRepresentableCurvature()
    {
        // f(x) = (x/2)(x/1e308) - 0.5e308 has its root at 1e308 and an exact curvature of 5e-309 (subnormal).
        var result = new MullerSolver<double>(Tolerance)
            .Solve(x => (x / 2) * (x / 1e308) - 0.5e308, 0, 1.5e308, 1.5e308 / 2.25);

        result.Converged.Should().BeTrue();
        (result.Root / 1e308).Should().BeApproximately(1.0, 1e-12);
    }

    [Fact]
    public void WideSpacing_WhoseSumOverflows_StillRecoversTheParabola()
    {
        var result = new MullerSolver<double>(Tolerance)
            .Solve(x => (x / 2) * (x / 1e308) - 0.5e308, -1.5e308, 0, 1.5e308);

        result.Converged.Should().BeTrue();
        Math.Abs(result.Root / 1e308).Should().BeApproximately(1.0, 1e-12);
    }

    [Fact]
    public void FirstSpacingBeyondTheTypeRange_KeepsTheUnscaledSecantStep()
    {
        // x1 - x0 overflows to +infinity, so the spacings have no usable binary exponent. The parabola
        // is then built unscaled (its curvature term is exactly zero) and reduces to a secant step
        // through the last two points, which is exact for this linear residual.
        var result = new MullerSolver<double>(Tolerance, 60)
            .Solve(x => (x - 1e300) * 1e-155, -1.5e308, 1.5e308, 1.45e308);

        result.Converged.Should().BeTrue();
        (result.Root / 1e300).Should().BeApproximately(1.0, 1e-9);
    }

    [Theory]
    [InlineData(-1000)]
    [InlineData(-532)]
    [InlineData(-100)]
    [InlineData(-3)]
    [InlineData(5)]
    [InlineData(200)]
    [InlineData(900)]
    public void PowerOfTwoAbscissaScale_ReproducesTheUnitScaleRunExactly(int exponent)
    {
        // Cubic with a real root, written in u = x / s. Power-of-two scaling is exact, so the whole
        // iteration (iterates, count, convergence flag) must match the unit-scale run bit for bit.
        static double Cubic(double u) => u * u * u - 2 * u * u + 0.5 * u - 3;
        var scale = Math.ScaleB(1.0, exponent);

        var unit = new MullerSolver<double>(1e-13, 40).Solve(Cubic, 1, 2, 3);
        var scaled = new MullerSolver<double>(1e-13, 40)
            .Solve(x => Cubic(x / scale), scale, 2 * scale, 3 * scale);

        scaled.Converged.Should().Be(unit.Converged);
        scaled.Iterations.Should().Be(unit.Iterations);
        (scaled.Root / scale).Should().Be(unit.Root);
        scaled.FinalResidual.Should().Be(unit.FinalResidual);
    }

    [Fact]
    public void OrdinaryUnitScaleProblem_KeepsItsConvergence()
    {
        var result = new MullerSolver<double>(1e-12).Solve(x => x * x - 2, 1, 1.5, 2);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(Math.Sqrt(2), 1e-12);
        result.Iterations.Should().BeLessThanOrEqualTo(2);
    }

    [Fact]
    public void QuadraticWithoutRealRoot_NeverReportsConvergence_AtSmallScale()
    {
        var scale = Math.ScaleB(1.0, -996);

        var result = new MullerSolver<double>(Tolerance, 30)
            .Solve(x => (x / scale) * (x / scale) + 1, scale, 1.5 * scale, 2 * scale);

        result.Converged.Should().BeFalse("x^2 + 1 has no real root, so no iterate can satisfy either stopping rule");
    }
}
