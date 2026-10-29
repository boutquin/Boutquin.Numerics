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
/// Muller's parabola is built in spacing-normalized coordinates, but that normalization must never
/// cost accuracy the unnormalized form had: when the two spacings differ by more than the type's
/// dynamic range, or the normalized slopes overflow, the first quadratic step must be the one the
/// plain divided-difference formulas give.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class MullerSolverSpacingRatioTests
{
    [Theory]
    [InlineData(0.0, 1e-300, 1e30)]
    [InlineData(0.0, 1e-320, 1e10)]
    [InlineData(0.0, 1e-310, 1e300)]
    [InlineData(1e30, 1e-300, 0.0)]
    public void LinearResidual_WithSpacingsBeyondOneDynamicRange_StillSolvesInTheSecantStep(double x0, double x1, double x2)
    {
        var result = new MullerSolver<double>(1e-12, 100).Solve(x => x - 5, x0, x1, x2);

        result.Converged.Should().BeTrue("the unscaled divided differences of a linear residual are exactly 1");
        result.Root.Should().BeApproximately(5.0, 1e-9);
    }

    [Theory]
    [InlineData(0.0, 1e-300, 1e30)]
    [InlineData(0.0, 1e-320, 1e10)]
    public void FirstTrialPoint_WhenTheNormalizedSpacingWouldFlush_IsTheUnnormalizedQuadraticStep(double x0, double x1, double x2)
    {
        static double F(double x) => x - 5;
        var evaluated = new List<double>();

        _ = new MullerSolver<double>(1e-12, 100).Solve(
            x =>
            {
                evaluated.Add(x);
                return F(x);
            },
            x0,
            x1,
            x2);

        evaluated.Should().HaveCountGreaterThan(3, "a finite quadratic step must be taken from the in-range unscaled parabola");
        evaluated[3].Should().Be(UnscaledMullerStep(x0, x1, x2, F), "the unscaled form is in range, so it is the contract");
    }

    [Fact]
    public void FirstTrialPoint_WhenNormalizedSlopesOverflowButPlainOnesDoNot_IsTheUnnormalizedQuadraticStep()
    {
        // Spacings 1 and 2^1000: dividing both by 2^1000 turns the first slope 1e9 into 1e9 * 2^1000, which
        // overflows, while the plain divided differences (1e9 and about 8e-292) are comfortably in range.
        static double F(double x) => (1e9 * Math.Min(x, 10)) - 5e9;
        var far = Math.ScaleB(1.0, 1000);
        var evaluated = new List<double>();

        _ = new MullerSolver<double>(1e-12, 100).Solve(
            x =>
            {
                evaluated.Add(x);
                return F(x);
            },
            0.0,
            1.0,
            far);

        evaluated[3].Should().Be(UnscaledMullerStep(0.0, 1.0, far, F));
    }

    // The first Muller trial point from the plain divided differences, with no spacing normalization.
    private static double UnscaledMullerStep(double x0, double x1, double x2, Func<double, double> f)
    {
        var f0 = f(x0);
        var f1 = f(x1);
        var f2 = f(x2);
        var h0 = x1 - x0;
        var h1 = x2 - x1;
        var delta0 = (f1 - f0) / h0;
        var delta1 = (f2 - f1) / h1;
        var a = (delta1 - delta0) / (h1 + h0);
        var b = (a * h1) + delta1;
        var sqrtDisc = Math.Sqrt(Math.Abs((b * b) - (4 * a * f2)));
        var plus = b + sqrtDisc;
        var minus = b - sqrtDisc;
        var denom = Math.Abs(plus) >= Math.Abs(minus) ? plus : minus;
        return x2 - (2 * f2 / denom);
    }
}
