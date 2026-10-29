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

/// <summary>
/// Range regressions for the Levenberg–Marquardt gradient stopping test. Every problem is a
/// set of identical affine residuals <c>s·(θ−c)</c> with Jacobian <c>s</c>, so the exact
/// gradient is <c>m·s²·(θ−c)</c> and the exact optimum is <c>θ = c</c>; the expected values
/// below come from that algebra, not from solver output.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class LevenbergMarquardtGradientRangeTests
{
    private const int TinyResidualCount = 16;
    private const double TinyScale = 1e-162;

    /// <summary>The analytic callback families that reach the gradient routine.</summary>
    public enum Family
    {
        AnalyticArray,
        Writer,
        FiniteDifference,
    }

    public static TheoryData<Family> AnalyticFamilies => new()
    {
        Family.AnalyticArray,
        Family.Writer,
    };

    public static TheoryData<Family> AllFamilies => new()
    {
        Family.AnalyticArray,
        Family.Writer,
        Family.FiniteDifference,
    };

    private static MultivariateSolverResult<double> Solve(
        Family family,
        LevenbergMarquardtSolver<double> solver,
        int residualCount,
        double scale,
        double centre,
        double start,
        double[]? lowerBounds = null)
    {
        switch (family)
        {
            case Family.Writer:
                return solver.Solve(
                    (ReadOnlySpan<double> x, Span<double> r) => r.Fill(scale * (x[0] - centre)),
                    [start],
                    residualCount,
                    (ReadOnlySpan<double> _, Span<double> j) => j.Fill(scale),
                    lowerBounds);
            case Family.FiniteDifference:
                return solver.Solve(
                    x => Enumerable.Repeat(scale * (x[0] - centre), residualCount).ToArray(),
                    [start],
                    null,
                    lowerBounds);
            default:
                return solver.Solve(
                    x => Enumerable.Repeat(scale * (x[0] - centre), residualCount).ToArray(),
                    [start],
                    _ =>
                    {
                        var j = new double[residualCount, 1];
                        for (var i = 0; i < residualCount; i++)
                        {
                            j[i, 0] = scale;
                        }

                        return j;
                    },
                    lowerBounds);
        }
    }

    private static LevenbergMarquardtSolver<double> ZeroGradientToleranceSolver(int maxIterations = 50)
        => new(maxIterations, 1e-14, 1e-14, 0, 1e-3);

    [Theory]
    [MemberData(nameof(AnalyticFamilies))]
    public void UnderflowedGradientAtTheStart_DoesNotSatisfyAZeroGradientTolerance(Family family)
    {
        // Sixteen residuals 1e-162·(θ−2), J = 1e-162, θ₀ = 1.999. The exact gradient there is
        // 16e-324·(−0.001) = −1.6e-326, which is not zero but rounds to zero as an absolute
        // value. gradientTolerance = 0 fires only on an exactly zero gradient, and the exact
        // Gauss–Newton step 1e-3 is representable, so the solver must take it.
        var result = Solve(family, ZeroGradientToleranceSolver(), TinyResidualCount, TinyScale, 2.0, 1.999);

        Assert.InRange(Math.Abs(result.Parameters[0] - 2.0), 0, 1e-9);
    }

    [Theory]
    [MemberData(nameof(AnalyticFamilies))]
    public void UnderflowedGradientAfterAnAcceptedStep_DoesNotSatisfyAZeroGradientTolerance(Family family)
    {
        // From θ₀ = 1 the first damped step lands at 1.999000999… where the exact gradient is
        // −1.6e-326. The post-acceptance gradient test must not certify that point either:
        // the exact optimum θ = 2 is one further representable step away.
        var result = Solve(family, ZeroGradientToleranceSolver(), TinyResidualCount, TinyScale, 2.0, 1.0);

        Assert.InRange(Math.Abs(result.Parameters[0] - 2.0), 0, 1e-9);
    }

    [Theory]
    [MemberData(nameof(AnalyticFamilies))]
    public void ExactlyZeroGradient_StillSatisfiesAZeroGradientTolerance(Family family)
    {
        // θ₀ = 2 is the exact minimum: every residual is exactly zero, so the gradient is
        // exactly zero and the documented exact-stationary-point gate must still fire.
        var result = Solve(family, ZeroGradientToleranceSolver(), TinyResidualCount, TinyScale, 2.0, 2.0);

        Assert.True(result.Converged);
        Assert.Equal(LmTerminationReason.GradientToleranceReached, result.TerminationReason);
        Assert.Equal(0, result.Iterations);
        Assert.Equal(2.0, result.Parameters[0]);
    }

    [Theory]
    [MemberData(nameof(AnalyticFamilies))]
    public void UnderflowedGradientBelowAPositiveTolerance_StillSatisfiesIt(Family family)
    {
        // Same θ₀ = 1.999 start, but gradientTolerance is the smallest positive double. The
        // exact |g| = 1.6e-326 is below it, so the gate must fire immediately.
        var solver = new LevenbergMarquardtSolver<double>(50, 1e-14, 1e-14, double.Epsilon, 1e-3);
        var result = Solve(family, solver, TinyResidualCount, TinyScale, 2.0, 1.999);

        Assert.True(result.Converged);
        Assert.Equal(LmTerminationReason.GradientToleranceReached, result.TerminationReason);
        Assert.Equal(0, result.Iterations);
        Assert.Equal(1.999, result.Parameters[0]);
    }

    [Theory]
    [MemberData(nameof(AnalyticFamilies))]
    public void UnderflowedGradientBlockedByALowerBound_IsExcludedFromTheProjectedNorm(Family family)
    {
        // θ₀ = 2.001 sits on its lower bound 2.001 with a positive exact gradient
        // 16e-324·0.001 = 1.6e-327 that rounds to zero as an absolute value. The coordinate is
        // blocked (descent points out of the box), so the projected gradient is exactly zero:
        // the exact-stationary gate fires with no iteration. The sign of the underflowed
        // gradient must still be honoured for that exclusion.
        var result = Solve(
            family, ZeroGradientToleranceSolver(), TinyResidualCount, TinyScale, 2.0, 2.001, [2.001]);

        Assert.True(result.Converged);
        Assert.Equal(LmTerminationReason.GradientToleranceReached, result.TerminationReason);
        Assert.Equal(0, result.Iterations);
        Assert.Equal(2.001, result.Parameters[0]);
    }

    [Theory]
    [MemberData(nameof(AllFamilies))]
    public void UnrepresentableGradient_DoesNotAbortTheSolve(Family family)
    {
        // Four residuals 1e200·(θ−1), J = 1e200, θ₀ = 0. The exact gradient 4e400·(−1) is beyond
        // the double range, but the residuals, the Jacobian, the Gauss–Newton step 1 and the
        // optimum θ = 1 are all representable, so the solve must run to the optimum.
        var result = Solve(family, ZeroGradientToleranceSolver(), 4, 1e200, 1.0, 0.0);

        Assert.InRange(Math.Abs(result.Parameters[0] - 1.0), 0, 1e-9);
    }

    [Theory]
    [MemberData(nameof(AllFamilies))]
    public void UnrepresentableGradient_NeverSatisfiesAGradientTolerance(Family family)
    {
        // The unrepresentable gradient is larger than every finite tolerance: even the largest
        // finite gradientTolerance must not certify the starting point.
        var solver = new LevenbergMarquardtSolver<double>(50, 1e-14, 1e-14, double.MaxValue, 1e-3);
        var result = Solve(family, solver, 4, 1e200, 1.0, 0.0);

        Assert.NotEqual(0, result.Iterations);
    }

    [Theory]
    [MemberData(nameof(AnalyticFamilies))]
    public void UnrepresentableGradientBlockedByALowerBound_IsExcludedFromTheProjectedNorm(Family family)
    {
        // θ₀ = 3 sits on its lower bound with a positive exact gradient 4e400·2 beyond the
        // range. The coordinate is blocked (descent leaves the box), so the projected gradient
        // is exactly zero and the exact-stationary gate fires at once; the sign of the
        // unrepresentable gradient must still be honoured for that exclusion.
        var result = Solve(family, ZeroGradientToleranceSolver(), 4, 1e200, 1.0, 3.0, [3.0]);

        Assert.True(result.Converged);
        Assert.Equal(LmTerminationReason.GradientToleranceReached, result.TerminationReason);
        Assert.Equal(0, result.Iterations);
        Assert.Equal(3.0, result.Parameters[0]);
    }
}
