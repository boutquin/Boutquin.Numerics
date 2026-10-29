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
/// Range-edge regressions for the Levenberg–Marquardt function- and parameter-tolerance
/// certificates. A trial point that is not representable because the step leaves the double
/// range is rejected and grows the damping; that damping then shrinks every coordinate's step,
/// so the tiny relative reductions it produces are not evidence of stationarity. Expected values
/// come from the decoupled closed forms of each problem, never from solver output.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class LevenbergMarquardtRangeEdgeConvergenceTests
{
    private static LevenbergMarquardtSolver<double> Solver() => new(200, 1e-12, 1e-12, 0, 1e-3);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OptimumBeyondTheRange_DoesNotCertifyAWrongOrdinaryCoordinate(bool writer)
    {
        // Residuals [1e-300·θ₀ − 1e9, θ₁ − 7] from (0, 0). θ₀'s optimum 1e309 is not
        // representable; θ₁ is decoupled with optimum exactly 7. Every full step leaves the
        // range, so the solve cannot reach a stationary point and must not report convergence
        // while θ₁ sits away from 7.
        var result = writer
            ? Solver().Solve(
                (p, r) => { r[0] = 1e-300 * p[0] - 1e9; r[1] = p[1] - 7; }, [0.0, 0.0], 2,
                (_, j) => { j[0] = 1e-300; j[1] = 0; j[2] = 0; j[3] = 1; })
            : Solver().Solve(
                p => [1e-300 * p[0] - 1e9, p[1] - 7], [0.0, 0.0],
                _ => new[,] { { 1e-300, 0.0 }, { 0.0, 1.0 } });

        AssertNotCertified(result, ordinaryIndex: 1, ordinaryOptimum: 7.0);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OptimumBeyondTheRangeWithOverflowingGradient_DoesNotCertifyAWrongOrdinaryCoordinate(bool writer)
    {
        // Residuals [1e-300·θ₀ − 1e9, 1e200·(θ₁ − 1e100), 1e200·(θ₁ − 1e100)] from (0, 0). θ₀ is
        // beyond the range as above; θ₁'s exact gradient (≈ 2e500) overflows but its optimum 1e100
        // is representable. A converged report must place θ₁ at 1e100.
        var result = writer
            ? Solver().Solve(
                (p, r) =>
                {
                    r[0] = 1e-300 * p[0] - 1e9;
                    r[1] = 1e200 * (p[1] - 1e100);
                    r[2] = 1e200 * (p[1] - 1e100);
                },
                [0.0, 0.0], 3,
                (_, j) => { j[0] = 1e-300; j[1] = 0; j[2] = 0; j[3] = 1e200; j[4] = 0; j[5] = 1e200; })
            : Solver().Solve(
                p => [1e-300 * p[0] - 1e9, 1e200 * (p[1] - 1e100), 1e200 * (p[1] - 1e100)], [0.0, 0.0],
                _ => new[,] { { 1e-300, 0.0 }, { 0.0, 1e200 }, { 0.0, 1e200 } });

        AssertNotCertified(result, ordinaryIndex: 1, ordinaryOptimum: 1e100);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UnrepresentableStep_DoesNotCertifyAWrongOrdinaryCoordinate(bool writer)
    {
        // Residuals [1e-320·θ₀ − 1, θ₁ − 7] from (0, 0). θ₀'s Gauss–Newton step (about 1e320) is
        // itself not representable, so the damped solve fails and the damping grows until that
        // step fits the range. The first representable step changes r₀ by a few parts in 1e14
        // and moves θ₁ by a correspondingly tiny amount: both relative reductions are tiny
        // because the range, not the model, sized λ. θ₁'s optimum is exactly 7.
        var result = writer
            ? Solver().Solve(
                (p, r) => { r[0] = 1e-320 * p[0] - 1; r[1] = p[1] - 7; }, [0.0, 0.0], 2,
                (_, j) => { j[0] = 1e-320; j[1] = 0; j[2] = 0; j[3] = 1; })
            : Solver().Solve(
                p => [1e-320 * p[0] - 1, p[1] - 7], [0.0, 0.0],
                _ => new[,] { { 1e-320, 0.0 }, { 0.0, 1.0 } });

        AssertNotCertified(result, ordinaryIndex: 1, ordinaryOptimum: 7.0);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RepresentableOptimumAfterARangeRejection_StillConverges(bool writer)
    {
        // Residuals [1e-300·θ − 5e7, 3.3e-300·θ − 1.2e8] from θ₀ = −1.5e308. The first
        // Gauss–Newton step (about 5.3e308) is not representable, so it is rejected, but the
        // least-squares optimum θ* = (1·5e7 + 3.3·1.2e8)/(1 + 3.3²)·1e300 ≈ 3.751e307 is. The two
        // residuals are inconsistent, so the gradient at θ* is not exactly zero and a zero
        // gradientTolerance cannot stop the solve: only the FTOL/XTOL certificates can, once the
        // damping has returned to its model-driven level.
        var result = writer
            ? Solver().Solve(
                (p, r) => { r[0] = 1e-300 * p[0] - 5e7; r[1] = 3.3e-300 * p[0] - 1.2e8; }, [-1.5e308], 2,
                (_, j) => { j[0] = 1e-300; j[1] = 3.3e-300; })
            : Solver().Solve(
                p => [1e-300 * p[0] - 5e7, 3.3e-300 * p[0] - 1.2e8], [-1.5e308],
                _ => new[,] { { 1e-300 }, { 3.3e-300 } });
        var optimum = (5e7 + 3.3 * 1.2e8) / (1 + 3.3 * 3.3) * 1e300;

        result.Converged.Should().BeTrue();
        result.Parameters[0].Should().BeApproximately(optimum, 1e-9 * optimum);
    }

    private static void AssertNotCertified(MultivariateSolverResult<double> result, int ordinaryIndex, double ordinaryOptimum)
    {
        var ordinaryIsAtItsOptimum =
            Math.Abs(result.Parameters[ordinaryIndex] - ordinaryOptimum) <= 1e-6 * Math.Abs(ordinaryOptimum);
        if (!ordinaryIsAtItsOptimum)
        {
            result.Converged.Should().BeFalse(
                $"θ{ordinaryIndex} = {result.Parameters[ordinaryIndex]:R} is not its optimum {ordinaryOptimum:R}, "
                + $"yet the solve reported {result.TerminationReason}");
        }

        result.TerminationReason.Should().NotBe(LmTerminationReason.FunctionToleranceReached);
        result.TerminationReason.Should().NotBe(LmTerminationReason.ParameterToleranceReached);
    }
}
