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

[Trait("Tier", "Unit")]
public sealed class Arena20260928LmTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void OneSidedDifference_BoundaryStepBalancesFirstOrderError(bool writer, bool upper)
    {
        var sign = upper ? -1.0 : 1.0;
        var solver = new LevenbergMarquardtSolver<double>(1, 1e-14, 1e-14, 1e-14, 1);
        double[] Residual(double[] x) => [sign * x[0] + x[0] * x[0] - 1];
        void WriteResidual(ReadOnlySpan<double> x, Span<double> r) => r[0] = sign * x[0] + x[0] * x[0] - 1;
        double[] lower = [upper ? -2 : 0];
        double[] higher = [upper ? 0 : 2];
        var result = writer
            ? solver.Solve(WriteResidual, [0], 1, lowerBounds: lower, upperBounds: higher)
            : solver.Solve(Residual, [0], lowerBounds: lower, upperBounds: higher);

        // At the boundary r=-1, J=sign and lambda=1 imply h=sign/2.
        // A one-sided derivative has O(h), not O(h²), truncation error.
        // Balancing epsilon/h with h gives sqrt(epsilon), NR §5.7.
        result.Parameters[0].Should().BeApproximately(sign * 0.5, 1e-8);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RejectedRoundedCostStep_MeetsIndependentFunctionTolerance(bool writer)
    {
        var solver = new LevenbergMarquardtSolver<double>(1, 1e-12, 1e-14, 1e-14, 1);
        double[] Residual(double[] x) => [1, x[0]];
        double[,] Jacobian(double[] _) => new[,] { { 0.0 }, { 1.0 } };
        void WriteResidual(ReadOnlySpan<double> x, Span<double> r) { r[0] = 1; r[1] = x[0]; }
        void WriteJacobian(ReadOnlySpan<double> x, Span<double> j) { j[0] = 0; j[1] = 1; }
        var result = writer
            ? solver.Solve(WriteResidual, [1e-8], 2, WriteJacobian)
            : solver.Solve(Residual, [1e-8], Jacobian);

        // Both costs round to .5; true model decrease is 3.75e-17.
        // MINPACK lmder lines398-434 tests ftol even after an unsuccessful iteration.
        result.Converged.Should().BeTrue();
        result.TerminationReason.Should().Be(LmTerminationReason.FunctionToleranceReached);
        result.Parameters[0].Should().Be(1e-8, "an unaccepted trial does not replace the current point");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CentralDifference_UsesRepresentedParameterDisplacement(bool writer)
    {
        var solver = new LevenbergMarquardtSolver<double>(1, 1e-14, 1e-14, 1e-14, 1);
        double[] Residual(double[] x) => [x[0] - 1.3];
        void WriteResidual(ReadOnlySpan<double> x, Span<double> r) => r[0] = x[0] - 1.3;
        var result = writer ? solver.Solve(WriteResidual, [1.1], 1) : solver.Solve(Residual, [1.1]);

        // The linear derivative is exactly one when using the represented spacing.
        result.Parameters[0].Should().BeApproximately(1.2, 2e-14);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CentralDifference_ExponentialStepMatchesAnalyticDerivative(bool writer)
    {
        var solver = new LevenbergMarquardtSolver<double>(1, 1e-14, 1e-14, 1e-14, 1);
        double[] Residual(double[] x) => [Math.Exp(x[0]) - 2];
        void WriteResidual(ReadOnlySpan<double> x, Span<double> r) => r[0] = Math.Exp(x[0]) - 2;
        var result = writer ? solver.Solve(WriteResidual, [0], 1) : solver.Solve(Residual, [0]);

        // At zero, r=-1 and J=1. Marquardt damping lambda=1 gives h=1/2.
        // NR 3e eqs.5.7.8-9 predict central-difference error O(epsilon^(2/3)).
        result.Parameters[0].Should().BeApproximately(0.5, 1e-10);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LegacyNullInitialGuess_PreservesArgumentNullContract(bool pooled)
    {
        var solver = new LevenbergMarquardtSolver();
        Action call = pooled
            ? () => solver.Solve(x => [x[0] - 1], null!, new LevenbergMarquardtBuffers(1, 1))
            : () => solver.Solve(x => [x[0] - 1], null!);

        call.Should().ThrowExactly<ArgumentNullException>().WithParameterName("initialGuess");
    }

    // Madsen, Nielsen and Tingleff (2004), p.25: L(0)-L(h)=-r'Jh-|Jh|²/2.
    // At the clipped h=(.1,.5), actual reduction=.53055 and model=.555.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ClippedStep_DoesNotMeetFunctionToleranceWhenModelExceedsIt(bool writer)
    {
        var solver = new LevenbergMarquardtSolver<double>(1, 0.54, 1e-12, 1e-12, 1);
        static double[] Residual(double[] x) => [-1 + 2 * x[0] - 3 * x[0] * x[0], x[1] - 1];
        static double[,] Jacobian(double[] x) => new[,] { { 2 - 6 * x[0], 0.0 }, { 0.0, 1.0 } };
        static void WriteResidual(ReadOnlySpan<double> x, Span<double> r)
        {
            r[0] = -1 + 2 * x[0] - 3 * x[0] * x[0];
            r[1] = x[1] - 1;
        }

        static void WriteJacobian(ReadOnlySpan<double> x, Span<double> j)
        {
            j[0] = 2 - 6 * x[0];
            j[1] = 0;
            j[2] = 0;
            j[3] = 1;
        }

        var result = writer
            ? solver.Solve(WriteResidual, [0, 0], 2, WriteJacobian, upperBounds: [0.1, 100])
            : solver.Solve(Residual, [0, 0], Jacobian, upperBounds: [0.1, 100]);

        result.Parameters[0].Should().Be(0.1);
        result.Parameters[1].Should().BeApproximately(0.5, 1e-15);
        result.Converged.Should().BeFalse("the projected quadratic model reduction .555 exceeds ftol .54");
    }
}
