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

public sealed class NewtonQualificationTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NearZeroSameSignEndpoints_StillRequireARootBracket(bool newton)
    {
        IBracketedRootSolver<double> solver = newton
            ? new NewtonRaphsonSolver<double>(1e-10, 1e-12, 1e-14)
            : new BisectionSolver<double>(1e-10, 1e-12);
        Action solve = () => solver.Solve(x => 1e-12 * (x * x + 1), -1, 1);
        solve.Should().Throw<InvalidOperationException>().WithMessage("*sign change*");
    }

    [Theory]
    [InlineData(10)]
    [InlineData(20)]
    public void BisectionStrictStepBound_IncludesEqualityIteration(int exponent)
    {
        var solver = new BisectionSolver<double>(1e-300, Math.ScaleB(1, -exponent), exponent + 1);
        var result = solver.Solve(x => x - 0.3, 0, 1);
        result.Converged.Should().BeTrue();
        result.Iterations.Should().Be(exponent + 1);
        result.EstimatedError.Should().BeLessThan(Math.ScaleB(1, -exponent));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(-1)]
    public void InfiniteDerivativeAtMidpoint_UsesBracketInsteadOfCertifyingZeroStep(int sign)
    {
        var solver = new NewtonRaphsonSolver<double>(1e-12, 1e-12, 1e-14, derivative:
            x => sign / (3 * Math.Pow(Math.Cbrt(x), 2)));
        var result = solver.Solve(x => sign * (Math.Cbrt(x) + 0.1), -1, 1);
        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(-0.001, 1e-12);
        Math.Abs(result.FinalResidual).Should().BeLessThan(1e-9);
    }

    [Theory]
    [InlineData(1e-31)]
    [InlineData(1e-100)]
    [InlineData(1e-149)]
    public void AffineObjectiveScale_DoesNotDisableNewton(double scale)
    {
        var solver = new NewtonRaphsonSolver<double>(1e-300, 1e-12, 1e-14, derivative: _ => scale);
        var result = solver.Solve(x => scale * (x - 0.3), 0, 1e6);
        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(0.3, 1e-12);
        result.Iterations.Should().BeLessThan(10);
    }

    [Theory]
    [InlineData(1e-7)]
    [InlineData(1e-100)]
    public void NumericalDerivative_RespectsSmallPositiveLogarithmScale(double scale)
    {
        var solver = new NewtonRaphsonSolver<double>(1e-12, 1e-12, 0);
        var result = solver.Solve(x => Math.Log(x / scale), 2 * scale);
        result.Converged.Should().BeTrue();
        (result.Root / scale).Should().BeApproximately(1, 1e-10);
    }
}
