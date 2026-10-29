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

namespace Boutquin.Numerics.Tests.Unit.GenericParity;

/// <summary>
/// Elementary problems with independently known roots, minima, and regression coefficients.
/// Arithmetic trajectories and historical result bits are not part of these checks.
/// </summary>
public sealed class Solvers_ParityTests
{
    [Fact]
    public void BrentSolver_RecoversSquareRootOfTwo()
    {
        var result = new BrentSolver<double>(tolerance: 1e-12, maxIterations: 100)
            .Solve(x => x * x - 2.0, 0.0, 2.0);
        AssertRoot(result, Math.Sqrt(2));
    }

    [Fact]
    public void BisectionSolver_RecoversUnitCubicRoot()
    {
        var result = new BisectionSolver<double>(functionTolerance: 1e-12, bracketTolerance: 1e-12, maxIterations: 200)
            .Solve(x => x * x * x - 1.0, 0.0, 2.0);
        AssertRoot(result, 1);
    }

    [Fact]
    public void SecantSolver_RecoversLogarithmOfThree()
    {
        var result = new SecantSolver<double>(tolerance: 1e-12, maxIterations: 50)
            .Solve(x => Math.Exp(x) - 3.0, 1.0);
        AssertRoot(result, Math.Log(3));
    }

    [Fact]
    public void MullerSolver_RecoversSquareRootOfFive()
    {
        var result = new MullerSolver<double>(tolerance: 1e-12, maxIterations: 50)
            .Solve(x => x * x - 5.0, 2.0);
        AssertRoot(result, Math.Sqrt(5));
    }

    [Fact]
    public void NewtonRaphsonSolver_Bracketed_RecoversSquareRootOfSeven()
    {
        var result = new NewtonRaphsonSolver<double>(functionTolerance: 1e-12, bracketTolerance: 1e-12, stepTolerance: 0.0)
            .Solve(x => x * x - 7.0, 1.0, 4.0);
        AssertRoot(result, Math.Sqrt(7));
    }

    [Fact]
    public void NewtonRaphsonSolver_Unbracketed_RecoversPi()
    {
        var result = new NewtonRaphsonSolver<double>(functionTolerance: 1e-12, bracketTolerance: 1e-12,
            stepTolerance: 0.0, derivative: Math.Cos).Solve(Math.Sin, 3.0);
        AssertRoot(result, Math.PI);
    }

    [Fact]
    public void LevenbergMarquardtSolver_RecoversRosenbrockMinimum()
    {
        static double[] Residuals(double[] p) => [10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]];
        var result = new LevenbergMarquardtSolver<double>(200, 1e-10, 1e-10, 1e-10, 1e-3)
            .Solve(Residuals, [-1.2, 1.0]);
        result.Parameters.Should().HaveCount(2);
        // Rosenbrock residuals vanish uniquely at (1,1); the corrected central
        // stencil intentionally supersedes the historical finite-difference bits.
        result.Parameters[0].Should().BeApproximately(1, 1e-10);
        result.Parameters[1].Should().BeApproximately(1, 1e-10);
        result.FinalCost.Should().BeLessThan(1e-20);
        result.Converged.Should().BeTrue();
    }

    [Fact]
    public void OrdinaryLeastSquares_RecoversAffineModel()
    {
        var x = new double[5, 1] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 } };
        var y = new double[] { 3.0, 5.0, 7.0, 9.0, 11.0 };
        var result = OrdinaryLeastSquares<double>.Fit(x, y);
        result.Coefficients.Should().HaveCount(2);
        result.Coefficients[0].Should().BeApproximately(1, 1e-12);
        result.Coefficients[1].Should().BeApproximately(2, 1e-12);
        result.ResidualSumOfSquares.Should().BeInRange(0, 1e-24);
        result.RSquared.Should().BeApproximately(1, 1e-12);
        result.DegreesOfFreedom.Should().Be(3);
    }

    private static void AssertRoot(RootSolverResult<double> result, double expected)
    {
        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(expected, 1e-10);
        Math.Abs(result.FinalResidual).Should().BeLessThanOrEqualTo(1e-10);
        result.EstimatedError.Should().BeGreaterThanOrEqualTo(0);
    }
}
