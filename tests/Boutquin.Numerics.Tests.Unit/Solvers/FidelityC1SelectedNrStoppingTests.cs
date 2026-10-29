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
public sealed class FidelityC1SelectedNrStoppingTests
{
    [Fact]
    public void NewtonSmallComputedStepStopsDespiteLargeResidual()
    {
        var result = new NewtonRaphsonSolver<double>(1e-20, 1e-20, 0.1, 1, x => 2e8 * x)
            .Solve(x => 1e8 * (x * x - 2), 1, 2);
        // At midpoint3/2, exact Newton dx=1/12<.1 and new x=17/12.
        Assert.True(result.Converged);
        Assert.InRange(Math.Abs(result.Root - 17.0 / 12), 0, 1e-15);
        Assert.True(Math.Abs(result.FinalResidual) > 1e5);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NewtonComputedStepBoundaryIsStrict(bool aboveBoundary)
    {
        var result = new NewtonRaphsonSolver<double>(1e-20, 1e-20,
            aboveBoundary ? Math.BitIncrement(0.5) : 0.5, 1, x => 2 * x)
            .Solve(x => x * x - 2, 0, 4);
        Assert.Equal(1.5, result.Root);
        Assert.Equal(aboveBoundary, result.Converged);
        Assert.Equal(0.25, result.FinalResidual);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(-1)]
    public void NewtonEndpointEqualityAloneDoesNotForceBisection(double sign)
    {
        // Monotone cubic f=x^3-3x^2+4x-1 has f0=-1,f2=3,f1=f'1=1.
        // Its first Newton candidate is endpoint0, while slow-decrease is equality.
        var result = new NewtonRaphsonSolver<double>(1e-20, 1e-20, 0, 1,
            x => sign * (3 * x * x - 6 * x + 4))
            .Solve(x => sign * (x * x * x - 3 * x * x + 4 * x - 1), 0, 2);
        Assert.Equal(0, result.Root);
        Assert.False(result.Converged);
        Assert.Equal(-sign, result.FinalResidual);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NewtonListingStagnationReturnsBeforeAnotherEvaluation(bool bisect)
    {
        var upper = Math.BitIncrement(1.0);
        var spacing = upper - 1;
        var calls = 0;
        double Objective(double x)
        {
            calls++;
            var u = (x - 1) / spacing;
            return bisect ? u * u * u - 0.125 : u - 0.5;
        }
        double Derivative(double x)
        {
            var u = (x - 1) / spacing;
            return bisect ? 3 * u * u / spacing : 1 / spacing;
        }
        var result = new NewtonRaphsonSolver<double>(1e-30, 1e-30, 0, 1, Derivative)
            .Solve(Objective, 1, upper);
        Assert.True(result.Converged);
        Assert.Equal(1, result.Root);
        Assert.Equal(3, calls);
        Assert.Equal(bisect ? -0.125 : -0.5, result.FinalResidual);
    }

    [Fact]
    public void SecantSmallStepDoesNotRequireResidualSignCrossing()
    {
        var result = new SecantSolver<double>(1, 1).Solve(x => 1e6 * (x * x - 2), 1, 2);
        // Smaller-residual seed1 is current; next4/3 has same negative residual sign.
        Assert.True(result.Converged);
        Assert.InRange(Math.Abs(result.Root - 4.0 / 3), 0, 1e-15);
        Assert.True(result.FinalResidual < -1e5);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SecantComputedStepBoundaryIsStrict(bool aboveBoundary)
    {
        var result = new SecantSolver<double>(aboveBoundary ? Math.BitIncrement(1.0) : 1, 1)
            .Solve(x => 100 * (x * x - 2), 0, 2);
        Assert.Equal(1, result.Root);
        Assert.Equal(aboveBoundary, result.Converged);
        Assert.Equal(-100, result.FinalResidual);
    }

    [Theory]
    [InlineData(1, false)]
    [InlineData(1, true)]
    [InlineData(-1, false)]
    [InlineData(-1, true)]
    public void BisectionHalvedStepBoundaryIsStrictAndReturnsNonpositiveSide(double sign, bool aboveBoundary)
    {
        var result = new BisectionSolver<double>(1e-20, aboveBoundary ? Math.BitIncrement(1.0) : 1, 1)
            .Solve(x => sign * 100 * (x * x - 2), 0, 2);
        Assert.Equal(aboveBoundary, result.Converged);
        Assert.Equal(aboveBoundary ? (sign > 0 ? 1 : 2) : 1.5, result.Root);
        Assert.Equal(aboveBoundary ? (sign > 0 ? -100 : -200) : sign * 25, result.FinalResidual);
        Assert.Equal(aboveBoundary ? 1 : 0.5, result.EstimatedError);
    }
}
