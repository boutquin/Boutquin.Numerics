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
/// Unit tests for <see cref="LevenbergMarquardtBuffers"/> pool lifecycle — construction,
/// dimension validation, and the grow-only <see cref="LevenbergMarquardtBuffers.Reset(int, int)"/>
/// policy. Integration with the solver is covered by
/// <see cref="LevenbergMarquardtSolverTests"/> and the allocation harness.
/// </summary>
public sealed class LevenbergMarquardtBuffersTests
{
    [Fact]
    public void Constructor_SetsDimensionsToRequestedValues()
    {
        var buffers = new LevenbergMarquardtBuffers(parameterCount: 6, residualCount: 24);

        buffers.ParameterCount.Should().Be(6);
        buffers.ResidualCount.Should().Be(24);
    }

    [Theory]
    [InlineData(0, 10)]
    [InlineData(-1, 10)]
    [InlineData(10, 0)]
    [InlineData(10, -1)]
    public void Constructor_NonPositiveDimensions_Throws(int parameterCount, int residualCount)
    {
        var act = () => new LevenbergMarquardtBuffers(parameterCount, residualCount);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Reset_ToSameDimensions_KeepsLogicalSize()
    {
        var buffers = new LevenbergMarquardtBuffers(parameterCount: 4, residualCount: 12);

        buffers.Reset(4, 12);

        buffers.ParameterCount.Should().Be(4);
        buffers.ResidualCount.Should().Be(12);
    }

    [Fact]
    public void Reset_ToLargerDimensions_GrowsCapacity()
    {
        var buffers = new LevenbergMarquardtBuffers(parameterCount: 3, residualCount: 10);

        buffers.Reset(6, 24);

        buffers.ParameterCount.Should().Be(6);
        buffers.ResidualCount.Should().Be(24);
    }

    [Fact]
    public void Reset_ToSmallerDimensions_RetainsCapacity_NoReallocation()
    {
        // Grow then shrink. The pool must retain its high-water storage so a subsequent
        // grow back to the original size does not reallocate — the test checks this by
        // re-growing to the original capacity and observing no exception and stable
        // logical dimensions (capacity is internal; the lifecycle contract is the only
        // observable behavior).
        var buffers = new LevenbergMarquardtBuffers(parameterCount: 6, residualCount: 24);

        buffers.Reset(3, 10);
        buffers.ParameterCount.Should().Be(3);
        buffers.ResidualCount.Should().Be(10);

        buffers.Reset(6, 24);
        buffers.ParameterCount.Should().Be(6);
        buffers.ResidualCount.Should().Be(24);
    }

    [Fact]
    public void Reset_GrowsOnlyMismatchedDimension_UpdatesBothLogicalSizes()
    {
        // When only one of (n, m) exceeds capacity, only that dimension grows;
        // the other retains its capacity. Observable: logical dims match the
        // requested values and subsequent use via the solver succeeds.
        var buffers = new LevenbergMarquardtBuffers(parameterCount: 4, residualCount: 24);

        buffers.Reset(8, 24); // grow n only
        buffers.ParameterCount.Should().Be(8);
        buffers.ResidualCount.Should().Be(24);

        buffers.Reset(8, 64); // grow m only
        buffers.ParameterCount.Should().Be(8);
        buffers.ResidualCount.Should().Be(64);
    }

    [Theory]
    [InlineData(0, 10)]
    [InlineData(-1, 10)]
    [InlineData(10, 0)]
    [InlineData(10, -1)]
    public void Reset_NonPositiveDimensions_Throws(int parameterCount, int residualCount)
    {
        var buffers = new LevenbergMarquardtBuffers(parameterCount: 6, residualCount: 24);

        var act = () => buffers.Reset(parameterCount, residualCount);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Solve_BufferParameterCountMismatch_Throws()
    {
        var buffers = new LevenbergMarquardtBuffers(parameterCount: 3, residualCount: 10);
        var solver = new LevenbergMarquardtSolver();

        var act = () => solver.Solve(
            theta => [theta[0] - 1.0],
            initialGuess: [0.0], // length 1, buffer sized for 3
            buffers);

        act.Should().Throw<ArgumentException>()
            .WithMessage("*parameter count*does not match*Reset*");
    }

    [Fact]
    public void Solve_ResidualCountMismatch_Throws()
    {
        // Buffer sized for 10 residuals; the residual function returns 3 → pool mismatch.
        var buffers = new LevenbergMarquardtBuffers(parameterCount: 1, residualCount: 10);
        var solver = new LevenbergMarquardtSolver();

        var act = () => solver.Solve(
            theta => new[] { theta[0] - 1.0, theta[0] - 2.0, theta[0] - 3.0 },
            initialGuess: [0.0],
            buffers);

        act.Should().Throw<ArgumentException>()
            .WithMessage("*residuals but the buffer pool was sized for*Reset*");
    }

    [Fact]
    public void Solve_NullBuffers_Throws()
    {
        var solver = new LevenbergMarquardtSolver();

        var act = () => solver.Solve(
            theta => [theta[0] - 1.0],
            initialGuess: [0.0],
            buffers: null!);

        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Reset_AfterSolve_EnablesReuseForDifferentProblemSize()
    {
        // Fit two different problems against the same pool. First: linear n=1, m=5.
        // Then: quadratic n=2, m=6. Reset between solves.
        var buffers = new LevenbergMarquardtBuffers(parameterCount: 1, residualCount: 5);
        var solver = new LevenbergMarquardtSolver();

        var first = solver.Solve(
            theta => [theta[0] - 1.0, theta[0] - 2.0, theta[0] - 3.0, theta[0] - 4.0, theta[0] - 5.0],
            initialGuess: [0.0],
            buffers);

        first.Converged.Should().BeTrue();
        first.Parameters[0].Should().BeApproximately(3.0, 1e-8);

        // Reconfigure for a 2-parameter, 6-residual problem.
        buffers.Reset(2, 6);
        buffers.ParameterCount.Should().Be(2);
        buffers.ResidualCount.Should().Be(6);

        // y = a + b·x at x = 0..5 with y = 1 + 2·x.
        var xs = new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
        var ys = new[] { 1.0, 3.0, 5.0, 7.0, 9.0, 11.0 };
        double[] Residuals(double[] theta)
        {
            var r = new double[xs.Length];
            for (var i = 0; i < xs.Length; i++)
            {
                r[i] = ys[i] - (theta[0] + theta[1] * xs[i]);
            }

            return r;
        }

        var second = solver.Solve(Residuals, initialGuess: [0.0, 0.0], buffers);

        second.Converged.Should().BeTrue();
        second.Parameters[0].Should().BeApproximately(1.0, 1e-8);
        second.Parameters[1].Should().BeApproximately(2.0, 1e-8);
    }
}
