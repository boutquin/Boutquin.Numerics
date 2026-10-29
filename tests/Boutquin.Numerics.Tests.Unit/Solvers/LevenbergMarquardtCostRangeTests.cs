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

public sealed class LevenbergMarquardtCostRangeTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TinyJacobian_PreservesFiniteCovariance(bool writer)
    {
        var result = SolveStationary(writer, 1e-160, 1e-160);

        result.Converged.Should().BeTrue();
        result.ParameterCovariance.Should().NotBeNull();
        result.ParameterCovariance![0, 0].Should().BeApproximately(1, 1e-4);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TinyJacobian_ExactFitHasZeroCovariance(bool writer)
    {
        var result = SolveStationary(writer, 0, 1e-160);

        result.Converged.Should().BeTrue();
        result.ParameterCovariance.Should().NotBeNull();
        result.ParameterCovariance![0, 0].Should().Be(0);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UnrepresentableCovariance_IsUnavailable(bool writer)
    {
        var result = SolveStationary(writer, 9e153, 0.1);

        result.Converged.Should().BeTrue();
        double.IsFinite(result.FinalCost).Should().BeTrue();
        result.ParameterCovariance.Should().BeNull();
    }

    private static MultivariateSolverResult<double> SolveStationary(bool writer, double residual, double jacobian)
    {
        var solver = new LevenbergMarquardtSolver<double>(10, 1e-10, 1e-10, 0, 1e-3);
        return writer
            ? solver.Solve((ReadOnlySpan<double> theta, Span<double> values) =>
                {
                    values[0] = (jacobian * theta[0]) - residual;
                    values[1] = (jacobian * theta[0]) + residual;
                }, [0.0], 2, (ReadOnlySpan<double> _, Span<double> values) =>
                {
                    values[0] = jacobian;
                    values[1] = jacobian;
                })
            : solver.Solve(theta => [(jacobian * theta[0]) - residual, (jacobian * theta[0]) + residual],
                [0.0], _ => new double[,] { { jacobian }, { jacobian } });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void StationarySingleResidual_PreservesRepresentableHalfSquare(bool writer)
    {
        const double residual = 1.5e154;
        var solver = new LevenbergMarquardtSolver<double>(10, 1e-10, 1e-10, 0, 1e-3);
        var result = writer
            ? solver.Solve((ReadOnlySpan<double> _, Span<double> values) => values[0] = residual,
                [0.0], 1, (ReadOnlySpan<double> _, Span<double> values) => values[0] = 0)
            : solver.Solve(_ => [residual], [0.0], _ => new double[,] { { 0.0 } });

        result.Converged.Should().BeTrue();
        result.FinalCost.Should().BeApproximately((residual / 2) * residual, 1e293);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BalancedLargeResiduals_PreserveRepresentableCovariance(bool writer)
    {
        const double scale = 1e154;
        var solver = new LevenbergMarquardtSolver<double>(10, 1e-10, 1e-10, 0, 1e-3);
        var result = writer
            ? solver.Solve((ReadOnlySpan<double> theta, Span<double> values) =>
                {
                    values[0] = theta[0] - scale;
                    values[1] = theta[0] + scale;
                }, [0.0], 2, (ReadOnlySpan<double> _, Span<double> values) =>
                {
                    values[0] = 1;
                    values[1] = 1;
                })
            : solver.Solve(theta => [theta[0] - scale, theta[0] + scale],
                [0.0], _ => new double[,] { { 1.0 }, { 1.0 } });

        result.Converged.Should().BeTrue();
        result.FinalCost.Should().BeApproximately(scale * scale, 1e293);
        result.ParameterCovariance.Should().NotBeNull();
        result.ParameterCovariance![0, 0].Should().BeApproximately(scale * scale, 1e293);
    }
}
