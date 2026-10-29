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

public sealed class LevenbergMarquardtActiveSetTests
{
    [Fact]
    public void FailedBufferResize_PreservesTheReusablePool()
    {
        var buffers = new LevenbergMarquardtBuffers<double>(1, 1);
        var originalJacobian = buffers.Jacobian;
        // The requested vector exceeds the CLR array length limit; no huge allocation is attempted.
        Action resize = () => buffers.Reset(int.MaxValue, 1);
        resize.Should().Throw<OutOfMemoryException>();
        buffers.ParameterCount.Should().Be(1);
        buffers.ResidualCount.Should().Be(1);
        buffers.Jacobian.Should().BeSameAs(originalJacobian);
        Action overflowingShape = () => buffers.Reset(50000, 50000);
        overflowingShape.Should().Throw<OverflowException>();
        buffers.Reset(2, 2);
        var solver = new LevenbergMarquardtSolver<double>(100, 1e-12, 1e-12, 1e-10, 1e-3);
        var result = solver.Solve(p => [p[0] - 1, p[1] - 2], [0.0, 0.0], buffers);
        result.Converged.Should().BeTrue();
        result.Parameters[0].Should().BeApproximately(1, 1e-8);
        result.Parameters[1].Should().BeApproximately(2, 1e-8);
    }

    [Theory]
    [InlineData(1e10)]
    [InlineData(1e12)]
    [InlineData(1e14)]
    public void LargeInitialDamping_DoesNotCertifyTheRosenbrockStart(double damping)
    {
        var solver = new LevenbergMarquardtSolver<double>(200, 1e-12, 1e-12, 1e-10, damping);
        var result = solver.Solve(p => [10 * (p[1] - p[0] * p[0]), 1 - p[0]], [-1.2, 1],
            p => new[,] { { -20 * p[0], 10 }, { -1, 0 } });
        result.Converged.Should().BeTrue();
        result.Parameters[0].Should().BeApproximately(1, 1e-6);
        result.Parameters[1].Should().BeApproximately(1, 1e-6);
    }

    [Theory]
    [InlineData(1e16)]
    [InlineData(1e20)]
    public void InitialDampingAtOrBeyondCeiling_IsRejected(double damping)
    {
        Action create = () => _ = new LevenbergMarquardtSolver<double>(200, 1e-12, 1e-12, 1e-10, damping);
        create.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("initialDamping");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void NullCallbackResults_AreReportedAsContractViolations(int callback)
    {
        var solver = new LevenbergMarquardtSolver<double>(200, 1e-12, 1e-12, 1e-10, 1e-3);
        var calls = 0;
        double[] Residual(double[] p) => callback == 0 || (callback == 1 && ++calls > 1) ? null! : [p[0] - 1];
        Action solve = () => solver.Solve(Residual, [0.0], callback == 2 ? _ => null! : null);
        solve.Should().Throw<InvalidOperationException>().WithMessage("*null*");
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ActiveInterceptBound_FitsFreeSlope(bool writer, bool centered)
    {
        const int count = 1000;
        var x = Enumerable.Range(1, count).Select(i => i - (centered ? 500.5 : 0)).ToArray();
        var y = x.Select((value, i) => 2 * value + 1 + 0.5 * Math.Sin(i + 1)).ToArray();
        var expected = x.Zip(y, (a, b) => a * b).Sum() / x.Sum(value => value * value);
        var solver = new LevenbergMarquardtSolver<double>(200, 1e-12, 1e-12, 1e-10, 1e-3);
        var result = writer
            ? solver.Solve((p, r) => { for (var i = 0; i < count; i++) { r[i] = p[0] * x[i] + p[1] - y[i]; } },
                [0.0, 0.0], count,
                (_, j) => { for (var i = 0; i < count; i++) { j[2 * i] = x[i]; j[2 * i + 1] = 1; } },
                upperBounds: [double.PositiveInfinity, 0])
            : solver.Solve(p => x.Select((value, i) => p[0] * value + p[1] - y[i]).ToArray(),
                [0.0, 0.0], _ =>
                {
                    var j = new double[count, 2];
                    for (var i = 0; i < count; i++) { j[i, 0] = x[i]; j[i, 1] = 1; }
                    return j;
                }, upperBounds: [double.PositiveInfinity, 0]);

        result.Converged.Should().BeTrue();
        result.Parameters[0].Should().BeApproximately(expected, 1e-9);
        result.Parameters[1].Should().Be(0);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void FixedCoordinate_LeavesSameFreeLeastSquaresProblem(bool writer, bool analytic)
    {
        var solver = new LevenbergMarquardtSolver<double>(200, 1e-12, 1e-12, 1e-10, 1e-3);
        var result = writer
            ? solver.Solve((p, r) => { r[0] = p[0] + p[1] - 3; r[1] = p[1] - 1; r[2] = p[0] - 2 * p[1]; r[3] = 0.5 * p[0] + p[1] - 2; },
                [0.5, 0.0], 4, analytic ? (_, j) => { double[] values = [1, 1, 0, 1, 1, -2, 0.5, 1]; values.CopyTo(j); }
        : null,
                [0.5, double.NegativeInfinity], [0.5, double.PositiveInfinity])
            : solver.Solve(p => [p[0] + p[1] - 3, p[1] - 1, p[0] - 2 * p[1], 0.5 * p[0] + p[1] - 2],
                [0.5, 0.0], analytic ? _ => new[,] { { 1.0, 1.0 }, { 0, 1 }, { 1, -2 }, { 0.5, 1 } } : null,
                [0.5, double.NegativeInfinity], [0.5, double.PositiveInfinity]);

        result.Converged.Should().BeTrue();
        result.Parameters[0].Should().Be(0.5);
        result.Parameters[1].Should().BeApproximately(6.25 / 7, 1e-8);
    }
}
