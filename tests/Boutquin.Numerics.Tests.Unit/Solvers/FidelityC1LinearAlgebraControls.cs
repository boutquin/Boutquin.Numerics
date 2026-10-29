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

using Boutquin.Numerics.LinearAlgebra.Internal;
using Boutquin.Numerics.Solvers;
using Boutquin.Numerics.Solvers.Internal;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

[Trait("Tier", "Unit")]
public sealed class FidelityC1LinearAlgebraControls
{
    [Fact]
    public void Qr_EqualTrailingNormsChooseEarliestRemainingColumn()
    {
        double[,] design = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 0, 0 } };
        double[] response = [2, 3, 4, 5];
        var diagonal = new double[3];
        var permutation = new int[3];
        Assert.True(HouseholderQr<double>.Factor(design, response, diagonal, permutation));
        Assert.Equal(new[] { 0, 1, 2 }, permutation);
        var beta = HouseholderQr<double>.SolveUpperTriangular(design, diagonal, response, 3, permutation);
        Assert.Equal(new[] { 2.0, 3.0, 4.0 }, beta);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(9)]
    [InlineData(1e12)]
    public void DampedDiagonalSystemMatchesIndependentRationalSolution(double lambda)
    {
        var buffers = new LevenbergMarquardtBuffers<double>(2, 2);
        buffers.Jacobian[0, 0] = 2;
        buffers.Jacobian[1, 1] = 3;
        Assert.True(DampedLinearSolve<double>.TrySolve([4, -9], lambda, buffers));
        // H=diag(4,9), g=(8,-27); each scalar equation solves independently.
        var expected0 = -2 / (1 + lambda);
        var expected1 = 3 / (1 + lambda);
        Assert.InRange(Math.Abs(buffers.Delta[0] - expected0), 0, 2e-14 * Math.Abs(expected0));
        Assert.InRange(Math.Abs(buffers.Delta[1] - expected1), 0, 2e-14 * Math.Abs(expected1));
    }

    [Fact]
    public void DampedStructuralZeroColumnRemainsNullAndHasZeroStep()
    {
        var buffers = new LevenbergMarquardtBuffers<double>(2, 2);
        buffers.Jacobian[0, 0] = 2;
        Assert.True(DampedLinearSolve<double>.TrySolve([4, -9], 9, buffers));
        Assert.Equal(0, buffers.NormalEquations[1, 1]);
        Assert.Equal(0, buffers.Gradient[1]);
        Assert.Equal(0, buffers.Delta[1]);
        Assert.InRange(Math.Abs(buffers.Delta[0] + 0.2), 0, 2e-15);
    }

    [Fact]
    public void WriterLoopAllocatesNoBytesBetweenSuccessiveResidualCallbacks()
    {
        // The nonlinear least-squares objective .5*(x²-2)² has minimum zero at sqrt2.
        // Callback-owned storage/delegates are allocated before warmup. The interval
        // ends at the last residual callback, excluding terminal result construction.
        var allocatedAtCall = new long[64];
        var record = false;
        var calls = 0;
        LmResidualWriter<double> residual = (ReadOnlySpan<double> x, Span<double> r) =>
        {
            if (record) { allocatedAtCall[calls++] = GC.GetAllocatedBytesForCurrentThread(); }
            r[0] = x[0] * x[0] - 2;
        };
        LmJacobianWriter<double> jacobian = static (ReadOnlySpan<double> x, Span<double> j) => j[0] = 2 * x[0];
        var solver = new LevenbergMarquardtSolver<double>(20, 1e-30, 1e-30, 1e-14, 0.001);
        var buffers = new LevenbergMarquardtBuffers<double>(1, 1);
        double[] start = [1];
        for (var i = 0; i < 32; i++) { _ = solver.Solve(residual, start, buffers, jacobian); }
        record = true;
        var result = solver.Solve(residual, start, buffers, jacobian);
        Assert.True(calls >= 4);
        Assert.True(result.Converged);
        Assert.InRange(Math.Abs(result.Parameters[0] - Math.Sqrt(2)), 0, 1e-12);
        Assert.InRange(result.FinalCost, 0, 1e-25);
        for (var i = 1; i < calls; i++) { Assert.Equal(0, allocatedAtCall[i] - allocatedAtCall[i - 1]); }
    }
}
