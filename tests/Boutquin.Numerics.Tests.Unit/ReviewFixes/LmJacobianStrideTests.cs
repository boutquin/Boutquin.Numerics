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

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Regression test for review finding #1: the analytic <see cref="LmJacobianWriter{T}"/> path
/// wrote a flat span at the logical parameter stride while reading the pooled Jacobian slab at
/// its (larger) capacity stride after a shrinking <see cref="LevenbergMarquardtBuffers{T}.Reset"/>,
/// silently corrupting the Jacobian.
/// </summary>
public sealed class LmJacobianStrideTests
{
    // Noisy data so the least-squares optimum has NONZERO residuals: this makes the
    // reported ParameterCovariance = (JᵀJ)⁻¹·σ² a deterministic, Jacobian-dependent
    // discriminator. A scrambled Jacobian yields a different covariance even if the
    // parameters happen to converge (LM measures actual cost reduction, so an exactly
    // linear/zero-residual problem is NOT a reliable discriminator).
    private static readonly double[] s_xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    private static readonly double[] s_ys = [1.0, 3.2, 4.8, 7.3, 8.9, 11.4];

    [Fact]
    public void AnalyticWriter_AfterShrinkingReset_MatchesUnpooledSolve()
    {
        var solver = new LevenbergMarquardtSolver<double>(
            maxIterations: 200,
            functionTolerance: 1e-12,
            parameterTolerance: 1e-12,
            gradientTolerance: 1e-12,
            initialDamping: 1e-3);

        // Reference: unpooled func-based solve (uses a private, exact-sized buffer).
        var reference = solver.Solve(
            residuals: ResidualsFunc,
            initialGuess: [0.0, 0.0],
            jacobian: JacobianFunc);

        // Pool allocated at n = 3, then shrunk to n = 2 — capacity column stride (3) now
        // exceeds the logical parameter count (2), which is exactly the corruption trigger.
        var buffers = new LevenbergMarquardtBuffers<double>(parameterCount: 3, residualCount: s_xs.Length);
        buffers.Reset(parameterCount: 2, residualCount: s_xs.Length);

        var pooled = solver.Solve(
            residuals: ResidualWriter,
            initialGuess: [0.0, 0.0],
            buffers: buffers,
            jacobian: JacobianWriter);

        pooled.Converged.Should().BeTrue();
        pooled.Parameters[0].Should().BeApproximately(reference.Parameters[0], 1e-9);
        pooled.Parameters[1].Should().BeApproximately(reference.Parameters[1], 1e-9);

        // The covariance is read straight off the (possibly corrupt) Jacobian slab.
        reference.ParameterCovariance.Should().NotBeNull();
        pooled.ParameterCovariance.Should().NotBeNull();
        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                pooled.ParameterCovariance![i, j].Should()
                    .BeApproximately(reference.ParameterCovariance![i, j], 1e-9);
            }
        }
    }

    private static double[] ResidualsFunc(double[] p)
    {
        var r = new double[s_xs.Length];
        for (var i = 0; i < s_xs.Length; i++)
        {
            r[i] = p[0] + p[1] * s_xs[i] - s_ys[i];
        }

        return r;
    }

    private static double[,] JacobianFunc(double[] p)
    {
        var j = new double[s_xs.Length, 2];
        for (var i = 0; i < s_xs.Length; i++)
        {
            j[i, 0] = 1.0;
            j[i, 1] = s_xs[i];
        }

        return j;
    }

    private static void ResidualWriter(ReadOnlySpan<double> p, Span<double> residuals)
    {
        for (var i = 0; i < s_xs.Length; i++)
        {
            residuals[i] = p[0] + p[1] * s_xs[i] - s_ys[i];
        }
    }

    private static void JacobianWriter(ReadOnlySpan<double> p, Span<double> jacobian)
    {
        // Row-major, stride = parameterCount (2): element (i, j) at jacobian[i * 2 + j].
        for (var i = 0; i < s_xs.Length; i++)
        {
            jacobian[i * 2] = 1.0;
            jacobian[i * 2 + 1] = s_xs[i];
        }
    }
}
