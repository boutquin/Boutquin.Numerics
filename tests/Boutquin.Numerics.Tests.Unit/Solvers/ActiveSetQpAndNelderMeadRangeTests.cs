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
/// Range behaviour at the edges of the documented domain: a mean-variance optimum or a
/// Nelder–Mead convergence test whose exact value is representable must not be lost to an
/// overflowing intermediate (partial sum, weight assembly or objective spread).
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpAndNelderMeadRangeTests
{
    private const double M = double.MaxValue;

    private static double[,] Diag(int n, double c)
    {
        var a = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            a[i, i] = c;
        }

        return a;
    }

    private static void AssertRelative(double[] actual, double[] expected, double relativeTolerance)
    {
        actual.Should().HaveCount(expected.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            var scale = Math.Max(Math.Abs(expected[i]), 1.0);
            (Math.Abs(actual[i] - expected[i]) / scale).Should().BeLessThanOrEqualTo(
                relativeTolerance, $"weight {i} is {actual[i]:R}, exact {expected[i]:R}");
        }
    }

    // ── Mean-variance QP: weight assembly ──────────────────────────────────

    [Theory]
    [InlineData(2.0, 1.0784666666666666e308)]
    [InlineData(4.0, 5.392333333333333e307)]
    public void QpMeanVariance_FreeWeightWhoseAssemblyOverflows_ReturnsTheRepresentableInBoxOptimum(
        double riskAversion, double expectedFirst)
    {
        const double R = 1.6177e298;
        double[] weights = [];
        var error = Record.Exception(() => weights = ActiveSetQpSolver<double>.SolveMeanVariance(
            Diag(3, 1e-10), [R, -R, -R], riskAversion, -1e308, 1.5e308));

        error.Should().BeNull("the optimum lies in the box and its release terms are representable");
        AssertRelative(weights, [expectedFirst, -expectedFirst / 2, -expectedFirst / 2], 1e-12);
    }

    // ── Mean-variance QP: Σb partial sum ───────────────────────────────────

    [Fact]
    public void QpMeanVariance_InverseSolveWhosePartialSumOverflows_ReturnsTheRepresentableInBoxOptimum()
    {
        double[] weights = [];
        var error = Record.Exception(() => weights = ActiveSetQpSolver<double>.SolveMeanVariance(
            Diag(4, 1e-10), [1.25e298, 1.25e298, -1.25e298, -1.05e298], 1, -1.5e308, 1.5e308));

        error.Should().BeNull("Σb is 2e307 although its first two terms sum past double.MaxValue");
        AssertRelative(weights, [1.2e308, 1.2e308, -1.3e308, -1.1e308], 1e-12);
    }

    // ── Mean-variance QP: fixed-weight partial sum ─────────────────────────

    [Fact]
    public void QpMeanVariance_FixedWeightSumWhosePartialSumOverflows_ReturnsTheRepresentableInBoxOptimum()
    {
        const double lc = 0.01 * 1e-10;
        double[] weights = [];
        var error = Record.Exception(() => weights = ActiveSetQpSolver<double>.SolveMeanVariance(
            Diag(4, 1e-10),
            [lc * 0.65 * M, lc * 0.65 * M, -lc * 0.75 * M, lc * (1 - (0.5 * M))],
            0.01, -0.7 * M, 0.6 * M));

        error.Should().BeNull("the fixed weights sum to 0.5·MaxValue although the first two sum past it");
        AssertRelative(weights, [0.6 * M, 0.6 * M, -0.7 * M, 1 - (0.5 * M)], 1e-12);
    }

    // ── Nelder–Mead: decimal objective spread ──────────────────────────────

    [Fact]
    public void NelderMead_DecimalObjectiveSpreadBeyondTheRange_IsNotConvergedRatherThanAnOverflowException()
    {
        // Every objective value lies in [-5e28, 5e28]; the first spread, 8.3e28, is not
        // representable in decimal, which means only "not converged".
        static decimal Bowl(decimal x)
        {
            var t = (x - 0.04m) / 0.03m;
            var q = t * t;
            if (q > 2m)
            {
                q = 2m;
            }

            return 5e28m * (q - 1m);
        }

        SimplexResult<decimal>? result = null;
        var error = Record.Exception(() => result = NelderMeadMinimizer<decimal>.Minimize(
            x => Bowl(x[0]), [0m]));

        error.Should().BeNull();
        result!.Converged.Should().BeTrue();
        ((double)result.Argmin[0]).Should().BeApproximately(0.04, 1e-6);
    }
}
