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
using Boutquin.Numerics.Tests.Unit.Solvers.NistStRD;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// Determinism-firewall regression guard (build-arena M-2, spec
/// <c>feature-solver-test-parallel-flakiness-hardening</c>): solves the same NIST
/// problem on many concurrent, shared-nothing tasks and asserts every result is
/// byte-identical to a single-threaded reference. Locks in the audited property
/// that <see cref="LevenbergMarquardtSolver"/> and <see cref="OrdinaryLeastSquares"/>
/// depend on no shared mutable state — no <c>static</c> accumulator, no pooled
/// buffer reused across concurrent calls — so a future edit that introduces one
/// fails this test rather than surfacing as an unreproduced flake under CI load.
/// </summary>
/// <remarks>
/// Companion to <c>SolverStaticStateTests</c> (architecture-level: fails if the
/// four solver types gain a non-<c>readonly</c> static field). This test is the
/// runtime/behavioral half of the same guarantee: even a structurally clean type
/// could still be handed a caller-shared mutable buffer, so the byte-identity
/// check is the ground truth the structural guard exists to protect.
/// </remarks>
public sealed class SolverDeterminismTests
{
    private const int ConcurrentTaskCount = 8;

    // ── LM path — Misra1a (14 obs, 2 params): fast to solve, exercises the
    //    nonlinear Levenberg-Marquardt iteration this spec's flake was observed on. ──

    [Fact]
    public async Task LevenbergMarquardt_Misra1a_ConcurrentSolves_AreByteIdenticalToSingleThreadedReference()
    {
        var data = NistStRDData.Misra1a();

        var reference = SolveMisra1a(data);

        var concurrent = await Task.WhenAll(
            Enumerable.Range(0, ConcurrentTaskCount).Select(_ => Task.Run(() => SolveMisra1a(data))));

        foreach (var result in concurrent)
        {
            AssertBitIdenticalParameters(reference.Parameters, result.Parameters);
            AssertBitIdenticalCost(reference.FinalCost, result.FinalCost);
        }
    }

    private static MultivariateSolverResult SolveMisra1a(NistStRDData.Problem data)
    {
        // Independent closure per call — captures only the (never-mutated) NIST
        // arrays and allocates its own residual buffer, so no state crosses calls.
        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var model = t[0] * (1.0 - Math.Exp(-t[1] * data.X[i]));
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        // The pool-free overload constructs a private, per-call buffer set
        // internally (see LevenbergMarquardtSolver's allocation-profile remarks) —
        // no buffer pool is shared across the concurrent tasks below.
        return new LevenbergMarquardtSolver(maxIterations: 500).Solve(Residuals, data.Start2);
    }

    // ── OLS path — Norris (36 obs, 1 predictor, well-conditioned): exercises the
    //    QR-backed OrdinaryLeastSquares path named alongside LM in the spec. ──

    [Fact]
    public async Task OrdinaryLeastSquares_Norris_ConcurrentFits_AreByteIdenticalToSingleThreadedReference()
    {
        var problem = NistLinearRegressionData.Norris();

        var reference = OrdinaryLeastSquares.Fit(problem.X, problem.Y, problem.IncludeIntercept);

        var concurrent = await Task.WhenAll(
            Enumerable.Range(0, ConcurrentTaskCount).Select(
                _ => Task.Run(() => OrdinaryLeastSquares.Fit(problem.X, problem.Y, problem.IncludeIntercept))));

        foreach (var result in concurrent)
        {
            AssertBitIdenticalVector(reference.Coefficients, result.Coefficients, "coefficient");
        }
    }

    // ── Shared bit-identity assertions ──────────────────────────────────────

    private static void AssertBitIdenticalParameters(double[] expected, double[] actual) =>
        AssertBitIdenticalVector(expected, actual, "parameter");

    private static void AssertBitIdenticalVector(double[] expected, double[] actual, string label)
    {
        actual.Length.Should().Be(expected.Length);

        for (var i = 0; i < expected.Length; i++)
        {
            BitConverter.DoubleToInt64Bits(actual[i]).Should().Be(
                BitConverter.DoubleToInt64Bits(expected[i]),
                because: $"{label} {i} must be bit-identical across concurrent shared-nothing solves: " +
                          $"reference={expected[i]:G17}, concurrent={actual[i]:G17}");
        }
    }

    private static void AssertBitIdenticalCost(double expected, double actual) =>
        BitConverter.DoubleToInt64Bits(actual).Should().Be(
            BitConverter.DoubleToInt64Bits(expected),
            because: $"final cost must be bit-identical across concurrent shared-nothing solves: " +
                      $"reference={expected:G17}, concurrent={actual:G17}");
}
