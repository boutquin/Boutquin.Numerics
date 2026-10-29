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

using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using Boutquin.Numerics.Internal;
using Boutquin.Numerics.Solvers;
using Xunit.Abstractions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// The cost of the conditioning guard of <see cref="ActiveSetQpSolver{T}"/> (<c>48·n·u·κ₁ ≥ 1</c> on every new working
/// set's free block and on the block that decides the result) for <c>decimal</c> covariances whose blocks have a
/// condition number between about 1e14 and 1e22: beyond the reach of a bound computed in <c>double</c>, yet many orders
/// of magnitude inside the margin for <c>u = 1e-28</c>. Decided by the exact adjugate on every such block, a solve of
/// order 100 to 120 costs 2,000 to 4,500 times the calibration workload (25,000 times with every asset free); each case
/// below decides no block by the exact adjugate, keeps every certified stage of the guard within its documented
/// frequency and O(m³) work (<see cref="ActiveSetQpStageWorkControlTests"/>), and, in a long-only box, stays under the
/// ceiling of <see cref="ActiveSetQpLargeProblemTests"/> in CPU time as well. Each covariance is positive definite
/// by construction (an exact Gram matrix plus a positive diagonal), and each result is checked against the exact
/// first-order conditions independently of the solver.
/// </summary>
[Collection(ActiveSetQpPerformanceCollection.Name)]
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpConditioningGuardCostTests(ITestOutputHelper output)
{
    // The calibrated ceiling is a wall-clock bound for a solve on four workers. The CPU time of that solve sums its workers
    // and so exceeds its wall time (by 1.3 to 1.6 times for the double refusal below on a lightly loaded host): the
    // ceiling on CPU time is that bound times this allowance.
    private const double SummedWorkerCpuAllowance = 2.0;

    // preparations: how many times the solve prepares the double-double eigenvalue bound of the whole equilibrated
    // covariance, documented as at most once per solve; 0 when the bound of the block in double settles every block.
    [Theory]
    [InlineData(119, 1e15, 1, 1)]
    [InlineData(120, 1e16, 1, 1)]
    [InlineData(110, 1e20, 1, 1)]
    public void SolveMinVariance_IllConditionedDecimalFactorModel_IsSolvedWithinTheCalibratedCeiling(int n, double kappa, int seed, int preparations)
    {
        var cov = ActiveSetQpExactFallbackCostTests.IllConditionedDecimal(n, kappa, seed);
        _ = AssertSolvedWithinCeiling(cov, 0m, 0.15m, preparations, cpuCeiling: true, string.Create(CultureInfo.InvariantCulture, $"decimal factor model n={n} kappa={kappa:E0} seed={seed} box [0, 0.15]"));
    }

    // Every asset is free at the optimum, so each working set's block grows to the full order: about one queried block
    // per asset, each of order up to n, so the per-block bounds dominate the guard's work. The cost of these solves is
    // pinned by the deterministic work of the guard's stages, which no host changes; their process CPU time is reported
    // and not asserted, because on unchanged code it crossed the calibrated ceiling by host scheduling alone (the order
    // 100 solve at 1151x in 1 of 30 executions, and 685x to 1254x of 1000x on one processor, where the first of the
    // best-of-three repetitions still run code the JIT has not yet optimised).
    [Theory]
    [InlineData(70, 1e15, 1)]
    [InlineData(100, 1e13, 1)]
    public void SolveMinVariance_IllConditionedDecimalFactorModelInALongShortBox_IsSolvedWithinItsDeterministicWork(int n, double kappa, int preparations)
    {
        var cov = ActiveSetQpExactFallbackCostTests.IllConditionedDecimal(n, kappa, seed: 20261031);
        var trace = AssertSolvedWithinCeiling(cov, -0.5m, 0.5m, preparations, cpuCeiling: false, string.Create(CultureInfo.InvariantCulture, $"decimal factor model n={n} kappa={kappa:E0} box [-0.5, 0.5]"));

        // About one queried block per asset (one working-set change each, plus the deciding block).
        Assert.True(trace.ConditioningQueries <= 2 * n, "queried blocks: " + trace.ConditioningQueries);
    }

    // A well-conditioned covariance: the bound of each block in double settles it, so the solve never prepares the
    // whole-matrix eigenvalue bound, in a long-only and in a long-short box.
    [Fact]
    public void SolveMinVariance_WellConditionedDecimalFactorModel_NeverPreparesTheWholeMatrixBound()
    {
        var cov = ActiveSetQpExactFallbackCostTests.IllConditionedDecimal(60, 1e3, seed: 20261031);
        foreach (var (lower, upper) in new[] { (0m, 0.15m), (-0.5m, 0.5m) })
        {
            var trace = new ActiveSetQpSolver<decimal>.RouteTrace();
            _ = ActiveSetQpSolver<decimal>.SolveMinVarianceUsingCore(cov, lower, upper, trace: trace);
            Assert.Equal(0, trace.ExactConditioningTests);
            Assert.Equal(0, trace.ConditioningPreparation.Runs);
        }
    }

    [Fact]
    public void SolveMinVariance_IllConditionedDoubleLongShortProblem_IsRefusedWithinTheCalibratedCeiling()
    {
        const int n = 100;
        var source = ActiveSetQpExactFallbackCostTests.IllConditionedDecimal(n, kappa: 1e13, seed: 20261031);
        var cov = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                cov[i, j] = (double)source[i, j];
            }
        }

        // Match the existing corpus timing gate: warm this production path before measuring steady-state solve cost.
        var warmup = Assert.Throws<InvalidOperationException>(() => ActiveSetQpSolver<double>.SolveMinVariance(cov, -0.5, 0.5));
        var ms = double.PositiveInfinity;
        var wallMs = double.NaN;
        var trace = new ActiveSetQpSolver<double>.RouteTrace();
        // Use the best of three repetitions, as the existing first-generation corpus timing gate does.
        for (var repetition = 0; repetition < 3; repetition++)
        {
            var attempt = new ActiveSetQpSolver<double>.RouteTrace();
            // The cost is the process CPU time of the refusal (as the decimal controls of this class measure it), summed
            // over the exact factor's row workers; the wall time rides along for the report line and is never asserted.
            var cpu0 = Process.GetCurrentProcess().TotalProcessorTime;
            var wall0 = Stopwatch.GetTimestamp();
            var error = Assert.Throws<InvalidOperationException>(() => ActiveSetQpSolver<double>.SolveMinVarianceUsingCore(cov, -0.5, 0.5, trace: attempt));
            var wall = Stopwatch.GetElapsedTime(wall0).TotalMilliseconds;
            var elapsed = (Process.GetCurrentProcess().TotalProcessorTime - cpu0).TotalMilliseconds;
            Assert.Contains("stationarity cannot be certified", error.Message, StringComparison.Ordinal);
            Assert.Equal(warmup.Message, error.Message);
            Assert.Equal(ActiveSetQpSolver<double>.ExactRouteReason.UnusableWorkingSet, attempt.ExactRouteReason);

            // The work of the refusal, the same on every host: the rounded route's working-set changes, the exact
            // route's changes over the cached exact factor, and no block decided by the exact adjugate (the exact
            // refusal witness proves the deciding block outside the margin; certified bounds settle the others).
            Assert.Equal(0, attempt.ExactConditioningTests);
            Assert.Equal(24, attempt.RoundedChanges);
            Assert.Equal(81, attempt.ExactChanges);
            Assert.True(attempt.CachedExactSolves > 0);
            if (elapsed < ms)
            {
                ms = elapsed;
                wallMs = wall;
                trace = attempt;
            }
        }
        // The exact factor's row update runs on Math.Min(4, Environment.ProcessorCount) workers, the calibration on one.
        // Wall time on w workers is therefore not a host-independent measure: a loaded host gives the workers far less
        // than w times the speed of one (about 1.4 on four against 4 assumed), so a ceiling on it scaled by 4/w fails on
        // unchanged code. The CPU time summed over the workers, as a multiple of the calibration, stays between 1.1 and
        // 1.4 thousand from a quiet host to a saturated one and is the same on one worker as on four, where the wall time
        // on four workers runs from 0.7 to 3.8 thousand. The ceiling is the calibrated wall ceiling doubled, which is
        // above the CPU time of unchanged code; a path three times slower (3.3 to 4.3 thousand times) is above it.
        var ceiling = ActiveSetQpLargeProblemTests.CalibratedCeiling * SummedWorkerCpuAllowance;
        var calibration = ActiveSetQpPerformanceTests.CalibrationMs();
        var ratio = ms / calibration;
        var line = string.Create(CultureInfo.InvariantCulture, $"double long-short refusal n=100: {ms:F1} ms CPU = {ratio:F1}x calibration ({calibration:F3} ms), ceiling {ceiling:F0}x CPU ({wallMs:F1} ms wall on {ExactLinearSolve.WorkerLimit} worker(s), not asserted); rounded changes {trace.RoundedChanges}, exact changes {trace.ExactChanges}, cached exact solves {trace.CachedExactSolves}");
        output.WriteLine(line);
        Assert.True(ratio <= ceiling, line);
    }

    // Covariances whose free blocks lie on both sides of the margin (48·n·u·κ₁ from about 1/16 to 16 times the
    // threshold): the certified guard must reach the exact test's verdict on every block a solve queries, so each solve
    // ends exactly as it does with the exact test alone, in its outcome and in the route it took.
    [Fact]
    public void ConditioningGuard_OnCovariancesStraddlingItsMargin_DecidesAsTheExactTestAlone()
    {
        var rng = new System.Random(20261061);
        var refused = 0;
        var settled = 0;
        for (var c = 0; c < 44; c++)
        {
            var n = 2 + ((c / 2) % 11);
            var target = Math.Pow(2.0, (8.0 * rng.NextDouble()) - 4.0);
            var (r, s) = c % 2 == 0
                ? CompareWithExactAlone(Straddling<double>(n, target, rng), -0.5, 0.5)
                : CompareWithExactAlone(Straddling<decimal>(n, target, rng), -0.5m, 0.5m);
            refused += r;
            settled += s;
        }

        // Both sides of the margin were exercised: blocks the exact test refuses, and blocks a certified bound settled.
        Assert.True(refused > 0, "no solve met a block outside the margin");
        Assert.True(settled > 0, "no block was settled by a certified bound");
    }

    // The same family concentrated around the margin, for double, where the guard also tries the exact refusal witness
    // ‖x‖₁/‖A·x‖₁ ≤ ‖A⁻¹‖₁ on a column x of the approximate inverse: 48·n·u·κ₁ of the whole matrix within ten percent of
    // the threshold for one covariance in four, within two percent for the rest. A witness whose threshold or norms
    // drift even slightly from the exact test's (a margin of 49 or 96 instead of 48, or a signed sum for ‖A·x‖₁, whose
    // entries off the unit column are about 1/(48·n) of it this close to the margin) refuses blocks just inside the
    // margin that the exact test accepts; here every solve must end exactly as it does with the exact test alone.
    [Fact]
    public void ConditioningGuard_OnDoubleCovariancesNearItsMargin_DecidesAsTheExactTestAlone()
    {
        var rng = new System.Random(20261071);
        var refused = 0;
        var accepted = 0;
        for (var c = 0; c < 2200; c++)
        {
            var n = 2 + (c % 11);
            var target = c % 4 == 0 ? 0.90 + (0.20 * rng.NextDouble()) : 0.98 + (0.04 * rng.NextDouble());
            var (r, _) = CompareWithExactAlone(Straddling<double>(n, target, rng), -0.5, 0.5);
            refused += r;
            accepted += 2 - r;
        }

        // Both sides of the margin were exercised, by solves the exact test alone refuses and by solves it accepts.
        output.WriteLine(string.Create(CultureInfo.InvariantCulture, $"near-margin double solves: {refused} refused, {accepted} accepted by the exact test alone"));
        Assert.True(refused > 0, "no solve met a block outside the margin");
        Assert.True(accepted > 0, "every solve met a block outside the margin");
    }

    [Fact]
    public void ExactFactorReuse_OnSeededProblems_MatchesFreshElimination()
    {
        CompareExactFactorReuse<double>();
        CompareExactFactorReuse<decimal>();
    }

    private static void CompareExactFactorReuse<T>()
        where T : IFloatingPoint<T>
    {
        foreach (var seed in new[] { 20261024, 20261025, 20261026 })
        {
            const int n = 16;
            var original = ActiveSetQpExactFallbackCostTests.IllConditionedDecimal(n, 1e6, seed);
            var cov = new T[n, n];
            var means = new T[n];
            for (var i = 0; i < n; i++)
            {
                means[i] = T.CreateChecked((i % 3) - 1) / T.CreateChecked(100);
                for (var j = 0; j < n; j++)
                {
                    cov[i, j] = T.CreateChecked(original[i, j]);
                }
            }

            foreach (var meanVariance in new[] { false, true })
            {
                var fresh = new ActiveSetQpSolver<T>.RouteTrace { ForceExactRoute = true, FractionFreeOnly = true };
                var reused = new ActiveSetQpSolver<T>.RouteTrace { ForceExactRoute = true };
                T[] Run(ActiveSetQpSolver<T>.RouteTrace trace) => meanVariance
                    ? ActiveSetQpSolver<T>.SolveMeanVarianceUsingCore(cov, means, T.CreateChecked(3), -T.One / T.CreateChecked(2), T.One / T.CreateChecked(2), trace: trace)
                    : ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(cov, -T.One / T.CreateChecked(2), T.One / T.CreateChecked(2), trace: trace);
                Assert.Equal(Run(fresh), Run(reused));
                Assert.Equal(fresh.ExactChanges, reused.ExactChanges);
                Assert.Equal(fresh.ExactRollbacks, reused.ExactRollbacks);
                // At these means the mean-variance optimum keeps fewer than 13 assets free, below the cached factor's
                // threshold, so that leg exercises the fraction-free path only; ActiveSetQpFactorReuseTests pins the
                // cached factor for mean variance on free blocks of 13 to 70.
                if (!meanVariance)
                {
                    Assert.True(reused.CachedExactSolves > 0);
                }
            }
        }
    }

    private static (int Refused, int Settled) CompareWithExactAlone<T>(T[,] cov, T lower, T upper)
        where T : IFloatingPoint<T>
    {
        var refused = 0;
        var settled = 0;
        foreach (var (l, h) in new[] { (T.Zero, T.One), (lower, upper) })
        {
            var certified = new ActiveSetQpSolver<T>.RouteTrace();
            var exactAlone = new ActiveSetQpSolver<T>.RouteTrace { ExactConditioningOnly = true };
            var expected = Outcome(() => ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(cov, l, h, trace: exactAlone));
            var actual = Outcome(() => ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(cov, l, h, trace: certified));
            var label = string.Create(CultureInfo.InvariantCulture, $"{typeof(T).Name} n={cov.GetLength(0)} box [{l}, {h}]");
            Assert.True(expected == actual, label + ": exact alone " + expected + " | certified " + actual);
            Assert.True(exactAlone.ExactRouteReason == certified.ExactRouteReason, label + ": exact alone " + exactAlone.ExactRouteReason + " | certified " + certified.ExactRouteReason);
            refused += exactAlone.ExactRouteReason == ActiveSetQpSolver<T>.ExactRouteReason.UnusableWorkingSet || expected.Contains("numerically singular", StringComparison.Ordinal) ? 1 : 0;
            settled += certified.ExactConditioningTests < exactAlone.ExactConditioningTests ? 1 : 0;
        }

        return (refused, settled);
    }

    private static string Outcome<T>(Func<T[]> solve)
    {
        try
        {
            return string.Join(" ", solve().Select(w => string.Create(CultureInfo.InvariantCulture, $"{w}")));
        }
        catch (InvalidOperationException ex)
        {
            return ex.Message;
        }
    }

    /// <summary>
    /// <c>G·G′ + ε·diag(d)</c> with <c>G</c> of order n × (n − 1) (entries multiples of 1/8 in [-1/4, 1/4], exact in
    /// <c>double</c> and <c>decimal</c>) and <c>d</c> in [1, 1.5]: positive definite, with one near-null direction of
    /// eigenvalue about <c>ε</c>; <c>ε = 48·n·u·‖G·G′‖₁/target</c> puts <c>48·n·u·κ₁</c> of the whole matrix near
    /// <c>target</c> (within a factor of about √n).
    /// </summary>
    private static T[,] Straddling<T>(int n, double target, System.Random rng)
        where T : IFloatingPoint<T>
    {
        var eighth = T.One / T.CreateChecked(8);
        var g = new T[n, n - 1];
        for (var i = 0; i < n; i++)
        {
            for (var k = 0; k < n - 1; k++)
            {
                g[i, k] = T.CreateChecked(rng.Next(-2, 3)) * eighth;
            }
        }

        var cov = new T[n, n];
        var norm = 0.0;
        for (var j = 0; j < n; j++)
        {
            var column = 0.0;
            for (var i = 0; i < n; i++)
            {
                var sum = T.Zero;
                for (var k = 0; k < n - 1; k++)
                {
                    sum += g[i, k] * g[j, k];
                }

                cov[i, j] = sum;
                column += Math.Abs(double.CreateChecked(sum));
            }

            norm = Math.Max(norm, column);
        }

        var unit = typeof(T) == typeof(decimal) ? 1e-28 : Math.ScaleB(1.0, -53);
        var epsilon = 48.0 * n * unit * Math.Max(norm, 1.0 / 64) / target;
        for (var i = 0; i < n; i++)
        {
            cov[i, i] += T.CreateChecked(epsilon * (1.0 + (rng.Next(0, 5) / 8.0)));
        }

        return cov;
    }

    private ActiveSetQpSolver<decimal>.RouteTrace AssertSolvedWithinCeiling(decimal[,] cov, decimal lower, decimal upper, int preparations, bool cpuCeiling, string label)
    {
        var n = cov.GetLength(0);
        _ = ActiveSetQpSolver<decimal>.SolveMinVariance(ActiveSetQpExactFallbackCostTests.IllConditionedDecimal(8, 1e3, seed: 1), 0m, 1m);
        var trace = new ActiveSetQpSolver<decimal>.RouteTrace();
        var weights = Array.Empty<decimal>();
        var ms = double.PositiveInfinity;
        var work = string.Empty;
        for (var repetition = 0; repetition < 3; repetition++)
        {
            var attempt = new ActiveSetQpSolver<decimal>.RouteTrace();
            var t0 = Process.GetCurrentProcess().TotalProcessorTime;
            weights = ActiveSetQpSolver<decimal>.SolveMinVarianceUsingCore(cov, lower, upper, trace: attempt);
            var elapsed = (Process.GetCurrentProcess().TotalProcessorTime - t0).TotalMilliseconds;
            Assert.Equal(0, attempt.ExactConditioningTests);
            var prepared = attempt.ConditioningPreparation.Runs;
            Assert.True(preparations == prepared, label + ": the whole-matrix eigenvalue bound was prepared " + prepared + " times, expected " + preparations);

            // The deterministic work of the guard: every certified stage within its documented frequency and O(m³) work.
            work = ActiveSetQpStageWorkControlTests.AssertConditioningWork(attempt, n, label);
            if (elapsed < ms)
            {
                ms = elapsed;
                trace = attempt;
            }
        }

        // Every block is far inside the margin, so a certified bound settles each one.
        Assert.Equal(0, trace.ExactConditioningTests);
        Assert.True(Math.Abs(weights.Sum() - 1m) <= 1e-25m, weights.Sum().ToString(CultureInfo.InvariantCulture));
        Assert.All(weights, w => Assert.True(w >= lower && w <= upper, w.ToString(CultureInfo.InvariantCulture)));
        ActiveSetQpLargeProblemTests.AssertOptimal(cov, weights, lower, upper, means: null, 1m, ActiveSetQpLargeProblemTests.DecimalStationarity);

        var calibration = ActiveSetQpPerformanceTests.CalibrationMs();
        var ratio = ms / calibration;
        var line = string.Create(CultureInfo.InvariantCulture, $"{label}: {ms:F1} ms = {ratio:F1}x the calibration workload ({calibration:F3} ms); ceiling {ActiveSetQpLargeProblemTests.CalibratedCeiling}x{(cpuCeiling ? "" : " (reported, not asserted)")}");
        output.WriteLine(line);
        output.WriteLine(work);
        Assert.True(!cpuCeiling || ratio <= ActiveSetQpLargeProblemTests.CalibratedCeiling, line);
        return trace;
    }
}
