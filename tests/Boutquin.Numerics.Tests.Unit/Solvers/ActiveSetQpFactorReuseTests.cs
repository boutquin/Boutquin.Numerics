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

using System.Globalization;
using System.Numerics;

using Boutquin.Numerics.Internal;
using Boutquin.Numerics.Solvers;
using Boutquin.Numerics.Tests.Unit.Internal;

using Xunit.Abstractions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// The factor reuse of <see cref="ActiveSetQpSolver{T}"/> across working sets: the rounded route's Cholesky prefix of
/// the previous reduced Hessian, and the exact route's cached fraction-free factor of the reduced Hessian with its
/// right-hand-side reuse and bounded parallel row update. Reuse may change the cost of a solve and nothing else: the
/// route a solve takes and every bit of its result must be those of a solve that factors every block afresh. The exact
/// factor serves only free blocks of 13 or more variables, beyond every case of the committed QP corpus (orders up to
/// 13), so the problems are generated here.
/// </summary>
// The order-70 test changes the exact factor's shared worker count; the non-parallel performance collection keeps it
// from overlapping the calibrated-ceiling timing tests.
[Collection(ActiveSetQpPerformanceCollection.Name)]
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpFactorReuseTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(12)]
    [InlineData(13)]
    [InlineData(14)]
    public void ExactRoute_AtTheCacheThreshold_MatchesFreshEliminationAndExercisesTheExpectedPath(int n)
    {
        CheckThreshold<double>(n);
        CheckThreshold<decimal>(n);
    }

    private static void CheckThreshold<T>(int n)
        where T : IFloatingPoint<T>
    {
        foreach (var meanVariance in new[] { false, true })
        {
            var problem = Problem<T>(n, 20261023, meanVariance, -50, 50);
            var (expected, fresh, freshPath) = problem.RunWithPath(new() { ForceExactRoute = true, FractionFreeOnly = true });
            var (actual, reused, reusedPath) = problem.RunWithPath(new() { ForceExactRoute = true });
            Assert.Equal(n, problem.FreeCount(actual));
            Assert.Equal(n >= 13, reused.CachedExactSolves > 0);
            Assert.Equal(0, fresh.CachedExactSolves);
            // In this wide box every working-set change releases one bound, so every cached solve after the first (which
            // has no previous transformed RHS) takes the O(n) RHS extension; fresh elimination never does.
            Assert.Equal(n >= 13 ? reused.CachedExactSolves - 1 : 0, reusedPath.RhsExtensions);
            Assert.Equal(0, freshPath.RhsExtensions);
            Assert.Equal(Bits(expected), Bits(actual));
            Assert.Equal(fresh.ExactChanges, reused.ExactChanges);
        }
    }

    // Free blocks of 13 or more variables, for minimum variance and mean variance alike (small means keep most assets
    // free at the mean-variance optimum too). The wide box keeps every asset free; the narrower boxes bind bounds on
    // the way, so the cached factor also loses variables (its suffix is rebuilt) and changes reference.
    [Theory]
    [InlineData(18, 20261111, -50, 50)]
    [InlineData(22, 20261112, -50, 50)]
    [InlineData(26, 20261113, -50, 50)]
    [InlineData(24, 20261115, -20, 30)]
    [InlineData(30, 20261116, -10, 20)]
    [InlineData(34, 20261117, -6, 15)]
    public void ExactRoute_OnLargeFreeBlocks_UsesTheCachedFactorAndMatchesFreshEliminationBitForBit(int n, int seed, int lowerPercent, int upperPercent)
    {
        foreach (var meanVariance in new[] { false, true })
        {
            CompareWithFreshElimination(Problem<double>(n, seed, meanVariance, lowerPercent, upperPercent));
            CompareWithFreshElimination(Problem<decimal>(n, seed, meanVariance, lowerPercent, upperPercent));
        }
    }

    // A free block above order 64 takes the bounded parallel row update when the cached factor grows. Its result must be
    // the serial branch's to the bit, and the weights must satisfy the first-order conditions checked independently of
    // the solver. Both worker counts are set explicitly, so the comparison holds on a one-processor host too.
    [Fact]
    public void ExactRoute_OnAFreeBlockAboveOrder64_ParallelRowsMatchTheSerialRowsBitForBit()
    {
        foreach (var meanVariance in new[] { false, true })
        {
            var problem = Problem<double>(70, 20261114, meanVariance, -50, 50);
            ActiveSetQpSolver<double>.RouteTrace serialTrace = null!;
            ActiveSetQpSolver<double>.RouteTrace parallelTrace = null!;
            double[] serial = [];
            double[] parallel = [];
            PathCounts serialPath = default;
            PathCounts parallelPath = default;
            var serialPeak = 0;
            var parallelPeak = 0;
            ExactLinearSolveTests.WithSerialRows(() => serialPeak = ExactLinearSolveWorkerBoundTests.PeakConcurrentRows(
                () => (serial, serialTrace, serialPath) = problem.RunWithPath(new() { ForceExactRoute = true })));
            ExactLinearSolveTests.WithParallelRows(() =>
            {
                parallelPeak = ExactLinearSolveWorkerBoundTests.PeakConcurrentRows(
                    () => (parallel, parallelTrace, parallelPath) = problem.RunWithPath(new() { ForceExactRoute = true }));
                // The rows of one update ran on at least two and at most the forced worker count at once.
                ExactLinearSolveWorkerBoundTests.AssertWithinWorkerBound(parallelPeak, "order-70 parallel rows");
            });
            output.WriteLine(problem.Describe(parallel, parallelTrace) + $", {parallelPath}, peak rows at once {parallelPeak} (serial {serialPeak})");

            Assert.True(problem.FreeCount(parallel) >= 66, problem.Describe(parallel, parallelTrace));
            Assert.True(parallelTrace.CachedExactSolves > 0, "the cached exact factor was never used");
            Assert.Equal(Bits(serial), Bits(parallel));
            Assert.Equal(serialTrace.ExactChanges, parallelTrace.ExactChanges);
            Assert.Equal(serialTrace.CachedExactSolves, parallelTrace.CachedExactSolves);
            // The parallel rows were actually dispatched, the serial ones never; the RHS path is the same on both.
            Assert.Equal(0, serialPath.ParallelUpdates);
            Assert.Equal(0, serialPath.DispatchedParallelUpdates);
            Assert.True(parallelPath.ParallelUpdates > 0, parallelPath.ToString());
            Assert.Equal(parallelPath.ParallelUpdates, parallelPath.DispatchedParallelUpdates);
            Assert.Equal(1, serialPeak);
            // Every bound is released on the way: each cached solve after the first extends the transformed RHS.
            Assert.Equal(serialTrace.CachedExactSolves - 1, serialPath.RhsExtensions);
            Assert.Equal(serialPath.RhsExtensions, parallelPath.RhsExtensions);
            ActiveSetQpLargeProblemTests.AssertOptimal(problem.Covariance, parallel, problem.Lower, problem.Upper, problem.Means, problem.RiskAversion, ActiveSetQpLargeProblemTests.DoubleStationarity);
        }
    }

    // The rounded route reuses the Cholesky factor of the previous working set's reduced Hessian only while the reference
    // variable (the free variable of smallest variance) and the leading other variables are unchanged. On these
    // well-conditioned factor models the reference changes during many solves, as bounds bind and release; a prefix
    // factored against another reference belongs to a different matrix, and reusing it makes the subspace minimiser
    // fail to refine, handing the solve to the far more expensive exact route. Every one of these solves must stay on
    // the rounded route, as a solve that factors every working set afresh does.
    [Fact]
    public void RoundedRoute_OnWellConditionedFactorModels_NeverHandsOffToTheExactRoute()
    {
        var diverted = new List<string>();
        for (var sample = 0; sample < 60; sample++)
        {
            var problem = FactorModel(20261120 + sample);
            var trace = new ActiveSetQpSolver<double>.RouteTrace();
            var weights = ActiveSetQpSolver<double>.SolveMinVarianceUsingCore(problem.Covariance, problem.Lower, problem.Upper, trace: trace);
            if (trace.ExactRouteReason != ActiveSetQpSolver<double>.ExactRouteReason.None || trace.ExactChanges != 0)
            {
                diverted.Add(string.Create(CultureInfo.InvariantCulture,
                    $"seed {problem.Seed} n={weights.Length} box [{problem.Lower}, {problem.Upper}]: {trace.ExactRouteReason}, rounded changes {trace.RoundedChanges}, exact changes {trace.ExactChanges}"));
            }

            ActiveSetQpLargeProblemTests.AssertOptimal(problem.Covariance, weights, problem.Lower, problem.Upper, means: null, 1.0, ActiveSetQpLargeProblemTests.DoubleStationarity);
        }

        Assert.True(diverted.Count == 0, string.Join(Environment.NewLine, diverted));
    }

    /// <summary>
    /// <c>B·B′ + diag(d)</c> with 2 to 5 factors of loadings in [-0.15, 0.35] and specific variances in [0.002, 0.022]
    /// (n from 8 to 35), in a long-only box [0, 0.3] or a long-short box [-0.5, 0.5].
    /// </summary>
    private static QpProblem<double> FactorModel(int seed)
    {
        var rng = new System.Random(seed);
        var n = 8 + rng.Next(0, 28);
        var f = 2 + rng.Next(0, 4);
        var b = new double[n, f];
        for (var i = 0; i < n; i++)
        {
            for (var k = 0; k < f; k++)
            {
                b[i, k] = Math.Round((rng.NextDouble() * 0.5) - 0.15, 3);
            }
        }

        var cov = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            var specific = Math.Round(0.002 + (rng.NextDouble() * 0.02), 4);
            for (var j = 0; j <= i; j++)
            {
                var sum = i == j ? specific : 0.0;
                for (var k = 0; k < f; k++)
                {
                    sum += b[i, k] * b[j, k];
                }

                cov[i, j] = cov[j, i] = sum;
            }
        }

        var (lower, upper) = seed % 2 == 0 ? (0.0, 0.3) : (-0.5, 0.5);
        return new QpProblem<double>(cov, null, 1.0, lower, upper, seed);
    }

    private void CompareWithFreshElimination<T>(QpProblem<T> problem)
        where T : IFloatingPoint<T>
    {
        var (expected, fresh, freshPath) = problem.RunWithPath(new() { ForceExactRoute = true, FractionFreeOnly = true });
        var (actual, reused, reusedPath) = problem.RunWithPath(new() { ForceExactRoute = true });
        output.WriteLine(problem.Describe(actual, reused) + $", {reusedPath}");

        Assert.Equal(Bits(expected), Bits(actual));
        Assert.Equal(fresh.ExactChanges, reused.ExactChanges);
        Assert.Equal(fresh.ExactRollbacks, reused.ExactRollbacks);
        Assert.Equal(0, fresh.CachedExactSolves);
        Assert.True(reused.CachedExactSolves > 0, "the cached exact factor was never used: " + problem.Describe(actual, reused));
        Assert.Equal(0, freshPath.RhsExtensions);
        // The first cached solve has no previous transformed RHS, and a bound that binds (a removal) invalidates it.
        Assert.True(reusedPath.RhsExtensions > 0, "the O(n) RHS extension never ran: " + problem.Describe(actual, reused));
        Assert.True(reusedPath.RhsExtensions <= reused.CachedExactSolves - 1, problem.Describe(actual, reused));
    }

    // The order-100 long-short refusal of the calibrated cost control, on the production worker configuration (no test
    // override active): its exact route extends the transformed RHS on the cached factor, and dispatches the large row
    // updates to the parallel loop wherever more than one production worker is configured, running at least two and at
    // most min(4, ProcessorCount) rows of one update at once. These counts, unlike the cost control's wall-clock ceiling,
    // do not depend on host load.
    [Fact]
    public void ExactRoute_OnTheOrder100LongShortRefusal_ExtendsTheRhsAndDispatchesRowsOnTheProductionWorkers()
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

        ExactLinearSolveTests.WithProductionRows(() =>
        {
            Assert.Equal(Math.Min(4, Environment.ProcessorCount), ExactLinearSolve.WorkerLimit);
            var trace = new ActiveSetQpSolver<double>.RouteTrace();
            var before = PathCounts.Now();
            InvalidOperationException error = null!;
            var peak = ExactLinearSolveWorkerBoundTests.PeakConcurrentRows(
                () => error = Assert.Throws<InvalidOperationException>(() => ActiveSetQpSolver<double>.SolveMinVarianceUsingCore(cov, -0.5, 0.5, trace: trace)));
            var path = PathCounts.Now() - before;
            output.WriteLine($"order-100 refusal on {ExactLinearSolve.WorkerLimit} worker(s): exact changes {trace.ExactChanges}, cached exact solves {trace.CachedExactSolves}, {path}, peak rows at once {peak}");

            Assert.Contains("stationarity cannot be certified", error.Message, StringComparison.Ordinal);
            Assert.True(trace.CachedExactSolves > 0);
            // The exact route only releases bounds here: every cached solve after the first extends the transformed RHS.
            Assert.True(path.RhsExtensions > 0, path.ToString());
            Assert.Equal(trace.CachedExactSolves - 1, path.RhsExtensions);
            Assert.True((path.ParallelUpdates > 0) == (ExactLinearSolve.WorkerLimit > 1), path.ToString());
            Assert.Equal(path.ParallelUpdates, path.DispatchedParallelUpdates);
            ExactLinearSolveWorkerBoundTests.AssertWithinWorkerBound(peak, "order-100 refusal on the production workers");
        });
    }

    /// <summary>Per-thread path counts of the exact factor (see <see cref="ExactLinearSolveTests.ThreadPathCounts"/>).</summary>
    private readonly record struct PathCounts(int RhsExtensions, int ParallelUpdates, int DispatchedParallelUpdates)
    {
        public static PathCounts Now()
        {
            var (rhs, parallel, dispatched) = ExactLinearSolveTests.ThreadPathCounts();
            return new(rhs, parallel, dispatched);
        }

        public static PathCounts operator -(PathCounts a, PathCounts b) =>
            new(a.RhsExtensions - b.RhsExtensions, a.ParallelUpdates - b.ParallelUpdates, a.DispatchedParallelUpdates - b.DispatchedParallelUpdates);
    }

    private static QpProblem<T> Problem<T>(int n, int seed, bool meanVariance, int lowerPercent, int upperPercent)
        where T : IFloatingPoint<T>
    {
        // A three-factor model with an idiosyncratic floor of 1/100 of its trace: well conditioned.
        var source = ActiveSetQpExactFallbackCostTests.IllConditionedDecimal(n, 1e2, seed);
        var cov = new T[n, n];
        var means = new T[n];
        for (var i = 0; i < n; i++)
        {
            means[i] = T.CreateChecked((i % 5) - 2) / T.CreateChecked(20000);
            for (var j = 0; j < n; j++)
            {
                cov[i, j] = T.CreateChecked(source[i, j]);
            }
        }

        var hundred = T.CreateChecked(100);
        return new QpProblem<T>(cov, meanVariance ? means : null, T.CreateChecked(3), T.CreateChecked(lowerPercent) / hundred, T.CreateChecked(upperPercent) / hundred, seed);
    }

    private static string[] Bits<T>(T[] weights) => weights
        .Select(w => w is double d
            ? BitConverter.DoubleToInt64Bits(d).ToString("X16", CultureInfo.InvariantCulture)
            : string.Join("/", decimal.GetBits((decimal)(object)w!).Select(x => x.ToString("X8", CultureInfo.InvariantCulture))))
        .ToArray();

    private sealed record QpProblem<T>(T[,] Covariance, T[]? Means, T RiskAversion, T Lower, T Upper, int Seed)
        where T : IFloatingPoint<T>
    {
        public (T[] Weights, ActiveSetQpSolver<T>.RouteTrace Trace) Run(ActiveSetQpSolver<T>.RouteTrace trace)
        {
            var weights = Means is null
                ? ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(Covariance, Lower, Upper, trace: trace)
                : ActiveSetQpSolver<T>.SolveMeanVarianceUsingCore(Covariance, Means, RiskAversion, Lower, Upper, trace: trace);
            return (weights, trace);
        }

        public (T[] Weights, ActiveSetQpSolver<T>.RouteTrace Trace, PathCounts Path) RunWithPath(ActiveSetQpSolver<T>.RouteTrace trace)
        {
            var before = PathCounts.Now();
            var (weights, _) = Run(trace);
            return (weights, trace, PathCounts.Now() - before);
        }

        public int FreeCount(T[] weights) => weights.Count(w => w > Lower && w < Upper);

        public string Describe(T[] weights, ActiveSetQpSolver<T>.RouteTrace trace) => string.Create(CultureInfo.InvariantCulture,
            $"{typeof(T).Name} {(Means is null ? "min-variance" : "mean-variance")} n={Covariance.GetLength(0)} seed={Seed}: free {FreeCount(weights)}, exact changes {trace.ExactChanges}, cached exact solves {trace.CachedExactSolves}");
    }
}
