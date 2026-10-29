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

using System.Numerics;

using Boutquin.Numerics.Internal;
using Boutquin.Numerics.Tests.Unit.Solvers;

namespace Boutquin.Numerics.Tests.Unit.Internal;

/// <summary>
/// The worker bound of the exact factor's parallel row update, at most <c>min(4, ProcessorCount)</c> workers, observed
/// where the loop applies it: the most rows of one pivot update that run at once
/// (<see cref="ExactLinearSolve.PeakConcurrentRows"/>). A rendezvous holds the first rows of the first parallel update
/// until as many rows as the configured worker count are running, then holds them a little longer. A loop given that
/// many workers reaches that count and cannot start another row while they are held; a loop serialized to one worker
/// never runs two rows; a loop allowed more workers starts another row meanwhile. None of this depends on how fast the
/// host runs the rows.
/// </summary>
// The worker-count overrides and the held rows must not overlap the calibrated-ceiling timing tests.
[Collection(ActiveSetQpPerformanceCollection.Name)]
[Trait("Tier", "Unit")]
public sealed class ExactLinearSolveWorkerBoundTests
{
    [Fact]
    public void TrySolve_WithForcedParallelRows_RunsAtLeastTwoAndAtMostWorkerLimitRowsAtOnce()
    {
        const int n = 72;
        var (matrix, rhs) = DiagonallyDominant(n);
        ExactLinearSolveTests.WithParallelRows(() =>
        {
            var solver = new ExactLinearSolve(matrix);
            var peak = PeakConcurrentRows(() => Assert.True(solver.TrySolve(Enumerable.Range(0, n).ToArray(), rhs, out _, out _)));
            Assert.True(solver.ParallelUpdates > 0);
            Assert.Equal(peak, solver.PeakConcurrentRows);
            AssertWithinWorkerBound(peak, "forced parallel rows, order 72");
        });
    }

    [Fact]
    public void TrySolve_OnOneWorker_RunsOneRowAtATime()
    {
        const int n = 72;
        var (matrix, rhs) = DiagonallyDominant(n);
        ExactLinearSolveTests.WithSerialRows(() =>
        {
            var solver = new ExactLinearSolve(matrix);
            var peak = PeakConcurrentRows(() => Assert.True(solver.TrySolve(Enumerable.Range(0, n).ToArray(), rhs, out _, out _)));
            Assert.Equal(0, solver.ParallelUpdates);
            Assert.Equal(1, peak);
            Assert.Equal(1, solver.PeakConcurrentRows);
        });
    }

    /// <summary>
    /// Runs <paramref name="solve"/> on this thread with the row rendezvous installed for the current worker count and
    /// returns the most rows of one pivot update that ran at once over every factor it used on this thread.
    /// </summary>
    internal static int PeakConcurrentRows(Action solve)
    {
        var rendezvous = new RowRendezvous(ExactLinearSolve.WorkerLimit);
        ExactLinearSolve.ResetThreadPeakConcurrentRows();
        ExactLinearSolve.ThreadRowEntryHook = rendezvous.Enter;
        try
        {
            solve();
        }
        finally
        {
            ExactLinearSolve.ThreadRowEntryHook = null;
        }

        return ExactLinearSolve.ThreadPeakConcurrentRows;
    }

    /// <summary>
    /// With more than one worker configured, the rendezvoused update ran at least two rows at once and never more than
    /// the worker limit; with one worker, rows ran one at a time.
    /// </summary>
    internal static void AssertWithinWorkerBound(int peak, string context)
    {
        var limit = ExactLinearSolve.WorkerLimit;
        if (limit > 1)
        {
            Assert.True(peak >= 2 && peak <= limit, $"{context}: {peak} row(s) ran at once on {limit} worker(s)");
        }
        else
        {
            Assert.Equal(1, peak);
        }
    }

    // The first `workers` rows to arrive wait until that many have arrived: a loop given that many workers delivers them.
    // They then wait until one more row arrives or a grace period passes. A loop capped at `workers` cannot start another
    // row while they are all held, so only a loop allowed more workers ends the grace early, by running a row beyond the
    // cap. Every later row runs freely. Both waits are bounded, so a serialized loop runs its rows one at a time.
    private sealed class RowRendezvous(int workers)
    {
        private static readonly TimeSpan s_arrival = TimeSpan.FromSeconds(10);
        private static readonly TimeSpan s_excess = TimeSpan.FromMilliseconds(500);
        private int _arrivals;

        public void Enter()
        {
            if (Interlocked.Increment(ref _arrivals) > workers)
            {
                return;
            }

            SpinWait.SpinUntil(() => Volatile.Read(ref _arrivals) >= workers, s_arrival);
            SpinWait.SpinUntil(() => Volatile.Read(ref _arrivals) > workers, s_excess);
        }
    }

    // Strictly diagonally dominant and symmetric: every principal block is positive definite. Order 72 meets the parallel
    // branch's size conditions.
    private static (BigInteger[,] Matrix, BigInteger[] Rhs) DiagonallyDominant(int n)
    {
        var matrix = new BigInteger[n, n];
        var rhs = new BigInteger[n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                matrix[i, j] = i == j ? 2 * n : ((i + j) % 3) - 1;
                rhs[i] += matrix[i, j] * ((j % 7) - 3);
            }
        }

        return (matrix, rhs);
    }
}
