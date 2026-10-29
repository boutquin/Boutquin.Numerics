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

namespace Boutquin.Numerics.Internal;

/// <summary>
/// Fraction-free factors of successive positive-definite principal blocks of one immutable integer matrix.
/// Adding a variable extends the existing Bareiss triangle in quadratic work; removing one rebuilds its affected suffix. A solve
/// returns an exact rational vector; every division is exact, as in a fresh Bareiss elimination.
/// </summary>
internal sealed class ExactLinearSolve(BigInteger[,] matrix)
{
    private static readonly ParallelOptions s_parallelOptions = new() { MaxDegreeOfParallelism = Math.Min(4, Environment.ProcessorCount) };
    internal static int WorkerLimit => s_parallelOptions.MaxDegreeOfParallelism;

    /// <summary>Pivot updates that actually entered the parallel branch, for deterministic path verification.</summary>
    internal int ParallelUpdates { get; private set; }

    /// <summary>
    /// Parallel-branch pivot updates whose rows ran inside a <see cref="Parallel"/> loop's tasks. It records that the
    /// loop ran the rows, not how many workers ran them: the loop runs even a one-worker update inside its own task, so
    /// the count is the same at any worker count. <see cref="PeakConcurrentRows"/> observes the worker bound.
    /// </summary>
    internal int DispatchedParallelUpdates { get; private set; }

    /// <summary>Solves whose transformed right-hand side was extended from the previous one in O(n), not eliminated again.</summary>
    internal int RhsExtensions { get; private set; }

    /// <summary>
    /// The most rows of one pivot update that ran at the same time: at most <see cref="WorkerLimit"/> in the parallel
    /// branch, 1 in the serial one, 0 before any row update. A test diagnostic of the worker bound the loop applies; no
    /// result depends on it.
    /// </summary>
    internal int PeakConcurrentRows { get; private set; }

    /// <summary>Solves that found a variable of the factor removed and truncated the factor before it (test diagnostics).</summary>
    internal int Truncations { get; private set; }

    /// <summary>
    /// The leading factor rows those truncations kept, summed: at each, the rows of the variables before the first removed
    /// one, the only rows that do not depend on it (test diagnostics; a full rebuild keeps none).
    /// </summary>
    internal int RetainedRows { get; private set; }

    /// <summary>Rows appended to the triangle, each extending it by one variable in quadratic work (test diagnostics).</summary>
    internal int AppendedRows { get; private set; }

    // The same counts summed (the peak: maximised) over every instance on the current thread. A solver builds its factor
    // internally and calls it on its own thread, so a solve's path is observable at its entry point; other threads never
    // touch them.
    [ThreadStatic]
    private static int s_threadParallelUpdates;

    [ThreadStatic]
    private static int s_threadDispatchedParallelUpdates;

    [ThreadStatic]
    private static int s_threadRhsExtensions;

    [ThreadStatic]
    private static int s_threadPeakConcurrentRows;

    // Test-only: run by every row of a parallel update once the row counts as running, on the row's worker. It is read on
    // the thread that starts the update, so it reaches only the updates of solves on the thread that set it.
    [ThreadStatic]
    private static Action? s_threadRowEntryHook;

    internal static int ThreadParallelUpdates => s_threadParallelUpdates;

    internal static int ThreadDispatchedParallelUpdates => s_threadDispatchedParallelUpdates;

    internal static int ThreadRhsExtensions => s_threadRhsExtensions;

    internal static int ThreadPeakConcurrentRows => s_threadPeakConcurrentRows;

    internal static Action? ThreadRowEntryHook
    {
        get => s_threadRowEntryHook;
        set => s_threadRowEntryHook = value;
    }

    internal static void ResetThreadPeakConcurrentRows() => s_threadPeakConcurrentRows = 0;

    private readonly BigInteger[,] _triangle = new BigInteger[matrix.GetLength(0), matrix.GetLength(0)];
    private readonly List<int> _order = [];
    private BigInteger[]? _lastRhs;
    private BigInteger[]? _lastWork;
    private readonly BigInteger[] _oddDivisors = new BigInteger[matrix.GetLength(0)];
    private readonly int[] _divisorShifts = new int[matrix.GetLength(0)];

    internal bool TrySolve(int[] indices, BigInteger[] rhs, out BigInteger[] numerators, out BigInteger denominator)
    {
        numerators = [];
        denominator = BigInteger.One;
        for (var i = 0; i < _order.Count; i++)
        {
            if (Array.IndexOf(indices, _order[i]) < 0)
            {
                // Only the suffix depends on a removed variable. Retain the unchanged leading factor.
                _order.RemoveRange(i, _order.Count - i);
                _lastWork = null;
                Truncations++;
                RetainedRows += _order.Count;
                break;
            }
        }

        foreach (var index in indices)
        {
            if (!_order.Contains(index) && !Append(index))
            {
                _order.Clear();
                _lastWork = null;
                return false;
            }
        }

        var n = indices.Length;
        if (n == 0)
        {
            return false;
        }
        var work = new BigInteger[n];
        for (var i = 0; i < n; i++)
        {
            work[i] = rhs[Array.IndexOf(indices, _order[i])];
        }

        var originalRhs = (BigInteger[])work.Clone();
        if (TryExtendRhs(work))
        {
            RhsExtensions++;
            s_threadRhsExtensions++;
        }
        else
        {
            for (var k = 0; k < n; k++)
            {
                var pivot = _triangle[k, k];
                for (var i = k + 1; i < n; i++)
                {
                    work[i] = Divide((pivot * work[i]) - (_triangle[k, i] * work[k]), k - 1);
                }

            }
        }

        _lastRhs = originalRhs;
        _lastWork = (BigInteger[])work.Clone();

        denominator = _triangle[n - 1, n - 1];
        var solution = new BigInteger[n];
        for (var i = n - 1; i >= 0; i--)
        {
            var sum = denominator * work[i];
            for (var j = i + 1; j < n; j++)
            {
                sum -= _triangle[i, j] * solution[j];
            }

            solution[i] = Divide(sum, i);
        }

        numerators = new BigInteger[n];
        for (var i = 0; i < n; i++)
        {
            numerators[Array.IndexOf(indices, _order[i])] = solution[i];
        }
        return true;
    }

    // Releasing one bound changes the old rows of b by a multiple of the newly appended column, plus a common
    // radix-grid rescaling. Bareiss elimination is linear in b: recover those two exact coefficients, verify the
    // relation on every old row, then extend the transformed RHS in O(n), rather than eliminating it again.
    private bool TryExtendRhs(BigInteger[] work)
    {
        var n = work.Length - 1;
        if (_lastWork is null || _lastWork.Length != n || n < 2)
        {
            return false;
        }

        var column = new BigInteger[n];
        for (var i = 0; i < n; i++)
        {
            column[i] = matrix[_order[i], _order[n]];
        }

        BigInteger divisor = 0, scale = 0, add = 0;
        var first = 0;
        while (first < n && _lastRhs![first].IsZero && column[first].IsZero)
        {
            first++;
        }

        if (first == n)
        {
            return false;
        }

        for (var j = first + 1; j < n; j++)
        {
            divisor = (_lastRhs![first] * column[j]) - (_lastRhs[j] * column[first]);
            if (!divisor.IsZero)
            {
                scale = (work[first] * column[j]) - (work[j] * column[first]);
                add = (_lastRhs[first] * work[j]) - (_lastRhs[j] * work[first]);
                break;
            }
        }

        if (divisor.IsZero)
        {
            return false;
        }

        var common = BigInteger.GreatestCommonDivisor(divisor, BigInteger.GreatestCommonDivisor(scale, add));
        divisor /= common;
        scale /= common;
        add /= common;

        for (var i = 0; i < n; i++)
        {
            if (work[i] * divisor != (scale * _lastRhs![i]) + (add * column[i]))
            {
                return false;
            }
        }

        var tail = work[n];
        for (var i = 0; i < n; i++)
        {
            work[i] = ((scale * _lastWork[i]) + (add * _triangle[i, n])) / divisor;
            tail = Divide((_triangle[i, i] * tail) - (_triangle[i, n] * work[i]), i - 1);
        }

        work[n] = tail;
        return true;
    }

    private bool Append(int index)
    {
        var n = _order.Count;
        var column = new BigInteger[n];
        for (var i = 0; i < n; i++)
        {
            column[i] = matrix[_order[i], index];
        }

        var diagonal = matrix[index, index];
        for (var k = 0; k < n; k++)
        {
            var pivot = _triangle[k, k];
            if (s_parallelOptions.MaxDegreeOfParallelism > 1 && n >= 64 && n - k >= 24)
            {
                ParallelUpdates++;
                s_threadParallelUpdates++;
                // Each row owns one output and reads the same immutable pivot; exact integer arithmetic makes the
                // result independent of scheduling. The column is complete before the next pivot is processed.
                // Parallel.For runs every row inside its own replica tasks, so a row seeing a task other than the
                // caller's records that the loop ran it (at any worker count); the rows running at once give the
                // worker count. The loop's completion publishes both to this thread.
                var caller = Task.CurrentId;
                var dispatched = false;
                var running = 0;
                var peak = 0;
                var hook = s_threadRowEntryHook;
                Parallel.For(k + 1, n, s_parallelOptions, i =>
                {
                    var now = Interlocked.Increment(ref running);
                    for (var seen = Volatile.Read(ref peak); now > seen; seen = Volatile.Read(ref peak))
                    {
                        if (Interlocked.CompareExchange(ref peak, now, seen) == seen)
                        {
                            break;
                        }
                    }

                    hook?.Invoke();
                    if (Task.CurrentId != caller)
                    {
                        dispatched = true;
                    }

                    column[i] = Divide((pivot * column[i]) - (_triangle[k, i] * column[k]), k - 1);
                    Interlocked.Decrement(ref running);
                });
                if (dispatched)
                {
                    DispatchedParallelUpdates++;
                    s_threadDispatchedParallelUpdates++;
                }

                RecordPeakConcurrentRows(peak);
            }
            else
            {
                for (var i = k + 1; i < n; i++)
                {
                    column[i] = Divide((pivot * column[i]) - (_triangle[k, i] * column[k]), k - 1);
                }

                if (k + 1 < n)
                {
                    RecordPeakConcurrentRows(1);
                }
            }

            diagonal = Divide((pivot * diagonal) - (column[k] * column[k]), k - 1);
            _triangle[k, n] = column[k];
        }

        if (diagonal.Sign <= 0)
        {
            return false;
        }

        _triangle[n, n] = diagonal;
        _divisorShifts[n] = checked((int)BigInteger.TrailingZeroCount(diagonal));
        _oddDivisors[n] = diagonal >> _divisorShifts[n];
        _order.Add(index);
        AppendedRows++;
        return true;
    }

    private void RecordPeakConcurrentRows(int rows)
    {
        PeakConcurrentRows = Math.Max(PeakConcurrentRows, rows);
        s_threadPeakConcurrentRows = Math.Max(s_threadPeakConcurrentRows, rows);
    }

    private BigInteger Divide(BigInteger numerator, int pivot)
    {
        if (pivot < 0 || numerator.IsZero)
        {
            return numerator;
        }

        // The quotient is integral, so removing the divisor's power of two from both operands is exact, including
        // for negative numerators. Cache the odd divisor rather than repeatedly dividing by its zero low limbs.
        return (numerator >> _divisorShifts[pivot]) / _oddDivisors[pivot];
    }
}
