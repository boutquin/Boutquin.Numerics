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
using System.Reflection;

using Boutquin.Numerics.Internal;
using Boutquin.Numerics.LinearAlgebra;
using Boutquin.Numerics.Tests.Unit.Solvers;

namespace Boutquin.Numerics.Tests.Unit.Internal;

// WithParallelRows / WithSerialRows change the solver's shared worker count; the non-parallel performance collection
// keeps them from overlapping the calibrated-ceiling timing tests.
[Collection(ActiveSetQpPerformanceCollection.Name)]
[Trait("Tier", "Unit")]
public sealed class ExactLinearSolveTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(70)]
    [InlineData(180)]
    public void TrySolve_SeededSystems_ReconstructsTheKnownRationalSolution(int bits)
    {
        var rng = new System.Random(20261022 + bits);
        for (var sample = 0; sample < 16; sample++)
        {
            var n = rng.Next(2, 25);
            var matrix = new BigInteger[n, n];
            var rhs = new BigInteger[n];
            var expected = new BigInteger[n];
            var denominator = (BigInteger.One << (bits + 3)) + 3;
            var coefficients = new BigInteger[n, n];
            for (var i = 0; i < n; i++)
            {
                expected[i] = rng.Next(-100, 101);
                for (var j = 0; j < n; j++)
                {
                    coefficients[i, j] = rng.Next(-10, 11);
                }
            }

            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    var gram = i == j ? BigInteger.One : BigInteger.Zero;
                    for (var k = 0; k < n; k++)
                    {
                        gram += coefficients[i, k] * coefficients[j, k];
                    }

                    matrix[i, j] = gram * denominator;
                    rhs[i] += gram * expected[j];
                }
            }

            var solver = new ExactLinearSolve(matrix);
            Assert.True(solver.TrySolve(Enumerable.Range(0, n).ToArray(), rhs, out var actual, out var divisor));
            Assert.True(divisor.Sign > 0);
            for (var i = 0; i < n; i++)
            {
                Assert.Equal(expected[i] * divisor, actual[i] * denominator);
            }

            // Grow in a shuffled order, then remove and reorder variables. Each block has the same independently
            // chosen rational solution restricted to that block; the RHS changes by the newly released column.
            var order = Enumerable.Range(0, n).ToArray();
            rng.Shuffle(order);
            var previousCount = n;
            foreach (var count in Enumerable.Range(2, n - 1).Concat([n / 2, n]))
            {
                var indices = order.Take(count).Reverse().ToArray();
                // Some growth steps deliberately change every old unknown independently of the appended column:
                // the affine-RHS fast path must decline them and re-eliminate the new RHS.
                BigInteger Target(int index) => expected[index] + (count % 3 == 1 ? count * (index % 5) * (index % 5) : 0);
                var blockRhs = new BigInteger[count];
                for (var i = 0; i < count; i++)
                {
                    for (var j = 0; j < count; j++)
                    {
                        blockRhs[i] += (matrix[indices[i], indices[j]] / denominator) * Target(indices[j]);
                    }
                }

                var extensions = solver.RhsExtensions;
                Assert.True(solver.TrySolve(indices, blockRhs, out actual, out divisor));
                for (var i = 0; i < count; i++)
                {
                    Assert.Equal(Target(indices[i]) * divisor, actual[i] * denominator);
                }

                // A growth by one variable (from a block of two or more) whose old rows keep the previous target is the
                // relation the O(n) extension accepts; a changed target, a removal or a growth by several must be declined.
                var affineGrowth = count == previousCount + 1 && previousCount >= 2 && count % 3 != 1 && previousCount % 3 != 1;
                Assert.Equal(affineGrowth ? extensions + 1 : extensions, solver.RhsExtensions);
                previousCount = count;
            }
        }
    }

    // Removing the last variable of a solved block and releasing two others, with the old rows of the new RHS equal to
    // the previous RHS, satisfies the affine relation that lets a transformed RHS be extended in place: the variable that
    // now stands where the removed one stood inherits its RHS entry. The transformed RHS of the removed variable must
    // not survive the removal, or it is applied to its successor and the solve returns a wrong exact vector.
    [Fact]
    public void TrySolve_RemovalThenGrowthByTwoUnderThePreviousRhs_SolvesTheNewBlockExactly()
    {
        const int n = 5;
        var matrix = new BigInteger[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                // Strictly diagonally dominant and symmetric: every principal block is positive definite.
                matrix[i, j] = matrix[j, i] = i == j ? 20 + (3 * i) : 1 + ((i + j) % 3) + (i * j % 2);
            }
        }

        var solver = new ExactLinearSolve(matrix);
        AssertSolvesExactly(solver, matrix, [0, 1, 2], [5, 7, 11]);
        AssertSolvesExactly(solver, matrix, [0, 1, 3, 4], [5, 7, 11, 13]);

        // The removal invalidated the transformed RHS: the relation must be declined and the RHS eliminated afresh.
        Assert.Equal(0, solver.RhsExtensions);

        // Removing variable 2 kept the two factor rows before it; only the released 3 and 4 were factored after the first
        // block's three rows.
        Assert.Equal((1, 2, 5), Rows(solver));
    }

    // Growing a solved block by one variable, with the old rows of the new RHS equal to c·(previous RHS) + a·(column of
    // the appended variable), is the relation the O(n) extension accepts: it must be taken, exactly once per such solve.
    // An unrelated RHS on the next growth must be declined. Every result is checked against the defining equations.
    [Theory]
    [InlineData(20261201)]
    [InlineData(20261202)]
    [InlineData(20261203)]
    public void TrySolve_GrowthByOneUnderAnAffineRhs_ExtendsTheTransformedRhsAndSolvesExactly(int seed)
    {
        var rng = new System.Random(seed);
        for (var sample = 0; sample < 12; sample++)
        {
            var n = rng.Next(6, 25);
            var matrix = RandomPositiveDefinite(n, rng);
            var solver = new ExactLinearSolve(matrix);
            var order = Enumerable.Range(0, n).ToArray();
            rng.Shuffle(order);
            var k = rng.Next(2, n - 2);
            var previousRhs = Enumerable.Range(0, k).Select(_ => (BigInteger)rng.Next(-50, 51)).ToArray();
            AssertSolvesExactly(solver, matrix, order.Take(k).ToArray(), previousRhs);
            Assert.Equal(0, solver.RhsExtensions);

            var indices = order.Take(k + 1).ToArray();
            var scale = (BigInteger)(rng.Next(1, 4) * (rng.Next(0, 2) == 0 ? -1 : 1));
            var add = (BigInteger)rng.Next(-3, 4);
            var rhs = new BigInteger[k + 1];
            for (var i = 0; i < k; i++)
            {
                rhs[i] = (scale * previousRhs[i]) + (add * matrix[indices[i], indices[k]]);
            }

            rhs[k] = rng.Next(-50, 51);
            AssertSolvesExactly(solver, matrix, indices, rhs);
            Assert.Equal(1, solver.RhsExtensions);

            // Every old row changes independently of the appended column: no affine relation holds.
            var unrelated = order.Take(k + 2).ToArray();
            var other = new BigInteger[k + 2];
            for (var i = 0; i < k + 2; i++)
            {
                other[i] = rhs.ElementAtOrDefault(i) + (i * i) + 1;
            }

            AssertSolvesExactly(solver, matrix, unrelated, other);
            Assert.Equal(1, solver.RhsExtensions);

            // Each growth extended the existing triangle by its new variables alone: k + 2 rows factored in all, none
            // truncated.
            Assert.Equal((0, 0, k + 2), Rows(solver));
        }
    }

    // The seeded family of the case above: a block is solved, any one of its variables is removed and two others are
    // released, and the new RHS on the retained-plus-first-released rows is c·(previous RHS) + a·(column of the last
    // released variable), the exact shape the RHS-reuse relation accepts. Every result is checked against the defining
    // equations A·x = d·b evaluated exactly (the solution of a positive-definite block is unique), never against
    // another run of the solver.
    [Theory]
    [InlineData(20261101)]
    [InlineData(20261102)]
    [InlineData(20261103)]
    [InlineData(20261104)]
    public void TrySolve_RemovalThenGrowthByTwoUnderAnAffineRhs_SolvesEveryBlockExactly(int seed)
    {
        var rng = new System.Random(seed);
        for (var sample = 0; sample < 12; sample++)
        {
            var n = rng.Next(8, 25);
            var matrix = RandomPositiveDefinite(n, rng);
            var solver = new ExactLinearSolve(matrix);
            var order = Enumerable.Range(0, n).ToArray();
            rng.Shuffle(order);
            var k = rng.Next(3, n - 2);
            var previous = order.Take(k).ToArray();
            var previousRhs = Enumerable.Range(0, k).Select(_ => (BigInteger)rng.Next(-50, 51)).ToArray();
            AssertSolvesExactly(solver, matrix, previous, previousRhs);
            Assert.Equal((0, 0, k), Rows(solver));

            // Remove the variable at position p of the solved order and release the next two of the shuffled order.
            var p = sample % 3 == 0 ? k - 1 : rng.Next(0, k);
            var (first, last) = (order[k], order[k + 1]);
            var indices = previous.Where((_, i) => i != p).Append(first).Append(last).ToArray();
            var scale = (BigInteger)(rng.Next(1, 4) * (rng.Next(0, 2) == 0 ? -1 : 1));
            var add = (BigInteger)rng.Next(-3, 4);
            var rhs = new BigInteger[k + 1];
            for (var i = 0; i < k; i++)
            {
                rhs[i] = (scale * previousRhs[i]) + (add * matrix[indices[i], last]);
            }

            rhs[k] = rng.Next(-50, 51);
            AssertSolvesExactly(solver, matrix, indices, rhs);

            // The removal invalidated the transformed RHS, so the relation must be declined.
            Assert.Equal(0, solver.RhsExtensions);

            // Only the factor rows from position p on depend on the removed variable: the factor kept the p rows before
            // it, rebuilt the k − 1 − p after it and appended the two released variables, 2k + 1 − p rows in all.
            Assert.Equal((1, p, (2 * k) + 1 - p), Rows(solver));
        }
    }

    [Fact]
    public void TrySolve_FailedAppendInvalidatesThePreviousRhsBeforeRebuilding()
    {
        var matrix = RandomPositiveDefinite(6, new System.Random(20261023));
        matrix[4, 4] = -1;
        var solver = new ExactLinearSolve(matrix);
        BigInteger[] previousRhs = [5, 7, 11];
        AssertSolvesExactly(solver, matrix, [0, 1, 2], previousRhs);

        // The negative pivot fails after a valid prefix. Rebuild a different block whose first three RHS rows
        // satisfy the old affine reuse relation: stale transformed rows must not survive the failed append.
        Assert.False(solver.TrySolve([0, 1, 2, 4], [5, 7, 11, 13], out _, out _));
        int[] indices = [1, 2, 3, 5];
        BigInteger[] rhs = [0, 0, 0, 17];
        for (var i = 0; i < 3; i++)
        {
            rhs[i] = previousRhs[i] + matrix[indices[i], 5];
        }
        AssertSolvesExactly(solver, matrix, indices, rhs);
        Assert.Equal(0, solver.RhsExtensions);
    }

    [Theory]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(66)]
    public void TrySolve_ParallelThreshold_EntersTheExpectedBranchAndSolvesExactly(int n)
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

        WithParallelRows(() =>
        {
            var solver = new ExactLinearSolve(matrix);
            AssertSolvesExactly(solver, matrix, Enumerable.Range(0, n).ToArray(), rhs);
            Assert.Equal(n > 64, solver.ParallelUpdates > 0);
            // Every update that entered the branch ran its rows inside the parallel loop's tasks (at any worker count:
            // ExactLinearSolveWorkerBoundTests observes how many rows ran at once).
            Assert.Equal(solver.ParallelUpdates, solver.DispatchedParallelUpdates);
        });
        WithSerialRows(() =>
        {
            var solver = new ExactLinearSolve(matrix);
            AssertSolvesExactly(solver, matrix, Enumerable.Range(0, n).ToArray(), rhs);
            Assert.Equal(0, solver.ParallelUpdates);
            Assert.Equal(0, solver.DispatchedParallelUpdates);
        });
    }

    // The production worker configuration, read with no test override active: min(4, ProcessorCount) workers.
    [Fact]
    public void WorkerLimit_WithNoOverrideActive_IsMinFourProcessorCount() =>
        WithProductionRows(() => Assert.Equal(Math.Min(4, Environment.ProcessorCount), ExactLinearSolve.WorkerLimit));

    // On the production worker configuration a large factor's row updates are dispatched to the parallel loop whenever
    // more than one worker is configured (every host with two or more processors), run on at least two and at most
    // WorkerLimit workers at once (one at a time on a one-processor host), and solve exactly.
    [Fact]
    public void TrySolve_OnTheProductionWorkers_DispatchesTheLargeRowUpdatesInParallel()
    {
        const int n = 72;
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

        WithProductionRows(() =>
        {
            var solver = new ExactLinearSolve(matrix);
            var peak = ExactLinearSolveWorkerBoundTests.PeakConcurrentRows(() => AssertSolvesExactly(solver, matrix, Enumerable.Range(0, n).ToArray(), rhs));
            Assert.True((solver.ParallelUpdates > 0) == (ExactLinearSolve.WorkerLimit > 1), $"parallel updates {solver.ParallelUpdates} on {ExactLinearSolve.WorkerLimit} worker(s)");
            Assert.Equal(solver.ParallelUpdates, solver.DispatchedParallelUpdates);
            ExactLinearSolveWorkerBoundTests.AssertWithinWorkerBound(peak, "production workers, order 72");
        });
    }

    [Fact]
    public void TrySolve_ZeroRightHandSide_IsExact()
    {
        var solver = new ExactLinearSolve(new BigInteger[,] { { 3, 2 }, { 2, 4 } });
        Assert.True(solver.TrySolve([0, 1], [0, 0], out var x, out _));
        Assert.All(x, value => Assert.Equal(BigInteger.Zero, value));
    }

    [Fact]
    public void TrySolve_SingularBlock_RequestsFractionFreeFallback()
    {
        var solver = new ExactLinearSolve(new BigInteger[,] { { 1, 1 }, { 1, 1 } });
        Assert.False(solver.TrySolve([0, 1], [1, 2], out _, out _));
    }

    [Fact]
    public void TrySolve_LargeFactorParallelRows_MatchAKnownRationalSolution()
    {
        const int n = 72;
        var denominator = (BigInteger.One << 48) + 7;
        var matrix = new BigInteger[n, n];
        var rhs = new BigInteger[n];
        var expected = new BigInteger[n];
        for (var i = 0; i < n; i++)
        {
            expected[i] = (i % 7) - 3;
        }

        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                // Strictly diagonally dominant and symmetric: positive definite, with a dense, multi-limb factor.
                var entry = i == j ? 2 * n : ((i + j) % 3) - 1;
                matrix[i, j] = entry * denominator;
                rhs[i] += entry * expected[j];
            }
        }

        // Order 72 meets the parallel branch's size conditions; enabling at least two workers makes the test take that
        // branch on a host that reports one processor as well.
        WithParallelRows(() =>
        {
            var solver = new ExactLinearSolve(matrix);
            Assert.True(solver.TrySolve(Enumerable.Range(0, n).ToArray(), rhs, out var actual, out var divisor));
            for (var i = 0; i < n; i++)
            {
                Assert.Equal(expected[i] * divisor, actual[i] * denominator);
            }
        });
    }

    [Fact]
    public void CholeskyPrefixReuse_MatchesEveryBitOfFreshFactors()
    {
        ComparePrefixes<double>();
        ComparePrefixes<decimal>();
    }

    private static void ComparePrefixes<T>()
        where T : IFloatingPoint<T>
    {
        const int n = 18;
        var matrix = new T[n, n];
        var rng = new System.Random(20261023);
        for (var i = 0; i < n; i++)
        {
            matrix[i, i] = T.CreateChecked(4 + i);
            for (var j = 0; j < i; j++)
            {
                matrix[i, j] = matrix[j, i] = T.CreateChecked(rng.Next(-10, 11)) / T.CreateChecked(100);
            }
        }

        var expected = CholeskyDecomposition<T>.Decompose(matrix);
        for (var prefix = 0; prefix <= n; prefix++)
        {
            var previous = new T[prefix, prefix];
            for (var i = 0; i < prefix; i++)
            {
                for (var j = 0; j <= i; j++)
                {
                    previous[i, j] = expected[i, j];
                }
            }

            var actual = CholeskyDecomposition<T>.DecomposeWithPrefix(matrix, previous, prefix);
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    if (typeof(T) == typeof(double))
                    {
                        Assert.Equal(BitConverter.DoubleToInt64Bits((double)(object)expected[i, j]), BitConverter.DoubleToInt64Bits((double)(object)actual[i, j]));
                    }
                    else
                    {
                        Assert.Equal(decimal.GetBits((decimal)(object)expected[i, j]), decimal.GetBits((decimal)(object)actual[i, j]));
                    }
                }
            }
        }
    }

    /// <summary>
    /// Runs <paramref name="action"/> with the bounded parallel row update of <see cref="ExactLinearSolve"/> enabled
    /// (at least two workers), so a test of that branch exercises it on a host that reports a single processor too.
    /// </summary>
    internal static void WithParallelRows(Action action) => WithRowWorkers(saved => Math.Max(2, saved), action);

    /// <summary>Runs <paramref name="action"/> with the row update of <see cref="ExactLinearSolve"/> on one worker (its serial branch).</summary>
    internal static void WithSerialRows(Action action) => WithRowWorkers(_ => 1, action);

    // The worker count lives in the solver's shared ParallelOptions instance. It is changed under one lock, so two tests
    // never interleave their save and restore, and it is restored afterwards. Exact integer arithmetic makes every
    // result independent of it; a concurrent solve only runs on more or fewer workers meanwhile.
    private static void WithRowWorkers(Func<int, int> workers, Action action)
    {
        var field = typeof(ExactLinearSolve).GetField("s_parallelOptions", BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(field);
        var options = Assert.IsType<ParallelOptions>(field.GetValue(null));
        lock (s_rowWorkersGate)
        {
            var saved = options.MaxDegreeOfParallelism;
            options.MaxDegreeOfParallelism = workers(saved);
            try
            {
                action();
            }
            finally
            {
                options.MaxDegreeOfParallelism = saved;
            }
        }
    }

    private static readonly object s_rowWorkersGate = new();

    /// <summary>
    /// Runs <paramref name="action"/> on the production worker configuration of <see cref="ExactLinearSolve"/>: under
    /// the override lock, so no test override is active meanwhile, and without changing it.
    /// </summary>
    internal static void WithProductionRows(Action action)
    {
        lock (s_rowWorkersGate)
        {
            action();
        }
    }

    // The per-thread totals of the exact factor's path counters (parallel updates that entered the branch, those whose
    // rows were dispatched to the parallel loop's tasks, accepted O(n) RHS extensions) over every instance a solve
    // creates on this thread: read before and after a solve at the solver's entry point.
    internal static (int RhsExtensions, int ParallelUpdates, int DispatchedParallelUpdates) ThreadPathCounts() =>
        (ExactLinearSolve.ThreadRhsExtensions, ExactLinearSolve.ThreadParallelUpdates, ExactLinearSolve.ThreadDispatchedParallelUpdates);

    // The factor's row counts: removals that truncated it, the leading rows those truncations kept, and the rows appended
    // (each extends the triangle by one row).
    internal static (int Truncations, int RetainedRows, int AppendedRows) Rows(ExactLinearSolve solver) =>
        (solver.Truncations, solver.RetainedRows, solver.AppendedRows);

    // A·x = d·b on the block, evaluated exactly, with d > 0: the defining equations of the unique rational solution.
    private static void AssertSolvesExactly(ExactLinearSolve solver, BigInteger[,] matrix, int[] indices, BigInteger[] rhs)
    {
        Assert.True(solver.TrySolve(indices, rhs, out var numerators, out var denominator));
        Assert.True(denominator.Sign > 0);
        for (var r = 0; r < indices.Length; r++)
        {
            BigInteger sum = 0;
            for (var c = 0; c < indices.Length; c++)
            {
                sum += matrix[indices[r], indices[c]] * numerators[c];
            }

            Assert.Equal(rhs[r] * denominator, sum);
        }
    }

    // I + C·C′ with small integer C: symmetric positive definite, dense, with every principal block positive definite.
    private static BigInteger[,] RandomPositiveDefinite(int n, System.Random rng)
    {
        var c = new int[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var k = 0; k < n; k++)
            {
                c[i, k] = rng.Next(-6, 7);
            }
        }

        var matrix = new BigInteger[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                BigInteger sum = i == j ? 1 : 0;
                for (var k = 0; k < n; k++)
                {
                    sum += c[i, k] * c[j, k];
                }

                matrix[i, j] = matrix[j, i] = sum;
            }
        }

        return matrix;
    }
}
