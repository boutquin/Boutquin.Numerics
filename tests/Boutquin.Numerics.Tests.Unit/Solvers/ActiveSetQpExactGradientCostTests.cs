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
using System.Reflection;

using Boutquin.Numerics.Solvers;

using Xunit.Abstractions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// The cost of the exact gradient <c>Σx</c> of <see cref="ActiveSetQpSolver{T}"/> on long-short problems, where the
/// feasible start releases about one variable per working-set change and every change evaluates the exact gradient
/// about four times. Formed with one allocating exact product and one exact sum per term, the gradient was more than
/// half the CPU time and about 70% of the allocation of a long-short solve of order 150 (a long-short mean-variance
/// solve of order 150 cost 1,100 to 1,600 times the calibration workload, and one of order 100 allocated about 690 MB).
/// The accumulated gradient must stay the same exact value, mantissa and exponent, as the term-by-term sum.
/// </summary>
[Collection(ActiveSetQpPerformanceCollection.Name)]
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpExactGradientCostTests(ITestOutputHelper output)
{
    /// <summary>
    /// Ceiling on the managed allocation of the long-short mean-variance solve of order 100 on its own thread. The
    /// term-by-term gradient allocates about 690 MB there; the accumulated one leaves about 220 MB.
    /// </summary>
    private const long AllocationCeilingBytes = 384L << 20;

    /// <summary>
    /// Ceiling on the managed allocation of the long-short mean-variance solve of order 150 on its own thread. The
    /// term-by-term gradient allocates about 2,400 MB there; the accumulated one leaves about 720 MB.
    /// </summary>
    private const long Order150AllocationCeilingBytes = 1280L << 20;

    [Fact]
    public void SolveMeanVariance_LongShortFactorModelOfOrder100_AllocatesBelowTheCeiling()
    {
        var (cov, means) = LongShortProblem(100);
        WarmUp();

        var trace = new ActiveSetQpSolver<double>.RouteTrace();
        var before = GC.GetAllocatedBytesForCurrentThread();
        var weights = ActiveSetQpSolver<double>.SolveMeanVarianceUsingCore(cov, means, 3.0, -0.5, 0.5, trace: trace);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        // The rounded route alone, which runs on the calling thread: the thread's allocation is the solve's.
        AssertRoundedRoute(trace);
        ActiveSetQpLargeProblemTests.AssertOptimal(cov, weights, -0.5, 0.5, means, 3.0, ActiveSetQpLargeProblemTests.DoubleStationarity);
        var line = string.Create(CultureInfo.InvariantCulture, $"long-short mean-variance n=100: {allocated / 1048576.0:F1} MB allocated over {trace.RoundedChanges} rounded changes; ceiling {AllocationCeilingBytes / 1048576.0:F0} MB");
        output.WriteLine(line);
        Assert.True(allocated <= AllocationCeilingBytes, line);
    }

    /// <summary>
    /// The order-150 solve of the same family, pinned by its deterministic work: the managed allocation of the solve on
    /// its own thread (the rounded route runs there alone), which the exact gradient dominates, and the length of its walk
    /// (about one working-set change per asset). Its process CPU time is reported and not asserted: on unchanged code it
    /// stayed between 1.40 and 1.58 s while the ratio to the calibration workload crossed the calibrated ceiling of 1000
    /// in 2 of 30 executions (up to 1015.5x), because that ratio sat at the ceiling whenever the calibration ran at its
    /// fastest; the allocation of the same solve is the same number on every host and every processor count.
    /// </summary>
    [Fact]
    public void SolveMeanVariance_LongShortFactorModelOfOrder150_StaysWithinItsDeterministicWork()
    {
        const int n = 150;
        var (cov, means) = LongShortProblem(n);
        WarmUp();

        var trace = new ActiveSetQpSolver<double>.RouteTrace();
        var t0 = Process.GetCurrentProcess().TotalProcessorTime;
        var before = GC.GetAllocatedBytesForCurrentThread();
        var weights = ActiveSetQpSolver<double>.SolveMeanVarianceUsingCore(cov, means, 3.0, -0.5, 0.5, trace: trace);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        var cpu = (Process.GetCurrentProcess().TotalProcessorTime - t0).TotalMilliseconds;

        AssertRoundedRoute(trace);
        ActiveSetQpLargeProblemTests.AssertOptimal(cov, weights, -0.5, 0.5, means, 3.0, ActiveSetQpLargeProblemTests.DoubleStationarity);
        var calibration = ActiveSetQpPerformanceTests.CalibrationMs();
        var line = string.Create(CultureInfo.InvariantCulture, $"long-short mean-variance n={n}: {allocated / 1048576.0:F1} MB allocated over {trace.RoundedChanges} rounded changes; ceiling {Order150AllocationCeilingBytes / 1048576.0:F0} MB; {cpu:F1} ms CPU = {cpu / calibration:F1}x the calibration workload ({calibration:F3} ms), reported and not asserted");
        output.WriteLine(line);
        Assert.True(trace.RoundedChanges <= 2 * n, line);
        Assert.True(allocated <= Order150AllocationCeilingBytes, line);
    }

    // The accumulated product against the term-by-term sum of the exact type's own product and sum operators, on seeded
    // matrices and vectors whose mantissas run to 200 bits (with trailing zeros) and whose exponents span from a few to
    // thousands of radix places, with zero entries, zero rows and rows whose partial sums cancel to exactly zero: every
    // component must carry the same mantissa and the same exponent.
    [Fact]
    public void ExactProduct_OnSeededMatricesOfMixedExponents_EqualsTheTermByTermSumInMantissaAndExponent()
    {
        var doubleRestarts = CompareWithTermByTerm<double>(seed: 20261101, radix: 2);
        var decimalRestarts = CompareWithTermByTerm<decimal>(seed: 20261102, radix: 10);
        var floatRestarts = CompareWithTermByTerm<float>(seed: 20261103, radix: 2);

        // Rows whose sum restarted after an exactly zero partial sum, so their exponent is not the row's smallest product
        // exponent, were compared for every type.
        Assert.True(doubleRestarts > 0 && decimalRestarts > 0 && floatRestarts > 0, string.Create(CultureInfo.InvariantCulture, $"restarted rows: double {doubleRestarts}, decimal {decimalRestarts}, float {floatRestarts}"));
    }

    // A row whose exponents span more than a million bits (beyond the accumulator's width) is still the term-by-term sum.
    [Fact]
    public void ExactProduct_OnARowOfExtremeExponentSpread_EqualsTheTermByTermSum()
    {
        var exact = new ExactType<double>();
        var matrix = exact.Matrix(1, 3);
        var vector = exact.Vector(3);
        matrix.SetValue(exact.Of(BigInteger.Parse("123456789012345678901", CultureInfo.InvariantCulture), -1_200_000), 0, 0);
        matrix.SetValue(exact.Of(-77, 0), 0, 1);
        matrix.SetValue(exact.Of(5, 3), 0, 2);
        vector.SetValue(exact.Of(3, 0), 0);
        vector.SetValue(exact.Of(BigInteger.One << 100, 7), 1);
        vector.SetValue(exact.Of(-9, 1), 2);

        var actual = exact.Times(matrix, vector);
        var expected = exact.TermByTerm(matrix, vector, 0);
        Assert.Equal(exact.Parts(expected), exact.Parts(actual.GetValue(0)!));
    }

    private static int CompareWithTermByTerm<T>(int seed, int radix)
        where T : IFloatingPoint<T>
    {
        var exact = new ExactType<T>();
        var rng = new System.Random(seed);
        var restarts = 0;
        for (var trial = 0; trial < 400; trial++)
        {
            var rows = 1 + rng.Next(10);
            var columns = 1 + rng.Next(12);
            var wide = trial % 3 == 0;
            var matrix = exact.Matrix(rows, columns);
            var vector = exact.Vector(columns);
            for (var j = 0; j < columns; j++)
            {
                vector.SetValue(RandomValue(exact, rng, radix, wide), j);
            }

            for (var i = 0; i < rows; i++)
            {
                var zeroRow = rng.Next(8) == 0;
                for (var j = 0; j < columns; j++)
                {
                    matrix.SetValue(zeroRow ? exact.Zero : RandomValue(exact, rng, radix, wide), i, j);
                }
            }

            // Rows whose leading products cancel exactly at a small exponent, followed by products of larger exponents:
            // the sum restarts at the exponent of the first product after the zero partial sum.
            if (columns >= 3 && rng.Next(2) == 0)
            {
                var i = rng.Next(rows);
                var j1 = rng.Next(columns - 2);
                var j2 = j1 + 1 + rng.Next(columns - j1 - 2);
                for (var j = 0; j < j2; j++)
                {
                    if (j != j1)
                    {
                        matrix.SetValue(exact.Zero, i, j);
                    }
                }

                var mantissa = RandomMantissa(rng, radix);
                matrix.SetValue(exact.Of(mantissa, -(radix == 2 ? 200 : 60)), i, j1);
                matrix.SetValue(exact.Of(-mantissa, -(radix == 2 ? 200 : 60)), i, j2);
                var shared = exact.Of(RandomMantissa(rng, radix), rng.Next(-5, 6));
                vector.SetValue(shared, j1);
                vector.SetValue(shared, j2);
                if (rng.Next(3) == 0)
                {
                    // Every product after the cancellation is zero, so the whole row sums to exactly zero.
                    for (var j = j2 + 1; j < columns; j++)
                    {
                        matrix.SetValue(exact.Zero, i, j);
                    }
                }
            }

            var actual = exact.Times(matrix, vector);
            for (var i = 0; i < rows; i++)
            {
                var expected = exact.TermByTerm(matrix, vector, i);
                var (expectedMantissa, expectedExponent) = exact.Parts(expected);
                var (actualMantissa, actualExponent) = exact.Parts(actual.GetValue(i)!);
                var label = string.Create(CultureInfo.InvariantCulture, $"{typeof(T).Name} trial {trial} row {i}: expected {expectedMantissa}·{radix}^{expectedExponent}, got {actualMantissa}·{radix}^{actualExponent}");
                Assert.True(expectedMantissa == actualMantissa && expectedExponent == actualExponent, label);
                if (!expectedMantissa.IsZero && expectedExponent > exact.SmallestProductExponent(matrix, vector, i))
                {
                    restarts++;
                }
            }
        }

        return restarts;
    }

    // A random exact value: zero one time in seven, otherwise a mantissa of 1 to 200 bits (sometimes with trailing radix
    // zeros) of either sign and an exponent of a few radix places or, in wide trials, of up to about a thousand.
    private static object RandomValue<T>(ExactType<T> exact, System.Random rng, int radix, bool wide)
        where T : IFloatingPoint<T>
    {
        if (rng.Next(7) == 0)
        {
            return exact.Zero;
        }

        var reach = radix == 2 ? (wide ? 1100 : 80) : (wide ? 300 : 30);
        return exact.Of(RandomMantissa(rng, radix), rng.Next(-reach, reach + 1));
    }

    private static BigInteger RandomMantissa(System.Random rng, int radix)
    {
        var bits = 1 + rng.Next(200);
        var bytes = new byte[((bits + 7) / 8) + 1];
        rng.NextBytes(bytes);
        bytes[^1] = 0;
        var mantissa = new BigInteger(bytes) >> ((8 * (bytes.Length - 1)) - bits);
        if (mantissa.IsZero)
        {
            mantissa = BigInteger.One;
        }

        if (rng.Next(3) == 0)
        {
            mantissa *= BigInteger.Pow(radix, 1 + rng.Next(radix == 2 ? 40 : 12));
        }

        return rng.Next(2) == 0 ? mantissa : -mantissa;
    }

    private static void AssertRoundedRoute(ActiveSetQpSolver<double>.RouteTrace trace)
    {
        Assert.Equal(0, trace.ExactChanges);
        Assert.Equal(0, trace.ExactConditioningTests);
        Assert.Equal(ActiveSetQpSolver<double>.ExactRouteReason.None, trace.ExactRouteReason);
    }

    // The committed three-factor model with means in [0.02, 0.10) at λ = 3 in the box [-0.5, 0.5]: the optimum holds
    // most assets strictly inside the box, so the walk from the feasible start makes about one change per asset.
    private static (double[,] Covariance, double[] Means) LongShortProblem(int n)
    {
        var cov = ActiveSetQpLargeProblemTests.FactorModel<double>(n, seed: 20261021, digits: 15);
        var rng = new System.Random(20261022);
        var means = new double[n];
        for (var i = 0; i < n; i++)
        {
            means[i] = 0.02 + (rng.NextDouble() * 0.08);
        }

        return (cov, means);
    }

    private static void WarmUp()
    {
        var (cov, means) = LongShortProblem(12);
        _ = ActiveSetQpSolver<double>.SolveMeanVariance(cov, means, 3.0, -0.5, 0.5);
    }

    /// <summary>The solver's private exact radix fraction for one <typeparamref name="T"/>, reached by reflection.</summary>
    private sealed class ExactType<T>
        where T : IFloatingPoint<T>
    {
        private const BindingFlags Any = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance;

        private readonly Type _exact;
        private readonly Type _productMatrix;
        private readonly MethodInfo _of;
        private readonly MethodInfo _multiply;
        private readonly MethodInfo _add;
        private readonly PropertyInfo _mantissa;
        private readonly PropertyInfo _exponent;

        public ExactType()
        {
            _exact = typeof(ActiveSetQpSolver<>).GetNestedType("Exact", Any)!.MakeGenericType(typeof(T));
            _productMatrix = typeof(ActiveSetQpSolver<>).GetNestedType("Exact", Any)!.GetNestedType("ProductMatrix", Any)!.MakeGenericType(typeof(T));
            _of = _exact.GetMethod("Of", Any)!;
            _multiply = _exact.GetMethod("op_Multiply", Any)!;
            _add = _exact.GetMethod("op_Addition", Any)!;
            _mantissa = _exact.GetProperty("Mantissa", Any)!;
            _exponent = _exact.GetProperty("Exponent", Any)!;
            Zero = Activator.CreateInstance(_exact)!;
        }

        public object Zero { get; }

        public object Of(BigInteger mantissa, int exponent) => _of.Invoke(null, [mantissa, exponent])!;

        public Array Matrix(int rows, int columns) => Array.CreateInstance(_exact, rows, columns);

        public Array Vector(int length) => Array.CreateInstance(_exact, length);

        public (BigInteger Mantissa, int Exponent) Parts(object value) => ((BigInteger)_mantissa.GetValue(value)!, (int)_exponent.GetValue(value)!);

        // The production product: the prepared matrix times the vector.
        public Array Times(Array matrix, Array vector)
        {
            var prepared = Activator.CreateInstance(_productMatrix, matrix)!;
            return (Array)_productMatrix.GetMethod("Times", Any)!.Invoke(prepared, [vector])!;
        }

        // The reference: the products a_ij·x_j added to zero in index order with the exact type's own operators.
        public object TermByTerm(Array matrix, Array vector, int i)
        {
            var sum = Zero;
            for (var j = 0; j < vector.Length; j++)
            {
                var product = _multiply.Invoke(null, [matrix.GetValue(i, j), vector.GetValue(j)])!;
                sum = _add.Invoke(null, [sum, product])!;
            }

            return sum;
        }

        // The smallest exponent among the row's nonzero products (int.MaxValue when there is none).
        public int SmallestProductExponent(Array matrix, Array vector, int i)
        {
            var smallest = int.MaxValue;
            for (var j = 0; j < vector.Length; j++)
            {
                var (am, ae) = Parts(matrix.GetValue(i, j)!);
                var (xm, xe) = Parts(vector.GetValue(j)!);
                if (!am.IsZero && !xm.IsZero)
                {
                    smallest = Math.Min(smallest, ae + xe);
                }
            }

            return smallest;
        }
    }
}
