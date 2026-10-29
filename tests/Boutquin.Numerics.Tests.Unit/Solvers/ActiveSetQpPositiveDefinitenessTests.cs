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

using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// The positive-definiteness requirement of <see cref="ActiveSetQpSolver{T}"/> for <c>λ &gt; 0</c> refuses exactly the
/// covariances whose exact leading principal minors are not all positive, with the documented
/// <see cref="InvalidOperationException"/>. The solver first tries to prove positive definiteness in <c>double</c> and
/// falls back to exact elimination; these cases sit on both sides of the boundary, within a rounding error of it, where a
/// floating-point decision that is not certified would accept a singular or indefinite covariance.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpPositiveDefinitenessTests
{
    private const string Refusal = "Covariance matrix must be positive definite (leading principal minor of order ";

    public static TheoryData<double[,], int> DoubleBoundaryRefusals => new()
    {
        // Singular positive semidefinite: the order-2 minor is exactly 0.
        { new double[,] { { 1, 1 }, { 1, 1 } }, 2 },

        // Indefinite by one unit in the last place: the order-2 minor is -2^-52, the smallest eigenvalue about -1.1e-16.
        { new double[,] { { 1, 1 }, { 1, 1 - Math.ScaleB(1.0, -53) } }, 2 },

        // Rank two of order five (v·v′ + w·w′): every minor from order 3 is exactly 0.
        { RankTwo(5), 3 },

        // A repeated row and column of a positive-definite block, the copy's diagonal lowered by 2^-40.
        { new double[,] { { 2, 1, 2 }, { 1, 2, 1 }, { 2, 1, 2 - Math.ScaleB(1.0, -40) } }, 3 },
    };

    [Theory]
    [MemberData(nameof(DoubleBoundaryRefusals))]
    public void NotPositiveDefiniteWithinARoundingErrorOfTheBoundary_IsRefusedWithTheExactMinorOrder(double[,] cov, int order)
    {
        var n = cov.GetLength(0);
        var means = new double[n];
        for (var i = 0; i < n; i++)
        {
            means[i] = 0.01 * (i + 1);
        }

        var minVariance = Assert.Throws<InvalidOperationException>(() => ActiveSetQpSolver<double>.SolveMinVariance(cov, 0.0, 1.0));
        var meanVariance = Assert.Throws<InvalidOperationException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(cov, means, 2.0, 0.0, 1.0));
        Assert.Equal(Refusal + order + " is not positive).", minVariance.Message);
        Assert.Equal(Refusal + order + " is not positive).", meanVariance.Message);
    }

    [Fact]
    public void DecimalNotPositiveDefiniteWithinARoundingErrorOfTheBoundary_IsRefusedWithTheExactMinorOrder()
    {
        var singular = new decimal[,] { { 1m, 1m }, { 1m, 1m } };
        var indefinite = new decimal[,] { { 1m, 1m }, { 1m, 0.9999999999999999999999999999m } };
        var repeated = new decimal[,] { { 2m, 1m, 2m }, { 1m, 2m, 1m }, { 2m, 1m, 1.99999999999999999999m } };
        foreach (var (cov, order) in new[] { (singular, 2), (indefinite, 2), (repeated, 3) })
        {
            var means = new decimal[cov.GetLength(0)];
            var minVariance = Assert.Throws<InvalidOperationException>(() => ActiveSetQpSolver<decimal>.SolveMinVariance(cov, 0m, 1m));
            var meanVariance = Assert.Throws<InvalidOperationException>(() => ActiveSetQpSolver<decimal>.SolveMeanVariance(cov, means, 2m, 0m, 1m));
            Assert.Equal(Refusal + order + " is not positive).", minVariance.Message);
            Assert.Equal(Refusal + order + " is not positive).", meanVariance.Message);
        }
    }

    [Fact]
    public void PositiveDefiniteWithinARoundingErrorOfTheBoundary_IsAccepted()
    {
        // Order-2 minors 2^-52 (double) and 1e-28 (decimal): positive definite, far below any floating-point margin.
        var d = ActiveSetQpSolver<double>.SolveMinVariance(new double[,] { { 1, 1 }, { 1, 1 + Math.ScaleB(1.0, -52) } }, 0.0, 1.0);
        var m = ActiveSetQpSolver<decimal>.SolveMinVariance(new decimal[,] { { 1m, 1m }, { 1m, 1.0000000000000000000000000001m } }, 0m, 1m);

        Assert.Equal([1.0, 0.0], d);
        Assert.Equal([1m, 0m], m);
    }

    /// <summary>
    /// 300 seeded covariances per type of order 2 to 6 that straddle positive definiteness (singular positive
    /// semidefinite, shifted by ± a few units in the last place, repeated rows, couplings across up to 60 binary or 6
    /// decimal orders of magnitude, random symmetric): the solver refuses a covariance for positive definiteness exactly
    /// when this test's own exact Sylvester computation on the input values finds a leading principal minor that is
    /// not positive.
    /// </summary>
    [Fact]
    public void SeededCovariancesStraddlingPositiveDefiniteness_AreRefusedExactlyWhenAnExactMinorIsNotPositive()
    {
        var rng = new System.Random(20261020);
        int doubleRefused = 0, decimalRefused = 0;
        for (var c = 0; c < 300; c++)
        {
            var cov = StraddlingDouble(rng, c);
            var expected = !ExactlyPositiveDefinite(cov, Exact);
            Assert.True(expected == RefusedForPositiveDefiniteness(() => ActiveSetQpSolver<double>.SolveMinVariance(cov, 0.0, 1.0)), "double case " + c);
            doubleRefused += expected ? 1 : 0;

            var dec = StraddlingDecimal(rng, c);
            var decExpected = !ExactlyPositiveDefinite(dec, Exact);
            Assert.True(decExpected == RefusedForPositiveDefiniteness(() => ActiveSetQpSolver<decimal>.SolveMinVariance(dec, 0m, 1m)), "decimal case " + c);
            decimalRefused += decExpected ? 1 : 0;
        }

        // Both sides of the boundary are populated.
        Assert.InRange(doubleRefused, 60, 240);
        Assert.InRange(decimalRefused, 60, 240);
    }

    /// <summary>
    /// 150 seeded covariances per type of order 8 to 24 around the resolution of the double-double certificate that
    /// decides what the double certificate cannot: dependent assets whose Schur complements are ±2^-k (double, k up to
    /// 100) or ±10^-k (decimal, k up to 28) or exactly 0, a repeated asset with its variance shifted either way or not
    /// at all, rounded sample covariances of fewer observations than assets, near-duplicate pairs, and dependent assets
    /// under a radix scaling. The solver refuses a covariance for positive definiteness, naming an order, exactly when
    /// this test's own exact computation on the input values finds that order to be the first leading principal minor
    /// that is not positive; and the double-double certificate decides both accepted and refused cases of both types,
    /// so a certificate that accepted, refused or named an order wrongly would be seen here.
    /// </summary>
    [Fact]
    public void SeededCovariancesAroundTheDoubleDoubleCertificate_AreDecidedAsTheExactMinorsDecide()
    {
        var rng = new System.Random(20261050);
        var decided = new int[2, 2]; // [type, refused]: cases decided by the double-double certificate
        for (var c = 0; c < 150; c++)
        {
            var cov = AroundDoubleDouble<double>(rng, c);
            var expected = FirstNonPositiveMinor(cov, Exact);
            var (order, route) = Decision(cov, 0.0, 1.0);
            Assert.True(expected == order, "double case " + c + ": expected " + expected + ", solver " + order + " (" + route + ")");
            decided[0, expected > 0 ? 1 : 0] += route == ActiveSetQpSolver<double>.PositiveDefinitenessRoute.DoubleDoubleCertificate ? 1 : 0;

            var dec = AroundDoubleDouble<decimal>(rng, c);
            var decExpected = FirstNonPositiveMinor(dec, Exact);
            var (decOrder, decRoute) = Decision(dec, 0m, 1m);
            Assert.True(decExpected == decOrder, "decimal case " + c + ": expected " + decExpected + ", solver " + decOrder + " (" + decRoute + ")");
            decided[1, decExpected > 0 ? 1 : 0] += decRoute == ActiveSetQpSolver<decimal>.PositiveDefinitenessRoute.DoubleDoubleCertificate ? 1 : 0;
        }

        Assert.True(decided[0, 0] >= 10 && decided[0, 1] >= 20 && decided[1, 0] >= 20 && decided[1, 1] >= 20,
            "double-double decisions: double accepted " + decided[0, 0] + ", refused " + decided[0, 1] + "; decimal accepted " + decided[1, 0] + ", refused " + decided[1, 1]);
    }

    /// <summary>
    /// The certified <c>double</c> test of positive definiteness, the solve's first decision, runs once per solve at the
    /// documented cost of O(n³) operations: its Cholesky factorisation and residual bound execute about <c>n³/3</c> loop
    /// bodies, counted inside the loops as they run (here asserted between <c>n³/4</c> and <c>n³/2</c>), at two orders,
    /// at both guaranteed types and through both objectives; on a covariance it cannot prove (a repeated asset, refused
    /// by the later tests) it still runs once and no more. A loop that ran longer than its documented range, even without
    /// changing a value, raises the count.
    /// </summary>
    [Theory]
    [InlineData(40)]
    [InlineData(120)]
    public void DoubleCertificate_RunsOncePerSolveWithinCubicWork(int n)
    {
        var source = ActiveSetQpExactFallbackCostTests.IllConditionedDecimal(n, 1e3, seed: 20261090);
        var repeated = new decimal[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                // The last asset repeats the first: singular, so no certificate in double can prove it.
                repeated[i, j] = source[i == n - 1 ? 0 : i, j == n - 1 ? 0 : j];
            }
        }

        AssertOneCubicCertificate(source, repeated, d => d);
        AssertOneCubicCertificate(source, repeated, d => (double)d);
    }

    private static void AssertOneCubicCertificate<T>(decimal[,] source, decimal[,] repeated, Func<decimal, T> convert)
        where T : IFloatingPoint<T>
    {
        var n = source.GetLength(0);
        var means = new T[n];
        for (var i = 0; i < n; i++)
        {
            means[i] = T.CreateChecked((i % 5) - 2) / T.CreateChecked(1000);
        }

        foreach (var (input, positiveDefinite) in new[] { (source, true), (repeated, false) })
        {
            var cov = new T[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    cov[i, j] = convert(input[i, j]);
                }
            }

            foreach (var meanVariance in new[] { false, true })
            {
                var trace = new ActiveSetQpSolver<T>.RouteTrace();
                var label = string.Create(CultureInfo.InvariantCulture, $"{typeof(T).Name} n={n} {(positiveDefinite ? "positive definite" : "repeated asset")} {(meanVariance ? "mean-variance" : "min-variance")}");
                try
                {
                    _ = meanVariance
                        ? ActiveSetQpSolver<T>.SolveMeanVarianceUsingCore(cov, means, T.CreateChecked(4), T.Zero, T.One, trace: trace)
                        : ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(cov, T.Zero, T.One, trace: trace);
                    Assert.True(positiveDefinite, label + ": solved");
                }
                catch (InvalidOperationException ex) when (ex.Message.StartsWith(Refusal, StringComparison.Ordinal))
                {
                    Assert.False(positiveDefinite, label + ": " + ex.Message);
                }

                Assert.True(positiveDefinite == (trace.PositiveDefinitenessRoute == ActiveSetQpSolver<T>.PositiveDefinitenessRoute.DoubleCertificate), label + ", decided by " + trace.PositiveDefinitenessRoute);
                var certificate = trace.DoubleCertificate;
                var work = certificate.Iterations;
                var cube = (long)n * n * n;
                Assert.True(certificate.Runs == 1 && certificate.RepeatedRuns == 0, label + ": the certificate ran " + certificate.Runs + " times");
                Assert.True(certificate.OrderCubes == cube, label + ": recorded order cubes " + certificate.OrderCubes + " against n³ = " + cube);
                Assert.True(work <= cube / 2 && (!positiveDefinite || work >= cube / 4), label + ": " + work + " loop bodies against n³ = " + cube);
            }
        }
    }

    // The order of the positive-definiteness refusal of a min-variance solve (0 when the covariance passes the test,
    // whatever the solve does next) and the route that decided it.
    private static (int Order, ActiveSetQpSolver<T>.PositiveDefinitenessRoute Route) Decision<T>(T[,] cov, T lower, T upper)
        where T : IFloatingPoint<T>
    {
        var trace = new ActiveSetQpSolver<T>.RouteTrace();
        try
        {
            _ = ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(cov, lower, upper, trace: trace);
        }
        catch (InvalidOperationException ex) when (ex.Message.StartsWith(Refusal, StringComparison.Ordinal))
        {
            var order = int.Parse(ex.Message.AsSpan(Refusal.Length, ex.Message.IndexOf(' ', Refusal.Length) - Refusal.Length), System.Globalization.CultureInfo.InvariantCulture);
            return (order, trace.PositiveDefinitenessRoute);
        }
        catch (InvalidOperationException)
        {
            // Another refusal: the covariance passed the test.
        }

        return (0, trace.PositiveDefinitenessRoute);
    }

    // Kinds by case index (orders 8 to 24): 0 dependent assets, 1 a repeated asset, 2 a rounded sample covariance,
    // 3 a near-duplicate pair, 4 kind 0 under a radix scaling D·A·D. Shifts are ±2^-k (k in [8, 100]) for binary types
    // and ±10^-k (k in [3, 28]) for decimal, or exactly 0; every value is rounded once into T.
    private static T[,] AroundDoubleDouble<T>(System.Random rng, int c)
        where T : IFloatingPoint<T>
    {
        var isDecimal = typeof(T) == typeof(decimal);
        var n = 8 + rng.Next(17);
        var kind = c % 5;
        // A positive shift small enough to defeat the double certificate yet representable beside the entries, or a
        // shift that is zero or negative down to far below the working precision.
        T Positive() => isDecimal ? T.CreateChecked(Pow10(-(8 + rng.Next(18)))) : T.CreateChecked(Math.ScaleB(1.0, -(24 + rng.Next(21))));
        T NonPositive() => rng.Next(4) == 0
            ? T.Zero
            : -(isDecimal ? T.CreateChecked(Pow10(-(3 + rng.Next(26)))) : T.CreateChecked(Math.ScaleB(1.0, -(8 + rng.Next(93)))));
        T Shift() => rng.Next(2) == 0 ? Positive() : NonPositive();

        T[,] Factor(int size)
        {
            var sixteenth = T.One / T.CreateChecked(16);
            var f = new T[size, 3];
            var d = new T[size];
            for (var i = 0; i < size; i++)
            {
                f[i, 0] = T.CreateChecked(rng.Next(8, 24)) * sixteenth;
                f[i, 1] = T.CreateChecked(rng.Next(-8, 9)) * sixteenth;
                f[i, 2] = T.CreateChecked(rng.Next(-8, 9)) * sixteenth;
                d[i] = T.CreateChecked(rng.Next(3, 13)) / T.CreateChecked(256);
            }

            var g = new T[size, size];
            for (var i = 0; i < size; i++)
            {
                for (var j = 0; j <= i; j++)
                {
                    var sum = i == j ? d[i] : T.Zero;
                    for (var k = 0; k < 3; k++)
                    {
                        sum += f[i, k] * f[j, k];
                    }

                    g[i, j] = g[j, i] = sum;
                }
            }

            return g;
        }

        var a = new T[n, n];
        switch (kind)
        {
            case 0:
            case 4:
                {
                    var r = 1 + rng.Next(n - 1);
                    var lead = Factor(r);
                    var coefficients = new T[n, r];
                    for (var t = r; t < n; t++)
                    {
                        for (var k = 0; k < r; k++)
                        {
                            coefficients[t, k] = T.CreateChecked(rng.Next(-4, 5)) / T.CreateChecked(4);
                        }
                    }

                    var column = new T[n, r];
                    for (var i = 0; i < r; i++)
                    {
                        for (var j = 0; j < r; j++)
                        {
                            a[i, j] = lead[i, j];
                        }
                    }

                    for (var t = r; t < n; t++)
                    {
                        for (var i = 0; i < r; i++)
                        {
                            var sum = T.Zero;
                            for (var k = 0; k < r; k++)
                            {
                                sum += lead[i, k] * coefficients[t, k];
                            }

                            column[t, i] = sum;
                            a[t, i] = a[i, t] = sum;
                        }
                    }

                    // Positive shifts before the dependent asset at index stop (none when stop = n), a non-positive one there.
                    var stop = r + rng.Next(n - r + 1);
                    for (var t = r; t < n; t++)
                    {
                        for (var u = r; u <= t; u++)
                        {
                            var sum = T.Zero;
                            for (var k = 0; k < r; k++)
                            {
                                sum += coefficients[t, k] * column[u, k];
                            }

                            a[t, u] = a[u, t] = t != u ? sum : sum + (t < stop ? Positive() : t == stop ? NonPositive() : Shift());
                        }
                    }

                    break;
                }

            case 1:
                {
                    var lead = Factor(n - 1);
                    var copy = rng.Next(n - 1);
                    for (var i = 0; i < n - 1; i++)
                    {
                        for (var j = 0; j < n - 1; j++)
                        {
                            a[i, j] = lead[i, j];
                        }

                        a[n - 1, i] = a[i, n - 1] = lead[copy, i];
                    }

                    a[n - 1, n - 1] = lead[copy, copy] + Shift();
                    break;
                }

            case 2:
                {
                    var observations = 2 + rng.Next(n + 4);
                    var x = new T[observations, n];
                    for (var t = 0; t < observations; t++)
                    {
                        for (var i = 0; i < n; i++)
                        {
                            x[t, i] = T.CreateChecked(rng.Next(-500, 501)) / T.CreateChecked(10000);
                        }
                    }

                    var mean = new T[n];
                    for (var i = 0; i < n; i++)
                    {
                        var sum = T.Zero;
                        for (var t = 0; t < observations; t++)
                        {
                            sum += x[t, i];
                        }

                        mean[i] = sum / T.CreateChecked(observations);
                    }

                    for (var i = 0; i < n; i++)
                    {
                        for (var j = 0; j <= i; j++)
                        {
                            var sum = T.Zero;
                            for (var t = 0; t < observations; t++)
                            {
                                sum += (x[t, i] - mean[i]) * (x[t, j] - mean[j]);
                            }

                            a[i, j] = a[j, i] = sum / T.CreateChecked(Math.Max(1, observations - 1));
                        }
                    }

                    break;
                }

            default:
                {
                    var lead = Factor(n);
                    var first = rng.Next(n - 1);
                    var second = first + 1 + rng.Next(n - 1 - first);
                    for (var i = 0; i < n; i++)
                    {
                        for (var j = 0; j < n; j++)
                        {
                            var ii = i == second ? first : i;
                            var jj = j == second ? first : j;
                            a[i, j] = lead[ii, jj];
                        }
                    }

                    // The pair shares every covariance; its variances keep the idiosyncratic part of the first and a shift.
                    a[second, second] = lead[first, first] + Shift();
                    break;
                }
        }

        if (kind == 4)
        {
            var e = new int[n];
            for (var i = 0; i < n; i++)
            {
                e[i] = isDecimal ? rng.Next(-3, 4) : rng.Next(-30, 31);
            }

            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j <= i; j++)
                {
                    var scale = isDecimal ? T.CreateChecked(Pow10(e[i] + e[j])) : T.CreateChecked(Math.ScaleB(1.0, e[i] + e[j]));
                    a[i, j] = a[j, i] = a[i, j] * scale;
                }
            }
        }

        return a;
    }

    // The order of the first leading principal minor of the exact input values that is not positive, 0 when every
    // one is positive: fraction-free elimination on the integer matrix over the finest common grid.
    private static int FirstNonPositiveMinor<TValue>(TValue[,] a, Func<TValue, (BigInteger Mantissa, int Exponent, int Radix)> exact)
    {
        var n = a.GetLength(0);
        var m = IntegerMatrix(a, exact);
        var previous = BigInteger.One;
        for (var k = 0; k < n; k++)
        {
            if (m[k, k].Sign <= 0)
            {
                return k + 1;
            }

            for (var i = k + 1; i < n; i++)
            {
                for (var j = k + 1; j < n; j++)
                {
                    m[i, j] = ((m[k, k] * m[i, j]) - (m[i, k] * m[k, j])) / previous;
                }
            }

            previous = m[k, k];
        }

        return 0;
    }

    private static bool RefusedForPositiveDefiniteness<T>(Func<T[]> solve)
    {
        try
        {
            _ = solve();
            return false;
        }
        catch (InvalidOperationException ex) when (ex.Message.StartsWith(Refusal, StringComparison.Ordinal))
        {
            return true;
        }
        catch (InvalidOperationException)
        {
            // Another refusal (for example a numerically singular deciding block): the covariance passed the test.
            return false;
        }
    }

    private static double[,] RankTwo(int n)
    {
        var cov = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                cov[i, j] = ((i + 1) * (j + 1)) + ((i % 2 == 0 ? 1 : -1) * (j % 2 == 0 ? 1 : -1));
            }
        }

        return cov;
    }

    // Kinds by case index: 0 singular G·G′ (rank below the order), 1 G·G′ ± 2^-k on the diagonal, 2 a repeated row
    // and column ± 2^-k on the copy's diagonal, 3 kind 1 under a radix scaling D·A·D with exponents in [-30, 30],
    // 4 a random symmetric matrix with a dominant-ish diagonal.
    private static double[,] StraddlingDouble(System.Random rng, int c)
    {
        var n = 2 + rng.Next(5);
        var kind = c % 5;
        var rank = kind == 0 ? 1 + rng.Next(n - 1) : n;
        var g = new double[n, rank];
        for (var i = 0; i < n; i++)
        {
            for (var k = 0; k < rank; k++)
            {
                g[i, k] = rng.Next(-3, 4);
            }
        }

        var a = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                var sum = 0.0;
                for (var k = 0; k < rank; k++)
                {
                    sum += g[i, k] * g[j, k];
                }

                a[i, j] = a[j, i] = sum;
            }
        }

        var delta = (rng.Next(2) == 0 ? -1.0 : 1.0) * Math.ScaleB(1.0, -(20 + rng.Next(36)));
        switch (kind)
        {
            case 1:
            case 3:
                for (var i = 0; i < n; i++)
                {
                    a[i, i] += delta;
                }

                break;
            case 2:
                for (var j = 0; j < n; j++)
                {
                    a[n - 1, j] = a[j, n - 1] = a[0, j];
                }

                a[n - 1, n - 1] = a[0, 0] + delta;
                a[0, n - 1] = a[n - 1, 0] = a[0, 0];
                break;
            case 4:
                for (var i = 0; i < n; i++)
                {
                    for (var j = 0; j <= i; j++)
                    {
                        a[i, j] = a[j, i] = i == j ? rng.NextDouble() * 2.0 : (rng.NextDouble() * 2.0) - 1.0;
                    }
                }

                break;
        }

        if (kind == 3)
        {
            var e = new int[n];
            for (var i = 0; i < n; i++)
            {
                e[i] = rng.Next(-30, 31);
            }

            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    a[i, j] = Math.ScaleB(a[i, j], e[i] + e[j]);
                }
            }
        }

        return a;
    }

    // The decimal analogue: shifts of ± 10^-k with k in [8, 26] and decimal scalings with exponents in [-3, 3].
    private static decimal[,] StraddlingDecimal(System.Random rng, int c)
    {
        var n = 2 + rng.Next(5);
        var kind = c % 5;
        var rank = kind == 0 ? 1 + rng.Next(n - 1) : n;
        var g = new decimal[n, rank];
        for (var i = 0; i < n; i++)
        {
            for (var k = 0; k < rank; k++)
            {
                g[i, k] = rng.Next(-3, 4);
            }
        }

        var a = new decimal[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                var sum = 0m;
                for (var k = 0; k < rank; k++)
                {
                    sum += g[i, k] * g[j, k];
                }

                a[i, j] = a[j, i] = sum;
            }
        }

        var delta = (rng.Next(2) == 0 ? -1m : 1m) * Pow10(-(8 + rng.Next(19)));
        switch (kind)
        {
            case 1:
            case 3:
                for (var i = 0; i < n; i++)
                {
                    a[i, i] += delta;
                }

                break;
            case 2:
                for (var j = 0; j < n; j++)
                {
                    a[n - 1, j] = a[j, n - 1] = a[0, j];
                }

                a[n - 1, n - 1] = a[0, 0] + delta;
                a[0, n - 1] = a[n - 1, 0] = a[0, 0];
                break;
            case 4:
                for (var i = 0; i < n; i++)
                {
                    for (var j = 0; j <= i; j++)
                    {
                        a[i, j] = a[j, i] = (decimal)(i == j ? rng.NextDouble() * 2.0 : (rng.NextDouble() * 2.0) - 1.0);
                    }
                }

                break;
        }

        if (kind == 3)
        {
            var e = new int[n];
            for (var i = 0; i < n; i++)
            {
                e[i] = rng.Next(-3, 4);
            }

            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j <= i; j++)
                {
                    a[i, j] = a[j, i] = a[i, j] * Pow10(e[i] + e[j]);
                }
            }
        }

        return a;
    }

    private static decimal Pow10(int k)
    {
        var x = 1m;
        for (var i = 0; i < Math.Abs(k); i++)
        {
            x = k > 0 ? x * 10m : x / 10m;
        }

        return x;
    }

    // An entry as an exact (integer mantissa, exponent, radix) triple.
    private static (BigInteger Mantissa, int Exponent, int Radix) Exact(double x)
    {
        if (x == 0.0)
        {
            return (BigInteger.Zero, 0, 2);
        }

        var bits = BitConverter.DoubleToInt64Bits(x);
        var biased = (int)((bits >> 52) & 0x7FF);
        var fraction = bits & 0xFFFFFFFFFFFFFL;
        var mantissa = biased == 0 ? fraction : fraction | (1L << 52);
        var exponent = (biased == 0 ? 1 : biased) - 1075;
        return (bits < 0 ? -(BigInteger)mantissa : mantissa, exponent, 2);
    }

    private static (BigInteger Mantissa, int Exponent, int Radix) Exact(decimal x)
    {
        Span<int> parts = stackalloc int[4];
        decimal.GetBits(x, parts);
        var magnitude = (new BigInteger((uint)parts[2]) << 64) | (new BigInteger((uint)parts[1]) << 32) | (uint)parts[0];
        var scale = (parts[3] >> 16) & 0xFF;
        return (parts[3] < 0 ? -magnitude : magnitude, -scale, 10);
    }

    // Sylvester's criterion on the exact values: the integer matrix over the finest common grid, then every pivot of
    // fraction-free elimination (the leading principal minors of that integer matrix) must be positive.
    private static bool ExactlyPositiveDefinite<TValue>(TValue[,] a, Func<TValue, (BigInteger Mantissa, int Exponent, int Radix)> exact) =>
        FirstNonPositiveMinor(a, exact) == 0;

    // The exact input values as integers over their finest common radix grid (a positive multiple of the matrix).
    private static BigInteger[,] IntegerMatrix<TValue>(TValue[,] a, Func<TValue, (BigInteger Mantissa, int Exponent, int Radix)> exact)
    {
        var n = a.GetLength(0);
        var entries = new (BigInteger Mantissa, int Exponent, int Radix)[n, n];
        var finest = int.MaxValue;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                entries[i, j] = exact(a[i, j]);
                if (!entries[i, j].Mantissa.IsZero)
                {
                    finest = Math.Min(finest, entries[i, j].Exponent);
                }
            }
        }

        var m = new BigInteger[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var (mantissa, exponent, radix) = entries[i, j];
                m[i, j] = mantissa.IsZero ? BigInteger.Zero : mantissa * BigInteger.Pow(radix, exponent - finest);
            }
        }

        return m;
    }
}
