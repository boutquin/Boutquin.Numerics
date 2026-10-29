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

using Xunit.Abstractions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// <see cref="ActiveSetQpSolver{T}"/> on problems well above the corpus's largest order (n = 12 for double, 13 for
/// decimal): the process CPU time of a solve of a well-conditioned problem whose optimum holds every asset, divided by the
/// calibration workload of <see cref="ActiveSetQpPerformanceTests"/>, stays below a fixed ceiling. The feasible start puts the
/// whole budget on the first assets and the iteration releases the others one at a time, so the free block grows to
/// the full order. A conditioning margin evaluated from scratch in exact arithmetic on every working set (about n^5 in
/// total: 7 s at n = 60, 293 s at n = 120) exceeds the ceiling by more than an order of magnitude; the ceiling leaves
/// several times the cost of the solve with a certified margin.
/// </summary>
[Collection(ActiveSetQpPerformanceCollection.Name)]
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpLargeProblemTests(ITestOutputHelper output)
{
    /// <summary>Ceiling on a large solve's time as a multiple of the calibration workload.</summary>
    internal const double CalibratedCeiling = 1000.0;

    [Fact]
    public void SolveMinVariance_SixtyEquicorrelatedAssets_SolvesExactlyWithinTheCalibratedCeiling()
    {
        const int n = 60;
        var cov = Equicorrelated(n, 0.04, 0.02);
        _ = ActiveSetQpSolver<double>.SolveMinVariance(Equicorrelated(4, 0.04, 0.02), 0.0, 1.0);

        var (weights, ms) = TimedCpu(() => ActiveSetQpSolver<double>.SolveMinVariance(cov, 0.0, 1.0));

        // The optimum of an equicorrelated problem is 1/n each.
        Assert.All(weights, w => Assert.Equal(1.0 / n, w, 1e-15));
        AssertWithinCeiling("double equicorrelated n=60", ms);
    }

    [Fact]
    public void SolveMinVariance_FiftySeededDiagonallyDominantAssets_IsFeasibleWithinTheCalibratedCeiling()
    {
        const int n = 50;
        var cov = DiagonallyDominant<double>(n, seed: 20261016, digits: 15);
        _ = ActiveSetQpSolver<double>.SolveMinVariance(DiagonallyDominant<double>(4, seed: 1, digits: 15), 0.0, 1.0);

        var (weights, ms) = TimedCpu(() => ActiveSetQpSolver<double>.SolveMinVariance(cov, 0.0, 0.2));

        Assert.Equal(1.0, weights.Sum(), 1e-12);
        Assert.All(weights, w => Assert.InRange(w, 0.0, 0.2));
        Assert.All(weights, w => Assert.True(w > 0.0));
        AssertOptimal(cov, weights, 0.0, 0.2, means: null, 1.0, DoubleStationarity);
        AssertWithinCeiling("double diagonally dominant n=50", ms);
    }

    [Fact]
    public void SolveMinVariance_FiftySeededDiagonallyDominantDecimalAssets_IsFeasibleWithinTheCalibratedCeiling()
    {
        const int n = 50;
        var cov = DiagonallyDominant<decimal>(n, seed: 20261017, digits: 14);
        _ = ActiveSetQpSolver<decimal>.SolveMinVariance(DiagonallyDominant<decimal>(4, seed: 2, digits: 14), 0m, 1m);

        var (weights, ms) = TimedCpu(() => ActiveSetQpSolver<decimal>.SolveMinVariance(cov, 0m, 1m));

        Assert.True(Math.Abs(weights.Sum() - 1m) <= 1e-25m, weights.Sum().ToString(CultureInfo.InvariantCulture));
        Assert.All(weights, w => Assert.True(w > 0m && w <= 1m, w.ToString(CultureInfo.InvariantCulture)));
        AssertOptimal(cov, weights, 0m, 1m, means: null, 1m, DecimalStationarity);
        AssertWithinCeiling("decimal diagonally dominant n=50", ms);
    }

    // The positive-definiteness test of the covariance runs once per λ > 0 solve. Decided by exact fraction-free
    // elimination of the whole matrix alone it costs about n^5 (8 s at n = 100, 19 s at n = 120 for a dense factor
    // model, more than 95% of the solve), several times the ceiling from n = 80 on.
    [Fact]
    public void SolveMinVariance_HundredSeededFactorModelAssets_IsFeasibleWithinTheCalibratedCeiling()
    {
        const int n = 100;
        var cov = FactorModel<double>(n, seed: 20261020, digits: 15);
        _ = ActiveSetQpSolver<double>.SolveMinVariance(FactorModel<double>(4, seed: 3, digits: 15), 0.0, 1.0);

        var (weights, ms) = TimedCpu(() => ActiveSetQpSolver<double>.SolveMinVariance(cov, 0.0, 0.15));

        Assert.Equal(1.0, weights.Sum(), 1e-12);
        Assert.All(weights, w => Assert.InRange(w, 0.0, 0.15));
        AssertOptimal(cov, weights, 0.0, 0.15, means: null, 1.0, DoubleStationarity);
        AssertWithinCeiling("double factor model n=100", ms);
    }

    [Fact]
    public void SolveMeanVariance_HundredTwentySeededFactorModelAssets_IsFeasibleWithinTheCalibratedCeiling()
    {
        const int n = 120;
        var cov = FactorModel<double>(n, seed: 20261021, digits: 15);
        var rng = new System.Random(20261022);
        var means = new double[n];
        for (var i = 0; i < n; i++)
        {
            means[i] = 0.02 + (rng.NextDouble() * 0.08);
        }

        _ = ActiveSetQpSolver<double>.SolveMeanVariance(FactorModel<double>(4, seed: 4, digits: 15), [0.05, 0.06, 0.07, 0.08], 3.0, 0.0, 1.0);

        var (weights, ms) = TimedCpu(() => ActiveSetQpSolver<double>.SolveMeanVariance(cov, means, 3.0, 0.0, 0.15));

        Assert.Equal(1.0, weights.Sum(), 1e-12);
        Assert.All(weights, w => Assert.InRange(w, 0.0, 0.15));
        AssertOptimal(cov, weights, 0.0, 0.15, means, 3.0, DoubleStationarity);
        AssertWithinCeiling("double factor model mean-variance n=120", ms);
    }

    [Fact]
    public void SolveMinVariance_SeventySeededFactorModelDecimalAssets_IsFeasibleWithinTheCalibratedCeiling()
    {
        const int n = 70;
        var cov = FactorModel<decimal>(n, seed: 20261023, digits: 14);
        _ = ActiveSetQpSolver<decimal>.SolveMinVariance(FactorModel<decimal>(4, seed: 5, digits: 14), 0m, 1m);

        var (weights, ms) = TimedCpu(() => ActiveSetQpSolver<decimal>.SolveMinVariance(cov, 0m, 0.15m));

        Assert.True(Math.Abs(weights.Sum() - 1m) <= 1e-25m, weights.Sum().ToString(CultureInfo.InvariantCulture));
        Assert.All(weights, w => Assert.True(w >= 0m && w <= 0.15m, w.ToString(CultureInfo.InvariantCulture)));
        AssertOptimal(cov, weights, 0m, 0.15m, means: null, 1m, DecimalStationarity);
        AssertWithinCeiling("decimal factor model n=70", ms);
    }

    // The decimal counterpart of the double n = 100 case: decided by exact elimination alone the positive-definiteness
    // test costs about 12 s at this order (about n^5), several times the ceiling, so a decimal covariance that the
    // certified pre-test no longer decides is visible here.
    [Fact]
    public void SolveMinVariance_HundredSeededFactorModelDecimalAssets_IsOptimalWithinTheCalibratedCeiling()
    {
        const int n = 100;
        var cov = FactorModel<decimal>(n, seed: 20261024, digits: 14);
        _ = ActiveSetQpSolver<decimal>.SolveMinVariance(FactorModel<decimal>(4, seed: 6, digits: 14), 0m, 1m);

        var (weights, ms) = TimedCpu(() => ActiveSetQpSolver<decimal>.SolveMinVariance(cov, 0m, 0.15m));

        Assert.True(Math.Abs(weights.Sum() - 1m) <= 1e-25m, weights.Sum().ToString(CultureInfo.InvariantCulture));
        Assert.All(weights, w => Assert.True(w >= 0m && w <= 0.15m, w.ToString(CultureInfo.InvariantCulture)));
        AssertOptimal(cov, weights, 0m, 0.15m, means: null, 1m, DecimalStationarity);
        AssertWithinCeiling("decimal factor model n=100", ms);
    }

    // The certified double Cholesky, not a later test, decides these well-conditioned covariances. A regression that
    // stops it certifying one type (the decimal conversion failing, say) changes no weight and only moves the decision
    // to a costlier test, so the route is asserted directly.
    [Fact]
    public void PositiveDefinitenessOfWellConditionedFactorModels_IsDecidedByTheDoubleCertificateForDecimalAndDouble()
    {
        foreach (var (n, seed) in new[] { (70, 20261023), (100, 20261024) })
        {
            var trace = new ActiveSetQpSolver<decimal>.RouteTrace();
            _ = ActiveSetQpSolver<decimal>.SolveMinVarianceUsingCore(FactorModel<decimal>(n, seed, digits: 14), 0m, 0.15m, trace: trace);
            Assert.Equal(ActiveSetQpSolver<decimal>.PositiveDefinitenessRoute.DoubleCertificate, trace.PositiveDefinitenessRoute);
        }

        var doubleTrace = new ActiveSetQpSolver<double>.RouteTrace();
        _ = ActiveSetQpSolver<double>.SolveMinVarianceUsingCore(FactorModel<double>(100, seed: 20261020, digits: 15), 0.0, 0.15, trace: doubleTrace);
        Assert.Equal(ActiveSetQpSolver<double>.PositiveDefinitenessRoute.DoubleCertificate, doubleTrace.PositiveDefinitenessRoute);
    }

    /// <summary>Stationarity tolerance of a double result, relative to each row's scale.</summary>
    internal const double DoubleStationarity = 1e-12;

    /// <summary>Stationarity tolerance of a decimal result, relative to each row's scale.</summary>
    internal const double DecimalStationarity = 1e-24;

    /// <summary>
    /// Asserts the first-order optimality conditions of <c>min (λ/2)·w′Σw − μ′w</c> (with <paramref name="means"/>
    /// <see langword="null"/>, <c>min w′Σw</c>) on <c>1′w = 1</c> and <c>lower ≤ w ≤ upper</c>, independently of the
    /// solver. The gradient <c>g = λΣw − μ</c> is formed exactly from the returned weights (integer arithmetic on the
    /// radix fractions of the inputs). The free weights (strictly inside the box) must share one gradient value
    /// <c>ν</c>, a weight at its lower bound must have <c>g_i ≥ ν</c> and one at its upper bound <c>g_i ≤ ν</c> (the
    /// multiplier signs), each within <paramref name="tolerance"/> times its row scale <c>Σ_j |λ·Σ_ij·w_j| + |μ_i| + |ν|</c>.
    /// A feasible vector that is not the optimum, such as the optimum with two unequal weights exchanged, violates them
    /// by many orders of magnitude more than the rounding of a correct result.
    /// </summary>
    internal static void AssertOptimal<T>(T[,] covariance, T[] weights, T lower, T upper, T[]? means, T riskAversion, double tolerance)
        where T : IFloatingPoint<T>
    {
        var n = weights.Length;
        var radix = typeof(T) == typeof(decimal) ? 10 : 2;
        var lambda = ExactOf(riskAversion);
        var gradient = new (BigInteger Mantissa, int Exponent)[n];
        var scale = new double[n];
        for (var i = 0; i < n; i++)
        {
            var sum = (BigInteger.Zero, 0);
            for (var j = 0; j < n; j++)
            {
                var term = Multiply(lambda, Multiply(ExactOf(covariance[i, j]), ExactOf(weights[j])));
                sum = Add(sum, term, radix);
                scale[i] += Math.Abs(double.CreateChecked(riskAversion) * double.CreateChecked(covariance[i, j]) * double.CreateChecked(weights[j]));
            }

            if (means is not null)
            {
                var mean = ExactOf(means[i]);
                sum = Add(sum, (-mean.Mantissa, mean.Exponent), radix);
                scale[i] += Math.Abs(double.CreateChecked(means[i]));
            }

            gradient[i] = sum;
        }

        var free = Enumerable.Range(0, n).Where(i => weights[i] > lower && weights[i] < upper).ToArray();
        Assert.True(free.Length > 0, "the optimum of these problems holds at least one weight strictly inside the box");
        var nu = gradient[free[0]];
        var nuMagnitude = Math.Abs(ToDouble(nu, radix));
        for (var i = 0; i < n; i++)
        {
            var difference = ToDouble(Add(gradient[i], (-nu.Mantissa, nu.Exponent), radix), radix);
            var allowed = tolerance * (scale[i] + nuMagnitude);
            var label = string.Create(CultureInfo.InvariantCulture, $"asset {i}: w = {weights[i]}, g_i − ν = {difference:E3}, allowed {allowed:E3}");
            if (weights[i] > lower && weights[i] < upper)
            {
                Assert.True(Math.Abs(difference) <= allowed, "free weight not stationary: " + label);
            }
            else if (weights[i] == lower)
            {
                Assert.True(difference >= -allowed, "lower-bound multiplier negative: " + label);
            }
            else
            {
                Assert.True(difference <= allowed, "upper-bound multiplier negative: " + label);
            }
        }
    }

    // A finite value of T as an exact radix fraction Mantissa·radix^Exponent (radix 10 for decimal, 2 otherwise).
    private static (BigInteger Mantissa, int Exponent) ExactOf<T>(T value)
        where T : IFloatingPoint<T>
    {
        if (value is decimal d)
        {
            Span<int> bits = stackalloc int[4];
            decimal.GetBits(d, bits);
            var magnitude = (new BigInteger((uint)bits[2]) << 64) | (new BigInteger((uint)bits[1]) << 32) | (uint)bits[0];
            return (bits[3] < 0 ? -magnitude : magnitude, -((bits[3] >> 16) & 0xFF));
        }

        var x = double.CreateChecked(value);
        if (x == 0.0)
        {
            return (BigInteger.Zero, 0);
        }

        var raw = BitConverter.DoubleToInt64Bits(x);
        var biased = (int)((raw >> 52) & 0x7FF);
        var fraction = raw & 0xFFFFFFFFFFFFFL;
        var mantissa = biased == 0 ? fraction : fraction | (1L << 52);
        return (raw < 0 ? -(BigInteger)mantissa : mantissa, (biased == 0 ? 1 : biased) - 1075);
    }

    private static (BigInteger Mantissa, int Exponent) Multiply((BigInteger Mantissa, int Exponent) a, (BigInteger Mantissa, int Exponent) b) =>
        (a.Mantissa * b.Mantissa, a.Exponent + b.Exponent);

    private static (BigInteger Mantissa, int Exponent) Add((BigInteger Mantissa, int Exponent) a, (BigInteger Mantissa, int Exponent) b, int radix)
    {
        if (a.Mantissa.IsZero)
        {
            return b;
        }

        if (b.Mantissa.IsZero)
        {
            return a;
        }

        var e = Math.Min(a.Exponent, b.Exponent);
        return ((a.Mantissa * BigInteger.Pow(radix, a.Exponent - e)) + (b.Mantissa * BigInteger.Pow(radix, b.Exponent - e)), e);
    }

    private static double ToDouble((BigInteger Mantissa, int Exponent) value, int radix)
    {
        if (value.Mantissa.IsZero)
        {
            return 0.0;
        }

        if (radix == 2)
        {
            var drop = Math.Max(0, (int)BigInteger.Abs(value.Mantissa).GetBitLength() - 62);
            return Math.ScaleB((double)(value.Mantissa >> drop), value.Exponent + drop);
        }

        return (double)value.Mantissa * Math.Pow(10.0, value.Exponent);
    }

    private void AssertWithinCeiling(string label, double ms)
    {
        var calibration = ActiveSetQpPerformanceTests.CalibrationMs();
        var ratio = ms / calibration;
        var line = string.Create(CultureInfo.InvariantCulture, $"{label}: {ms:F1} ms CPU = {ratio:F1}x the calibration workload ({calibration:F3} ms); ceiling {CalibratedCeiling}x");
        output.WriteLine(line);
        Assert.True(ratio <= CalibratedCeiling, line);
    }

    /// <summary>
    /// Runs <paramref name="action"/> three times and returns the result of the cheapest repetition with its process CPU
    /// time in milliseconds. A cost ceiling asserts how much work the code does, not how much of a core the host gives it:
    /// wall time on a loaded host runs to several times the CPU time of the same unchanged code (a solve of 155 ms wall
    /// took 1.4 s niced), while the CPU time stays within about the same factor of the calibration workload.
    /// </summary>
    internal static (T Result, double CpuMs) TimedCpu<T>(Func<T> action)
    {
        var best = double.PositiveInfinity;
        var result = default(T)!;
        for (var repetition = 0; repetition < 3; repetition++)
        {
            var cpu0 = Environment.CpuUsage.TotalTime;
            var attempt = action();
            var elapsed = (Environment.CpuUsage.TotalTime - cpu0).TotalMilliseconds;
            if (elapsed < best)
            {
                best = elapsed;
                result = attempt;
            }
        }

        return (result, best);
    }

    private static double[,] Equicorrelated(int n, double variance, double covariance)
    {
        var cov = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                cov[i, j] = i == j ? variance : covariance;
            }
        }

        return cov;
    }

    // Variances in [0.04, 0.09) and covariances in [-0.0005, 0.0005), parsed from strings of the given number of
    // decimal places and mirrored: exactly symmetric, strictly diagonally dominant (so positive definite and well
    // conditioned), every input carrying a full mantissa, and an optimum strictly inside a box of [0, 0.2] or wider.
    private static T[,] DiagonallyDominant<T>(int n, int seed, int digits)
        where T : System.Numerics.IFloatingPoint<T>
    {
        var rng = new System.Random(seed);
        var format = "F" + digits.ToString(CultureInfo.InvariantCulture);
        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                var value = i == j ? 0.04 + (rng.NextDouble() * 0.05) : (rng.NextDouble() - 0.5) * 0.001;
                cov[i, j] = cov[j, i] = T.Parse(value.ToString(format, CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);
            }
        }

        return cov;
    }

    // A dense three-factor model Σ = B·F·B′ + D: loadings on a market factor in [0.5, 1.5) and on two style factors in
    // [-0.5, 0.5), factor variances 0.03, 0.01 and 0.005, idiosyncratic variances in [0.01, 0.05). Every entry is
    // parsed from a string of the given number of decimal places and mirrored, so the matrix is exactly symmetric; the
    // idiosyncratic floor keeps it positive definite (smallest eigenvalue above 0.009) with every entry nonzero.
    internal static T[,] FactorModel<T>(int n, int seed, int digits)
        where T : System.Numerics.IFloatingPoint<T>
    {
        var rng = new System.Random(seed);
        double[] factorVariance = [0.03, 0.01, 0.005];
        var loadings = new double[n, 3];
        var idiosyncratic = new double[n];
        for (var i = 0; i < n; i++)
        {
            loadings[i, 0] = 0.5 + rng.NextDouble();
            loadings[i, 1] = rng.NextDouble() - 0.5;
            loadings[i, 2] = rng.NextDouble() - 0.5;
            idiosyncratic[i] = 0.01 + (rng.NextDouble() * 0.04);
        }

        var format = "F" + digits.ToString(CultureInfo.InvariantCulture);
        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                var value = i == j ? idiosyncratic[i] : 0.0;
                for (var f = 0; f < 3; f++)
                {
                    value += loadings[i, f] * factorVariance[f] * loadings[j, f];
                }

                cov[i, j] = cov[j, i] = T.Parse(value.ToString(format, CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);
            }
        }

        return cov;
    }
}
