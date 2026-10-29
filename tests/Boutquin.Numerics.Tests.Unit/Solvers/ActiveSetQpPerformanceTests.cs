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
using System.Text.Json;

using Boutquin.Numerics.LinearAlgebra;
using Boutquin.Numerics.Solvers;

using Xunit.Abstractions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// xunit collection that keeps the solver timing gate out of the suite's parallel bucket: while it runs, no other
/// test collection runs concurrently, so the measurement is not inflated by a loaded machine.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class ActiveSetQpPerformanceCollection
{
    /// <summary>The xunit collection name shared between this definition and the timing test class.</summary>
    public const string Name = "ActiveSetQpPerformance (non-parallel)";
}

/// <summary>
/// The solve-time gate of <see cref="ActiveSetQpSolver{T}"/>: on the committed regression corpus's first-generation
/// case set (the cases the committed solve-time ratios were recorded on) the median and p99 solve time of the
/// solver, each divided by a fixed calibration workload (a 200 x 200 <see cref="CholeskyDecomposition{T}"/>
/// factorisation timed in the same test), may not exceed 20x (median) and 50x (p99) the ratios committed in
/// <c>tests/Verification/qp-corpus/baseline.json</c>. The ceilings catch a pathological regression, such as an
/// unbounded refinement loop or an exact path that became cubic per iteration, not the expected cost of exact
/// residuals. The committed baseline is a default-tiered-JIT measurement, so the gate measures under default tiering.
/// </summary>
[Collection(ActiveSetQpPerformanceCollection.Name)]
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpPerformanceTests(ITestOutputHelper output)
{
    /// <summary>Ceiling on the median calibrated ratio, as a multiple of the committed median ratio.</summary>
    internal const double MedianCeiling = 20.0;

    /// <summary>Ceiling on the p99 calibrated ratio, as a multiple of the committed p99 ratio.</summary>
    internal const double P99Ceiling = 50.0;

    private const int TimingRepetitions = 3;

    // The families of the first-generation corpus: the case set the committed ratios were recorded on. Cases of
    // families added later are excluded.
    private static readonly string[] s_firstGenerationFamilies =
    [
        "dbl-corr", "dbl-minvar", "dec-corr", "dec-diag-count", "r5-dbl-disp-mv",
        "r5-dec-rng-mean", "r5-dec-rng-budget", "r5-dec-rng-budget-min", "r5-dec-rng-zc", "r5-dec-zc2",
        "r5-rng-dbl-mean", "r5-rng-dbl-budget", "r5-rng-dbl-budget-min", "r5-rng-dbl-zc", "hd-hand",
    ];

    private static string FixtureDirectory => Path.Combine(AppContext.BaseDirectory, "Verification", "qp-corpus");

    // ------------------------------------------------------------------------------------------
    // The gate
    // ------------------------------------------------------------------------------------------

    [Fact]
    public void SolveTimeRatios_OnTheFirstGenerationCaseSet_StayWithinTheCeilingsOfTheCommittedRatios()
    {
        var committed = CommittedRatios.Load(Path.Combine(FixtureDirectory, "baseline.json"));
        var cases = LoadCases();
        Assert.Equal(committed.TotalCases, cases.Count);

        // Warm-up pass: every case once, so first-call JIT work is not part of any timed solve.
        foreach (var c in cases)
        {
            TimedSolve(c);
        }

        var calibrationBefore = CalibrationMs();
        var minima = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var type in new[] { "double", "decimal" })
        {
            var ofType = cases.Where(c => c.Type == type).ToList();
            Assert.Equal(committed.Cases[type], ofType.Count);
            var best = new double[ofType.Count];
            Array.Fill(best, double.PositiveInfinity);
            for (var repetition = 0; repetition < TimingRepetitions; repetition++)
            {
                for (var i = 0; i < ofType.Count; i++)
                {
                    best[i] = Math.Min(best[i], TimedSolve(ofType[i]));
                }
            }

            minima[type] = best;
        }

        // The calibration is the faster of the two readings taken around the solves, as an idle machine would give it.
        var calibration = Math.Min(calibrationBefore, CalibrationMs());

        var failures = new List<string>();
        var report = new List<string>();
        foreach (var (type, perCase) in minima)
        {
            var (median, p99) = Estimate(perCase);
            var medianRatio = median / calibration;
            var p99Ratio = p99 / calibration;
            var medianMultiple = medianRatio / committed.MedianRatio[type];
            var p99Multiple = p99Ratio / committed.P99Ratio[type];
            var line = string.Create(
                CultureInfo.InvariantCulture,
                $"{type}: {perCase.Length} cases, calibration {calibration:F4} ms; median {median:F6} ms = {medianRatio:F5}x calibration = {medianMultiple:F2}x the committed {committed.MedianRatio[type]:F5} (ceiling {MedianCeiling}x); p99 {p99:F6} ms = {p99Ratio:F5}x calibration = {p99Multiple:F2}x the committed {committed.P99Ratio[type]:F5} (ceiling {P99Ceiling}x)");
            output.WriteLine(line);
            report.Add(line);
            failures.AddRange(CeilingFailures(type, medianMultiple, p99Multiple));
        }

        // The measured ratios are left beside the fixtures of the test output for the build report.
        File.WriteAllLines(Path.Combine(FixtureDirectory, "observed-performance.txt"), report);
        Assert.True(failures.Count == 0, string.Join("; ", failures) + " | " + string.Join(" | ", report));
    }

    // ------------------------------------------------------------------------------------------
    // The ceilings and the estimator are pinned independently of the machine's speed
    // ------------------------------------------------------------------------------------------

    [Theory]
    [InlineData(19.99, 49.99, 0)]
    [InlineData(20.0, 50.0, 0)]
    [InlineData(20.01, 10.0, 1)]
    [InlineData(10.0, 50.01, 1)]
    [InlineData(40.0, 100.0, 2)]
    public void Ceilings_FailOnlyWhenTheMedianExceeds20xOrTheP99Exceeds50xTheCommittedRatio(
        double medianMultiple, double p99Multiple, int expectedFailures)
    {
        Assert.Equal(expectedFailures, CeilingFailures("double", medianMultiple, p99Multiple).Count());
    }

    [Fact]
    public void Estimate_IsTheMedianAndTheCeilingP99IndexOverThePerCaseMinima()
    {
        // Odd count: the middle element. p99 index = ceil(0.99 * 5) - 1 = 4, the largest.
        var (oddMedian, oddP99) = Estimate([5.0, 1.0, 3.0, 2.0, 4.0]);
        Assert.Equal(3.0, oddMedian);
        Assert.Equal(5.0, oddP99);

        // Even count: the mean of the two middle elements.
        var (evenMedian, _) = Estimate([4.0, 1.0, 3.0, 2.0]);
        Assert.Equal(2.5, evenMedian);

        // 100 cases 1..100: p99 index = ceil(0.99 * 100) - 1 = 98, the 99th smallest.
        var hundred = Enumerable.Range(1, 100).Select(i => (double)i).Reverse().ToArray();
        Assert.Equal(99.0, Estimate(hundred).P99);
    }

    [Fact]
    public void FirstGenerationCaseSet_IsTheCaseSetOfTheCommittedRatios()
    {
        var committed = CommittedRatios.Load(Path.Combine(FixtureDirectory, "baseline.json"));
        var cases = LoadCases();
        Assert.Equal(1708, committed.TotalCases);
        Assert.Equal(committed.TotalCases, cases.Count);
        Assert.Equal(committed.Cases["double"], cases.Count(c => c.Type == "double"));
        Assert.Equal(committed.Cases["decimal"], cases.Count(c => c.Type == "decimal"));
        Assert.All(cases, c => Assert.Contains(c.Family, s_firstGenerationFamilies));
    }

    private static IEnumerable<string> CeilingFailures(string type, double medianMultiple, double p99Multiple)
    {
        if (medianMultiple > MedianCeiling)
        {
            yield return string.Create(
                CultureInfo.InvariantCulture,
                $"{type} median calibrated ratio is {medianMultiple:F2}x the committed M-1 ratio (ceiling {MedianCeiling}x)");
        }

        if (p99Multiple > P99Ceiling)
        {
            yield return string.Create(
                CultureInfo.InvariantCulture,
                $"{type} p99 calibrated ratio is {p99Multiple:F2}x the committed M-1 ratio (ceiling {P99Ceiling}x)");
        }
    }

    // Median and p99 over the per-case minima: the p99 is the element at index ceil(0.99 n) - 1 of the sorted minima.
    private static (double Median, double P99) Estimate(double[] minima)
    {
        var sorted = minima.OrderBy(x => x).ToArray();
        var p99Index = Math.Max(0, (int)Math.Ceiling(0.99 * sorted.Length) - 1);
        var median = sorted.Length % 2 == 1
            ? sorted[sorted.Length / 2]
            : (sorted[(sorted.Length / 2) - 1] + sorted[sorted.Length / 2]) / 2;
        return (median, sorted[p99Index]);
    }

    // ------------------------------------------------------------------------------------------
    // The timed unit: the same one the committed ratios were recorded on. Parsing the case's decimal strings, the
    // production entry point, and the exact rational conversion of every returned weight are all inside the timed
    // region; a refusal is timed like a result.
    // ------------------------------------------------------------------------------------------

    private static double TimedSolve(PerformanceCase c)
    {
        var t0 = Stopwatch.GetTimestamp();
        try
        {
            _ = c.Type == "double" ? SolveDouble(c) : SolveDecimal(c);
        }
        catch (InvalidOperationException)
        {
            // The documented refusal.
        }

        return Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
    }

    private static (BigInteger Numerator, BigInteger Denominator)[] SolveDouble(PerformanceCase c)
    {
        static double Parse(string s) => double.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture);
        var n = c.Cov.Length;
        var cov = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                cov[i, j] = Parse(c.Cov[i][j]);
            }
        }

        var weights = c.Mode == "minvar"
            ? ActiveSetQpSolver<double>.SolveMinVariance(cov, Parse(c.Lb), Parse(c.Ub))
            : ActiveSetQpSolver<double>.SolveMeanVariance(cov, c.Mu.Select(Parse).ToArray(), Parse(c.Lambda), Parse(c.Lb), Parse(c.Ub));
        return weights.Select(ExactFromDouble).ToArray();
    }

    private static (BigInteger Numerator, BigInteger Denominator)[] SolveDecimal(PerformanceCase c)
    {
        static decimal Parse(string s) => decimal.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture);
        var n = c.Cov.Length;
        var cov = new decimal[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                cov[i, j] = Parse(c.Cov[i][j]);
            }
        }

        var weights = c.Mode == "minvar"
            ? ActiveSetQpSolver<decimal>.SolveMinVariance(cov, Parse(c.Lb), Parse(c.Ub))
            : ActiveSetQpSolver<decimal>.SolveMeanVariance(cov, c.Mu.Select(Parse).ToArray(), Parse(c.Lambda), Parse(c.Lb), Parse(c.Ub));
        return weights.Select(ExactFromDecimal).ToArray();
    }

    // The weight as a reduced fraction, as the committed timing converted every result before classifying it.
    private static (BigInteger Numerator, BigInteger Denominator) ExactFromDouble(double x)
    {
        if (!double.IsFinite(x))
        {
            throw new ArithmeticException("non-finite weight returned");
        }

        var bits = BitConverter.DoubleToInt64Bits(x);
        var exponent = (int)((bits >> 52) & 0x7FF);
        var mantissa = bits & 0xFFFFFFFFFFFFFL;
        if (exponent == 0)
        {
            exponent = 1;
        }
        else
        {
            mantissa |= 1L << 52;
        }

        exponent -= 1075;
        BigInteger numerator = bits < 0 ? -mantissa : mantissa;
        return exponent >= 0 ? Reduce(numerator << exponent, BigInteger.One) : Reduce(numerator, BigInteger.One << -exponent);
    }

    private static (BigInteger Numerator, BigInteger Denominator) ExactFromDecimal(decimal x)
    {
        var bits = decimal.GetBits(x);
        var magnitude = (new BigInteger((uint)bits[2]) << 64) | (new BigInteger((uint)bits[1]) << 32) | new BigInteger((uint)bits[0]);
        var scale = (bits[3] >> 16) & 0xFF;
        return Reduce(bits[3] < 0 ? -magnitude : magnitude, BigInteger.Pow(10, scale));
    }

    private static (BigInteger Numerator, BigInteger Denominator) Reduce(BigInteger numerator, BigInteger denominator)
    {
        if (numerator.IsZero)
        {
            return (BigInteger.Zero, BigInteger.One);
        }

        var gcd = BigInteger.GreatestCommonDivisor(numerator, denominator);
        return gcd.IsOne ? (numerator, denominator) : (numerator / gcd, denominator / gcd);
    }

    // The calibration workload: a 200 x 200 CholeskyDecomposition<double> factorisation of a fixed seeded matrix,
    // the best of seven runs.
    internal static double CalibrationMs()
    {
        const int n = 200;
        var rng = new System.Random(20261008);
        var b = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                b[i, j] = rng.NextDouble() - 0.5;
            }
        }

        var a = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var sum = i == j ? n : 0.0;
                for (var k = 0; k < n; k++)
                {
                    sum += b[i, k] * b[j, k];
                }

                a[i, j] = sum;
            }
        }

        var best = double.PositiveInfinity;
        for (var repetition = 0; repetition < 7; repetition++)
        {
            var t0 = Stopwatch.GetTimestamp();
            _ = CholeskyDecomposition<double>.Decompose(a);
            best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
        }

        return best;
    }

    // ------------------------------------------------------------------------------------------
    // Fixtures
    // ------------------------------------------------------------------------------------------

    private sealed record PerformanceCase(
        string Type, string Family, string Mode, string[][] Cov, string[] Mu, string Lambda, string Lb, string Ub);

    private static List<PerformanceCase> LoadCases()
    {
        var cases = new List<PerformanceCase>();
        foreach (var type in new[] { "double", "decimal" })
        {
            var path = Path.Combine(FixtureDirectory, "corpus-" + type + ".json");
            Assert.True(File.Exists(path), "the corpus fixture is missing: " + path);
            using var document = JsonDocument.Parse(File.ReadAllBytes(path));
            foreach (var c in document.RootElement.GetProperty("cases").EnumerateArray())
            {
                var family = c.GetProperty("family").GetString()!;
                if (!s_firstGenerationFamilies.Contains(family))
                {
                    continue;
                }

                var mode = c.GetProperty("mode").GetString()!;
                var meanVariance = mode == "meanvar";
                cases.Add(new PerformanceCase(
                    type,
                    family,
                    mode,
                    c.GetProperty("cov").EnumerateArray().Select(r => r.EnumerateArray().Select(x => x.GetString()!).ToArray()).ToArray(),
                    meanVariance ? c.GetProperty("mu").EnumerateArray().Select(x => x.GetString()!).ToArray() : [],
                    meanVariance ? c.GetProperty("lambda").GetString()! : "",
                    c.GetProperty("lb").GetString()!,
                    c.GetProperty("ub").GetString()!));
            }
        }

        return cases;
    }

    private sealed class CommittedRatios
    {
        public Dictionary<string, double> MedianRatio { get; } = [];

        public Dictionary<string, double> P99Ratio { get; } = [];

        public Dictionary<string, int> Cases { get; } = [];

        public int TotalCases => Cases.Values.Sum();

        public static CommittedRatios Load(string path)
        {
            Assert.True(File.Exists(path), "the committed baseline is missing: " + path);
            var ratios = new CommittedRatios();
            using var document = JsonDocument.Parse(File.ReadAllBytes(path));
            var timing = document.RootElement.GetProperty("timing");
            foreach (var type in new[] { "double", "decimal" })
            {
                var t = timing.GetProperty(type);
                ratios.MedianRatio[type] = t.GetProperty("median_ratio").GetDouble();
                ratios.P99Ratio[type] = t.GetProperty("p99_ratio").GetDouble();
                ratios.Cases[type] = t.GetProperty("cases").GetInt32();
                Assert.True(ratios.MedianRatio[type] > 0 && ratios.P99Ratio[type] >= ratios.MedianRatio[type], "the committed ratios of " + type + " must be positive and ordered");
            }

            return ratios;
        }
    }
}
