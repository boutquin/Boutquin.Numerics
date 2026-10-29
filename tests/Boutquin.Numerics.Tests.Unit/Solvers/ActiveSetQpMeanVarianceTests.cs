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
/// <see cref="ActiveSetQpSolver{T}.SolveMeanVariance"/> on the primal-feasible active-set core: the λ = 0 greedy LP
/// branch reaches the exact greedy optimum, the mean-variance entry point reaches the same machinery as the min-variance one,
/// and the failure modes keep their exception types (one test per mode, double and decimal). The accuracy statement on
/// the corpus lives in <see cref="ActiveSetQpCorpusTests"/> (the <c>ActiveSetQpMeanVariance_*</c> theories).
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpMeanVarianceTests
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ZeroRisk_RoundedFeasibilityFrontierRespectsBoundsAndBudgetTolerance(bool lowerBound)
    {
        double[,] covariance = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        double[] means = [3, 2, 1];
        // Feasibility products use working arithmetic. These exact-real products
        // straddle one, but both round to one and satisfy the numerical budget.
        var lower = lowerBound ? Math.BitIncrement(1d / 3) : 0;
        var upper = lowerBound ? 1 : 1d / 3;
        var weights = ActiveSetQpSolver<double>.SolveMeanVariance(covariance, means, 0, lower, upper);
        Assert.All(weights, weight => Assert.InRange(weight, lower, upper));
        Assert.InRange(Math.Abs(weights.Sum() - 1d), 0d, 3d * (Math.BitIncrement(1d) - 1d));
    }

    // ------------------------------------------------------------------------------------------
    // The mean-variance entry point runs the same machinery as the min-variance one
    // ------------------------------------------------------------------------------------------

    /// <summary>
    /// With zero means the mean-variance problem has the same optimum as the minimum-variance problem.
    /// Compare that mathematical equivalence within the working type's tolerance; arithmetic paths need not match.
    /// </summary>
    [Fact]
    public void ActiveSetQpMeanVariance_ZeroMeans_ReturnsTheSameMinimumVarianceOptimum()
    {
        var mismatches = new List<string>();
        Compare<double>([1.0, 4.0, 0.25], mismatches);
        Compare<decimal>([1m, 100m, 0.01m], mismatches);
        Assert.True(mismatches.Count == 0, mismatches.Count + " zero-mean solves differ from SolveMinVariance: " + string.Join(" | ", mismatches.Take(10)));
    }

    [Fact]
    public void ActiveSetQpMeanVariance_RepeatedCalls_AreBitIdentical()
    {
        var mismatches = 0;
        foreach (var (cov, lower, upper, k) in SpdCases<decimal>(60))
        {
            var means = Means<decimal>(cov.GetLength(0), k);
            var first = Outcome(() => ActiveSetQpSolver<decimal>.SolveMeanVariance(cov, means, 3m, lower, upper));
            var second = Outcome(() => ActiveSetQpSolver<decimal>.SolveMeanVariance(cov, means, 3m, lower, upper));
            if (first != second)
            {
                mismatches++;
            }
        }

        Assert.Equal(0, mismatches);
    }

    /// <summary>
    /// An in-domain decimal problem whose optimum <c>(1e13 + 0.5, −1e13 + 0.5)</c> is representable while the gradient
    /// products <c>λ·Σᵢⱼ·wⱼ</c> reach 1e29, beyond the decimal range (each gradient row is a representable difference
    /// of such terms): the solver returns the representable optimum. The exact
    /// bound of the result is asserted against the rational oracle by the corpus witness test of the same name.
    /// </summary>
    [Fact]
    public void ActiveSetQpMeanVariance_DomainEdgeRangeWitness_ReturnsTheRepresentableOptimum()
    {
        decimal[,] cov = { { 100000000m, 99999999.9999999m }, { 99999999.9999999m, 100000000m } };
        decimal[] means = [100000000000000m, -100000000000000m];
        var generic = ActiveSetQpSolver<decimal>.SolveMeanVariance(cov, means, 100000000m, -100000000000000m, 100000000000000m);
        Assert.True(Math.Abs(generic[0] - 10000000000000.5m) <= 0.001m && Math.Abs(generic[1] + 9999999999999.5m) <= 0.001m, string.Join(", ", generic));
    }

    /// <summary>
    /// The documented range refusal is reserved for a gradient product <c>λ·Σᵢⱼ·wⱼ</c> that is not representable together
    /// with one of its factors <c>λ·Σᵢⱼ</c> or <c>Σᵢⱼ·wⱼ</c>, which only an input outside the operating domain can produce.
    /// Here <c>λ = 1e19</c> (outside the domain) makes <c>λ·Σᵢᵢ = 1e29</c> unrepresentable, but every gradient product
    /// <c>λ·Σᵢᵢ·wᵢ = 5e28</c> is representable and the result <c>(½, ½)</c> is certified, so it is returned, as before;
    /// with <c>Σ = 7e28·I</c> and <c>λ = 4</c> the product <c>λ·Σᵢᵢ·wᵢ = 1.4e29</c> and its factor <c>λ·Σᵢᵢ</c> both
    /// overflow and the documented exception is raised.
    /// </summary>
    [Fact]
    public void ActiveSetQpMeanVariance_RangeRefusal_RequiresAnUnrepresentableGradientProductAndFactor()
    {
        var weights = ActiveSetQpSolver<decimal>.SolveMeanVariance(
            new decimal[,] { { 10000000000m, 0m }, { 0m, 10000000000m } }, [0m, 0m], 10000000000000000000m, 0m, 1m);
        Assert.Equal(new[] { 0.5m, 0.5m }, weights);
        var error = Assert.Throws<InvalidOperationException>(() => ActiveSetQpSolver<decimal>.SolveMeanVariance(
            new decimal[,] { { 7e28m, 0m }, { 0m, 7e28m } }, [0m, 0m], 4m, 0m, 1m));
        Assert.Equal("QP release gradient must be finite.", error.Message);
    }

    /// <summary>
    /// The solver's equilibration exponents depend on the covariance alone: it reads no scaled bound or mean, so the
    /// exactness of their scaling cannot hold an exponent at 0. <c>Σ = diag(1e14, 1e-14)</c> is diagonal (κ = 1 once
    /// equilibrated) and the optimum (½, ½) is interior; a 28-place bound <c>±2/3</c> or a 28-place mean <c>1e-28</c>
    /// is inexact under any decimal shift, and before the fix either one left the block unscaled, where the
    /// conditioning guard refused it as numerically singular. Every representation of the same problem returns the
    /// optimum.
    /// </summary>
    [Fact]
    public void ActiveSetQpMeanVariance_DiagonalDecimalProblemWithFullScaleBoundsAndMeans_ReturnsTheInteriorOptimum()
    {
        var cov = new decimal[,] { { 100000000000000m, 0m }, { 0m, 0.00000000000001m } };
        var twoThirds = 2m / 3m;
        foreach (var (mean, bound) in new[] { (0.0000000000000000000000000001m, twoThirds), (0.0000000000000000000000000001m, 0.7m), (0m, twoThirds), (0m, 0.7m) })
        {
            decimal[] weights = [];
            var refusal = Record.Exception(() => weights = ActiveSetQpSolver<decimal>.SolveMeanVariance(cov, [mean, -50000000000000m], 1m, -bound, bound));
            Assert.True(refusal is null, string.Create(CultureInfo.InvariantCulture, $"mean {mean}, bound {bound}: {refusal?.Message}"));
            Assert.True(
                Math.Abs(weights[0] - 0.5m) <= 1e-27m && Math.Abs(weights[1] - 0.5m) <= 1e-27m,
                string.Create(CultureInfo.InvariantCulture, $"mean {mean}, bound {bound}: {weights[0]}, {weights[1]}"));
        }
    }

    // ------------------------------------------------------------------------------------------
    // Failure modes are unchanged (type and condition, one test per mode, double and decimal)
    // ------------------------------------------------------------------------------------------

    [Fact]
    public void ActiveSetQpMeanVariance_NonPositiveDefiniteCovariance_IsTheDocumentedInvalidOperationException()
    {
        var d = Assert.Throws<InvalidOperationException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1, 2 }, { 2, 1 } }, [0.1, 0.2], 1, 0, 1));
        var m = Assert.Throws<InvalidOperationException>(() => ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[,] { { 1, 2 }, { 2, 1 } }, [0.1m, 0.2m], 1, 0, 1));
        Assert.Contains("positive definite", d.Message, StringComparison.Ordinal);
        Assert.Contains("positive definite", m.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// At λ = 0 the greedy LP branch (kept unchanged bit for bit by the spec) reads the covariance only for its
    /// validation: square, symmetric and finite. An indefinite, negative-definite or zero covariance returns the same
    /// bits as the identity at λ = 0, while the same covariance is refused for any λ &gt; 0; an asymmetric or non-finite
    /// one is refused at λ = 0 too.
    /// </summary>
    [Fact]
    public void ActiveSetQpMeanVariance_ZeroRiskAversion_ReadsTheCovarianceOnlyForSymmetryAndFiniteness()
    {
        double[] means = [0.1, 0.2, 0.15];
        var identity = Outcome(() => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }, means, 0.0, 0.0, 0.5));
        var notPositiveDefinite = new[]
        {
            new double[,] { { 1, 2, 0 }, { 2, 1, 0 }, { 0, 0, 1 } },
            new double[,] { { -1, 0, 0 }, { 0, -1, 0 }, { 0, 0, -1 } },
            new double[3, 3],
        };
        foreach (var cov in notPositiveDefinite)
        {
            Assert.Equal(identity, Outcome(() => ActiveSetQpSolver<double>.SolveMeanVariance(cov, means, 0.0, 0.0, 0.5)));
            Assert.Throws<InvalidOperationException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(cov, means, 1e-300, 0.0, 0.5));
        }

        var decimalIdentity = Outcome(() => ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[,] { { 1, 0 }, { 0, 1 } }, [0.1m, 0.2m], 0m, 0m, 1m));
        Assert.Equal(decimalIdentity, Outcome(() => ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[,] { { 1, 2 }, { 2, 1 } }, [0.1m, 0.2m], 0m, 0m, 1m)));
        Assert.Equal("[1]", "[" + string.Join(",", ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[,] { { -1m } }, [0.1m], 0m, 0m, 1m)) + "]");

        Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1, 2 }, { 0, 1 } }, [0.1, 0.2], 0.0, 0.0, 1.0));
        Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1, double.NaN }, { double.NaN, 1 } }, [0.1, 0.2], 0.0, 0.0, 1.0));
    }

    /// <summary>
    /// The published contract matches that behaviour: in the XML documentation of
    /// <see cref="ActiveSetQpSolver{T}.SolveMeanVariance"/> (which the legacy decimal facade inherits), positive
    /// definiteness is required, and its absence raises <see cref="InvalidOperationException"/>, only for λ &gt; 0.
    /// </summary>
    [Fact]
    public void ActiveSetQpMeanVariance_PublishedContract_RequiresPositiveDefinitenessOnlyForPositiveRiskAversion()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Boutquin.Numerics.xml");
        Assert.True(File.Exists(path), "the XML documentation of the library is missing: " + path);
        var member = System.Xml.Linq.XDocument.Load(path).Descendants("member")
            .Single(m => ((string?)m.Attribute("name"))!.StartsWith("M:Boutquin.Numerics.Solvers.ActiveSetQpSolver`1.SolveMeanVariance(", StringComparison.Ordinal));
        static string Text(System.Xml.Linq.XElement e) => string.Join(" ", e.Value.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
        var refusal = Text(member.Elements("exception").Single(e => ((string?)e.Attribute("cref"))!.EndsWith("InvalidOperationException", StringComparison.Ordinal)));
        var covariance = Text(member.Elements("param").Single(e => (string?)e.Attribute("name") == "covariance"));

        Assert.StartsWith("For λ > 0, the covariance is not positive definite", refusal, StringComparison.Ordinal);
        Assert.DoesNotContain("; or, for λ > 0, the active-set", refusal, StringComparison.Ordinal);
        Assert.Contains("positive definite for λ > 0", covariance, StringComparison.Ordinal);
        Assert.Contains("symmetry and finiteness", covariance, StringComparison.Ordinal);
    }

    [Fact]
    public void ActiveSetQpMeanVariance_NonSquareCovariance_IsAnArgumentExceptionOnTheCovariance()
    {
        var d = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(new double[2, 3], [0.1, 0.2], 1, 0, 1));
        var m = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[2, 3], [0.1m, 0.2m], 1, 0, 1));
        Assert.Equal("covariance", d.ParamName);
        Assert.Equal("covariance", m.ParamName);
        Assert.Contains("square", d.Message, StringComparison.Ordinal);
        Assert.Contains("square", m.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ActiveSetQpMeanVariance_NonFiniteCovariance_IsAnArgumentExceptionOnTheCovariance()
    {
        var d = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1, 0 }, { 0, double.PositiveInfinity } }, [0.1, 0.2], 1, 0, 1));
        Assert.Equal("covariance", d.ParamName);
        Assert.Contains("finite", d.Message, StringComparison.Ordinal);
        var n = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { double.NaN, 0 }, { 0, 1 } }, [0.1, 0.2], 1, 0, 1));
        Assert.Equal("covariance", n.ParamName);
    }

    [Fact]
    public void ActiveSetQpMeanVariance_WronglySizedCovariance_IsAnArgumentException()
    {
        var d = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(new double[0, 0], [], 1, 0, 1));
        var m = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[0, 0], [], 1, 0, 1));
        Assert.Equal("covariance", d.ParamName);
        Assert.Equal("covariance", m.ParamName);
        var mismatch = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[,] { { 1, 0 }, { 0, 1 } }, [0.1m, 0.2m, 0.3m], 1, 0, 1));
        Assert.Equal("means", mismatch.ParamName);
    }

    [Fact]
    public void ActiveSetQpMeanVariance_AsymmetricCovariance_IsAnArgumentExceptionOnTheCovariance()
    {
        var d = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1, 0.5 }, { 0.4, 1 } }, [0.1, 0.2], 1, 0, 1));
        var m = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[,] { { 1, 0.5m }, { 0.4m, 1 } }, [0.1m, 0.2m], 1, 0, 1));
        Assert.Equal("covariance", d.ParamName);
        Assert.Equal("covariance", m.ParamName);
        Assert.Contains("symmetric", m.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ActiveSetQpMeanVariance_NonFiniteMeans_IsAnArgumentExceptionOnTheMeans()
    {
        var d = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1, 0 }, { 0, 1 } }, [0.1, double.PositiveInfinity], 1, 0, 1));
        Assert.Equal("means", d.ParamName);
        Assert.Contains("finite", d.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ActiveSetQpMeanVariance_NonFiniteOrCrossedBoundsAndInvalidRiskAversion_AreArgumentExceptions()
    {
        var cov = new double[,] { { 1, 0 }, { 0, 1 } };
        double[] means = [0.1, 0.2];
        Assert.Equal("minWeight", Assert.Throws<ArgumentOutOfRangeException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(cov, means, 1, double.NaN, 1)).ParamName);
        Assert.Equal("maxWeight", Assert.Throws<ArgumentOutOfRangeException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(cov, means, 1, 0, double.PositiveInfinity)).ParamName);
        Assert.Equal("minWeight", Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(cov, means, 1, 0.7, 0.6)).ParamName);
        Assert.Equal("minWeight", Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[,] { { 1, 0 }, { 0, 1 } }, [0.1m, 0.2m], 1, 0.7m, 0.6m)).ParamName);
        Assert.Equal("riskAversion", Assert.Throws<ArgumentOutOfRangeException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(cov, means, -1, 0, 1)).ParamName);
        Assert.Equal("riskAversion", Assert.Throws<ArgumentOutOfRangeException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(cov, means, double.NaN, 0, 1)).ParamName);
        Assert.Equal("riskAversion", Assert.Throws<ArgumentOutOfRangeException>(() => ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[,] { { 1, 0 }, { 0, 1 } }, [0.1m, 0.2m], -1m, 0, 1)).ParamName);
    }

    [Theory]
    [InlineData(0.0, 0.2)]
    [InlineData(0.6, 1.0)]
    public void ActiveSetQpMeanVariance_InfeasibleBounds_KeepTheDocumentedClampAndTheLpRefusal(double lower, double upper)
    {
        // n·ub < 1 or n·lb > 1: for λ > 0 only the offending bound is relaxed to 1/n and a feasible vector comes back.
        var cov = new double[,] { { 1, 0.2, 0 }, { 0.2, 2, 0.1 }, { 0, 0.1, 3 } };
        var weights = ActiveSetQpSolver<double>.SolveMeanVariance(cov, [0.1, 0.2, 0.3], 2, lower, upper);
        Assert.All(weights, w => Assert.InRange(w, Math.Min(lower, 1.0 / 3), Math.Max(upper, 1.0 / 3)));
        Assert.InRange(Math.Abs(weights.Sum() - 1), 0, 1e-15);

        var decimalWeights = ActiveSetQpSolver<decimal>.SolveMeanVariance(
            new decimal[,] { { 1, 0.2m, 0 }, { 0.2m, 2, 0.1m }, { 0, 0.1m, 3 } }, [0.1m, 0.2m, 0.3m], 2m, (decimal)lower, (decimal)upper);
        Assert.All(decimalWeights, w => Assert.InRange(w, Math.Min((decimal)lower, 1m / 3), Math.Max((decimal)upper, 1m / 3)));
        Assert.InRange(Math.Abs(decimalWeights.Sum() - 1m), 0m, 1e-26m);

        // The λ = 0 greedy LP keeps its own documented refusal of infeasible bounds.
        var lp = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<decimal>.SolveMeanVariance(
            new decimal[,] { { 1, 0.2m, 0 }, { 0.2m, 2, 0.1m }, { 0, 0.1m, 3 } }, [0.1m, 0.2m, 0.3m], 0m, (decimal)lower, (decimal)upper));
        Assert.Equal(lower > 0 ? "minWeight" : "maxWeight", lp.ParamName);
    }

    [Fact]
    public void ActiveSetQpMeanVariance_HalfPrecision_IsRejectedWithNotSupportedException()
    {
        Half[,] cov = { { (Half)1, (Half)0 }, { (Half)0, (Half)1 } };
        Assert.Throws<NotSupportedException>(() => ActiveSetQpSolver<Half>.SolveMeanVariance(cov, [(Half)0.1, (Half)0.2], (Half)1, (Half)0, (Half)1));
    }

    [Fact]
    public void ActiveSetQpMeanVariance_NullCovariance_IsAnArgumentNullException()
    {
        Assert.Equal("covariance", Assert.Throws<ArgumentNullException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(null!, [0.1], 1, 0, 1)).ParamName);
    }

    // ------------------------------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------------------------------

    private static void Compare<T>(T[] riskAversions, List<string> mismatches)
        where T : IFloatingPoint<T>
    {
        var index = 0;
        foreach (var (cov, lower, upper, _) in SpdCases<T>(120))
        {
            var n = cov.GetLength(0);
            var minVariance = Run(() => ActiveSetQpSolver<T>.SolveMinVariance(cov, lower, upper));
            var tolerance = typeof(T) == typeof(decimal) ? T.CreateChecked(1e-22m) : T.CreateChecked(1e-10);
            foreach (var lambda in riskAversions)
            {
                var meanVariance = Run(() => ActiveSetQpSolver<T>.SolveMeanVariance(cov, new T[n], lambda, lower, upper));
                if (minVariance.Error != meanVariance.Error ||
                    (minVariance.Weights is not null && meanVariance.Weights is not null &&
                     minVariance.Weights.Zip(meanVariance.Weights).Any(pair =>
                         T.Abs(pair.First - pair.Second) > tolerance * T.Max(T.One, T.Abs(pair.First)))))
                {
                    mismatches.Add(typeof(T).Name + " case " + index + " λ=" + lambda + ": the optima or acceptance decisions differ");
                }
            }

            index++;
        }
    }

    private static (T[]? Weights, Type? Error) Run<T>(Func<T[]> solve)
    {
        try
        {
            return (solve(), null);
        }
        catch (Exception ex)
        {
            return (null, ex.GetType());
        }
    }

    // Seeded symmetric positive-definite problems: Σ = s·(AA′ + shift·I) for an integer A, scaled per asset, with
    // bounds from the long-only, boxed, short-allowed and tight regimes.
    private static IEnumerable<(T[,] Cov, T Lower, T Upper, int Seed)> SpdCases<T>(int count)
        where T : IFloatingPoint<T>
    {
        var rng = new System.Random(typeof(T) == typeof(decimal) ? 4402 : 4401);
        static T P(string s) => T.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture);
        for (var k = 0; k < count; k++)
        {
            var n = rng.Next(2, 11);
            var a = new int[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    a[i, j] = rng.Next(-4, 5);
                }
            }

            var scale = new T[n];
            for (var i = 0; i < n; i++)
            {
                scale[i] = P(new[] { "1", "0.5", "2", "0.1", "3" }[rng.Next(5)]);
            }

            var shift = rng.Next(1, 4);
            var cov = new T[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = i; j < n; j++)
                {
                    long acc = 0;
                    for (var q = 0; q < n; q++)
                    {
                        acc += a[i, q] * a[j, q];
                    }

                    var v = T.CreateChecked(acc + (i == j ? shift : 0)) * scale[i] * scale[j];
                    cov[i, j] = v;
                    cov[j, i] = v;
                }
            }

            (T, T) bounds = rng.Next(5) switch
            {
                0 => (T.Zero, T.One),
                1 => (T.Zero, P(Math.Round(1.0 / n * (1.1 + rng.NextDouble()), 4).ToString("R", CultureInfo.InvariantCulture))),
                2 => (P("-0.25"), P("2")),
                3 => (P(Math.Round(rng.NextDouble() / n, 4).ToString("R", CultureInfo.InvariantCulture)), T.One),
                _ => (P("-1000"), P("1000")),
            };
            yield return (cov, bounds.Item1, bounds.Item2, rng.Next());
        }
    }

    private static T[] Means<T>(int n, int seed)
        where T : IFloatingPoint<T>
    {
        var rng = new System.Random(seed);
        var means = new T[n];
        for (var i = 0; i < n; i++)
        {
            means[i] = T.Parse(Math.Round(rng.NextDouble() * 0.2 - 0.05, 4).ToString("R", CultureInfo.InvariantCulture), NumberStyles.Float, CultureInfo.InvariantCulture);
        }

        return means;
    }

    private static string Outcome<T>(Func<T[]> solve, bool typeOnly = false)
        where T : IFloatingPoint<T>
    {
        try
        {
            return string.Join(",", solve().Select(Bits));
        }
        catch (Exception ex)
        {
            return "EX:" + ex.GetType().Name + (typeOnly ? "" : ":" + ((ex as ArgumentException)?.ParamName ?? ex.Message));
        }
    }

    private static string Bits<T>(T x) => x is double d
        ? BitConverter.DoubleToInt64Bits(d).ToString("x16", CultureInfo.InvariantCulture)
        : string.Join(".", decimal.GetBits((decimal)(object)x!).Select(b => b.ToString("x8", CultureInfo.InvariantCulture)));
}
