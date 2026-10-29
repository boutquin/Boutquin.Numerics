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
using System.Text.Json;

using Boutquin.Numerics.Internal;
using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// The primal-feasible active-set core of <see cref="ActiveSetQpSolver{T}.SolveMinVariance"/>: termination on
/// degenerate problems, the unchanged failure modes, and the SLSQP verification vectors replayed through the
/// production entry point.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpMinVarianceTests
{
    // ------------------------------------------------------------------------------------------
    // Termination: seeded degenerate problems never reach the iteration budget
    // ------------------------------------------------------------------------------------------

    /// <summary>
    /// Seeded degenerate min-variance problems (repeated covariances: equal variances and equal correlations,
    /// repeated diagonals, twin blocks; coinciding bounds: lb = ub = 1/n and bounds at the equal-weight
    /// frontier; ties at a bound: an upper bound equal to an optimal weight) at <c>double</c> and
    /// <c>decimal</c>. Every problem is positive definite and in the operating domain, so every solve must
    /// terminate with a certified optimum; a solve that exhausts the iteration budget (or refuses at all)
    /// fails the property.
    /// </summary>
    [Fact]
    public void DegenerateProblems_NeverReachTheIterationBudget_AndReturnACertifiedOptimum()
    {
        var failures = new List<string>();
        var rng = new System.Random(20261008);
        var count = 0;
        for (var trial = 0; trial < 400; trial++)
        {
            var problem = DegenerateProblem(rng, trial);
            count++;
            Check<double>(problem, failures);
            Check<decimal>(problem, failures);
        }

        Assert.True(count == 400, "the property must cover the seeded problems");
        Assert.True(failures.Count == 0, failures.Count + " degenerate solves failed: " + string.Join(" | ", failures.Take(12)));
    }

    /// <summary>
    /// The seeded degenerate problems above through both entry points (minimum variance, and mean variance with means
    /// tied across all assets or tied within the groups of repeated rows) at <c>double</c> and <c>decimal</c>: neither
    /// route of the core reaches its iteration budget. The production solve must not spend the rounded route's budget
    /// (a spent budget is handed to the exact route, so only the route diagnostics show it), and the exact route run on
    /// its own from the exact feasible start, where every release of these problems begins at a vertex or with free
    /// variables on their bounds, must not spend its own.
    /// </summary>
    [Fact]
    public void DegenerateProblems_NeitherRouteReachesItsIterationBudget_AtBothTypesAndBothEntryPoints()
    {
        var failures = new List<string>();
        var rng = new System.Random(20261024);
        for (var trial = 0; trial < 200; trial++)
        {
            var problem = DegenerateProblem(rng, trial);
            CheckRoutes<double>(problem, failures);
            CheckRoutes<decimal>(problem, failures);
        }

        Assert.True(failures.Count == 0, failures.Count + " solves did not terminate with a result (a spent budget or a refusal): " + string.Join(" | ", failures.Take(12)));
    }

    private static void CheckRoutes<T>(Degenerate problem, List<string> failures)
        where T : IFloatingPoint<T>
    {
        var n = problem.Numerator.GetLength(0);
        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                cov[i, j] = T.CreateChecked(problem.Numerator[i, j]);
            }
        }

        var lower = T.CreateChecked(problem.LowerNumerator) / T.CreateChecked(problem.BoundDenominator);
        var upper = T.CreateChecked(problem.UpperNumerator) / T.CreateChecked(problem.BoundDenominator);

        // Tied means: the same mean everywhere, or one mean per distinct covariance row (assets of one group tie).
        var tied = Enumerable.Repeat(T.CreateChecked(3), n).ToArray();
        var grouped = new T[n];
        for (var i = 0; i < n; i++)
        {
            var row = 0;
            while (row < i && !Enumerable.Range(0, n).All(j => problem.Numerator[row, j] == problem.Numerator[i, j]))
            {
                row++;
            }

            grouped[i] = T.CreateChecked(row % 3) / T.CreateChecked(4);
        }

        foreach (var means in new T[]?[] { null, tied, grouped })
        {
            foreach (var force in new[] { false, true })
            {
                var trace = new ActiveSetQpSolver<T>.RouteTrace { ForceExactRoute = force };
                var label = typeof(T).Name + " " + problem.Name + (means is null ? " min-variance" : " mean-variance") + (force ? " exact route" : "");
                try
                {
                    _ = means is null
                        ? ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(cov, lower, upper, forcedReference: -1, trace: trace)
                        : ActiveSetQpSolver<T>.SolveMeanVarianceUsingCore(cov, means, T.One, lower, upper, trace: trace);
                }
                catch (Exception ex)
                {
                    failures.Add(label + ": " + ex.GetType().Name + ": " + ex.Message);
                    continue;
                }

                if (trace.RoundedBudgetReached)
                {
                    failures.Add(label + ": the rounded route spent its budget (" + trace.RoundedChanges + " working-set changes)");
                }

                // A release on a strictly negative exact multiplier lowers the exact objective: the exact route never
                // undoes one, in particular never a release whose step had length zero.
                if (trace.ExactRollbacks > 0)
                {
                    failures.Add(label + ": the exact route rolled back " + trace.ExactRollbacks + " release(s)");
                }
            }
        }
    }

    private sealed record Degenerate(string Name, int[,] Numerator, int Denominator, int LowerNumerator, int UpperNumerator, int BoundDenominator);

    // Integer-ratio problems, exactly representable at both types up to the rounding of the bound ratios.
    private static Degenerate DegenerateProblem(System.Random rng, int trial)
    {
        var n = rng.Next(2, 13);
        var cov = new int[n, n];
        string family;
        switch (trial % 6)
        {
            case 4:
            case 5:
                {
                    // Factor model with repeated integer loadings: Σ = B·B′ + d·I, B ∈ {-2..2}^(n×2) drawn from
                    // two repeated rows, so groups of assets tie in every covariance and every multiplier.
                    family = "tied-factor";
                    int[][] rows = [[rng.Next(-2, 3), rng.Next(-2, 3)], [rng.Next(-2, 3), rng.Next(-2, 3)], [2, -1]];
                    var d = rng.Next(1, 3);
                    var loadings = new int[n][];
                    for (var i = 0; i < n; i++)
                    {
                        loadings[i] = rows[rng.Next(rows.Length)];
                    }

                    for (var i = 0; i < n; i++)
                    {
                        for (var j = 0; j < n; j++)
                        {
                            cov[i, j] = (loadings[i][0] * loadings[j][0]) + (loadings[i][1] * loadings[j][1]) + (i == j ? d : 0);
                        }
                    }

                    break;
                }

            case 0:
                {
                    // Equal variances, equal correlations: Σ = v·((1 − ρ)I + ρ11′) with ρ = c/v.
                    family = "equal-correlation";
                    var v = rng.Next(1, 5) * 4;
                    var c = rng.Next(0, v);
                    for (var i = 0; i < n; i++)
                    {
                        for (var j = 0; j < n; j++)
                        {
                            cov[i, j] = i == j ? v : c;
                        }
                    }

                    break;
                }

            case 1:
                {
                    family = "repeated-diagonal";
                    int[] values = [1, 2, 4];
                    for (var i = 0; i < n; i++)
                    {
                        cov[i, i] = values[rng.Next(values.Length)];
                    }

                    break;
                }

            case 2:
                {
                    // Two blocks of identical assets: block variance v, intra-block covariance c, cross 0.
                    family = "twin-blocks";
                    var v = rng.Next(2, 6) * 2;
                    var c = rng.Next(0, v);
                    var half = n / 2;
                    for (var i = 0; i < n; i++)
                    {
                        for (var j = 0; j < n; j++)
                        {
                            var same = (i < half) == (j < half);
                            cov[i, j] = i == j ? v : same ? c : 0;
                        }
                    }

                    break;
                }

            default:
                {
                    // Diagonal-dominant with tied off-diagonals: every pair shares the same covariance.
                    family = "tied-offdiagonal";
                    var c = rng.Next(1, 3);
                    for (var i = 0; i < n; i++)
                    {
                        for (var j = 0; j < n; j++)
                        {
                            cov[i, j] = i == j ? (n * c) + rng.Next(1, 3) : c;
                        }
                    }

                    for (var i = 0; i < n; i++)
                    {
                        for (var j = 0; j < i; j++)
                        {
                            cov[i, j] = cov[j, i];
                        }
                    }

                    break;
                }
        }

        // Bounds as integer ratios lo/den and hi/den.
        var choice = rng.Next(6);
        var (lo, hi, den) = choice switch
        {
            0 => (0, 1, 1),
            1 => (0, 1, n),                      // upper bound at the equal-weight frontier
            2 => (1, n, n),                      // lower bound at the equal-weight frontier
            3 => (1, 1, n),                      // coinciding bounds lb = ub = 1/n
            4 => (0, 1, Math.Max(1, n - 1)),     // ties at the upper bound
            _ => (-1, 1, 2),                     // long-short box
        };

        return new Degenerate(family + "-" + trial + "-n" + n + "-b" + choice, cov, 1, lo, hi, den);
    }

    private static void Check<T>(Degenerate problem, List<string> failures)
        where T : IFloatingPoint<T>
    {
        var n = problem.Numerator.GetLength(0);
        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                cov[i, j] = T.CreateChecked(problem.Numerator[i, j]);
            }
        }

        var lower = T.CreateChecked(problem.LowerNumerator) / T.CreateChecked(problem.BoundDenominator);
        var upper = T.CreateChecked(problem.UpperNumerator) / T.CreateChecked(problem.BoundDenominator);
        T[] weights;
        try
        {
            weights = ActiveSetQpSolver<T>.SolveMinVariance(cov, lower, upper);
        }
        catch (Exception ex)
        {
            failures.Add(typeof(T).Name + " " + problem.Name + ": " + ex.GetType().Name + ": " + ex.Message);
            return;
        }

        // The documented clamp of infeasible bounds to the equal-weight frontier. The returned vector must be
        // certified independently, and with the same fourfold margin under the solver's exit scale κ that the
        // corpus calibration asserts: some power-of-two scale at most κ/4 accepts it.
        var inverse = T.One / T.CreateChecked(n);
        var effectiveLower = T.Min(lower, inverse);
        var effectiveUpper = T.Max(upper, inverse);
        var quarter = ActiveSetQpSolver<T>.CertificateToleranceScale / T.CreateChecked(4);
        var kappa = T.One;
        while (kappa <= quarter && !QpKktCertificate<T>.Check(cov, null, T.One, effectiveLower, effectiveUpper, weights, kappa).Accepted)
        {
            kappa += kappa;
        }

        if (kappa > quarter)
        {
            var detail = QpKktCertificate<T>.Check(cov, null, T.One, effectiveLower, effectiveUpper, weights, ActiveSetQpSolver<T>.CertificateToleranceScale).Detail;
            failures.Add(typeof(T).Name + " " + problem.Name + ": not certified within κ/4: " + detail);
        }
    }

    /// <summary>
    /// <c>float</c> is best-effort: the same degenerate problems at <c>float</c> are solved and their refusals
    /// recorded in <c>observed-degenerate-float.txt</c> beside the test assembly for the build report; nothing is
    /// asserted beyond the solve completing or refusing with a documented exception type.
    /// </summary>
    [Fact]
    public void DegenerateProblems_AtFloat_AreBestEffort_RefusalsAreDocumentedExceptions()
    {
        var rng = new System.Random(20261008);
        var refusals = new List<string>();
        var solved = 0;
        for (var trial = 0; trial < 400; trial++)
        {
            var problem = DegenerateProblem(rng, trial);
            var n = problem.Numerator.GetLength(0);
            var cov = new float[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    cov[i, j] = problem.Numerator[i, j];
                }
            }

            try
            {
                _ = ActiveSetQpSolver<float>.SolveMinVariance(
                    cov, (float)problem.LowerNumerator / problem.BoundDenominator, (float)problem.UpperNumerator / problem.BoundDenominator);
                solved++;
            }
            catch (InvalidOperationException ex)
            {
                refusals.Add(problem.Name + ": " + ex.Message);
            }
        }

        File.WriteAllText(
            Path.Combine(AppContext.BaseDirectory, "observed-degenerate-float.txt"),
            "solved " + solved + " of 400; refused " + refusals.Count + Environment.NewLine + string.Join(Environment.NewLine, refusals.Take(20)));
        Assert.Equal(400, solved + refusals.Count);
    }

    // ------------------------------------------------------------------------------------------
    // Diagnostics: a refusal names the failed certificate component
    // ------------------------------------------------------------------------------------------

    /// <summary>
    /// A positive-definite covariance whose reduced Hessian on the budget plane is singular at <c>decimal</c>'s
    /// working precision: assets 0 and 1 differ by one unit of the 28th decimal place, so the equilibrated
    /// reduced Hessian <c>[[1, 1 − 1e-28/3], …]</c> rounds to a singular matrix. The problem is not qualified;
    /// the documented refusal must name the certificate component it cannot establish.
    /// </summary>
    [Fact]
    public void NumericallySingularReducedHessian_IsRefusedNamingTheStationarityComponent()
    {
        decimal[,] cov =
        {
            { 2m, 1.9999999999999999999999999999m, 0m },
            { 1.9999999999999999999999999999m, 2m, 0m },
            { 0m, 0m, 1m },
        };

        var error = Assert.Throws<InvalidOperationException>(() => ActiveSetQpSolver<decimal>.SolveMinVariance(cov, -1m, 2m));
        Assert.Contains("did not converge", error.Message, StringComparison.Ordinal);
        Assert.Contains("stationarity", error.Message, StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------------------------------
    // Failure modes are unchanged (type and condition, one test per mode, double and decimal)
    // ------------------------------------------------------------------------------------------

    [Fact]
    public void NonPositiveDefiniteCovariance_IsTheDocumentedInvalidOperationException()
    {
        var d = Assert.Throws<InvalidOperationException>(() => ActiveSetQpSolver<double>.SolveMinVariance(new double[,] { { 1, 2 }, { 2, 1 } }, 0, 1));
        var m = Assert.Throws<InvalidOperationException>(() => ActiveSetQpSolver<decimal>.SolveMinVariance(new decimal[,] { { 1, 2 }, { 2, 1 } }, 0, 1));
        Assert.Contains("positive definite", d.Message, StringComparison.Ordinal);
        Assert.Contains("positive definite", m.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NonSquareCovariance_IsAnArgumentExceptionOnTheCovariance()
    {
        var d = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMinVariance(new double[2, 3], 0, 1));
        var m = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<decimal>.SolveMinVariance(new decimal[2, 3], 0, 1));
        Assert.Equal("covariance", d.ParamName);
        Assert.Equal("covariance", m.ParamName);
        Assert.Contains("square", d.Message, StringComparison.Ordinal);
        Assert.Contains("square", m.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NonFiniteCovariance_IsAnArgumentExceptionOnTheCovariance()
    {
        var d = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMinVariance(new double[,] { { 1, 0 }, { 0, double.PositiveInfinity } }, 0, 1));
        Assert.Equal("covariance", d.ParamName);
        Assert.Contains("finite", d.Message, StringComparison.Ordinal);
        var n = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMinVariance(new double[,] { { double.NaN, 0 }, { 0, 1 } }, 0, 1));
        Assert.Equal("covariance", n.ParamName);
    }

    [Fact]
    public void WronglySizedCovariance_IsAnArgumentExceptionOnTheCovariance()
    {
        var d = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMinVariance(new double[0, 0], 0, 1));
        var m = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<decimal>.SolveMinVariance(new decimal[0, 0], 0, 1));
        Assert.Equal("covariance", d.ParamName);
        Assert.Equal("covariance", m.ParamName);
        var mismatch = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1, 0 }, { 0, 1 } }, [0.1], 1, 0, 1));
        Assert.Equal("means", mismatch.ParamName);
    }

    [Fact]
    public void AsymmetricCovariance_IsAnArgumentExceptionOnTheCovariance()
    {
        var d = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMinVariance(new double[,] { { 1, 0.5 }, { 0.4, 1 } }, 0, 1));
        var m = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<decimal>.SolveMinVariance(new decimal[,] { { 1, 0.5m }, { 0.4m, 1 } }, 0, 1));
        Assert.Equal("covariance", d.ParamName);
        Assert.Equal("covariance", m.ParamName);
        Assert.Contains("symmetric", d.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NonFiniteMeans_IsAnArgumentExceptionOnTheMeans()
    {
        var d = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1, 0 }, { 0, 1 } }, [double.NaN, 0], 1, 0, 1));
        Assert.Equal("means", d.ParamName);
        Assert.Contains("finite", d.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NonFiniteOrCrossedBounds_AreArgumentExceptions()
    {
        var cov = new double[,] { { 1, 0 }, { 0, 1 } };
        Assert.Equal("minWeight", Assert.Throws<ArgumentOutOfRangeException>(() => ActiveSetQpSolver<double>.SolveMinVariance(cov, double.NaN, 1)).ParamName);
        Assert.Equal("maxWeight", Assert.Throws<ArgumentOutOfRangeException>(() => ActiveSetQpSolver<double>.SolveMinVariance(cov, 0, double.PositiveInfinity)).ParamName);
        Assert.Equal("minWeight", Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMinVariance(cov, 0.7, 0.6)).ParamName);
        Assert.Equal("minWeight", Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<decimal>.SolveMinVariance(new decimal[,] { { 1, 0 }, { 0, 1 } }, 0.7m, 0.6m)).ParamName);
    }

    [Theory]
    [InlineData(0.0, 0.2)]
    [InlineData(0.6, 1.0)]
    public void InfeasibleBounds_KeepTheDocumentedClampToTheEqualWeightFrontier(double lower, double upper)
    {
        // n·ub < 1 or n·lb > 1: only the offending bound is relaxed to 1/n, and a feasible vector comes back.
        var cov = new double[,] { { 1, 0.2, 0 }, { 0.2, 2, 0.1 }, { 0, 0.1, 3 } };
        var weights = ActiveSetQpSolver<double>.SolveMinVariance(cov, lower, upper);
        var effectiveLower = Math.Min(lower, 1.0 / 3);
        var effectiveUpper = Math.Max(upper, 1.0 / 3);
        Assert.All(weights, w => Assert.InRange(w, effectiveLower, effectiveUpper));
        Assert.InRange(Math.Abs(weights.Sum() - 1), 0, 1e-15);

        var decimalWeights = ActiveSetQpSolver<decimal>.SolveMinVariance(
            new decimal[,] { { 1, 0.2m, 0 }, { 0.2m, 2, 0.1m }, { 0, 0.1m, 3 } }, (decimal)lower, (decimal)upper);
        Assert.All(decimalWeights, w => Assert.InRange(w, Math.Min((decimal)lower, 1m / 3), Math.Max((decimal)upper, 1m / 3)));
        Assert.InRange(Math.Abs(decimalWeights.Sum() - 1m), 0m, 1e-26m);

        // The λ = 0 greedy LP keeps its own documented refusal of infeasible bounds.
        var lp = Assert.Throws<ArgumentException>(() => ActiveSetQpSolver<double>.SolveMeanVariance(cov, [0.1, 0.2, 0.3], 0, lower, upper));
        Assert.Equal(lower > 0 ? "minWeight" : "maxWeight", lp.ParamName);
    }

    [Fact]
    public void HalfPrecision_IsRejectedWithNotSupportedException()
    {
        Half[,] cov = { { (Half)1, (Half)0 }, { (Half)0, (Half)1 } };
        Assert.Throws<NotSupportedException>(() => ActiveSetQpSolver<Half>.SolveMinVariance(cov, (Half)0, (Half)1));
    }

    [Fact]
    public void NullCovariance_IsAnArgumentNullException()
    {
        Assert.Equal("covariance", Assert.Throws<ArgumentNullException>(() => ActiveSetQpSolver<double>.SolveMinVariance(null!, 0, 1)).ParamName);
    }

    // ------------------------------------------------------------------------------------------
    // The five SLSQP verification vectors (5e-6 per weight), replayed through the production entry points
    // ------------------------------------------------------------------------------------------

    [Fact]
    public void SlsqpVerificationVectors_StillMatchWithin5e6PerWeight()
    {
        var path = FindRepositoryFile(Path.Combine("tests", "Verification", "vectors", "qp_solver.json"));
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var cases = doc.RootElement.GetProperty("cases").EnumerateArray().ToList();
        Assert.Equal(5, cases.Count);
        Assert.Equal(3, cases.Count(c => c.GetProperty("kind").GetString() == "minvar"));

        foreach (var c in cases)
        {
            var name = c.GetProperty("name").GetString()!;
            var rows = c.GetProperty("covariance").EnumerateArray().Select(r => r.EnumerateArray().Select(x => x.GetDouble()).ToArray()).ToArray();
            var n = rows.Length;
            var cov = new double[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    cov[i, j] = rows[i][j];
                }
            }

            var lb = c.GetProperty("min_weight").GetDouble();
            var ub = c.GetProperty("max_weight").GetDouble();
            var expected = c.GetProperty("weights").EnumerateArray().Select(x => x.GetDouble()).ToArray();
            var actual = c.GetProperty("kind").GetString() == "minvar"
                ? ActiveSetQpSolver<double>.SolveMinVariance(cov, lb, ub)
                : ActiveSetQpSolver<double>.SolveMeanVariance(
                    cov,
                    c.GetProperty("means").EnumerateArray().Select(x => x.GetDouble()).ToArray(),
                    c.GetProperty("risk_aversion").GetDouble(),
                    lb,
                    ub);

            Assert.Equal(expected.Length, actual.Length);
            for (var i = 0; i < expected.Length; i++)
            {
                Assert.True(Math.Abs(actual[i] - expected[i]) <= 5e-6, name + "[" + i + "]: " + actual[i] + " vs SLSQP " + expected[i]);
            }
        }
    }

    private static string FindRepositoryFile(string relative)
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, relative);
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new FileNotFoundException("repository file not found above the test output directory", relative);
    }
}
