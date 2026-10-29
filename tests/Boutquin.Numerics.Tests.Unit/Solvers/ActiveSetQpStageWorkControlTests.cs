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
/// The work of the certified stages of <see cref="ActiveSetQpSolver{T}"/>, observed deterministically: each stage counts
/// the loop-body executions of its own loops as they run, and the run's order, so a stage that ran more often than
/// documented, or whose loops did more than their documented O(n³) work, fails here on every host. The stages: the
/// double-double test of positive definiteness (<c>CertifiedLeadingMinors</c>, once per solve when the double
/// certificate cannot decide), the conditioning guard's whole-matrix eigenvalue bound (at most once per solve), and the
/// guard's two per-block bounds (the bound in <c>double</c> and the double-double eigenvalue bound of the block, each at
/// most once per queried block). Every case runs at both guaranteed types through both objectives.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpStageWorkControlTests(ITestOutputHelper output)
{
    private const string Refusal = "Covariance matrix must be positive definite (leading principal minor of order ";

    /// <summary>
    /// The double-double certificate runs once per solve the double certificate cannot decide, and its factorisation,
    /// residual bound and refusal witness execute at most <c>2n³/3 + 8n²</c> loop bodies (the square term the exact
    /// products of the integer witnesses, O(k²) for a refusal at order k), whether it accepts, refuses with an
    /// order, or leaves the decision to the exact elimination: about <c>n³/3</c> to prove the covariance (asserted at
    /// least <c>n³/4</c>), up to about <c>n³/2</c> to refuse it (the witness factors the leading block once more); a
    /// covariance the double certificate proves never reaches it. A second run, or a factorisation repeated within one,
    /// exceeds the bound at both orders.
    /// </summary>
    [Theory]
    [InlineData(60)]
    [InlineData(120)]
    public void LeadingMinors_RunsOncePerUndecidedSolveWithinCubicWork(int n)
    {
        var undecided = n == 120 ? 48 : 24;
        AssertLeadingMinors(ActiveSetQpExactFallbackCostTests.IllConditionedDecimal(n, kappa: 1e16, seed: 20261111), refusedOrder: 0, Route<decimal>.DoubleDouble);
        AssertLeadingMinors(ActiveSetQpExactFallbackCostTests.NearDuplicatePair(n, seed: 20261112, deltaBits: 40), refusedOrder: 0, Route<double>.DoubleDouble);
        AssertLeadingMinors(ActiveSetQpExactFallbackCostTests.Dependent<double>(n, independent: n / 2, seed: 20261113), refusedOrder: (n / 2) + 1, Route<double>.DoubleDouble);
        AssertLeadingMinors(ActiveSetQpExactFallbackCostTests.Repeated<decimal>(n, seed: 20261114), refusedOrder: n, Route<decimal>.DoubleDouble);
        AssertLeadingMinors(ActiveSetQpExactFallbackCostTests.IllConditionedDecimal(n, kappa: 1e3, seed: 20261115), refusedOrder: 0, Route<decimal>.Double);
        AssertLeadingMinors(ThirdDependent<double>(undecided, seed: 20261116), refusedOrder: undecided, Route<double>.Exact);
        AssertLeadingMinors(ThirdDependent<decimal>(undecided, seed: 20261117), refusedOrder: undecided, Route<decimal>.Exact);
    }

    /// <summary>
    /// <c>9·G·2⁻⁸</c> for an integer positive-definite three-factor Gram matrix <c>G</c> of order <c>n − 1</c>, followed by
    /// the asset <c>(x₀ + x₁)/3</c>: exactly singular, with the null vector <c>(1, 1, 0, …, −3)</c>, whose
    /// <c>(−y, 1)</c> form has components 1/3 on no binary or decimal grid. No integer witness of the double-double
    /// certificate is an exact null vector, so the exact elimination decides the refusal, at order <c>n</c>. Every entry
    /// is an integer times 2⁻⁸, exact in double and in decimal.
    /// </summary>
    internal static T[,] ThirdDependent<T>(int n, int seed)
        where T : IFloatingPoint<T>
    {
        var rng = new System.Random(seed);
        var m = n - 1;
        var f = new int[m, 3];
        var d = new int[m];
        for (var i = 0; i < m; i++)
        {
            for (var k = 0; k < 3; k++)
            {
                f[i, k] = rng.Next(-8, 9);
            }

            d[i] = rng.Next(3, 13);
        }

        var g = new long[m, m];
        for (var i = 0; i < m; i++)
        {
            for (var j = 0; j < m; j++)
            {
                long sum = i == j ? d[i] : 0;
                for (var k = 0; k < 3; k++)
                {
                    sum += f[i, k] * f[j, k];
                }

                g[i, j] = sum;
            }
        }

        var scale = T.One / T.CreateChecked(256);
        var cov = new T[n, n];
        for (var i = 0; i < m; i++)
        {
            for (var j = 0; j < m; j++)
            {
                cov[i, j] = T.CreateChecked(9 * g[i, j]) * scale;
            }

            cov[m, i] = cov[i, m] = T.CreateChecked(3 * (g[0, i] + g[1, i])) * scale;
        }

        cov[m, m] = T.CreateChecked(g[0, 0] + (2 * g[0, 1]) + g[1, 1]) * scale;
        return cov;
    }

    private static void AssertLeadingMinors<T>(T[,] cov, int refusedOrder, ActiveSetQpSolver<T>.PositiveDefinitenessRoute route)
        where T : IFloatingPoint<T>
    {
        var n = cov.GetLength(0);
        var means = new T[n];
        for (var i = 0; i < n; i++)
        {
            means[i] = T.CreateChecked((i % 5) - 2) / T.CreateChecked(1000);
        }

        foreach (var meanVariance in new[] { false, true })
        {
            var trace = new ActiveSetQpSolver<T>.RouteTrace();
            var label = string.Create(CultureInfo.InvariantCulture, $"{typeof(T).Name} n={n} {route} {(meanVariance ? "mean-variance" : "min-variance")}");
            string? message = null;
            try
            {
                _ = meanVariance
                    ? ActiveSetQpSolver<T>.SolveMeanVarianceUsingCore(cov, means, T.CreateChecked(4), T.Zero, T.One, trace: trace)
                    : ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(cov, T.Zero, T.One, trace: trace);
            }
            catch (InvalidOperationException ex)
            {
                message = ex.Message;
            }

            Assert.True(route == trace.PositiveDefinitenessRoute, label + ": decided by " + trace.PositiveDefinitenessRoute);
            if (refusedOrder > 0)
            {
                Assert.Equal(Refusal + refusedOrder.ToString(CultureInfo.InvariantCulture) + " is not positive).", message);
            }
            else
            {
                Assert.False(message is not null && message.StartsWith(Refusal, StringComparison.Ordinal), label + ": " + message);
            }

            var cube = (long)n * n * n;
            var certificate = trace.DoubleCertificate;
            var minors = trace.LeadingMinors;
            Assert.True(certificate.Runs == 1 && certificate.RepeatedRuns == 0, label + ": the double certificate ran " + certificate.Runs + " times");
            if (route == ActiveSetQpSolver<T>.PositiveDefinitenessRoute.DoubleCertificate)
            {
                Assert.True(minors.Runs == 0 && minors.Iterations == 0, label + ": the double-double certificate ran " + minors.Runs + " times");
                continue;
            }

            Assert.True(minors.Runs == 1 && minors.RepeatedRuns == 0, label + ": the double-double certificate ran " + minors.Runs + " times");
            Assert.True(minors.OrderCubes == cube, label + ": recorded order cubes " + minors.OrderCubes + " against n³ = " + cube);
            Assert.True(minors.Iterations <= (2 * cube / 3) + (8L * n * n), label + ": " + minors.Iterations + " loop bodies against n³ = " + cube);
            Assert.True(refusedOrder > 0 || minors.Iterations >= cube / 4, label + ": " + minors.Iterations + " loop bodies against n³ = " + cube);
        }
    }

    /// <summary>
    /// The conditioning guard's stages, on factor models whose blocks the bound in <c>double</c> settles (κ = 1e3), whose
    /// blocks it cannot settle (κ = 1e16, so the whole-matrix eigenvalue bound is prepared and settles them), and with a
    /// near-duplicate of asset 0 appended (variance raised by 1e-26), whose free blocks neither the whole-matrix bound nor
    /// the block's bound in double settles, so that several blocks of one solve reach the block's own double-double
    /// eigenvalue bound; in <c>double</c>, a long-short box. Per solve: the whole-matrix bound is prepared at most once
    /// and then on the whole order, within <c>n³/2 + 4n²</c> loop bodies; each per-block bound runs at most once per
    /// queried block, the bound in double within <c>3·Σm³ + 8·Σm²</c> and the double-double one within
    /// <c>Σm³/2 + 4·Σm²</c> over the orders <c>m</c> of the blocks it ran on.
    /// </summary>
    [Theory]
    [InlineData(30)]
    [InlineData(60)]
    public void ConditioningStages_RunOncePerSolveAndOncePerQueriedBlockWithinCubicWork(int n)
    {
        var totals = new long[4]; // block-bound loop bodies, its Σm³, eigenvalue-bound loop bodies, its Σm³
        var population = new int[3]; // solves that prepared, solves whose eigenvalue bound ran on two or more blocks, double solves whose bound in double ran
        var settled = ActiveSetQpExactFallbackCostTests.IllConditionedDecimal(n, kappa: 1e3, seed: 20261121);
        var prepared = ActiveSetQpExactFallbackCostTests.IllConditionedDecimal(n, kappa: 1e16, seed: 20261122);
        var duplicated = AppendNearDuplicate(ActiveSetQpExactFallbackCostTests.IllConditionedDecimal(n, kappa: 1e16, seed: 20261123), 0.00000000000000000000000001m);
        var source = ActiveSetQpExactFallbackCostTests.IllConditionedDecimal(n, kappa: 1e13, seed: 20261124);
        var binary = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                binary[i, j] = (double)source[i, j];
            }
        }

        AssertConditioningStages(settled, -0.5m, 0.5m, totals, population);
        AssertConditioningStages(prepared, 0m, 0.15m, totals, population);
        AssertConditioningStages(duplicated, -0.5m, 0.5m, totals, population);
        AssertConditioningStages(binary, -0.5, 0.5, totals, population);

        var label = string.Create(CultureInfo.InvariantCulture, $"n={n}: prepared {population[0]}, eigenvalue bound on two or more blocks {population[1]}, double solves with the bound in double {population[2]}");
        Assert.True(population[0] > 0 && population[1] > 0 && population[2] > 0, label);

        // A stage that stopped counting its loops would pass every ceiling above: the work observed is at least the
        // leading term of its documented cost.
        Assert.True(totals[0] >= totals[1] && totals[2] >= totals[3] / 4, label + string.Create(CultureInfo.InvariantCulture, $"; block bound {totals[0]} loop bodies over Σm³ = {totals[1]}, eigenvalue bound {totals[2]} over Σm³ = {totals[3]}"));
    }

    // Appends a near-duplicate of asset 0 whose variance exceeds asset 0's by delta.
    private static decimal[,] AppendNearDuplicate(decimal[,] cov, decimal delta)
    {
        var n = cov.GetLength(0);
        var a = new decimal[n + 1, n + 1];
        for (var i = 0; i <= n; i++)
        {
            for (var j = 0; j <= n; j++)
            {
                a[i, j] = cov[i == n ? 0 : i, j == n ? 0 : j];
            }
        }

        a[n, n] += delta;
        return a;
    }

    private void AssertConditioningStages<T>(T[,] cov, T lower, T upper, long[] totals, int[] population)
        where T : IFloatingPoint<T>
    {
        var n = cov.GetLength(0);
        var means = new T[n];
        for (var i = 0; i < n; i++)
        {
            means[i] = T.CreateChecked((i % 5) - 2) / T.CreateChecked(1000);
        }

        foreach (var meanVariance in new[] { false, true })
        {
            var trace = new ActiveSetQpSolver<T>.RouteTrace();
            var label = string.Create(CultureInfo.InvariantCulture, $"{typeof(T).Name} n={n} box [{lower}, {upper}] {(meanVariance ? "mean-variance" : "min-variance")}");
            try
            {
                _ = meanVariance
                    ? ActiveSetQpSolver<T>.SolveMeanVarianceUsingCore(cov, means, T.CreateChecked(4), lower, upper, trace: trace)
                    : ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(cov, lower, upper, trace: trace);
            }
            catch (InvalidOperationException ex) when (ex.Message.Contains("numerically singular", StringComparison.Ordinal))
            {
                // The guard's documented refusal: its stages ran on the blocks queried before it.
            }

            output.WriteLine(AssertConditioningWork(trace, n, label));
            var block = trace.BlockMarginBound;
            var eigenvalue = trace.BlockEigenvalueBound;
            var preparation = trace.ConditioningPreparation;
            totals[0] += block.Iterations;
            totals[1] += block.OrderCubes;
            totals[2] += eigenvalue.Iterations;
            totals[3] += eigenvalue.OrderCubes;
            population[0] += preparation.Runs;
            population[1] += eigenvalue.Runs >= 2 ? 1 : 0;
            population[2] += typeof(T) == typeof(double) && block.Runs > 0 ? 1 : 0;
        }
    }

    /// <summary>
    /// The documented frequency and work of the conditioning guard's stages over one solve of order <paramref name="n"/>
    /// (see <see cref="ConditioningStages_RunOncePerSolveAndOncePerQueriedBlockWithinCubicWork"/>); returns the line that
    /// reports them.
    /// </summary>
    internal static string AssertConditioningWork<T>(ActiveSetQpSolver<T>.RouteTrace trace, int n, string label)
        where T : IFloatingPoint<T>
    {
        var whole = (long)n * n;
        var queries = trace.ConditioningQueries;
        var preparation = trace.ConditioningPreparation;
        var block = trace.BlockMarginBound;
        var eigenvalue = trace.BlockEigenvalueBound;
        var line = string.Create(CultureInfo.InvariantCulture, $"{label}: {queries} queried blocks; preparation {Describe(preparation)}; bound in double {Describe(block)}; eigenvalue bound {Describe(eigenvalue)}");

        Assert.True(preparation.Runs <= 1 && preparation.RepeatedRuns == 0, line + ": the whole-matrix bound was prepared " + preparation.Runs + " times");
        Assert.True(preparation.Runs == 0 || preparation.OrderCubes == whole * n, line);
        Assert.True(preparation.Iterations <= (whole * n / 2) + (4 * whole), line);
        Assert.True(block.RepeatedRuns == 0 && block.Runs <= queries, line + ": the bound in double ran twice on one block");
        Assert.True(block.Iterations <= (3 * block.OrderCubes) + (8 * block.OrderSquares), line);
        Assert.True(eigenvalue.RepeatedRuns == 0 && eigenvalue.Runs <= queries, line + ": the eigenvalue bound ran twice on one block");
        Assert.True(eigenvalue.Iterations <= (eigenvalue.OrderCubes / 2) + (4 * eigenvalue.OrderSquares), line);
        return line;
    }

    internal static string Describe<T>(ActiveSetQpSolver<T>.StageWork work)
        where T : IFloatingPoint<T> =>
        string.Create(CultureInfo.InvariantCulture, $"{work.Runs} runs ({work.RepeatedRuns} repeated), {work.Iterations} loop bodies over Σm³ = {work.OrderCubes}");

    private static class Route<T>
        where T : IFloatingPoint<T>
    {
        public const ActiveSetQpSolver<T>.PositiveDefinitenessRoute Double = ActiveSetQpSolver<T>.PositiveDefinitenessRoute.DoubleCertificate;
        public const ActiveSetQpSolver<T>.PositiveDefinitenessRoute DoubleDouble = ActiveSetQpSolver<T>.PositiveDefinitenessRoute.DoubleDoubleCertificate;
        public const ActiveSetQpSolver<T>.PositiveDefinitenessRoute Exact = ActiveSetQpSolver<T>.PositiveDefinitenessRoute.ExactElimination;
    }
}
