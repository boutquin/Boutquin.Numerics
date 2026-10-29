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
using System.Reflection;

using Boutquin.Numerics.Internal;
using Boutquin.Numerics.Solvers;

using Xunit.Abstractions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// The positive-definiteness test of <see cref="ActiveSetQpSolver{T}"/> on the covariances that a <c>double</c>
/// Cholesky certificate cannot decide: ill-conditioned <c>decimal</c> covariances the conditioning guard accepts, a
/// near-duplicate pair of assets, and covariances that must be refused (dependent assets, the commonest invalid input,
/// and an indefinite covariance whose first non-positive leading principal minor is the last one). Decided by exact
/// fraction-free elimination of the whole matrix alone, each costs about n^5 on multi-digit integers (2 to 7 s at
/// n = 100 to 130, several times the ceiling of <see cref="ActiveSetQpLargeProblemTests"/>). Each case's outcome is
/// known independently of the solver: positive definite by construction, or refused at an order fixed by construction
/// (an exact Schur complement of <c>-2^-10</c> or exactly 0), or, for the rounded sample covariance, at the order the exact-only test
/// named.
/// </summary>
[Collection(ActiveSetQpPerformanceCollection.Name)]
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpExactFallbackCostTests(ITestOutputHelper output)
{
    private const string Refusal = "Covariance matrix must be positive definite (leading principal minor of order ";

    [Fact]
    public void SolveMinVariance_IllConditionedDecimalFactorModelOfOrder120_IsSolvedWithinTheCalibratedCeiling()
    {
        const int n = 120;
        var cov = IllConditionedDecimal(n, kappa: 1e13, seed: 20261031);
        WarmUp();

        var (weights, ms) = ActiveSetQpLargeProblemTests.TimedCpu(() => ActiveSetQpSolver<decimal>.SolveMinVariance(cov, 0m, 0.15m));

        Assert.True(Math.Abs(weights.Sum() - 1m) <= 1e-25m, weights.Sum().ToString(CultureInfo.InvariantCulture));
        Assert.All(weights, w => Assert.True(w >= 0m && w <= 0.15m, w.ToString(CultureInfo.InvariantCulture)));
        ActiveSetQpLargeProblemTests.AssertOptimal(cov, weights, 0m, 0.15m, means: null, 1m, ActiveSetQpLargeProblemTests.DecimalStationarity);
        AssertWithinCeiling("decimal ill-conditioned factor model n=120 kappa=1e13", ms);
    }

    [Fact]
    public void SolveMinVariance_NearDuplicatePairOfOrder130_IsSolvedWithinTheCalibratedCeiling()
    {
        const int n = 130;
        var cov = NearDuplicatePair(n, seed: 20261032, deltaBits: 40);
        WarmUp();

        var (weights, ms) = ActiveSetQpLargeProblemTests.TimedCpu(() => ActiveSetQpSolver<double>.SolveMinVariance(cov, 0.0, 0.15));

        Assert.Equal(1.0, weights.Sum(), 1e-12);
        Assert.All(weights, w => Assert.InRange(w, 0.0, 0.15));
        ActiveSetQpLargeProblemTests.AssertOptimal(cov, weights, 0.0, 0.15, means: null, 1.0, ActiveSetQpLargeProblemTests.DoubleStationarity);
        AssertWithinCeiling("double near-duplicate pair n=130", ms);
    }

    [Fact]
    public void SolveMinVariance_DependentAssetsOfOrder130_AreRefusedAtTheFirstDependentOrderWithinTheCalibratedCeiling()
    {
        var cov = Dependent<double>(n: 130, independent: 70, seed: 20261033);
        WarmUp();

        var (message, ms) = TimedRefusal(() => ActiveSetQpSolver<double>.SolveMinVariance(cov, 0.0, 0.15));

        Assert.Equal(Refusal + "71 is not positive).", message);
        AssertWithinCeiling("double dependent assets n=130 (refused at order 71)", ms);
    }

    [Fact]
    public void SolveMinVariance_RoundedSampleCovarianceOfSixtyObservationsOfOrder100_IsRefusedWithinTheCalibratedCeiling()
    {
        // The exact-only test (the solver before its certified decisions) names order 60 for this input.
        var cov = SampleCovariance(n: 100, observations: 60, seed: 20261034);
        WarmUp();

        var (message, ms) = TimedRefusal(() => ActiveSetQpSolver<double>.SolveMinVariance(cov, 0.0, 0.15));

        Assert.Equal(Refusal + "60 is not positive).", message);
        AssertWithinCeiling("double rounded sample covariance n=100 T=60", ms);
    }

    [Fact]
    public void SolveMinVariance_LastMinorNegativeOfOrder130_IsRefusedAtOrder130WithinTheCalibratedCeiling()
    {
        var cov = Dependent<double>(n: 130, independent: 129, seed: 20261035);
        WarmUp();

        var (message, ms) = TimedRefusal(() => ActiveSetQpSolver<double>.SolveMinVariance(cov, 0.0, 0.15));

        Assert.Equal(Refusal + "130 is not positive).", message);
        AssertWithinCeiling("double last minor negative n=130", ms);
    }

    [Fact]
    public void SolveMeanVariance_LastMinorNegativeOfOrder130_IsRefusedAtOrder130WithinTheCalibratedCeiling()
    {
        const int n = 130;
        var cov = Dependent<double>(n, independent: n - 1, seed: 20261036);
        var means = Enumerable.Range(0, n).Select(i => 0.02 + (0.0005 * i)).ToArray();
        WarmUp();

        var (message, ms) = TimedRefusal(() => ActiveSetQpSolver<double>.SolveMeanVariance(cov, means, 2.0, 0.0, 0.15));

        Assert.Equal(Refusal + "130 is not positive).", message);
        AssertWithinCeiling("double last minor negative mean-variance n=130", ms);
    }

    [Fact]
    public void SolveMinVariance_DecimalLastMinorNegativeOfOrder110_IsRefusedAtOrder110WithinTheCalibratedCeiling()
    {
        var cov = Dependent<decimal>(n: 110, independent: 109, seed: 20261037);
        WarmUp();

        var (message, ms) = TimedRefusal(() => ActiveSetQpSolver<decimal>.SolveMinVariance(cov, 0m, 0.15m));

        Assert.Equal(Refusal + "110 is not positive).", message);
        AssertWithinCeiling("decimal last minor negative n=110", ms);
    }

    [Fact]
    public void SolveMinVariance_RepeatedAssetOfOrder130_IsRefusedAtOrder130WithinTheCalibratedCeiling()
    {
        // The last asset repeats asset 0 exactly: the order-130 minor is exactly zero.
        var cov = Repeated<double>(n: 130, seed: 20261038);
        WarmUp();

        var (message, ms) = TimedRefusal(() => ActiveSetQpSolver<double>.SolveMinVariance(cov, 0.0, 0.15));

        Assert.Equal(Refusal + "130 is not positive).", message);
        AssertWithinCeiling("double repeated asset n=130", ms);
    }

    [Fact]
    public void PositiveDefiniteness_InMarginNearDuplicateDecimalCovarianceOfOrder240_IsDecidedByTheDoubleDoubleCertificateWithinTheCalibratedCeiling()
    {
        // Smallest eigenvalue 1.25e-24 against a 1-norm of 1: 48·n·u·κ₁ = 0.92 with u = 1e-28, inside the conditioning
        // guard's margin, so the positive-definiteness decision must not fall to the exact elimination (about n^5).
        var cov = InMarginNearDuplicate(240, delta: 0.00000000000000000000000125m);
        WarmUp();

        var (trace, message, ms) = DecidePositiveDefiniteness(cov);

        Assert.Null(message);
        Assert.Equal(ActiveSetQpSolver<decimal>.PositiveDefinitenessRoute.DoubleDoubleCertificate, trace.PositiveDefinitenessRoute);

        // The deterministic cost of the decision, the same on every host: one run of each certificate, the double-double
        // one within its documented O(n³) (about n³/3 loop bodies to prove the covariance).
        const long cube = 240L * 240 * 240;
        var certificate = trace.DoubleCertificate;
        var minors = trace.LeadingMinors;
        Assert.True(certificate.Runs == 1 && minors.Runs == 1 && minors.RepeatedRuns == 0, "certificate runs: double " + certificate.Runs + ", double-double " + minors.Runs);
        Assert.True(minors.Iterations <= cube / 2 && minors.Iterations >= cube / 4, "double-double certificate: " + minors.Iterations + " loop bodies against n³ = " + cube);
        AssertWithinCeiling("decimal in-margin near-duplicate pair n=240 (positive-definiteness decision)", ms);
    }

    [Fact]
    public void SolveMinVariance_ExactlyDependentAssetsOfOrder130_AreRefusedAtTheFirstDependentOrderWithinTheCalibratedCeiling()
    {
        // The first later asset is an exact quarter-coefficient combination of the first 70: its Schur complement is 0.
        var cov = Dependent<double>(n: 130, independent: 70, seed: 20261051, exactlyDependent: true);
        WarmUp();

        var (message, ms) = TimedRefusal(() => ActiveSetQpSolver<double>.SolveMinVariance(cov, 0.0, 0.15));

        Assert.Equal(Refusal + "71 is not positive).", message);
        AssertWithinCeiling("double exactly dependent assets n=130 (refused at order 71)", ms);
    }

    // Each of these covariances is decided by the certified double-double test, not by the exact elimination, with the
    // outcome fixed by construction: positive definite, or refused at the stated order.
    [Fact]
    public void CovariancesTheDoubleCertificateCannotDecide_AreDecidedByTheDoubleDoubleCertificate()
    {
        AssertDecided(IllConditionedDecimal(80, kappa: 1e14, seed: 20261041), 0m, 0.15m, refusedOrder: 0);
        AssertDecided(IllConditionedDecimal(60, kappa: 1e16, seed: 20261042), 0m, 0.15m, refusedOrder: 0);
        AssertDecided(NearDuplicatePair(100, seed: 20261043, deltaBits: 40), 0.0, 0.15, refusedOrder: 0);
        AssertDecided(Dependent<double>(100, independent: 60, seed: 20261044), 0.0, 0.15, refusedOrder: 61);
        AssertDecided(Dependent<double>(100, independent: 99, seed: 20261045), 0.0, 0.15, refusedOrder: 100);
        AssertDecided(Dependent<decimal>(80, independent: 79, seed: 20261046), 0m, 0.15m, refusedOrder: 80);
        AssertDecided(Repeated<double>(100, seed: 20261047), 0.0, 0.15, refusedOrder: 100);
        AssertDecided(Repeated<decimal>(80, seed: 20261048), 0m, 0.15m, refusedOrder: 80);
        AssertDecided(SampleCovariance(100, observations: 60, seed: 20261034), 0.0, 0.15, refusedOrder: 60);

        // An exact dependence with quarter coefficients: only a witness on a grid finer than the unit (1/4 for binary,
        // 1/100 for decimal) is an exact null vector of the first singular leading block.
        AssertDecided(Dependent<double>(100, independent: 60, seed: 20261052, exactlyDependent: true), 0.0, 0.15, refusedOrder: 61);
        AssertDecided(Dependent<double>(100, independent: 99, seed: 20261053, exactlyDependent: true), 0.0, 0.15, refusedOrder: 100);
        AssertDecided(Dependent<decimal>(80, independent: 79, seed: 20261054, exactlyDependent: true), 0m, 0.15m, refusedOrder: 80);
    }

    private static void AssertDecided<T>(T[,] cov, T lower, T upper, int refusedOrder)
        where T : IFloatingPoint<T>
    {
        var trace = new ActiveSetQpSolver<T>.RouteTrace();
        string? message = null;
        try
        {
            _ = ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(cov, lower, upper, trace: trace);
        }
        catch (InvalidOperationException ex)
        {
            message = ex.Message;
        }

        var label = typeof(T).Name + " n=" + cov.GetLength(0).ToString(CultureInfo.InvariantCulture) + ": " + (message ?? "solved");
        Assert.True(ActiveSetQpSolver<T>.PositiveDefinitenessRoute.DoubleDoubleCertificate == trace.PositiveDefinitenessRoute, label + ", decided by " + trace.PositiveDefinitenessRoute);
        if (refusedOrder > 0)
        {
            Assert.Equal(Refusal + refusedOrder.ToString(CultureInfo.InvariantCulture) + " is not positive).", message);
        }
        else
        {
            // A covariance proved positive definite is solved: no refusal of any kind.
            Assert.True(message is null, label);
        }
    }

    // The solve's first decision alone (RequirePositiveDefinite on the equilibrated covariance), as SolveMinVariance makes it.
    private static (ActiveSetQpSolver<decimal>.RouteTrace Trace, string? Message, double Ms) DecidePositiveDefiniteness(decimal[,] cov)
    {
        var n = cov.GetLength(0);
        var scaled = QpScaling<decimal>.EquilibrateCovariance(cov).Covariance;
        var solver = typeof(ActiveSetQpSolver<decimal>);
        var exact = solver.GetNestedType("Exact", BindingFlags.NonPublic)!.MakeGenericType(typeof(decimal));
        var primal = solver.GetNestedType("PrimalActiveSet", BindingFlags.NonPublic)!.MakeGenericType(typeof(decimal));
        var from = exact.GetMethod("From", BindingFlags.Public | BindingFlags.Static)!;
        var decide = primal.GetMethod("RequirePositiveDefinite", BindingFlags.NonPublic | BindingFlags.Static)!;
        var sigma = Array.CreateInstance(exact, n, n);
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                sigma.SetValue(from.Invoke(null, [scaled[i, j]]), i, j);
            }
        }

        var ((trace, message), cpuMs) = ActiveSetQpLargeProblemTests.TimedCpu(() =>
        {
            var attempt = new ActiveSetQpSolver<decimal>.RouteTrace();
            string? refusal = null;
            try
            {
                _ = decide.Invoke(null, [sigma, n, attempt]);
            }
            catch (TargetInvocationException ex) when (ex.InnerException is InvalidOperationException inner)
            {
                refusal = inner.Message;
            }

            return (attempt, refusal);
        });

        return (trace, message, cpuMs);
    }

    private static void WarmUp()
    {
        _ = ActiveSetQpSolver<double>.SolveMinVariance(Dependent<double>(6, 6, seed: 1), 0.0, 1.0);
        _ = ActiveSetQpSolver<decimal>.SolveMinVariance(Dependent<decimal>(6, 6, seed: 1), 0m, 1m);
    }

    private void AssertWithinCeiling(string label, double ms)
    {
        var calibration = ActiveSetQpPerformanceTests.CalibrationMs();
        var ratio = ms / calibration;
        var line = string.Create(CultureInfo.InvariantCulture, $"{label}: {ms:F1} ms CPU = {ratio:F1}x the calibration workload ({calibration:F3} ms); ceiling {ActiveSetQpLargeProblemTests.CalibratedCeiling}x");
        output.WriteLine(line);
        Assert.True(ratio <= ActiveSetQpLargeProblemTests.CalibratedCeiling, line);
    }

    private static (string Message, double CpuMs) TimedRefusal<T>(Func<T[]> solve) =>
        ActiveSetQpLargeProblemTests.TimedCpu(() => Assert.Throws<InvalidOperationException>(() => solve()).Message);

    /// <summary>
    /// A dense three-factor model <c>B·B′ + ε·diag(1 … 1.5)</c> in <see cref="decimal"/>: loadings of four decimal places
    /// in [-0.1, 0.3), so the Gram matrix is exact; <c>ε</c> sets the condition number near <paramref name="kappa"/>.
    /// Positive definite by construction (an exact positive semidefinite Gram matrix plus a positive diagonal).
    /// </summary>
    internal static decimal[,] IllConditionedDecimal(int n, double kappa, int seed)
    {
        var rng = new System.Random(seed);
        var b = new decimal[n, 3];
        var top = 0.0;
        for (var i = 0; i < n; i++)
        {
            for (var k = 0; k < 3; k++)
            {
                b[i, k] = Math.Round((decimal)((rng.NextDouble() * 0.4) - 0.1), 4);
                top += (double)(b[i, k] * b[i, k]);
            }
        }

        var epsilon = decimal.Parse((top / kappa).ToString("E6", CultureInfo.InvariantCulture), NumberStyles.Float, CultureInfo.InvariantCulture);
        var cov = new decimal[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                var sum = 0m;
                for (var k = 0; k < 3; k++)
                {
                    sum += b[i, k] * b[j, k];
                }

                if (i == j)
                {
                    sum += epsilon * (1m + Math.Round((decimal)rng.NextDouble() * 0.5m, 3));
                }

                cov[i, j] = cov[j, i] = sum;
            }
        }

        return cov;
    }

    /// <summary>
    /// <c>0.5·I</c> in <see cref="decimal"/> with the last two assets a near-duplicate pair (covariance <c>0.5 − δ</c>):
    /// <c>(e_{n−2} − e_{n−1})/√2</c> is an exact eigenvector with eigenvalue <c>δ</c>, every other eigenvalue is 0.5 or
    /// <c>1 − δ</c>, <c>‖A‖₁ = 1 − δ</c> and <c>‖A⁻¹‖₁ = 1/δ</c>, so <c>κ₁ = (1 − δ)/δ</c> and the whole matrix is inside
    /// the conditioning guard's margin <c>48·n·u·κ₁ &lt; 1</c> (<c>u = 1e-28</c>) when <c>δ &gt; 48·n·u</c>. Positive
    /// definite by construction.
    /// </summary>
    internal static decimal[,] InMarginNearDuplicate(int n, decimal delta)
    {
        var cov = new decimal[n, n];
        for (var i = 0; i < n; i++)
        {
            cov[i, i] = 0.5m;
        }

        cov[n - 2, n - 1] = cov[n - 1, n - 2] = 0.5m - delta;
        return cov;
    }

    /// <summary>
    /// A three-factor model <c>F·F′ + diag(d)</c> computed exactly in <see cref="double"/> (loadings multiples of 2^-4,
    /// idiosyncratic variances multiples of 2^-8 in [0.012, 0.05)) in which asset 1 repeats asset 0's loadings and both
    /// carry the idiosyncratic variance <c>2^-deltaBits</c>: positive definite by construction, the pair's correlation
    /// about <c>1 − 2^-deltaBits</c>.
    /// </summary>
    internal static double[,] NearDuplicatePair(int n, int seed, int deltaBits)
    {
        var rng = new System.Random(seed);
        var (f, d) = Factors<double>(n, rng);
        for (var k = 0; k < 3; k++)
        {
            f[1, k] = f[0, k];
        }

        d[0] = d[1] = Math.ScaleB(1.0, -deltaBits);
        return Gram(f, d);
    }

    /// <summary>
    /// The first <paramref name="independent"/> assets form a three-factor model <c>F·F′ + diag(d)</c> (exact in
    /// <typeparamref name="T"/>: loadings multiples of 1/16, idiosyncratic variances multiples of 1/256). Every later asset
    /// is an exact combination <c>c</c> of them (coefficients multiples of 1/4 in [-1, 1]) whose variance is
    /// <c>c′Σc − 2^-10</c>, so the Schur complement of the first later asset is exactly <c>-2^-10</c> (exactly 0 when
    /// <paramref name="exactlyDependent"/>, the variance then being <c>c′Σc</c>): the leading block of order
    /// <paramref name="independent"/> is positive definite and the first non-positive leading principal minor has order
    /// <c>independent + 1</c>. Every value is a short binary fraction, exact in double and in decimal.
    /// </summary>
    internal static T[,] Dependent<T>(int n, int independent, int seed, bool exactlyDependent = false)
        where T : IFloatingPoint<T>
    {
        var rng = new System.Random(seed);
        var (f, d) = Factors<T>(independent, rng);
        var lead = Gram(f, d);
        var quarter = T.One / T.CreateChecked(4);
        var eta = exactlyDependent ? T.Zero : T.One / T.CreateChecked(1024);
        var c = new T[n, independent];
        for (var t = independent; t < n; t++)
        {
            for (var k = 0; k < independent; k++)
            {
                c[t, k] = T.CreateChecked(rng.Next(-4, 5)) * quarter;
            }
        }

        var cov = new T[n, n];
        for (var i = 0; i < independent; i++)
        {
            for (var j = 0; j < independent; j++)
            {
                cov[i, j] = lead[i, j];
            }
        }

        // a_t = Σ_lead·c_t, then the later block c_t′·Σ_lead·c_u.
        var a = new T[n, independent];
        for (var t = independent; t < n; t++)
        {
            for (var i = 0; i < independent; i++)
            {
                var sum = T.Zero;
                for (var k = 0; k < independent; k++)
                {
                    sum += lead[i, k] * c[t, k];
                }

                a[t, i] = sum;
                cov[t, i] = cov[i, t] = sum;
            }
        }

        for (var t = independent; t < n; t++)
        {
            for (var u = independent; u <= t; u++)
            {
                var sum = T.Zero;
                for (var k = 0; k < independent; k++)
                {
                    sum += c[t, k] * a[u, k];
                }

                cov[t, u] = cov[u, t] = t == u ? sum - eta : sum;
            }
        }

        return cov;
    }

    /// <summary>
    /// A three-factor model of order <c>n − 1</c> (as in <see cref="Dependent{T}"/>) followed by an exact copy of its
    /// asset 0: the leading block of order <c>n − 1</c> is positive definite and the minor of order <c>n</c> is exactly 0.
    /// </summary>
    internal static T[,] Repeated<T>(int n, int seed)
        where T : IFloatingPoint<T>
    {
        var rng = new System.Random(seed);
        var (f, d) = Factors<T>(n - 1, rng);
        var lead = Gram(f, d);
        var cov = new T[n, n];
        for (var i = 0; i < n - 1; i++)
        {
            for (var j = 0; j < n - 1; j++)
            {
                cov[i, j] = lead[i, j];
            }

            cov[n - 1, i] = cov[i, n - 1] = lead[0, i];
        }

        cov[n - 1, n - 1] = lead[0, 0];
        return cov;
    }

    /// <summary>
    /// The sample covariance of <paramref name="observations"/> seeded uniform returns of <paramref name="n"/> assets,
    /// rounded in double (rank-deficient before rounding when there are fewer observations than assets).
    /// </summary>
    internal static double[,] SampleCovariance(int n, int observations, int seed)
    {
        var rng = new System.Random(seed);
        var x = new double[observations, n];
        for (var t = 0; t < observations; t++)
        {
            for (var i = 0; i < n; i++)
            {
                x[t, i] = 0.01 * (rng.NextDouble() - 0.5);
            }
        }

        var mean = new double[n];
        for (var i = 0; i < n; i++)
        {
            for (var t = 0; t < observations; t++)
            {
                mean[i] += x[t, i];
            }

            mean[i] /= observations;
        }

        var cov = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                var sum = 0.0;
                for (var t = 0; t < observations; t++)
                {
                    sum += (x[t, i] - mean[i]) * (x[t, j] - mean[j]);
                }

                cov[i, j] = cov[j, i] = sum / (observations - 1);
            }
        }

        return cov;
    }

    private static (T[,] Loadings, T[] Idiosyncratic) Factors<T>(int n, System.Random rng)
        where T : IFloatingPoint<T>
    {
        var sixteenth = T.One / T.CreateChecked(16);
        var unit = T.One / T.CreateChecked(256);
        var f = new T[n, 3];
        var d = new T[n];
        for (var i = 0; i < n; i++)
        {
            f[i, 0] = T.CreateChecked(rng.Next(8, 24)) * sixteenth;
            f[i, 1] = T.CreateChecked(rng.Next(-8, 9)) * sixteenth;
            f[i, 2] = T.CreateChecked(rng.Next(-8, 9)) * sixteenth;
            d[i] = T.CreateChecked(rng.Next(3, 13)) * unit;
        }

        return (f, d);
    }

    private static T[,] Gram<T>(T[,] f, T[] d)
        where T : IFloatingPoint<T>
    {
        var n = d.Length;
        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                var sum = i == j ? d[i] : T.Zero;
                for (var k = 0; k < 3; k++)
                {
                    sum += f[i, k] * f[j, k];
                }

                cov[i, j] = cov[j, i] = sum;
            }
        }

        return cov;
    }
}
