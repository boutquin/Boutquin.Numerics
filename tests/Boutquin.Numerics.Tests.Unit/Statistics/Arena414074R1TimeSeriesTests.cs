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

using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

/// <summary>
/// Representability and conditioning controls for the EWMA covariance, Newey-West
/// long-run covariance, GARCH(1,1) and graphical-lasso estimators. Every expected
/// value comes from an exact closed form or exact rational arithmetic, never from
/// the implementation under test.
/// </summary>
public sealed class Arena414074R1TimeSeriesTests
{
    // ── EWMA ────────────────────────────────────────────────────────────

    [Fact]
    public void Ewma_RemoteNegligibleWeightAnchor_DoesNotCorruptWeightedCenter()
    {
        // lambda 1/2, 1000 rows, only the OLDEST row is nonzero. Its exact weight is
        // w0 = 2^-1000 / (1 - 2^-1000); the exact weighted variance is
        // w0 (1 - w0) * 1e200 = 2^-1000 * 1e200 to well under 1e-12 relative.
        const int rows = 1000;
        var returns = new double[rows, 1];
        returns[0, 0] = 1e100;

        var actual = new ExponentiallyWeightedCovarianceEstimator<double>(0.5).Estimate(returns)[0, 0];

        var expected = Math.ScaleB(1e200, -1000);
        (Math.Abs(actual - expected) / expected).Should().BeLessThan(1e-9,
            "a negligible-weight remote observation must not be used as the centering anchor for the other 999 rows");
    }

    [Fact]
    public void Ewma_RemoteNegligibleWeightAnchor_DecimalRoute_MatchesExactWeightedVariance()
    {
        // 300 rows, row 0 = 7e27, rows k mod 3 afterwards, lambda 1/2. The exact weighted
        // variance (exact rational weights) is 0.53061224489795918367... = 26/49.
        const int rows = 300;
        var returns = new decimal[rows, 1];
        for (var k = 0; k < rows; k++)
        {
            returns[k, 0] = k % 3;
        }

        returns[0, 0] = 7_000_000_000_000_000_000_000_000_000m;

        var generic = new ExponentiallyWeightedCovarianceEstimator<decimal>(0.5m).Estimate(returns)[0, 0];

        generic.Should().BeApproximately(26m / 49m, 1e-20m);
    }

    [Fact]
    public void Ewma_CrossCovariance_IsRepresentedIndependentlyOfColumnOrder()
    {
        // Row 0 = (1e-100, 1e100), all other rows zero, lambda 1/2, 1000 rows. The exact
        // cross-covariance is w0 (1 - w0) * 1e-100 * 1e100 = 2^-1000 (to 1e-12 relative); every
        // factor and the final product are representable, so no column order may lose it.
        const int rows = 1000;
        var forward = new double[rows, 2];
        var swapped = new double[rows, 2];
        forward[0, 0] = 1e-100;
        forward[0, 1] = 1e100;
        swapped[0, 0] = 1e100;
        swapped[0, 1] = 1e-100;

        var estimator = new ExponentiallyWeightedCovarianceEstimator<double>(0.5);
        var a = estimator.Estimate(forward);
        var b = estimator.Estimate(swapped);

        var expected = Math.ScaleB(1.0, -1000);
        (Math.Abs(a[0, 1] - expected) / expected).Should().BeLessThan(1e-9);
        (Math.Abs(b[0, 1] - expected) / expected).Should().BeLessThan(1e-9);
        a[1, 0].Should().Be(a[0, 1]);
        b[1, 0].Should().Be(b[0, 1]);
    }

    [Fact]
    public void Ewma_DecimalCrossCovariance_IsIndependentOfColumnOrder()
    {
        // 40 rows, lambda 1/2, row 0 = (1e-14, 1e14), all other rows zero. The weight of row 0 is
        // w0 = 1 / (2^40 - 1), so the exact cross-covariance is w0 (1 - w0) = (2^40 - 2) / (2^40 - 1)^2
        // = 9.0949470177292824e-13. A left-associated weight * 1e-14 * 1e14 rounds the 9e-27 intermediate
        // to decimal's 1e-28 resolution and loses digits in one column order only.
        var forward = new decimal[40, 2];
        var swapped = new decimal[40, 2];
        forward[0, 0] = 1e-14m;
        forward[0, 1] = 1e14m;
        swapped[0, 0] = 1e14m;
        swapped[0, 1] = 1e-14m;
        var expected = 1_099_511_627_774m / (1_099_511_627_775m * 1_099_511_627_775m);

        var generic = new ExponentiallyWeightedCovarianceEstimator<decimal>(0.5m);

        generic.Estimate(forward)[0, 1].Should().BeApproximately(expected, 1e-26m);
        generic.Estimate(swapped)[0, 1].Should().BeApproximately(expected, 1e-26m);
    }

    // ── GARCH ───────────────────────────────────────────────────────────

    [Fact]
    public void Garch_VarianceTargeting_TargetIsInvariantToLocation_Double()
    {
        // 40 returns 1e16 + {0 or 2}: 14 of them carry the 2, so the exact population variance of the
        // stored inputs is 4*14/40 - (2*14/40)^2 = 0.91. omega = (1 - alpha - beta) * target.
        var returns = new double[40];
        for (var k = 0; k < returns.Length; k++)
        {
            returns[k] = 1e16 + (k % 3 == 0 ? 2.0 : 0.0);
        }

        var fit = GarchOneOneEstimator<double>.Fit(returns, new GarchOptions<double> { VarianceTargeting = true });

        var recoveredTarget = fit.Omega / (1.0 - fit.Alpha - fit.Beta);
        recoveredTarget.Should().BeApproximately(0.91, 1e-6);
    }

    [Fact]
    public void Garch_VarianceTargeting_TargetIsInvariantToLocation_Float()
    {
        // 16,000,000 + {0 or 2} is exact in float; the exact population variance is 0.91.
        var returns = new float[40];
        for (var k = 0; k < returns.Length; k++)
        {
            returns[k] = 16_000_000f + (k % 3 == 0 ? 2f : 0f);
        }

        var fit = GarchOneOneEstimator<float>.Fit(returns, new GarchOptions<float> { VarianceTargeting = true });

        var recoveredTarget = fit.Omega / (1f - fit.Alpha - fit.Beta);
        recoveredTarget.Should().BeApproximately(0.91f, 1e-3f);
    }

    [Fact]
    public void Garch_VarianceTargeting_SubnormalVarianceBranch_TargetIsInvariantToLocation()
    {
        // Location 1e-139 with five equally frequent offsets of -2..2 ulp(1e-139): the exact population
        // variance of the stored inputs is 2 ulp^2 = 6.95e-310, a subnormal, so Fit takes its
        // normalization branch. Dividing the raw returns by max|r| first rounds each of them by up to
        // half an ulp of 1 - about a tenth of the spread - and the target absorbs that noise.
        var location = 1e-139;
        var ulp = Math.BitIncrement(location) - location;
        var returns = new double[40];
        for (var k = 0; k < returns.Length; k++)
        {
            returns[k] = location + ulp * (((k * 7) % 5) - 2);
        }

        var fit = GarchOneOneEstimator<double>.Fit(returns, new GarchOptions<double> { VarianceTargeting = true });

        var recoveredTarget = fit.Omega / (1.0 - fit.Alpha - fit.Beta);
        Relative(recoveredTarget, 2 * ulp * ulp).Should().BeLessThan(1e-6);
    }

    [Fact]
    public void Garch_Ar1_SubnormalVarianceBranch_InterceptIncludesTheAnchorShift()
    {
        // An AR(1) series around 5 fitted at scale 2^-520 has a subnormal variance, so Fit shifts the series
        // by its first observation and rescales it. mu = scale * mu_normalized + (1 - phi) * anchor must
        // map the fitted intercept back: it agrees with the fit at the safe scale, whose mu is about 1.8,
        // and cannot be recovered from the rescaled intercept alone.
        var state = 7u;
        var series = new double[200];
        var previous = 5.0;
        for (var i = 0; i < series.Length; i++)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            previous = (5.0 * 0.4) + (0.6 * previous) + ((state / (double)uint.MaxValue) - 0.5) * 1.5;
            series[i] = previous;
        }

        var options = new GarchOptions<double>(IncludeAr1Mean: true, MaxIterations: 400);
        var baseline = GarchOneOneEstimator<double>.Fit(series, options);
        var scaled = GarchOneOneEstimator<double>.Fit(series.Select(x => Math.ScaleB(x, -520)).ToArray(), options);

        Math.ScaleB(scaled.Mu, 520).Should().BeApproximately(baseline.Mu, 0.05 * Math.Abs(baseline.Mu));
        scaled.Phi.Should().BeApproximately(baseline.Phi, 0.02);
    }

    [Fact]
    public void Garch_MultiStepForecast_IsNotPoisonedByAnUnrepresentableFirstStep()
    {
        // omega 1, alpha 1/2, beta 0.4, h_T 1e308, s_T 2. The exact one-step forecast
        // 2.4e308 exceeds MaxValue, but the exact recursion f <- omega + (alpha+beta) f decays
        // to a representable value after the third step (exact rational recursion on the stored doubles).
        var fit = new GarchFitResult<double>(0, 0, 1, 0.5, 0.4, [1e308], [2.0], 0, true);

        fit.ForecastVariance(1).Should().Be(double.PositiveInfinity);
        fit.ForecastVariance(2).Should().Be(double.PositiveInfinity);
        fit.ForecastVariance(3).Should().Be(double.PositiveInfinity);
        Relative(fit.ForecastVariance(4), 1.7496000000000002e308).Should().BeLessThan(1e-12);
        Relative(fit.ForecastVariance(10), 9.298091736000003e307).Should().BeLessThan(1e-12);
        Relative(fit.ForecastVariance(1000), 4.6609900045938526e262).Should().BeLessThan(1e-11);
    }

    [Fact]
    public void Garch_MultiStepForecast_KeepsTheInterceptWhenAnUnrepresentableFirstStepDecays()
    {
        // omega 3.5 (not a power of two), alpha 0.3, beta 0.65, h_T 1.5e308, s_T 1.7: h1 = 2.28e308 is
        // unrepresentable and the exact values below follow the exact rational recursion on the stored
        // doubles. At horizon 20000 the decayed first step is gone and the forecast has returned to the
        // intercept-driven level omega / (1 - alpha - beta) = 70, so omega must enter at its own scale.
        var fit = new GarchFitResult<double>(0, 0, 3.5, 0.3, 0.65, [1.5e308], [1.7], 0, true);

        fit.ForecastVariance(3).Should().Be(double.PositiveInfinity);
        Relative(fit.ForecastVariance(10), 1.434132531828348e308).Should().BeLessThan(1e-11);
        Relative(fit.ForecastVariance(100), 1.4181225516705277e306).Should().BeLessThan(1e-11);
        Relative(fit.ForecastVariance(1000), 1.2675308065782414e286).Should().BeLessThan(1e-9);
        Relative(fit.ForecastVariance(5000), 9.939846525336846e196).Should().BeLessThan(1e-6);
        Relative(fit.ForecastVariance(20000), 69.99999999999994).Should().BeLessThan(1e-9);
        Relative(fit.ForecastVariance(int.MaxValue), 69.99999999999994).Should().BeLessThan(1e-9);
    }

    [Fact]
    public void Garch_MultiStepForecast_CountsTheInterceptInAnUnrepresentableFirstStep()
    {
        // omega 8e306 is about 4% of the unrepresentable first step (2.02e308), so a mis-scaled
        // intercept shows at the first digit. Exact rational recursion on the stored doubles:
        // the forecast first becomes representable at horizon 16 and settles at omega/(1 - alpha - beta) = 1.6e308.
        var fit = new GarchFitResult<double>(0, 0, 8e306, 0.15, 0.8, [1.7e308], [1.5], 0, true);

        fit.ForecastVariance(15).Should().Be(double.PositiveInfinity);
        Relative(fit.ForecastVariance(16), 1.7916867464785992e308).Should().BeLessThan(1e-11);
        Relative(fit.ForecastVariance(20), 1.756130053048985e308).Should().BeLessThan(1e-11);
        Relative(fit.ForecastVariance(100), 1.6025785462788581e308).Should().BeLessThan(1e-10);
        Relative(fit.ForecastVariance(5000), 1.6000000000000022e308).Should().BeLessThan(1e-10);
    }

    [Fact]
    public void Garch_MultiStepForecast_IsNotPoisonedByAnUnrepresentableFirstStep_Float()
    {
        // Exact rational recursion on the stored floats: h1 = 7.2e38 exceeds float.MaxValue;
        // the exact 10-step value is 2.789e38 and the 1000-step value is 10.0000 (omega/(1-alpha-beta)).
        var fit = new GarchFitResult<float>(0f, 0f, 1f, 0.5f, 0.4f, [3e38f], [2f], 0f, true);

        fit.ForecastVariance(2).Should().Be(float.PositiveInfinity);
        Relative(fit.ForecastVariance(10), 2.7894268677882037e38).Should().BeLessThan(1e-4);
        Relative(fit.ForecastVariance(1000), 9.999997755640777).Should().BeLessThan(1e-4);
    }

    // ── Graphical lasso ─────────────────────────────────────────────────

    [Fact]
    public void GraphicalLasso_ScaleEquivariance_HoldsAtTheTopOfTheRepresentableRange()
    {
        double[,] s0 =
        {
            { 1.0, 0.98, 0.65 },
            { 0.98, 1.0, 0.55 },
            { 0.65, 0.55, 1.0 },
        };
        var baseline = GraphicalLassoSolver<double>.Solve(s0, 1e-6);

        var scaled = new double[3, 3];
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                scaled[i, j] = Math.ScaleB(s0[i, j], 1023);
            }
        }

        var solve = () => GraphicalLassoSolver<double>.Solve(scaled, Math.ScaleB(1e-6, 1023));
        var actual = solve.Should().NotThrow("the scaled problem is valid and its exact W and Theta are representable").Subject;

        actual.Converged.Should().Be(baseline.Converged);
        var covariance = actual.Covariance;
        var baselineCovariance = baseline.Covariance;
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                Relative(covariance[i, j], Math.ScaleB(baselineCovariance[i, j], 1023)).Should().BeLessThan(1e-9);
            }
        }
    }

    [Fact]
    public void GraphicalLassoBic_ScaleEquivariance_HoldsAtTheTopOfTheRepresentableRange()
    {
        double[,] s0 =
        {
            { 1.0, 0.98, 0.65 },
            { 0.98, 1.0, 0.55 },
            { 0.65, 0.55, 1.0 },
        };
        var baseline = GraphicalLassoSolver<double>.SolveWithBic(s0, 100, [1e-6, 1e-3]);
        var scaled = new double[3, 3];
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                scaled[i, j] = Math.ScaleB(s0[i, j], 1023);
            }
        }

        var solve = () => GraphicalLassoSolver<double>.SolveWithBic(scaled, 100, [Math.ScaleB(1e-6, 1023), Math.ScaleB(1e-3, 1023)]);
        var actual = solve.Should().NotThrow("the scaled grid is valid and every exact W is representable").Subject;

        Relative(actual.Lambda, Math.ScaleB(baseline.Lambda, 1023)).Should().BeLessThan(1e-12);
        Relative(actual.Covariance[0, 1], Math.ScaleB(baseline.Covariance[0, 1], 1023)).Should().BeLessThan(1e-9);
    }

    // ── Newey-West automatic long-run covariance ────────────────────────

    [Fact]
    public void NeweyWestAutomatic_ScaleEquivariance_HoldsWhereTheRawGramExceedsTheRange_Double()
    {
        // At 2^509 the raw Gram sum of squares exceeds double.MaxValue (the unfixed estimator throws from
        // k = 508 on this series), although the exact output (base * 2^1018, base about 0.05) is representable.
        // Newey-West (1994) is homogeneous of degree two in the series, so the output must equal the
        // base output times 2^1018.
        var baseSeries = BinaryFractionSeries(100, seed: 20261004);
        var baseline = NeweyWestVariance<double>.MeanVarianceAutomatic(baseSeries);
        var scaledSeries = baseSeries.Select(x => Math.ScaleB(x, 509)).ToArray();

        var run = () => NeweyWestVariance<double>.MeanVarianceAutomatic(scaledSeries);

        var actual = run.Should().NotThrow("the exact long-run variance is representable").Subject;
        Relative(actual, Math.ScaleB(baseline, 1018)).Should().BeLessThan(1e-10);
    }

    [Fact]
    public void NeweyWestAutomatic_ScaleEquivariance_HoldsWhereTheRawGramExceedsTheRange_Float()
    {
        var baseSeries = BinaryFractionSeries(100, seed: 20261004).Select(x => (float)x).ToArray();
        var baseline = NeweyWestVariance<float>.MeanVarianceAutomatic(baseSeries);
        var scaledSeries = baseSeries.Select(x => MathF.ScaleB(x, 60)).ToArray();

        var run = () => NeweyWestVariance<float>.MeanVarianceAutomatic(scaledSeries);

        var actual = run.Should().NotThrow("the exact long-run variance is representable in float").Subject;
        Relative(actual, MathF.ScaleB(baseline, 120)).Should().BeLessThan(1e-4);
    }

    [Fact]
    public void NeweyWestAutomatic_ScaleEquivariance_HoldsWhereTheRawGramExceedsTheRange_Decimal()
    {
        // Binary fractions k/512 times 2^44 are exact decimals. The exact output base * 2^88 is
        // about 1.6e25, far below decimal.MaxValue, while the raw Gram sum of squares exceeds it.
        var baseSeries = BinaryFractionSeries(100, seed: 20261004).Select(x => (decimal)x).ToArray();
        var baseline = NeweyWestVariance<decimal>.MeanVarianceAutomatic(baseSeries);
        var factor = 17_592_186_044_416m; // 2^44
        var scaledSeries = baseSeries.Select(x => x * factor).ToArray();

        var run = () => NeweyWestVariance<decimal>.MeanVarianceAutomatic(scaledSeries);

        var actual = run.Should().NotThrow("the exact long-run variance is representable in decimal").Subject;
        var expected = baseline * factor * factor;
        ((double)(Math.Abs(actual - expected) / expected)).Should().BeLessThan(1e-15);
    }

    [Fact]
    public void NeweyWestAutomatic_PilotSpectralSums_DoNotOverflowWhenOneObservationDominates()
    {
        // One nonzero observation of 2^513. The weighted residual 0.99 * 2^513 squares to about 3.5e308,
        // so the pilot product weighted[t] * weighted[t] exceeds double.MaxValue before it is divided by
        // T - 1, although the exact long-run variance is about 7e304 and the output is homogeneous of
        // degree two in the series: output(2^513 e_50) = output(e_50) * 2^1026.
        var unit = new double[100];
        unit[50] = 1.0;
        var scaled = new double[100];
        scaled[50] = Math.ScaleB(1.0, 513);
        var baseline = NeweyWestVariance<double>.MeanVarianceAutomatic(unit);

        var run = () => NeweyWestVariance<double>.MeanVarianceAutomatic(scaled);

        var actual = run.Should().NotThrow("the exact long-run variance is representable").Subject;
        Relative(actual, Math.ScaleB(baseline, 1026)).Should().BeLessThan(1e-10);
    }

    [Fact]
    public void NeweyWestLongRun_DiagonalInstrumentScaling_IsEquivariantAtTheTopOfTheRange()
    {
        // Z = [2^k, z]. With pilot weights (2^-k, 1) the pilot series, the bandwidth and the VAR
        // are invariant, so S scales as D S D with D = diag(2^k, 1). At k = 509 the raw Gram entry
        // for the first instrument exceeds double.MaxValue although every S entry is representable.
        const int rows = 100;
        const int k = 509;
        var residuals = BinaryFractionSeries(rows, seed: 11, span: 1024);
        var z = BinaryFractionSeries(rows, seed: 12, span: 1024);
        double[,] Instruments(int exponent)
        {
            var matrix = new double[rows, 2];
            for (var i = 0; i < rows; i++)
            {
                matrix[i, 0] = Math.ScaleB(1.0, exponent);
                matrix[i, 1] = z[i];
            }

            return matrix;
        }

        var baseline = NeweyWestVariance<double>.LongRunCovariance(Instruments(0), residuals, [1.0, 1.0], 4);

        var run = () => NeweyWestVariance<double>.LongRunCovariance(Instruments(k), residuals, [Math.ScaleB(1.0, -k), 1.0], 4);

        var actual = run.Should().NotThrow("every entry of S is representable").Subject;
        actual.Lags.Should().Be(baseline.Lags);
        Relative(actual.Covariance[0, 0], Math.ScaleB(baseline.Covariance[0, 0], 2 * k)).Should().BeLessThan(1e-10);
        Relative(actual.Covariance[0, 1], Math.ScaleB(baseline.Covariance[0, 1], k)).Should().BeLessThan(1e-10);
        Relative(actual.Covariance[1, 1], baseline.Covariance[1, 1]).Should().BeLessThan(1e-10);
    }

    private static double[] BinaryFractionSeries(int count, uint seed, int span = 2048)
    {
        // Exact binary fractions in [-span/512, span/512], generated by an xorshift32 stream.
        var state = seed == 0 ? 1u : seed;
        var series = new double[count];
        for (var i = 0; i < count; i++)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            series[i] = ((int)(state % (uint)(2 * span + 1)) - span) / 512.0;
        }

        return series;
    }

    private static double Relative(double actual, double expected) => Math.Abs(actual - expected) / Math.Abs(expected);
}
