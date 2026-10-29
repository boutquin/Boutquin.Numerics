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

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Result of the Harvey-Liu (2015) Sharpe-ratio haircut. Contains the
/// originally observed Sharpe, the haircut applied (in Sharpe units), the
/// haircut Sharpe ratio, and the multiple-testing-adjusted p-value under
/// each of the three correction methods.
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <param name="ObservedSharpe">Originally observed annualized Sharpe.</param>
/// <param name="HaircutSharpe">
/// Annualized Sharpe whose single-test p-value equals <see cref="BonferroniPValue"/>
/// (Harvey-Liu 2015 eq. 5, the Bonferroni-adjusted Sharpe of their Exhibit 1). It carries
/// the sign of the observed Sharpe and never exceeds it in magnitude.
/// </param>
/// <param name="HaircutAmount">
/// <c>ObservedSharpe − HaircutSharpe</c>: the shrinkage toward zero, in annualized Sharpe
/// units (negative when the observed Sharpe is negative).
/// </param>
/// <param name="BonferroniPValue">Bonferroni-adjusted p-value <c>min(1, M · p_S)</c> of the two-sided single test.</param>
/// <param name="HolmPValue">
/// Holm step-down adjusted p-value. ComputePanel uses all observed hypotheses;
/// scalar Compute assumes every other trial has p=1, giving Bonferroni for the target.
/// </param>
/// <param name="BhyPValue">
/// BHY adjusted p-value. ComputePanel uses the full ordered empirical panel.
/// Scalar Compute assumes the other M − 1 trials carry no evidence (p = 1):
/// <c>min(1, M · c(M) · p_S)</c>, <c>c(M) = Σ_{j≤M} 1/j</c>. A full panel whose other
/// trials are also significant can only lower it, so it is an upper bound on the panel value.
/// </param>
public sealed record HaircutSharpeResult<T>(
    T ObservedSharpe,
    T HaircutSharpe,
    T HaircutAmount,
    T BonferroniPValue,
    T HolmPValue,
    T BhyPValue)
    where T : IFloatingPoint<T>;

/// <summary>
/// Harvey-Liu (2015) haircut Sharpe ratio — adjusts an observed Sharpe
/// for multiple-testing bias using three correction methods (Bonferroni,
/// Holm, Benjamini-Hochberg-Yekutieli) and reports the resulting haircut.
/// More stringent than the Deflated Sharpe Ratio for large trial counts.
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <remarks>
/// <para>
/// Reference: Harvey, C. R. &amp; Liu, Y. (2015). "Backtesting." Journal of
/// Portfolio Management, 42(1). The multiple-testing framework behind the
/// Bonferroni, Holm and BHY adjustments is Harvey, C. R., Liu, Y. &amp; Zhu, H.
/// (2016). "...and the Cross-Section of Expected Returns." Review of Financial
/// Studies, 29(1), 5–68.
/// </para>
/// <para>
/// The annualized Sharpe and the observation count are put on the same sampling
/// frequency: with <c>T = backTestYears · tradingDaysPerYear</c> observations the
/// t-ratio is <c>(SR / √tradingDaysPerYear) · √T = SR · √backTestYears</c>, and the
/// single-test p-value is two-sided on a Student t with <c>T − 1</c> degrees of freedom
/// (Harvey-Liu 2015 eqs. 1–3). The haircut Sharpe inverts the same test at the
/// Bonferroni-adjusted p-value (eq. 5) and is re-annualized.
/// </para>
/// <para>
/// Tier A with transcendental tail: arithmetic body, scalar Student-t tail/inverse finish in double.
/// </para>
/// </remarks>
public static class HaircutSharpe<T>
    where T : IFloatingPoint<T>
{
    private const int MaximumTrialCount = 1_000_000;

    /// <summary>
    /// Adjusts a complete empirical panel of p-values by Bonferroni, Holm and the
    /// Harvey-Liu (2015), pages 15-16, BHY recursion, preserving original hypothesis order.
    /// </summary>
    /// <param name="pValues">One finite p-value in [0,1] per tested hypothesis; between 1 and 1,000,000 entries.</param>
    /// <returns>Adjusted arrays in input order. BHY uses the paper's terminal value p_(M), then a descending minimum with M c(M) p_(i)/i.</returns>
    /// <remarks>The harmonic correction is c(M)=sum(1/j). Ties are ordered by original index and receive equal adjusted values. Arithmetic remains in T except for Half, whose exactly widened p-values are adjusted in double and rounded once at the output.</remarks>
    public static (T[] Bonferroni, T[] Holm, T[] Bhy) AdjustPValues(ReadOnlySpan<T> pValues)
    {
        if (pValues.IsEmpty || pValues.Length > MaximumTrialCount)
        {
            throw new ArgumentOutOfRangeException(nameof(pValues));
        }

        var values = pValues.ToArray();
        var order = Enumerable.Range(0, values.Length).ToArray();
        for (var i = 0; i < values.Length; i++)
        {
            if (!T.IsFinite(values[i]) || values[i] < T.Zero || values[i] > T.One)
            {
                throw new ArgumentOutOfRangeException(nameof(pValues), "P-values must be finite and between zero and one.");
            }
        }

        Array.Sort(order, (left, right) =>
        {
            var comparison = values[left].CompareTo(values[right]);
            return comparison != 0 ? comparison : left.CompareTo(right);
        });
        var count = T.CreateChecked(values.Length);
        if (!T.IsFinite(count))
        {
            throw new ArgumentOutOfRangeException(nameof(pValues), "The panel length must be representable in the working type.");
        }

        if (typeof(T) == typeof(Half))
        {
            // Half rounds both reciprocal indices and products in long panels.
            // Exact input widening avoids harmonic stagnation and rounds each
            // adjusted probability only once, retaining the count guard above.
            return AdjustHalfPanel(values);
        }

        var harmonic = T.Zero;
        for (var j = 1; j <= values.Length; j++)
        {
            harmonic += T.One / T.CreateChecked(j);
        }

        var bonferroni = new T[values.Length];
        var holm = new T[values.Length];
        var bhy = new T[values.Length];
        var runningMaximum = T.Zero;
        for (var rank = 0; rank < values.Length; rank++)
        {
            var index = order[rank];
            bonferroni[index] = T.Min(T.One, count * values[index]);
            runningMaximum = T.Max(runningMaximum, T.Min(T.One, T.CreateChecked(values.Length - rank) * values[index]));
            holm[index] = runningMaximum;
        }

        var runningMinimum = values[order[^1]];
        bhy[order[^1]] = runningMinimum;
        for (var rank = values.Length - 2; rank >= 0; rank--)
        {
            var index = order[rank];
            runningMinimum = T.Min(runningMinimum, (values[index] * count / T.CreateChecked(rank + 1)) * harmonic);
            bhy[index] = runningMinimum;
        }

        return (bonferroni, holm, bhy);
    }

    private static (T[] Bonferroni, T[] Holm, T[] Bhy) AdjustHalfPanel(T[] values)
    {
        var wide = HaircutSharpe<double>.AdjustPValues(Array.ConvertAll(values, double.CreateChecked));
        return (Array.ConvertAll(wide.Bonferroni, T.CreateChecked),
            Array.ConvertAll(wide.Holm, T.CreateChecked), Array.ConvertAll(wide.Bhy, T.CreateChecked));
    }

    /// <summary>
    /// Computes Sharpe haircuts using every observed strategy's p-value, with the
    /// complete ordered-panel Holm and BHY procedures.
    /// </summary>
    /// <param name="observedSharpes">Annualized Sharpes, in original hypothesis order.</param>
    /// <param name="backTestYears">Common backtest length in years.</param>
    /// <param name="tradingDaysPerYear">Common observations per year.</param>
    /// <returns>One result per hypothesis. HaircutSharpe continues to use Bonferroni; HolmPValue and BhyPValue use the complete panel.</returns>
    /// <remarks>The single-test p-values come from the double Student-t tail and are adjusted in double, so a p-value below the working type's least positive value still yields its representable adjusted value and haircut; each reported p-value and haircut is rounded to T once.</remarks>
    public static HaircutSharpeResult<T>[] ComputePanel(ReadOnlySpan<T> observedSharpes, T backTestYears, int tradingDaysPerYear = 252)
    {
        if (observedSharpes.IsEmpty || observedSharpes.Length > MaximumTrialCount)
        {
            throw new ArgumentOutOfRangeException(nameof(observedSharpes));
        }

        // The single-test p-values come from the double Student-t tail. Keep them in double
        // through the M-fold adjustment and the inverse: a p-value below T's least positive
        // value (Half, float, decimal) is representable once multiplied by M, but rounds to
        // zero if narrowed first, and a zero Bonferroni p-value would skip the haircut.
        var results = new HaircutSharpeResult<T>[observedSharpes.Length];
        var pValues = new double[observedSharpes.Length];
        for (var i = 0; i < results.Length; i++)
        {
            ComputeCore(observedSharpes[i], 1, backTestYears, tradingDaysPerYear, out var pSingle);
            pValues[i] = Math.Min(1.0, pSingle);
        }

        var adjusted = HaircutSharpe<double>.AdjustPValues(pValues);
        for (var i = 0; i < results.Length; i++)
        {
            var years = double.CreateChecked(backTestYears > T.Zero ? backTestYears : T.One);
            var original = double.CreateChecked(observedSharpes[i]);
            var probability = adjusted.Bonferroni[i];
            var magnitude = probability > 0.0
                ? Math.Min(Math.Abs(original), StudentTTail.InverseTwoSided(probability, years * tradingDaysPerYear - 1.0) / Math.Sqrt(years))
                : Math.Abs(original);
            var haircut = T.CreateChecked(Math.CopySign(magnitude, original));
            results[i] = new HaircutSharpeResult<T>(observedSharpes[i], haircut, observedSharpes[i] - haircut,
                T.CreateChecked(adjusted.Bonferroni[i]), T.CreateChecked(adjusted.Holm[i]), T.CreateChecked(adjusted.Bhy[i]));
        }

        return results;
    }

    /// <summary>
    /// Computes the scalar compatibility haircut assuming every other trial has p=1.
    /// For observed trial panels use ComputePanel; this overload does not infer their p-values.
    /// </summary>
    /// <param name="observedSharpe">Observed annualized Sharpe ratio.</param>
    /// <param name="numTrials">Number of trials in the search. Values above 1,000,000 are rejected.</param>
    /// <param name="backTestYears">Backtest length in years; non-positive finite values use one year.</param>
    /// <param name="tradingDaysPerYear">
    /// Return observations per year — the sampling frequency of the backtest (252 daily,
    /// 52 weekly, 12 monthly). Default 252.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when a floating-point input is non-finite, <paramref name="numTrials"/> exceeds
    /// 1,000,000, <paramref name="tradingDaysPerYear"/> is not positive, or the backtest spans
    /// one observation or fewer after replacing non-positive finite years with one
    /// (<c>effectiveYears · tradingDaysPerYear ≤ 1</c>).
    /// </exception>
    public static HaircutSharpeResult<T> Compute(
        T observedSharpe,
        int numTrials,
        T backTestYears,
        int tradingDaysPerYear = 252)
        => ComputeCore(observedSharpe, numTrials, backTestYears, tradingDaysPerYear, out _);

    private static HaircutSharpeResult<T> ComputeCore(
        T observedSharpe,
        int numTrials,
        T backTestYears,
        int tradingDaysPerYear,
        out double pSingle)
    {
        if (tradingDaysPerYear <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(tradingDaysPerYear),
                tradingDaysPerYear,
                "Trading days per year must be positive.");
        }

        if (!T.IsFinite(observedSharpe))
        {
            throw new ArgumentOutOfRangeException(nameof(observedSharpe), observedSharpe, "Observed Sharpe must be finite.");
        }

        if (!T.IsFinite(backTestYears))
        {
            throw new ArgumentOutOfRangeException(nameof(backTestYears), backTestYears, "Backtest years must be finite.");
        }

        if (numTrials > MaximumTrialCount)
        {
            throw new ArgumentOutOfRangeException(
                nameof(numTrials), numTrials, $"Trial count must not exceed {MaximumTrialCount}.");
        }

        if (numTrials < 1)
        {
            numTrials = 1;
        }

        if (backTestYears <= T.Zero)
        {
            backTestYears = T.One;
        }

        // Harvey-Liu (2015) eqs. 1-3: t-ratio = SR_period · √T with
        // SR_period = SR_annual / √periods and T = years · periods, i.e.
        // SR_annual · √years (Sharpe and T on the same sampling frequency);
        // the single-test p-value is two-sided on a Student t with T − 1 dof.
        var yearsDbl = double.CreateChecked(backTestYears);
        var observations = yearsDbl * tradingDaysPerYear;
        var dof = observations - 1.0;
        if (!(dof > 0.0))
        {
            throw new ArgumentOutOfRangeException(
                nameof(backTestYears),
                backTestYears,
                "The backtest must span more than one observation (backTestYears × tradingDaysPerYear > 1).");
        }

        var srDbl = double.CreateChecked(observedSharpe);

        var tRatio = srDbl * Math.Sqrt(yearsDbl);
        pSingle = StudentTTail.TwoSided(tRatio, dof);

        var pBonferroni = Math.Min(1.0, numTrials * pSingle);

        // Holm (step-down) and BHY (step-up, factor M·c(M)/i) for the tested
        // strategy: it is the best of the M trials (rank 1) and the other
        // trials carry no evidence (p = 1), so Holm reduces to Bonferroni and
        // BHY to min(1, M·c(M)·p_S), c(M) = Σ 1/j.
        var pHolm = pBonferroni;

        var harmonic = 0.0;
        for (var i = 1; i <= numTrials; i++)
        {
            harmonic += 1.0 / i;
        }

        var pBhy = Math.Min(1.0, numTrials * harmonic * pSingle);

        // Eq. 5: the haircut Sharpe is the Sharpe whose single-test p-value
        // equals the Bonferroni-adjusted p-value (the paper's Exhibit 1 HSR),
        // re-annualized and carrying the observed sign (the test is two-sided).
        var haircutSr = observedSharpe;
        if (pBonferroni > 0.0)
        {
            var tHaircut = StudentTTail.InverseTwoSided(pBonferroni, dof);
            var hsrMagnitude = tHaircut / Math.Sqrt(yearsDbl);
            if (hsrMagnitude < Math.Abs(srDbl))
            {
                haircutSr = T.CreateChecked(Math.CopySign(hsrMagnitude, srDbl));
            }
        }

        var haircutAmount = observedSharpe - haircutSr;

        return new HaircutSharpeResult<T>(
            ObservedSharpe: observedSharpe,
            HaircutSharpe: haircutSr,
            HaircutAmount: haircutAmount,
            BonferroniPValue: T.CreateChecked(pBonferroni),
            HolmPValue: T.CreateChecked(pHolm),
            BhyPValue: T.CreateChecked(pBhy));
    }
}

/// <summary>
/// Result of the Harvey-Liu (2015) Sharpe-ratio haircut. Contains the
/// originally observed Sharpe, the haircut applied (in Sharpe units), the
/// haircut Sharpe ratio, and the multiple-testing-adjusted p-value under
/// each of the three correction methods.
/// </summary>
/// <param name="ObservedSharpe">Originally observed annualized Sharpe.</param>
/// <param name="HaircutSharpe">
/// Annualized Sharpe whose single-test p-value equals <see cref="BonferroniPValue"/>
/// (Harvey-Liu 2015 eq. 5, the Bonferroni-adjusted Sharpe of their Exhibit 1). It carries
/// the sign of the observed Sharpe and never exceeds it in magnitude.
/// </param>
/// <param name="HaircutAmount">
/// <c>ObservedSharpe − HaircutSharpe</c>: the shrinkage toward zero, in annualized Sharpe
/// units (negative when the observed Sharpe is negative).
/// </param>
/// <param name="BonferroniPValue">Bonferroni-adjusted p-value <c>min(1, M · p_S)</c> of the two-sided single test.</param>
/// <param name="HolmPValue">
/// Holm step-down adjusted p-value. ComputePanel uses all observed hypotheses;
/// scalar Compute assumes every other trial has p=1, giving Bonferroni for the target.
/// </param>
/// <param name="BhyPValue">
/// BHY adjusted p-value. ComputePanel uses the full ordered empirical panel.
/// Scalar Compute assumes the other M − 1 trials carry no evidence (p = 1):
/// <c>min(1, M · c(M) · p_S)</c>, <c>c(M) = Σ_{j≤M} 1/j</c>. A full panel whose other
/// trials are also significant can only lower it, so it is an upper bound on the panel value.
/// </param>
public sealed record HaircutSharpeResult(
    decimal ObservedSharpe,
    decimal HaircutSharpe,
    decimal HaircutAmount,
    decimal BonferroniPValue,
    decimal HolmPValue,
    decimal BhyPValue);

/// <summary>
/// Harvey-Liu (2015) haircut Sharpe ratio — adjusts an observed Sharpe
/// for multiple-testing bias using three correction methods (Bonferroni,
/// Holm, Benjamini-Hochberg-Yekutieli) and reports the resulting haircut.
/// More stringent than the Deflated Sharpe Ratio for large trial counts.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Harvey, C. R. &amp; Liu, Y. (2015). "Backtesting." Journal of
/// Portfolio Management, 42(1). The multiple-testing framework behind the
/// Bonferroni, Holm and BHY adjustments is Harvey, C. R., Liu, Y. &amp; Zhu, H.
/// (2016). "...and the Cross-Section of Expected Returns." Review of Financial
/// Studies, 29(1), 5–68.
/// </para>
/// <para>
/// The haircut converts each adjusted p-value into the Sharpe ratio that
/// would have produced it under the null. Subtracting the haircut from
/// the observed Sharpe gives the multiple-testing-corrected estimate.
/// </para>
/// <para>
/// Legacy facade: delegates to <see cref="HaircutSharpe{T}"/> at <c>T = decimal</c>.
/// </para>
/// </remarks>
[Obsolete(
    "Use HaircutSharpe<decimal> instead (the facade preserves the original decimal precision; use <double> deliberately if you want IEEE precision). This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class HaircutSharpe
{
    /// <summary>
    /// Computes the scalar compatibility haircut assuming every other trial has p=1.
    /// For observed trial panels use ComputePanel; this overload does not infer their p-values.
    /// </summary>
    /// <param name="observedSharpe">Observed annualized Sharpe ratio.</param>
    /// <param name="numTrials">Number of trials in the search.</param>
    /// <param name="backTestYears">Backtest length in years; non-positive finite values use one year.</param>
    /// <param name="tradingDaysPerYear">Return observations per year (sampling frequency). Default 252.</param>
    public static HaircutSharpeResult Compute(
        decimal observedSharpe,
        int numTrials,
        decimal backTestYears,
        int tradingDaysPerYear = 252)
    {
        var result = HaircutSharpe<decimal>.Compute(observedSharpe, numTrials, backTestYears, tradingDaysPerYear);
        return new HaircutSharpeResult(
            result.ObservedSharpe,
            result.HaircutSharpe,
            result.HaircutAmount,
            result.BonferroniPValue,
            result.HolmPValue,
            result.BhyPValue);
    }
}

/// <summary>
/// Two-sided Student-t tail probability <c>Pr(|r| &gt; t)</c> for
/// <c>r ~ t(dof)</c>, and its inverse, evaluated in <see cref="double"/>.
/// The tail is the regularized incomplete beta
/// <c>I_{dof/(dof+t²)}(dof/2, 1/2)</c> evaluated by Lentz's continued fraction.
/// </summary>
file static class StudentTTail
{
    private const double Epsilon = 1e-16;
    private const double TinyDenominator = 1e-300;
    private const int MaximumContinuedFractionTerms = 100_000;

    /// <summary>Returns <c>Pr(|r| &gt; |t|)</c> for <c>r ~ t(dof)</c>.</summary>
    public static double TwoSided(double t, double dof)
    {
        var tSquared = t * t;
        if (tSquared == 0.0)
        {
            return 1.0;
        }

        if (double.IsPositiveInfinity(tSquared))
        {
            // For a large finite statistic, x = dof/(dof+t²) can underflow
            // although I_x(dof/2, 1/2) is representable. Evaluate its leading
            // small-x term in log space. Here dof < 1e8 makes the omitted
            // relative O(x) correction smaller than double precision.
            if (!double.IsFinite(t) || dof >= 1e8)
            {
                return 0.0;
            }

            var a = 0.5 * dof;
            var logX = Math.Log(dof) - 2.0 * Math.Log(Math.Abs(t));
            return Math.Exp(a * logX - Math.Log(a)
                + LogGamma(a + 0.5) - LogGamma(a) - LogGamma(0.5));
        }

        if (dof >= 1e8)
        {
            // The first Student-t asymptotic correction avoids cancellation in
            // dof/(dof+t²) and the log-gamma difference. For |t| <= 40 the
            // omitted O(dof^-2) probability term is below double tail accuracy;
            // larger statistics have a tail below the representable range here.
            var magnitude = Math.Abs(t);
            if (magnitude > 40.0)
            {
                return 0.0;
            }

            var normalTail = 2.0 * Distributions.NormalDistribution<double>.Cdf(-magnitude);
            var density = Math.Exp(-0.5 * tSquared) / Math.Sqrt(2.0 * Math.PI);
            return normalTail + density * magnitude * (tSquared + 1.0) / (2.0 * dof);
        }

        return RegularizedIncompleteBeta(dof / (dof + tSquared), 0.5 * dof, 0.5);
    }

    /// <summary>
    /// Returns the non-negative <c>t</c> with <c>Pr(|r| &gt; t) = p</c> for
    /// <c>p</c> in <c>(0, 1]</c>, by bisection on the monotone tail.
    /// </summary>
    public static double InverseTwoSided(double p, double dof)
    {
        if (p >= 1.0)
        {
            return 0.0;
        }

        var lower = 0.0;
        var upper = 1.0;
        while (TwoSided(upper, dof) > p)
        {
            lower = upper;
            upper *= 2.0;
            if (double.IsPositiveInfinity(upper))
            {
                return double.PositiveInfinity;
            }
        }

        for (var iteration = 0; iteration < 200 && upper - lower > 4.0 * double.Epsilon * upper; iteration++)
        {
            var middle = 0.5 * (lower + upper);
            if (TwoSided(middle, dof) > p)
            {
                lower = middle;
            }
            else
            {
                upper = middle;
            }
        }

        return 0.5 * (lower + upper);
    }

    private static double RegularizedIncompleteBeta(double x, double a, double b)
    {
        if (x <= 0.0)
        {
            return 0.0;
        }

        if (x >= 1.0)
        {
            return 1.0;
        }

        var logFront = LogGamma(a + b) - LogGamma(a) - LogGamma(b) + (a * Math.Log(x)) + (b * Math.Log(1.0 - x));
        var front = Math.Exp(logFront);

        return x < (a + 1.0) / (a + b + 2.0)
            ? front * ContinuedFraction(x, a, b) / a
            : 1.0 - (front * ContinuedFraction(1.0 - x, b, a) / b);
    }

    private static double ContinuedFraction(double x, double a, double b)
    {
        var qab = a + b;
        var qap = a + 1.0;
        var qam = a - 1.0;
        var c = 1.0;
        var d = 1.0 - (qab * x / qap);
        d = 1.0 / (Math.Abs(d) < TinyDenominator ? TinyDenominator : d);
        var h = d;

        for (var m = 1; m <= MaximumContinuedFractionTerms; m++)
        {
            var m2 = 2 * m;
            var aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + (aa * d);
            d = 1.0 / (Math.Abs(d) < TinyDenominator ? TinyDenominator : d);
            c = 1.0 + (aa / c);
            c = Math.Abs(c) < TinyDenominator ? TinyDenominator : c;
            h *= d * c;

            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + (aa * d);
            d = 1.0 / (Math.Abs(d) < TinyDenominator ? TinyDenominator : d);
            c = 1.0 + (aa / c);
            c = Math.Abs(c) < TinyDenominator ? TinyDenominator : c;
            var delta = d * c;
            h *= delta;
            if (Math.Abs(delta - 1.0) < Epsilon)
            {
                break;
            }
        }

        return h;
    }

    /// <summary>
    /// ln Γ(x) for x &gt; 0: upward recurrence to x ≥ 10, then Stirling's
    /// series through the 1/(156 x¹³) term (truncation below 1e-16 there).
    /// </summary>
    private static double LogGamma(double x)
    {
        var shift = 0.0;
        while (x < 10.0)
        {
            shift -= Math.Log(x);
            x += 1.0;
        }

        var inverse = 1.0 / x;
        var inverseSquared = inverse * inverse;
        var series = inverse * ((1.0 / 12.0) + (inverseSquared * ((-1.0 / 360.0) + (inverseSquared * ((1.0 / 1260.0)
            + (inverseSquared * ((-1.0 / 1680.0) + (inverseSquared * ((1.0 / 1188.0)
            + (inverseSquared * ((-691.0 / 360360.0) + (inverseSquared * (1.0 / 156.0)))))))))))));

        return shift + ((x - 0.5) * Math.Log(x)) - x + (0.5 * Math.Log(2.0 * Math.PI)) + series;
    }
}
