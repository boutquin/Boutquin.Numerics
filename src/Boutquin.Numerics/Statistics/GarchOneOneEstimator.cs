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

using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Options controlling AR(1)-GARCH(1,1) estimation.
/// </summary>
/// <typeparam name="T">IEEE 754 floating-point type.</typeparam>
/// <param name="IncludeAr1Mean">
/// When <see langword="true"/>, the mean equation includes an AR(1) term
/// (<c>r_t = μ + φ·r_{t−1} + ε_t</c>), constrained to |φ| &lt; 1; otherwise φ is fixed at zero.
/// </param>
/// <param name="MinObservations">
/// Minimum number of observations required; fewer raises <see cref="ArgumentException"/>.
/// Must be at least 2. Default: 20.
/// </param>
/// <param name="MaxIterations">
/// Maximum Nelder-Mead iterations per phase. Must be at least 2 so the coarse and
/// fine phases each receive a positive budget. Default: 5 000.
/// </param>
/// <param name="Tolerance">
/// Convergence tolerance passed to <see cref="NelderMeadMinimizer{T}"/>
/// for both <c>XTolerance</c> and <c>FTolerance</c>. Must be finite and strictly
/// positive. Default: the greater of 1e-8 and one representable step above 1
/// for the selected floating-point type.
/// </param>
public sealed record GarchOptions<T>(
    bool IncludeAr1Mean = false,
    int MinObservations = 20,
    int MaxIterations = 5000,
    T? Tolerance = default)
    where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>Fixes both μ and φ at zero. Cannot be combined with AR(1).</summary>
    public bool FixedZeroMean { get; init; }

    /// <summary>
    /// Constrains ω at every trial to (1−α−β) times the fixed input second moment
    /// about the initial mean, divided by n. The initial mean is zero in fixed-zero-mean
    /// mode and the sample mean otherwise. The target stays fixed when fitting AR(1).
    /// </summary>
    public bool VarianceTargeting { get; init; }
}

/// <summary>
/// The result of an AR(1)-GARCH(1,1) QMLE fit.
/// </summary>
/// <typeparam name="T">IEEE 754 floating-point type.</typeparam>
/// <param name="Mu">Estimated intercept of the mean equation.</param>
/// <param name="Phi">Estimated AR(1) coefficient (zero when <see cref="GarchOptions{T}.IncludeAr1Mean"/> is <see langword="false"/>).</param>
/// <param name="Omega">Estimated GARCH intercept ω &gt; 0.</param>
/// <param name="Alpha">Estimated ARCH coefficient α ∈ (0, 1).</param>
/// <param name="Beta">Estimated GARCH coefficient β ∈ (0, 1); α + β &lt; 1 (enforced by construction).</param>
/// <param name="ConditionalVariances">
/// Filtered conditional variance series h_1, …, h_T. The constructor and
/// accessor defensively copy the series so later caller mutations cannot
/// change forecast state.
/// </param>
/// <param name="StandardizedResiduals">
/// Standardised residual series ε_t / √h_t. The constructor and accessor
/// defensively copy the series.
/// </param>
/// <param name="LogLikelihood">Average per-observation Gaussian QMLE log-likelihood at the optimum (without the constant −½ln(2π)).</param>
/// <param name="Converged"><see langword="true"/> when the Nelder-Mead simplex converged before the iteration cap.</param>
public sealed record GarchFitResult<T>(
    T Mu,
    T Phi,
    T Omega,
    T Alpha,
    T Beta,
    T[] ConditionalVariances,
    T[] StandardizedResiduals,
    T LogLikelihood,
    bool Converged)
    where T : struct, IFloatingPointIeee754<T>
{
    private T[] _conditionalVariances = CloneSeries(ConditionalVariances);
    private T[] _standardizedResiduals = CloneSeries(StandardizedResiduals);

    /// <summary>Filtered conditional variance series h_1, …, h_T.</summary>
    public T[] ConditionalVariances
    {
        get => CloneSeries(_conditionalVariances);
        init => _conditionalVariances = CloneSeries(value);
    }

    /// <summary>Standardised residual series ε_t / √h_t.</summary>
    public T[] StandardizedResiduals
    {
        get => CloneSeries(_standardizedResiduals);
        init => _standardizedResiduals = CloneSeries(value);
    }

    /// <summary>
    /// Returns the GARCH(1,1) variance forecast <c>h_{T+horizon}</c>.
    /// </summary>
    /// <param name="horizon">Forecast horizon (≥ 1).</param>
    /// <returns>
    /// For <paramref name="horizon"/> = 1: ω + α·ε²_T + β·h_T where ε_T is recovered from
    /// the last standardised residual.
    /// For <paramref name="horizon"/> &gt; 1: unconditional recursion
    /// h_{T+k} = ω/(1−α−β) + (α+β)^(k−1)·(h_{T+1} − ω/(1−α−β)).
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="horizon"/> is less than 1.
    /// </exception>
    /// <exception cref="ArgumentException">The stored series are empty or have different lengths, or the terminal variance, residual, or GARCH coefficients are invalid.</exception>
    public T ForecastVariance(int horizon)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(horizon, 1);
        if (_conditionalVariances.Length == 0)
        {
            throw new ArgumentException(
                "Conditional variance series must contain at least one value.");
        }

        if (_standardizedResiduals.Length == 0)
        {
            throw new ArgumentException(
                "Standardized residual series must contain at least one value.");
        }

        if (_standardizedResiduals.Length != _conditionalVariances.Length)
        {
            throw new ArgumentException(
                "Conditional variance and standardized residual series must have the same length.");
        }

        var hT = _conditionalVariances[_conditionalVariances.Length - 1];
        var sT = _standardizedResiduals[_standardizedResiduals.Length - 1];
        if (!T.IsFinite(hT) || hT < T.Zero || !T.IsFinite(sT)
            || !T.IsFinite(Omega) || Omega <= T.Zero
            || !T.IsFinite(Alpha) || Alpha < T.Zero || !T.IsFinite(Beta) || Beta < T.Zero)
        {
            throw new ArgumentException("Forecast state requires finite nonnegative terminal variance, finite residual, positive omega and nonnegative alpha and beta.");
        }

        // A zero ARCH weight makes the innovation irrelevant, including when
        // recovering the unweighted residual would overflow.
        var innovation = T.Zero;
        if (Alpha != T.Zero)
        {
            var epsT = sT * T.Sqrt(hT);
            innovation = Alpha * epsT * epsT;
            if ((!T.IsFinite(innovation) || innovation == T.Zero) && sT != T.Zero && hT != T.Zero)
            {
                // Combine bounded significands before applying the total exponent:
                // α s² h can fit even when the unweighted ε or ε² cannot.
                var alphaExponent = T.ILogB(Alpha);
                var residualExponent = T.ILogB(T.Abs(sT));
                var varianceExponent = T.ILogB(hT);
                var alphaMantissa = T.ScaleB(Alpha, -alphaExponent);
                var residualMantissa = T.ScaleB(sT, -residualExponent);
                var varianceMantissa = T.ScaleB(hT, -varianceExponent);
                innovation = T.ScaleB(alphaMantissa * residualMantissa * residualMantissa * varianceMantissa,
                    alphaExponent + 2 * residualExponent + varianceExponent);
            }
        }

        var h1 = Omega + innovation + Beta * hT;
        if (horizon == 1)
        {
            return h1;
        }

        // Finite-horizon recursion avoids requiring a representable long-run variance.
        var ab = Alpha + Beta;
        if (ab >= T.One)
        {
            throw new InvalidOperationException(
                "Non-stationary GARCH: α+β≥1 makes the long-run variance undefined.");
        }

        // The recursion is monotone toward its fixed point omega / (1 - alpha - beta), so a
        // representable first step is never followed by an unrepresentable step that later
        // becomes representable again: plain evaluation is exact in range and overflows only
        // when the forecast itself does. An unrepresentable first step decays toward the fixed
        // point, so carry the value as significand and binary exponent instead.
        if (T.IsFinite(h1))
        {
            var forecast = h1;
            for (var k = 1; k < horizon; k++)
            {
                forecast = Omega + ab * forecast;
            }

            return forecast;
        }

        return ScaledForecast(ab, hT, sT, horizon);
    }

    /// <summary>
    /// Evaluates <c>h_{T+horizon}</c> with the running value held as <c>m · 2^e</c>, so an
    /// unrepresentable early step cannot poison a representable later one.
    /// </summary>
    private T ScaledForecast(T ab, T hT, T sT, int horizon)
    {
        // First step h1 = omega + alpha s^2 h + beta h, each term as (significand product, exponent).
        var omegaExponent = T.ILogB(Omega);
        var exponent = omegaExponent;
        var hasInnovation = Alpha != T.Zero && sT != T.Zero && hT != T.Zero;
        var hasPersistence = Beta != T.Zero && hT != T.Zero;
        var innovationExponent = 0;
        var persistenceExponent = 0;
        var innovationSignificand = T.Zero;
        var persistenceSignificand = T.Zero;
        if (hasInnovation)
        {
            var alphaExponent = T.ILogB(Alpha);
            var residualExponent = T.ILogB(T.Abs(sT));
            var varianceExponent = T.ILogB(hT);
            var residualSignificand = T.ScaleB(sT, -residualExponent);
            innovationSignificand = T.ScaleB(Alpha, -alphaExponent) * residualSignificand * residualSignificand
                * T.ScaleB(hT, -varianceExponent);
            innovationExponent = alphaExponent + 2 * residualExponent + varianceExponent;
            exponent = Math.Max(exponent, innovationExponent);
        }

        if (hasPersistence)
        {
            var betaExponent = T.ILogB(Beta);
            var varianceExponent = T.ILogB(hT);
            persistenceSignificand = T.ScaleB(Beta, -betaExponent) * T.ScaleB(hT, -varianceExponent);
            persistenceExponent = betaExponent + varianceExponent;
            exponent = Math.Max(exponent, persistenceExponent);
        }

        var significand = T.ScaleB(Omega, -exponent);
        if (hasInnovation)
        {
            significand += T.ScaleB(innovationSignificand, innovationExponent - exponent);
        }

        if (hasPersistence)
        {
            significand += T.ScaleB(persistenceSignificand, persistenceExponent - exponent);
        }

        // Every later step is at least omega, so omega * 2^-exponent never exceeds 2 after renormalization.
        for (var k = 1; k < horizon; k++)
        {
            var next = T.ScaleB(Omega, -exponent) + ab * significand;
            var shift = T.ILogB(next);
            var settled = shift == 0 && next == significand;
            significand = T.ScaleB(next, -shift);
            exponent += shift;
            if (settled)
            {
                break;
            }
        }

        return T.ScaleB(significand, exponent);
    }

    private static T[] CloneSeries(T[] source)
    {
        ArgumentNullException.ThrowIfNull(source);
        return (T[])source.Clone();
    }
}

/// <summary>
/// AR(1)-mean + GARCH(1,1) quasi-maximum-likelihood estimator.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier: B</b> — requires <see cref="IFloatingPointIeee754{TSelf}"/> for
/// <c>T.Log</c> and <c>T.Exp</c>. <c>T = decimal</c> produces a correct <c>CS0315</c> compile error.
/// </para>
/// <para>
/// Estimates the AR(1)-GARCH(1,1) model:
/// <code>r_t = μ + φ·r_{t−1} + ε_t,   ε_t = √h_t · z_t,   z_t ∼ N(0,1)</code>
/// <code>h_t = ω + α·ε²_{t−1} + β·h_{t−1}</code>
/// via Gaussian quasi-maximum likelihood (QMLE). Stationarity (α + β &lt; 1) is enforced by
/// construction through a logistic reparameterisation of the (α, β) parameter space, with the
/// persistence sum capped a tiny margin below 1 so the invariant holds even at the IEEE
/// rounding boundary. On integrated / near-integrated (IGARCH-like) data the estimate
/// <c>α̂ + β̂</c> <b>saturates just below that ceiling</b> — a value pinned near 1 there is the
/// enforced stationary bound, not a coincidence or a fitting artefact.
/// </para>
/// <para>
/// <b>Reparameterisation (unconstrained search):</b>
/// <code>ω = exp(ln_ω),   α+β = σ(logit_ab),   α = (α+β)·σ(logit_af)</code>
/// where σ(x) = 1/(1+e^{−x}). Initial values use the input second moment about the initial mean:
/// α₀ = 0.05, β₀ = 0.90, ω₀ = (1−α₀−β₀)·ŝ².
/// </para>
/// <para>
/// Optimisation is delegated to <see cref="NelderMeadMinimizer{T}"/>. The conditional
/// variance and squared shock are both seeded at the candidate residual second moment divided by n
/// (Bollerslev 1986, footnote 5). The first lagged return equals the first observation.
/// The objective and reported likelihood are averages per observation, omitting the Gaussian constant.
/// Optional variance targeting fixes the input second moment about the initial mean as the target
/// throughout optimization. Optional AR(1) stationarity is an additional library mean-model boundary;
/// the publication's innovation-variance stationarity theorem does not establish return stationarity.
/// </para>
/// <para>
/// <b>Thread safety:</b> stateless static class; concurrent calls are independent.
/// </para>
/// </remarks>
/// <typeparam name="T">IEEE 754 floating-point type (<c>double</c>, <c>float</c>).</typeparam>
public static class GarchOneOneEstimator<T>
    where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>
    /// Fits an AR(1)-GARCH(1,1) model to <paramref name="returns"/> via QMLE.
    /// </summary>
    /// <param name="returns">Return series (length ≥ <see cref="GarchOptions{T}.MinObservations"/>).</param>
    /// <param name="options">Estimation options; <see langword="null"/> uses defaults.</param>
    /// <returns>Fitted GARCH parameters, conditional variances, and standardised residuals.</returns>
    /// <exception cref="ArgumentNullException">
    /// <paramref name="returns"/> is <see langword="null"/>.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="returns"/> has fewer than <see cref="GarchOptions{T}.MinObservations"/>
    /// observations, contains a non-finite value, or has non-finite or numerically zero
    /// variance (a constant series is not GARCH-identifiable), or an initial or fitted
    /// intercept or conditional variance cannot be represented as finite and strictly
    /// positive at the selected floating-point precision.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// An option has an invalid iteration limit, observation minimum, or tolerance.
    /// </exception>
    public static GarchFitResult<T> Fit(T[] returns, GarchOptions<T>? options = null)
    {
        ArgumentNullException.ThrowIfNull(returns);

        options ??= new GarchOptions<T>();

        if (options.MinObservations < 2)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options), options.MinObservations,
                "MinObservations must be at least 2 so sample variance is defined.");
        }

        if (options.MaxIterations < 2)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options), options.MaxIterations,
                "MaxIterations must be at least 2 so both optimization phases run.");
        }

        var tol = options.Tolerance ?? T.Max(
            T.CreateChecked(1e-8), T.BitIncrement(T.One) - T.One);
        if (!T.IsFinite(tol) || tol <= T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options), tol, "Tolerance must be finite and strictly positive.");
        }

        if (options.FixedZeroMean && options.IncludeAr1Mean)
        {
            throw new ArgumentException("FixedZeroMean cannot be combined with IncludeAr1Mean.", nameof(options));
        }

        var n = returns.Length;
        if (n < options.MinObservations)
        {
            throw new ArgumentException(
                $"Insufficient observations: {n} < {options.MinObservations} (MinObservations).",
                nameof(returns));
        }

        var hasVariation = false;
        for (var i = 0; i < n; i++)
        {
            if (!T.IsFinite(returns[i]))
            {
                throw new ArgumentException(
                    $"Return series entry at index {i} is not finite (value {returns[i]}).",
                    nameof(returns));
            }

            if (i > 0 && returns[i] != returns[0])
            {
                hasVariation = true;
            }
        }

        if (!hasVariation)
        {
            throw new ArgumentException(
                "Series has zero variance; a constant series is not identifiable as a GARCH process.",
                nameof(returns));
        }

        // An AR(1) location shift changes only mu -> mu + (1-phi)*anchor.
        // Searching on absolute levels instead couples mu and phi along a steep
        // ridge and can falsely converge far from the same residual likelihood.
        if (options.IncludeAr1Mean && returns[0] != T.Zero)
        {
            var searchAnchor = returns[0];
            var centered = new T[n];
            for (var i = 0; i < n; i++)
            {
                centered[i] = returns[i] - searchAnchor;
            }

            if (centered.All(T.IsFinite))
            {
                // centered[0]=0, so the recursive fit does not re-anchor.
                var fit = Fit(centered, options);
                return fit with { Mu = fit.Mu + (T.One - fit.Phi) * searchAnchor };
            }
        }

        // Half cannot represent long observation counts or reliably sum their
        // contributions. Widen the moment reduction, then narrow its two scalars.
        var mean = T.Zero;
        var sampleVar = T.Zero;
        if (typeof(T) == typeof(Half))
        {
            var wideMean = 0.0;
            for (var i = 0; i < n; i++)
            {
                wideMean += double.CreateChecked(returns[i]);
            }

            wideMean = options.FixedZeroMean ? 0.0 : wideMean / n;
            var wideVariance = 0.0;
            for (var i = 0; i < n; i++)
            {
                var deviation = double.CreateChecked(returns[i]) - wideMean;
                wideVariance += deviation * deviation;
            }

            mean = T.CreateChecked(wideMean);
            sampleVar = T.CreateChecked(wideVariance / n);
        }
        else
        {
            // Measure every return from a data-row anchor and keep the mean offset separate.
            // Subtracting a rounded absolute mean instead leaves its half-ulp error at the
            // location's magnitude, and that error is squared into the second moment, so the
            // variance target would grow with the location of the series.
            var count = T.CreateChecked(n);
            var anchor = options.FixedZeroMean ? T.Zero : returns[0];
            var offset = T.Zero;
            if (!options.FixedZeroMean)
            {
                for (var i = 0; i < n; i++)
                {
                    offset += (returns[i] - anchor) / count;
                }
            }

            mean = anchor + offset;

            for (var i = 0; i < n; i++)
            {
                var d = (returns[i] - anchor) - offset;
                sampleVar += d * d;
            }

            sampleVar /= count;
        }

        var alpha0 = T.CreateChecked(0.05);
        var beta0 = T.CreateChecked(0.90);
        var omega0 = (T.One - alpha0 - beta0) * sampleVar;

        // Nelder-Mead's default log-intercept simplex vertex adds
        // .05*max(abs(log(omega0)),1). Detect overflow at that actual vertex,
        // even when sample variance and the unperturbed intercept are finite.
        var unsafeInitialOmegaVertex = omega0 > T.Zero && T.IsFinite(omega0) &&
            !T.IsFinite(T.Exp(T.Log(omega0) + T.CreateChecked(0.05) * T.Max(T.Abs(T.Log(omega0)), T.One)));
        var subnormalScale = T.One;
        if (T.IsSubnormal(sampleVar))
        {
            subnormalScale = T.Zero;
            for (var i = 0; i < n; i++)
            {
                subnormalScale = T.Max(subnormalScale, T.Abs(returns[i]));
            }
        }

        if (!T.IsFinite(sampleVar) || sampleVar <= T.Zero || CanImproveSubnormalVariance(sampleVar, subnormalScale) || unsafeInitialOmegaVertex)
        {
            if (!T.IsFinite(sampleVar) || T.IsSubnormal(sampleVar) || unsafeInitialOmegaVertex)
            {
                // Subnormal variance loses relative precision and can overflow the
                // likelihood at ordinary-sized simplex mean perturbations. Fit in
                // return units whose squared values remain normal and finite.
                // Scaling the fit back preserves dimensionless ARCH coefficients
                // and standardized residuals while restoring variance units.
                //
                // Measure the series from a data-row anchor before dividing by its largest
                // deviation. Dividing the raw returns first rounds each by up to half an ulp of
                // one, which is a large fraction of the spread when the location dwarfs it. The
                // mean equation is affine, so the anchor moves only the intercept:
                // mu = scale * mu_normalized + (1 - phi) * anchor. A fixed-zero mean cannot be
                // shifted, and a deviation that overflows falls back to the unshifted series.
                var anchor = options.FixedZeroMean ? T.Zero : returns[0];
                var scale = T.Zero;
                for (var i = 0; i < n; i++)
                {
                    scale = T.Max(scale, T.Abs(returns[i] - anchor));
                }

                if (!T.IsFinite(scale) || scale == T.Zero)
                {
                    anchor = T.Zero;
                    scale = T.Zero;
                    for (var i = 0; i < n; i++)
                    {
                        scale = T.Max(scale, T.Abs(returns[i]));
                    }
                }

                var normalized = new T[n];
                for (var i = 0; i < n; i++)
                {
                    normalized[i] = (returns[i] - anchor) / scale;
                }

                var normalizedMean = T.Zero;
                for (var i = 0; i < n; i++)
                {
                    normalizedMean += normalized[i];
                }

                normalizedMean = options.FixedZeroMean ? T.Zero : normalizedMean / T.CreateChecked(n);
                var normalizedVariance = T.Zero;
                for (var i = 0; i < n; i++)
                {
                    var deviation = normalized[i] - normalizedMean;
                    normalizedVariance += deviation * deviation;
                }

                var restoredVariance = ((normalizedVariance / T.CreateChecked(n)) * scale) * scale;
                if (!T.IsFinite(restoredVariance) || restoredVariance <= T.Zero)
                {
                    throw new ArgumentException(
                        "Series variance is non-finite or numerically zero at the selected floating-point " +
                        "precision; the GARCH process is not identifiable.",
                        nameof(returns));
                }

                var fit = Fit(normalized, options);
                var restoredOmega = (fit.Omega * scale) * scale;
                if (!T.IsFinite(restoredOmega) || restoredOmega <= T.Zero)
                {
                    throw new ArgumentException(
                        "Fitted GARCH intercept is not representable as finite and strictly positive " +
                        "at the selected floating-point precision.", nameof(returns));
                }

                var variances = fit.ConditionalVariances;
                for (var i = 0; i < variances.Length; i++)
                {
                    variances[i] = (variances[i] * scale) * scale;
                    if (!T.IsFinite(variances[i]) || variances[i] <= T.Zero)
                    {
                        throw new ArgumentException(
                            "Fitted GARCH variance is not representable as finite and strictly positive " +
                            "at the selected floating-point precision.", nameof(returns));
                    }
                }

                return fit with
                {
                    Mu = (fit.Mu * scale) + ((T.One - fit.Phi) * anchor),
                    Omega = restoredOmega,
                    ConditionalVariances = variances,
                    LogLikelihood = fit.LogLikelihood - T.Log(scale),
                };
            }

            throw new ArgumentException(
                "Series variance is non-finite or numerically zero at the selected floating-point " +
                "precision; the GARCH process is not identifiable.",
                nameof(returns));
        }

        var ar1 = options.IncludeAr1Mean;

        // ── Variance-targeting initial guess ────────────────────────────────────

        if (!T.IsFinite(omega0) || omega0 <= T.Zero)
        {
            throw new ArgumentException(
                "The variance-targeting initial intercept is not representable as finite and strictly " +
                "positive at the selected floating-point precision; normalization cannot improve " +
                "the variance scale.", nameof(returns));
        }

        var lnOmega0 = T.Log(omega0);
        var abSum0 = alpha0 + beta0;                               // 0.95
        var logitAbSum0 = T.Log(abSum0 / (T.One - abSum0));
        var alphaFrac0 = alpha0 / abSum0;                          // 0.05/0.95
        var logitAFrac0 = T.Log(alphaFrac0 / (T.One - alphaFrac0));

        var initial = new List<T>();
        if (!options.FixedZeroMean)
        {
            initial.Add(mean);
        }
        if (ar1)
        {
            initial.Add(T.Zero);
        }
        if (!options.VarianceTargeting)
        {
            initial.Add(lnOmega0);
        }
        initial.Add(logitAbSum0);
        initial.Add(logitAFrac0);
        var x0 = initial.ToArray();

        void Decode(T[] p, out T mu, out T phi, out T omega, out T alpha, out T beta)
        {
            var index = 0;
            mu = options.FixedZeroMean ? T.Zero : p[index++];
            // This stationary mean boundary is a library choice, not a GARCH theorem.
            phi = ar1 ? T.Clamp(T.Tanh(p[index++]), -T.BitDecrement(T.One), T.BitDecrement(T.One)) : T.Zero;
            var freeOmega = options.VarianceTargeting ? T.Zero : T.Exp(p[index++]);
            var persistence = StrictlyBelowOne(Sigmoid(p[index++]));
            alpha = persistence * Sigmoid(p[index]);
            beta = persistence - alpha;
            omega = options.VarianceTargeting ? (T.One - alpha - beta) * sampleVar : freeOmega;
        }

        T ResidualSecondMoment(T mu, T phi)
        {
            var sum = T.Zero;
            var wideSum = 0.0;
            for (var t = 0; t < n; t++)
            {
                var lag = t > 0 ? returns[t - 1] : returns[t];
                var residual = returns[t] - mu - phi * lag;
                if (typeof(T) == typeof(Half))
                {
                    var wide = double.CreateChecked(residual);
                    wideSum += wide * wide;
                }
                else
                {
                    sum += residual * residual;
                }
            }
            return typeof(T) == typeof(Half) ? T.CreateChecked(wideSum / n) : sum / T.CreateChecked(n);
        }

        // Bollerslev (1986), footnote 5: both presample h and epsilon squared
        // use the mean squared residual at the candidate mean parameters.
        T NegQll(T[] p)
        {
            Decode(p, out var mu, out var phi, out var omega, out var alpha, out var beta);
            var seed = ResidualSecondMoment(mu, phi);
            var h = seed;
            var weightedInnovation = alpha * seed;
            var ll = T.Zero;
            var wideLikelihood = 0.0;
            var half = T.CreateChecked(0.5);
            for (var t = 0; t < n; t++)
            {
                h = omega + weightedInnovation + beta * h;
                if (!T.IsFinite(h) || h <= T.Zero)
                {
                    return T.PositiveInfinity;
                }
                var lag = t > 0 ? returns[t - 1] : returns[t];
                var eps = returns[t] - mu - phi * lag;
                var standardized = eps / T.Sqrt(h);
                var term = half * T.Log(h) + (half * standardized) * standardized;
                if (typeof(T) == typeof(Half))
                {
                    wideLikelihood += double.CreateChecked(term);
                }
                else
                {
                    ll += term / T.CreateChecked(n);
                }
                // Apply alpha before squaring: the innovation itself may overflow
                // even when its weighted contribution to the next variance fits.
                weightedInnovation = (alpha * eps) * eps;
            }
            return typeof(T) == typeof(Half) ? T.CreateChecked(wideLikelihood / n) : ll;
        }

        // ── Optimise (two-phase warm restart for better convergence) ─────────────
        // Phase 1: coarse pass to escape the neighbourhood of the initial guess.
        // Phase 2: fine pass from phase-1's solution to tighten convergence.
        // Two phases improve robustness vs. a single long run when the Nelder-Mead
        // simplex collapses along poorly-scaled dimensions (the logit parameters
        // span a different scale than ln_omega, causing unbalanced exploration).
        var coarseOpts = new SimplexOptions<T>(
            MaxIterations: options.MaxIterations / 2,
            XTolerance: T.CreateChecked(1e-4),
            FTolerance: T.CreateChecked(1e-4));

        var coarseResult = NelderMeadMinimizer<T>.Minimize(NegQll, x0, coarseOpts);

        var fineOpts = new SimplexOptions<T>(
            MaxIterations: options.MaxIterations,
            XTolerance: tol,
            FTolerance: tol);

        var optResult = NelderMeadMinimizer<T>.Minimize(NegQll, coarseResult.Argmin, fineOpts);

        // ── Extract constrained parameters ───────────────────────────────────────
        var pBest = optResult.Argmin;
        Decode(pBest, out var muFit, out var phiFit, out var omegaFit, out var alphaFit, out var betaFit);

        // ── Compute filtered conditional variances and standardised residuals ────
        // Use the same GARCH filter convention: h_t uses eps_{t-1}
        var hArr = new T[n];
        var sArr = new T[n];
        var hLast = ResidualSecondMoment(muFit, phiFit);
        var weightedInnovationFit = alphaFit * hLast;
        for (var t = 0; t < n; t++)
        {
            var hCur = omegaFit + weightedInnovationFit + betaFit * hLast;
            hArr[t] = hCur;
            var rt1 = t > 0 ? returns[t - 1] : returns[t];
            var epsT = returns[t] - muFit - phiFit * rt1;
            sArr[t] = epsT / T.Sqrt(hCur);
            hLast = hCur;
            weightedInnovationFit = (alphaFit * epsT) * epsT;
        }

        var ll2 = -NegQll(pBest);

        return new GarchFitResult<T>(
            Mu: muFit,
            Phi: phiFit,
            Omega: omegaFit,
            Alpha: alphaFit,
            Beta: betaFit,
            ConditionalVariances: hArr,
            StandardizedResiduals: sArr,
            LogLikelihood: ll2,
            Converged: optResult.Converged);
    }

    private static T Sigmoid(T x) => T.One / (T.One + T.Exp(-x));

    /// <summary>
    /// The largest value <c>α+β</c> may take: a tiny margin strictly below 1, so the
    /// documented stationarity invariant <c>α + β &lt; 1</c> holds by construction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The reparameterisation forms <c>α+β = σ(logit_ab)</c>, which is mathematically in
    /// <c>(0, 1)</c> but rounds to <b>exactly 1.0</b> in IEEE arithmetic once <c>e^{−logit_ab}</c>
    /// underflows relative to 1 (<c>logit_ab ≳ 37</c> for <c>double</c>). Because the inner
    /// Nelder-Mead search over <c>logit_ab</c> is unbounded, near-integrated (IGARCH-like) data
    /// can drive it past that boundary and make <c>Fit()</c> return <c>α+β == 1.0</c>. Capping the
    /// sum a margin below 1 restores the invariant.
    /// </para>
    /// <para>
    /// The margin is <c>max(1e-8, 16·ulp(1))</c>: the <c>1e-8</c> floor is numerically negligible
    /// for any realistic fit at <c>double</c> precision, and the <c>16·ulp(1)</c> term guarantees
    /// the ceiling is strictly representable below 1 for the narrower IEEE widths
    /// (<c>float</c>/<c>Half</c>), where a fixed <c>1e-8</c> literal would round back up to 1.
    /// </para>
    /// </remarks>
    private static readonly T s_abSumCeiling =
        T.One - T.Max(T.CreateChecked(1e-8), T.CreateChecked(16) * (T.BitIncrement(T.One) - T.One));

    // Dividing returns by a maximum magnitude below one increases variance.
    // At scale one normalization copies the same values and cannot make progress;
    // above one it decreases an already subnormal variance. Never recurse there.
    private static bool CanImproveSubnormalVariance(T variance, T scale) =>
        T.IsSubnormal(variance) && scale < T.One;

    /// <summary>
    /// Clamps <paramref name="s"/> (a candidate <c>α+β</c> in <c>(0, 1]</c>) strictly below 1,
    /// so the estimator enforces the documented stationarity invariant <c>α + β &lt; 1</c>.
    /// A no-op for every value already at or below <see cref="s_abSumCeiling"/>.
    /// </summary>
    private static T StrictlyBelowOne(T s) => T.Min(s, s_abSumCeiling);
}
