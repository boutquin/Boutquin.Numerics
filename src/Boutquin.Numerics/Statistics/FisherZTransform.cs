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
using Boutquin.Numerics.Distributions;

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Confidence interval for a Pearson correlation coefficient via the
/// Fisher z-transform. The transform <c>z = atanh(r)</c>
/// is approximately normal with variance <c>1/(n−3)</c>, allowing a
/// symmetric Gaussian CI in the z-domain to be back-transformed via
/// <c>tanh</c> into an asymmetric CI on r ∈ (−1, 1).
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <remarks>
/// Reference: Press et al. (2007), Numerical Recipes, third edition, §14.5,
/// equations 14.5.6–8, for the transform and approximate variance 1/(n−3).
/// Fisher (1915), p.521, mentions the transform but does not derive this interval.
/// The normal approximation assumes a joint distribution near binormal and
/// moderately large samples (Numerical Recipes describes approximately N >= 10).
/// The API's n >= 4 requirement only makes the standard error finite; it does not
/// establish exact small-sample coverage. The interval is the selected uncorrected
/// approximation, rather than the source's bias-corrected significance tests.
/// <para>
/// Tier B: Fully transcendental computation.
/// </para>
/// </remarks>
public static class FisherZTransform<T>
    where T : IFloatingPointIeee754<T>
{
    /// <summary>Forward transform z = atanh(r) = 0.5·ln((1+r)/(1−r)).</summary>
    public static T Forward(T r)
    {
        if (!T.IsFinite(r) || r < -T.One || r > T.One)
        {
            throw new ArgumentOutOfRangeException(nameof(r), r, "Correlation must be finite and lie in [-1, 1].");
        }

        return T.Atanh(r);
    }

    /// <summary>Inverse transform r = tanh(z).</summary>
    public static T Inverse(T z) => T.Tanh(z);

    /// <summary>
    /// Two-sided confidence interval for a Pearson correlation r based on
    /// <paramref name="n"/> paired observations.
    /// </summary>
    /// <param name="r">Finite sample correlation in [−1, 1].</param>
    /// <param name="n">Sample size (must satisfy n ≥ 4).</param>
    /// <param name="confidenceLevel">Two-sided confidence level in (0, 1). Default 0.95.</param>
    /// <returns>(lower, upper) bounds for the population correlation.</returns>
    public static (T Lower, T Upper) ConfidenceInterval(T r, int n, double confidenceLevel = 0.95)
    {
        if (n < 4)
        {
            throw new ArgumentOutOfRangeException(nameof(n), n, "Need at least 4 observations for a Fisher z CI.");
        }

        if (!double.IsFinite(confidenceLevel) || confidenceLevel <= 0.0 || confidenceLevel >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), confidenceLevel, "Confidence level must lie in (0, 1).");
        }

        if (!T.IsFinite(r))
        {
            throw new ArgumentOutOfRangeException(nameof(r), r, "Sample correlation must be finite.");
        }

        if (r < -T.One || r > T.One)
        {
            throw new ArgumentOutOfRangeException(nameof(r), r, "Sample correlation must lie in [-1, 1].");
        }

        if (r == -T.One || r == T.One)
        {
            return (r, r);
        }

        var z = Forward(r);
        var se = T.CreateChecked(1.0 / Math.Sqrt(n - 3.0));
        var tail = (1.0 - confidenceLevel) / 2.0;
        var critical = T.CreateChecked(-NormalDistribution<double>.InverseCdf(tail));

        var lo = z - critical * se;
        var hi = z + critical * se;
        return (Inverse(lo), Inverse(hi));
    }
}

/// <summary>
/// Confidence interval for a Pearson correlation coefficient via the
/// Fisher z-transform. The transform <c>z = atanh(r)</c>
/// is approximately normal with variance <c>1/(n−3)</c>, allowing a
/// symmetric Gaussian CI in the z-domain to be back-transformed via
/// <c>tanh</c> into an asymmetric CI on r ∈ (−1, 1).
/// </summary>
/// <remarks>
/// Reference: Press et al. (2007), Numerical Recipes, third edition, §14.5,
/// equations 14.5.6–8, for the transform and approximate variance 1/(n−3).
/// Fisher (1915), p.521, mentions the transform but does not derive this interval.
/// The normal approximation assumes a joint distribution near binormal and
/// moderately large samples (Numerical Recipes describes approximately N >= 10).
/// The API's n >= 4 requirement only makes the standard error finite; it does not
/// establish exact small-sample coverage. The interval is the selected uncorrected
/// approximation, rather than the source's bias-corrected significance tests.
/// <para>
/// Legacy facade: delegates to <see cref="FisherZTransform{T}"/> at <c>T = double</c>.
/// </para>
/// </remarks>
[Obsolete(
    "Use FisherZTransform<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class FisherZTransform
{
    /// <summary>Forward transform z = atanh(r) = 0.5·ln((1+r)/(1−r)).</summary>
    public static double Forward(double r) => FisherZTransform<double>.Forward(r);

    /// <summary>Inverse transform r = tanh(z).</summary>
    public static double Inverse(double z) => FisherZTransform<double>.Inverse(z);

    /// <summary>
    /// Two-sided confidence interval for a Pearson correlation r based on
    /// <paramref name="n"/> paired observations.
    /// </summary>
    /// <param name="r">Finite sample correlation in [−1, 1].</param>
    /// <param name="n">Sample size (must satisfy n ≥ 4).</param>
    /// <param name="confidenceLevel">Two-sided confidence level in (0, 1). Default 0.95.</param>
    /// <returns>(lower, upper) bounds for the population correlation.</returns>
    public static (double Lower, double Upper) ConfidenceInterval(double r, int n, double confidenceLevel = 0.95)
        => FisherZTransform<double>.ConfidenceInterval(r, n, confidenceLevel);
}
