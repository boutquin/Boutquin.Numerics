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
/// Fisher z-transform (Fisher 1915). The transform <c>z = atanh(r)</c>
/// is approximately normal with variance <c>1/(n−3)</c>, allowing a
/// symmetric Gaussian CI in the z-domain to be back-transformed via
/// <c>tanh</c> into an asymmetric CI on r ∈ (−1, 1).
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <remarks>
/// Reference: Fisher, R. A. (1915). "Frequency Distribution of the Values
/// of the Correlation Coefficient in Samples from an Indefinitely Large
/// Population." Biometrika, 10(4), 507–521.
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
        var half = T.CreateChecked(0.5);
        return half * T.Log((T.One + r) / (T.One - r));
    }

    /// <summary>Inverse transform r = tanh(z).</summary>
    public static T Inverse(T z) => T.Tanh(z);

    /// <summary>
    /// Two-sided confidence interval for a Pearson correlation r based on
    /// <paramref name="n"/> paired observations.
    /// </summary>
    /// <param name="r">Sample correlation in (−1, 1).</param>
    /// <param name="n">Sample size (must satisfy n ≥ 4).</param>
    /// <param name="confidenceLevel">Two-sided confidence level in (0, 1). Default 0.95.</param>
    /// <returns>(lower, upper) bounds for the population correlation.</returns>
    public static (T Lower, T Upper) ConfidenceInterval(T r, int n, T confidenceLevel)
    {
        if (n < 4)
        {
            throw new ArgumentOutOfRangeException(nameof(n), n, "Need at least 4 observations for a Fisher z CI.");
        }

        if (confidenceLevel <= T.Zero || confidenceLevel >= T.One)
        {
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), confidenceLevel, "Confidence level must lie in (0, 1).");
        }

        if (r <= -T.One || r >= T.One)
        {
            return (r, r);
        }

        var z = Forward(r);
        var se = T.One / T.Sqrt(T.CreateChecked(n - 3));
        var tail = (T.One - confidenceLevel) / T.CreateChecked(2);
        var critical = NormalDistribution<T>.InverseCdf(T.One - tail);

        var lo = z - critical * se;
        var hi = z + critical * se;
        return (Inverse(lo), Inverse(hi));
    }
}

/// <summary>
/// Confidence interval for a Pearson correlation coefficient via the
/// Fisher z-transform (Fisher 1915). The transform <c>z = atanh(r)</c>
/// is approximately normal with variance <c>1/(n−3)</c>, allowing a
/// symmetric Gaussian CI in the z-domain to be back-transformed via
/// <c>tanh</c> into an asymmetric CI on r ∈ (−1, 1).
/// </summary>
/// <remarks>
/// Reference: Fisher, R. A. (1915). "Frequency Distribution of the Values
/// of the Correlation Coefficient in Samples from an Indefinitely Large
/// Population." Biometrika, 10(4), 507–521.
/// <para>
/// Legacy facade: delegates to <see cref="FisherZTransform{T}"/> at <c>T = double</c>.
/// </para>
/// </remarks>
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
    /// <param name="r">Sample correlation in (−1, 1).</param>
    /// <param name="n">Sample size (must satisfy n ≥ 4).</param>
    /// <param name="confidenceLevel">Two-sided confidence level in (0, 1). Default 0.95.</param>
    /// <returns>(lower, upper) bounds for the population correlation.</returns>
    public static (double Lower, double Upper) ConfidenceInterval(double r, int n, double confidenceLevel = 0.95)
        => FisherZTransform<double>.ConfidenceInterval(r, n, confidenceLevel);
}
