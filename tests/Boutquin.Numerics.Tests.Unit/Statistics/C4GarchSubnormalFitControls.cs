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
using System.Reflection;
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

[Trait("Tier", "Unit")]
public sealed class C4GarchSubnormalFitControls
{
    [Fact]
    public void DoubleRepresentableSubnormalFit_PreservesScaleLaw() => CheckScale<double>(1e-160, 1e-320);

    [Fact]
    public void FloatRepresentableSubnormalFit_PreservesScaleLaw() => CheckScale<float>(1e-21, 1e-42);

    [Theory]
    [InlineData(3e-162)]
    [InlineData(2e-162)]
    public void DoubleUnrepresentableIntercept_IsExplicitlyRejected(double scale) => CheckRejection<double>(scale);

    [Fact]
    public void FloatUnrepresentableIntercept_IsExplicitlyRejected() => CheckRejection<float>(4e-23);

    [Fact]
    public void HalfUnrepresentableIntercept_IsExplicitlyRejected() => CheckRejection<Half>(.0003);

    [Fact]
    public void HalfNearConstantUnitPanel_RejectsUnrepresentableInitialInterceptWithoutRetry()
    {
        var upper = Half.BitIncrement(Half.One);
        var values = Enumerable.Range(0, 40).Select(i => i % 2 == 0 ? Half.One : upper).ToArray();
        // Independent exact values: deviations are +/-2^-11, unbiased variance
        // (40/39)*2^-22 rounds to 2^-22; the .05 variance target rounds to zero.
        // Dividing by upper gives {1023/1024,1}, still variance 2^-22.
        var act = () => GarchOneOneEstimator<Half>.Fit(values);
        act.Should().Throw<ArgumentException>().WithParameterName("returns")
            .WithMessage("The variance-targeting initial intercept is not representable*");
    }

    [Fact]
    public void HalfNearUnitRepresentableInitialIntercept_RemainsFittable()
    {
        var values = Enumerable.Range(0, 40).Select(i => (Half)(i % 2 == 0 ? 1 : 1.01)).ToArray();
        var act = () => GarchOneOneEstimator<Half>.Fit(values);
        var fit = act.Should().NotThrow().Which;
        (fit.Omega > Half.Zero).Should().BeTrue();
    }

    [Fact]
    public void NormalizationAtUnitScale_IsAFixedPointAndMustNotRetry()
    {
        // Check the actual production decision directly so a deliberately wrong
        // equality mutation is detected without executing infinite recursion.
        var method = typeof(GarchOneOneEstimator<Half>).GetMethod(
            "CanImproveSubnormalVariance", BindingFlags.NonPublic | BindingFlags.Static)!;
        var variance = (Half)Math.ScaleB(1, -22);
        ((bool)method.Invoke(null, [variance, Half.One])!).Should().BeFalse();
        ((bool)method.Invoke(null, [variance, (Half)2])!).Should().BeFalse();
        ((bool)method.Invoke(null, [variance, (Half).5])!).Should().BeTrue();
    }

    [Fact]
    public void DoubleFiniteLargeVariance_PreservesScaleLaw() => CheckScale<double>(1e153, 1e306);

    private static void CheckScale<T>(double scale, double varianceScale) where T : struct, IFloatingPointIeee754<T>
    {
        var values = Enumerable.Range(0, 40).Select(i => T.CreateChecked(i % 2 == 0 ? scale : -scale)).ToArray();
        Func<GarchFitResult<T>> act = () => GarchOneOneEstimator<T>.Fit(values);
        var fit = act.Should().NotThrow("the finite positive variance and intercept are representable").Which;
        T.IsFinite(fit.Omega).Should().BeTrue();
        (fit.Omega > T.Zero).Should().BeTrue();
        foreach (var variance in fit.ConditionalVariances)
        {
            T.IsFinite(variance).Should().BeTrue();
            (variance > T.Zero).Should().BeTrue();
        }

        // Independent GARCH homogeneity: r'=c*r gives mu'=c*mu,
        // omega'=c^2*omega, h'=c^2*h; alpha, beta and standardized
        // innovations are invariant. The per-observation likelihood changes by -log(c).
        // This is a metamorphic scale identity, not parameter recovery parity.
        var normalized = Enumerable.Range(0, 40).Select(i => T.CreateChecked(i % 2 == 0 ? 1 : -1)).ToArray();
        var reference = GarchOneOneEstimator<T>.Fit(normalized);
        double.CreateChecked(fit.LogLikelihood).Should().BeApproximately(
            double.CreateChecked(reference.LogLikelihood) - Math.Log(scale), typeof(T) == typeof(double) ? 1e-10 : 1e-3);
        var quantum = double.CreateChecked(T.Epsilon);
        double.CreateChecked(fit.Omega).Should().BeApproximately(double.CreateChecked(reference.Omega) * varianceScale, Math.Max(2 * quantum, varianceScale * 1e-14));
        fit.Alpha.Should().Be(reference.Alpha);
        fit.Beta.Should().Be(reference.Beta);
        for (var i = 0; i < 40; i++)
        {
            double.CreateChecked(fit.ConditionalVariances[i]).Should().BeApproximately(
                double.CreateChecked(reference.ConditionalVariances[i]) * varianceScale, Math.Max(2 * quantum, varianceScale * 1e-14));
            fit.StandardizedResiduals[i].Should().Be(reference.StandardizedResiduals[i]);
        }
    }

    private static void CheckRejection<T>(double scale) where T : struct, IFloatingPointIeee754<T>
    {
        var values = Enumerable.Range(0, 40).Select(i => T.CreateChecked(i % 2 == 0 ? scale : -scale)).ToArray();
        var act = () => GarchOneOneEstimator<T>.Fit(values);
        act.Should().Throw<ArgumentException>()
            .WithParameterName("returns")
            .WithMessage("Fitted GARCH intercept is not representable*");
    }
}
