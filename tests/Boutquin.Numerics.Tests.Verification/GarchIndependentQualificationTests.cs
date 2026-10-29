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

namespace Boutquin.Numerics.Tests.Verification;

/// <summary>
/// Exercises GARCH fitting against arch 8.0.0 variance/Normal implementations,
/// fitted by constrained SLSQP in natural coordinates rather than the library's
/// transformed Nelder-Mead path. Presample and likelihood conventions are aligned.
/// </summary>
public sealed class GarchIndependentQualificationTests : CrossLanguageVerificationBase
{
    [Theory]
    [InlineData(1.0, false)]
    [InlineData(10.0, false)]
    [InlineData(10000.0, false)]
    [InlineData(1.0, true)]
    [InlineData(10.0, true)]
    [InlineData(10000.0, true)]
    public void Ar1Mean_LocationShiftPreservesResidualLikelihood(double location, bool targeting)
    {
        using var source = LoadVector("garch");
        var innovations = GetDoubleArray(source.RootElement.GetProperty("returns"));
        var observations = new double[innovations.Length];
        for (var i = 0; i < observations.Length; i++)
        {
            observations[i] = innovations[i] + (i == 0 ? 0 : .3 * observations[i - 1]);
        }
        var options = new GarchOptions<double>(IncludeAr1Mean: true, Tolerance: 1e-12)
        {
            VarianceTargeting = targeting,
        };
        var baseline = GarchOneOneEstimator<double>.Fit(observations, options);
        var shifted = GarchOneOneEstimator<double>.Fit(observations.Select(x => x + location).ToArray(), options);
        // Under r'=r+c, mu'=mu+(1-phi)c. Every innovation and its likelihood
        // are unchanged, including the first observation/lag convention.
        Assert.True(baseline.Converged && shifted.Converged);
        AssertScalarWithin(shifted.Phi, baseline.Phi, 1e-5, "AR coefficient after location shift");
        AssertScalarWithin(shifted.Alpha, baseline.Alpha, 1e-5, "ARCH coefficient after location shift");
        AssertScalarWithin(shifted.Beta, baseline.Beta, 1e-5, "GARCH coefficient after location shift");
        AssertScalarWithin(shifted.Omega, baseline.Omega, 1e-7, "variance intercept after location shift");
        AssertScalarWithin(shifted.Mu, baseline.Mu + (1 - baseline.Phi) * location,
            1e-5 * Math.Max(1, location), "restored mean intercept");
        AssertScalarWithin(shifted.LogLikelihood, baseline.LogLikelihood, 1e-8, "likelihood after location shift");
    }

    [Theory]
    [InlineData("constant_mean")]
    [InlineData("zero_mean")]
    [InlineData("ar1_mean")]
    [InlineData("variance_target")]
    public void Fit_FilterLikelihoodAndForecastMatchIndependentArchFit(string name)
    {
        using var input = LoadVector("garch");
        using var reference = LoadVector("garch_independent_qualification");
        var root = reference.RootElement;
        Assert.Equal("8.0.0", root.GetProperty("arch_version").GetString());
        Assert.Equal(4, root.GetProperty("cases").GetArrayLength());
        var item = root.GetProperty("cases").EnumerateArray().Single(row => row.GetProperty("name").GetString() == name);
        var scale = root.GetProperty("input_scale").GetDouble();
        var observations = GetDoubleArray(input.RootElement.GetProperty("returns")).Select(value => scale * value).ToArray();
        Assert.Equal(2000, observations.Length);
        Assert.Equal(observations.Length, root.GetProperty("source_observations").GetInt32());
        Assert.Equal(3, item.GetProperty("optimizer_successes").GetInt32());
        Assert.InRange(item.GetProperty("start_objective_spread").GetDouble(), 0, 1e-8);

        var fit = GarchOneOneEstimator<double>.Fit(observations, new GarchOptions<double>(
            IncludeAr1Mean: item.GetProperty("ar1_mean").GetBoolean(), Tolerance: 1e-12)
        {
            FixedZeroMean = item.GetProperty("fixed_zero_mean").GetBoolean(),
            VarianceTargeting = item.GetProperty("variance_targeting").GetBoolean(),
        });

        Assert.True(fit.Converged, $"{name}: the production fit must report convergence");
        // Same observations and local optimum: these are numerical tolerances,
        // not statistical sampling allowances. The independent starts agree to 3e-13.
        AssertScalarWithin(fit.Mu, item.GetProperty("mu").GetDouble(), 1e-5, $"{name}: mu");
        AssertScalarWithin(fit.Phi, item.GetProperty("phi").GetDouble(), 1e-5, $"{name}: phi");
        AssertScalarWithin(fit.Omega, item.GetProperty("omega").GetDouble(), 1e-5, $"{name}: omega");
        AssertScalarWithin(fit.Alpha, item.GetProperty("alpha").GetDouble(), 1e-5, $"{name}: alpha");
        AssertScalarWithin(fit.Beta, item.GetProperty("beta").GetDouble(), 1e-5, $"{name}: beta");
        AssertScalarWithin(fit.LogLikelihood, item.GetProperty("log_likelihood").GetDouble(), 1e-9, $"{name}: likelihood");

        var variances = fit.ConditionalVariances;
        var residuals = fit.StandardizedResiduals;
        var expectedVariances = GetDoubleArray(item.GetProperty("conditional_variances"));
        var expectedResiduals = GetDoubleArray(item.GetProperty("standardized_residuals"));
        Assert.Equal(observations.Length, expectedVariances.Length);
        Assert.Equal(observations.Length, expectedResiduals.Length);
        Assert.Equal(observations.Length, variances.Length);
        Assert.Equal(observations.Length, residuals.Length);
        for (var i = 0; i < observations.Length; i++)
        {
            AssertScalarWithin(variances[i], expectedVariances[i], 5e-5, $"{name}: variance[{i}]");
            AssertScalarWithin(residuals[i], expectedResiduals[i], 5e-5, $"{name}: residual[{i}]");
        }

        var forecasts = GetDoubleArray(item.GetProperty("forecasts"));
        Assert.Equal(10, forecasts.Length);
        for (var horizon = 1; horizon <= forecasts.Length; horizon++)
        {
            AssertScalarWithin(fit.ForecastVariance(horizon), forecasts[horizon - 1], 5e-5, $"{name}: forecast[{horizon}]");
        }
    }
}
