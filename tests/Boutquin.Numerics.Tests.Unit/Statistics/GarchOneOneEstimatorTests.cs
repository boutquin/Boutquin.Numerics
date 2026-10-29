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

public sealed class GarchOneOneEstimatorTests
{
    [Fact]
    public void Fit_LargeFiniteSeries_PreservesRepresentableVarianceScale()
    {
        // Raw squared-deviation sum overflows: 1000*1e306 > maxDouble,
        // while physical sample variance (1000/999)*1e306 is representable.
        var returns = Enumerable.Range(0, 1000).Select(i => i % 2 == 0 ? 1e153 : -1e153).ToArray();
        var result = GarchOneOneEstimator<double>.Fit(returns);

        double.IsFinite(result.Omega).Should().BeTrue();
        result.Omega.Should().BePositive();
        double.IsFinite(result.LogLikelihood).Should().BeTrue();
        result.ConditionalVariances.Should().HaveCount(1000);
        result.ConditionalVariances.Should().OnlyContain(value => double.IsFinite(value) && value > 0);
    }

    [Fact]
    public void Fit_SparseWideSeries_RejectsUnrepresentableFittedVariance()
    {
        var normalized = new double[1000];
        normalized[0] = 1;
        var normalizedFit = GarchOneOneEstimator<double>.Fit(normalized);
        normalizedFit.Alpha.Should().BeGreaterThan(.9);
        Math.Abs(normalizedFit.Mu).Should().BeLessThan(.01);

        // Independent recurrence lower bound: positive omega/beta/h imply
        // h1 >= alpha*(1-mu)^2 > .9*.99^2 = .88209. Restoring returns
        // scaled by 1e155 multiplies h by about1e310. Since maxDouble<2e308,
        // any normalized h>.02 is unrepresentable in original variance units.
        // This certifies the range failure, not optimizer parameter parity.
        var lowerBound = normalizedFit.Alpha * Math.Pow(1 - Math.Abs(normalizedFit.Mu), 2);
        lowerBound.Should().BeGreaterThan(.88209);
        var returns = new double[1000];
        returns[0] = 1e155;
        var act = () => GarchOneOneEstimator<double>.Fit(returns);
        act.Should().Throw<ArgumentException>().WithParameterName("returns")
            .WithMessage("Fitted GARCH variance is not representable*");
    }

    [Fact]
    public void Fit_UnrepresentableVariance_StillRejectsSeries()
    {
        var returns = new double[1000];
        returns[0] = 1e200;

        var act = () => GarchOneOneEstimator<double>.Fit(returns);

        act.Should().Throw<ArgumentException>().WithMessage("*variance is non-finite*");
    }

    // Simulate GARCH(1,1) path for testing (pure C#, no external deps)
    private static double[] SimulateGarch(int n, double omega, double alpha, double beta, double mu = 0, int seed = 42)
    {
        var rng = new System.Random(seed);
        var returns = new double[n];
        var h = omega / (1 - alpha - beta); // unconditional variance
        for (var t = 0; t < n; t++)
        {
            // Box-Muller for standard normal
            var u1 = rng.NextDouble();
            var u2 = rng.NextDouble();
            var z = Math.Sqrt(-2 * Math.Log(u1 + 1e-15)) * Math.Cos(2 * Math.PI * u2);
            var eps = Math.Sqrt(h) * z;
            returns[t] = mu + eps;
            h = omega + alpha * eps * eps + beta * h;
        }
        return returns;
    }

    [Fact]
    public void Fit_SimulatedGarch_RecoversTrueParams()
    {
        var returns = SimulateGarch(2000, omega: 1e-5, alpha: 0.10, beta: 0.85);
        var result = GarchOneOneEstimator<double>.Fit(returns);
        result.Converged.Should().BeTrue();
        result.Alpha.Should().BeApproximately(0.10, 0.05);
        result.Beta.Should().BeApproximately(0.85, 0.05);
        result.Omega.Should().BeApproximately(1e-5, 1e-4);
    }

    [Fact]
    public void Fit_ConditionalVariancesAllPositive()
    {
        var returns = SimulateGarch(500, omega: 1e-5, alpha: 0.10, beta: 0.85);
        var result = GarchOneOneEstimator<double>.Fit(returns);
        result.ConditionalVariances.Should().AllSatisfy(h => h.Should().BePositive());
    }

    [Fact]
    public void Fit_HalfWithDefaultTolerance_ReturnsFiniteFit()
    {
        var returns = Enumerable.Range(0, 20)
            .Select(i => (Half)(i % 3 - 1))
            .ToArray();

        GarchFitResult<Half>? result = null;
        var act = () => result = GarchOneOneEstimator<Half>.Fit(returns);

        act.Should().NotThrow();
        Half.IsFinite(result!.Omega).Should().BeTrue();
        result.ConditionalVariances.Should().HaveCount(returns.Length);
    }

    [Fact]
    public void Fit_StationarityInvariant()
    {
        var returns = SimulateGarch(500, omega: 1e-5, alpha: 0.10, beta: 0.85);
        var result = GarchOneOneEstimator<double>.Fit(returns);
        (result.Alpha + result.Beta).Should().BeLessThan(1.0);
    }

    [Fact]
    public void ForecastVariance_OneStep_MatchesAnalytic()
    {
        var returns = SimulateGarch(200, omega: 1e-5, alpha: 0.10, beta: 0.85);
        var result = GarchOneOneEstimator<double>.Fit(returns);
        var n = returns.Length;
        var sT = result.StandardizedResiduals[n - 1];
        var hT = result.ConditionalVariances[n - 1];
        var epsT = sT * Math.Sqrt(hT);
        var expected = result.Omega + result.Alpha * epsT * epsT + result.Beta * hT;
        result.ForecastVariance(1).Should().BeApproximately(expected, 1e-10);
    }

    [Fact]
    public void Fit_TooFewObservations_Throws()
    {
        var returns = new double[] { 0.01, 0.02, 0.03 };
        var act = () => GarchOneOneEstimator<double>.Fit(returns);
        act.Should().Throw<ArgumentException>().WithMessage("*Insufficient*");
    }

    [Fact]
    public void Fit_ConstantSeries_Throws()
    {
        var returns = Enumerable.Repeat(0.01, 100).ToArray();
        var act = () => GarchOneOneEstimator<double>.Fit(returns);
        act.Should().Throw<ArgumentException>().WithMessage("*zero variance*");
    }

    [Fact]
    public void Fit_StandardizedResiduals_HaveUnitVariance()
    {
        var returns = SimulateGarch(500, omega: 1e-5, alpha: 0.10, beta: 0.85);
        var result = GarchOneOneEstimator<double>.Fit(returns);
        var std = result.StandardizedResiduals;
        var mean = std.Average();
        var variance = std.Select(s => (s - mean) * (s - mean)).Average();
        variance.Should().BeApproximately(1.0, 0.3); // statistical tolerance
    }

    [Fact]
    public void Fit_NullReturns_ThrowsArgumentNullException()
    {
        // F-LQ-7: a null returns array must be rejected with ArgumentNullException,
        // not a NullReferenceException from the unguarded returns.Length access.
        var act = () => GarchOneOneEstimator<double>.Fit(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Fit_NonFiniteReturnElement_ThrowsArgumentException()
    {
        // M-3 guard: a NaN/±∞ element must be rejected explicitly, naming the
        // parameter and index. Pre-guard, a NaN silently poisons the sample-mean/
        // variance accumulation and either surfaces as an unrelated Nelder-Mead
        // "non-finite objective" ArgumentException (different message) or a
        // silently NaN-corrupted fit — neither matches "*index 10*".
        var returns = SimulateGarch(50, omega: 1e-5, alpha: 0.10, beta: 0.85);
        returns[10] = double.NaN;

        var act = () => GarchOneOneEstimator<double>.Fit(returns);

        act.Should().Throw<ArgumentException>().WithMessage("*index 10*");
    }

    // Pure-integrated (IGARCH-like) clustered series: omega = 0, alpha + beta = 1
    // exactly, so the variance process is a random walk with no mean reversion.
    // This maximises the QMLE incentive to drive persistence toward 1, so the
    // UNCLAMPED optimizer pushes logit_ab past the Sigmoid saturation point and
    // Fit() returns alpha + beta == 1.0 bit-exactly (the F-VS1-GARCH defect).
    private static double[] SimulateIntegratedGarch(int n, int seed)
    {
        var rng = new System.Random(seed);
        var returns = new double[n];
        var h = 1.0;
        const double alpha = 0.12, beta = 0.88; // alpha + beta == 1 (integrated)
        for (var t = 0; t < n; t++)
        {
            var u1 = rng.NextDouble();
            var u2 = rng.NextDouble();
            var z = Math.Sqrt(-2 * Math.Log(u1 + 1e-15)) * Math.Cos(2 * Math.PI * u2);
            var eps = Math.Sqrt(h) * z;
            returns[t] = eps;
            h = alpha * eps * eps + beta * h; // omega = 0 => pure integrated
        }
        return returns;
    }

    [Fact]
    public void Fit_NearIntegratedSeries_ReturnsAlphaPlusBetaStrictlyBelowOne()
    {
        // F-VS1-GARCH (AC1). On a pure-integrated series the unclamped optimizer
        // pushes logit_ab past the Sigmoid saturation point, so the PRE-FIX code
        // returns alpha + beta == 1.0 bit-exactly (RED). After the StrictlyBelowOne
        // clamp, the documented invariant "alpha + beta < 1 (enforced by
        // construction)" holds strictly for every fit (GREEN).
        var returns = SimulateIntegratedGarch(1000, seed: 16);

        var result = GarchOneOneEstimator<double>.Fit(returns);

        (result.Alpha + result.Beta).Should().BeLessThan(1.0);
    }

    [Fact]
    public void Fit_NormalSeries_SatisfiesAuthorPresampleAndAverageLikelihood()
    {
        var returns = SimulateGarch(500, omega: 1e-5, alpha: 0.10, beta: 0.85);
        var result = GarchOneOneEstimator<double>.Fit(returns);
        var second = GarchOneOneEstimator<double>.Fit(returns);
        result.Mu.Should().Be(second.Mu);
        result.Omega.Should().Be(second.Omega);
        result.Alpha.Should().Be(second.Alpha);
        result.Beta.Should().Be(second.Beta);
        var seed = returns.Select(x => (x - result.Mu) * (x - result.Mu)).Average();
        var variance = seed;
        var previousSquare = seed;
        var sum = 0.0;
        for (var i = 0; i < returns.Length; i++)
        {
            variance = result.Omega + result.Alpha * previousSquare + result.Beta * variance;
            result.ConditionalVariances[i].Should().BeApproximately(variance, 1e-15);
            var residual = returns[i] - result.Mu;
            result.StandardizedResiduals[i].Should().BeApproximately(residual / Math.Sqrt(variance), 1e-12);
            sum += -.5 * (Math.Log(variance) + residual * residual / variance);
            previousSquare = residual * residual;
        }
        result.LogLikelihood.Should().BeApproximately(sum / returns.Length, 1e-12);
        (result.Alpha + result.Beta).Should().BeLessThan(.999);
    }

    [Fact]
    public void ForecastVariance_MultiStep_OnFittedNearIntegratedResult_DoesNotThrow()
    {
        // F-VS1-GARCH (AC3 — Fit/Forecast consistency). PRE-FIX: the near-integrated
        // fit returns alpha + beta == 1.0, so ForecastVariance(horizon > 1) throws
        // "Non-stationary GARCH" (RED). POST-FIX: because Fit() now guarantees
        // alpha + beta < 1, the multi-step forecast succeeds and is finite (GREEN).
        var returns = SimulateIntegratedGarch(1000, seed: 16);
        var result = GarchOneOneEstimator<double>.Fit(returns);

        var forecast = result.ForecastVariance(5);

        forecast.Should().BePositive();
        double.IsFinite(forecast).Should().BeTrue();
    }

    [Fact]
    public void ForecastVariance_MultiStep_OnHandBuiltNonStationaryRecord_StillThrows()
    {
        // F-VS1-GARCH (AC3, second half). The ForecastVariance guard is retained for
        // hand-constructed records: a caller can build a GarchFitResult with
        // alpha + beta >= 1 directly, and the long-run variance is genuinely
        // undefined there, so the throw must still fire. This path is untouched by
        // the fit-side clamp.
        var result = new GarchFitResult<double>(
            Mu: 0.0,
            Phi: 0.0,
            Omega: 1e-5,
            Alpha: 0.30,
            Beta: 0.70, // alpha + beta == 1 exactly, set by hand
            ConditionalVariances: new[] { 1e-4, 1e-4 },
            StandardizedResiduals: new[] { 0.5, 0.5 },
            LogLikelihood: 0.0,
            Converged: true);

        var act = () => result.ForecastVariance(2);

        act.Should().Throw<InvalidOperationException>().WithMessage("*Non-stationary*");
    }

    [Fact]
    public void Fit_LowScaleNonConstantSeries_IsNotRejectedAsZeroVariance()
    {
        var returns = Enumerable.Range(0, 30)
            .Select(i => (i % 2 == 0 ? 1.0 : -1.0) * 1e-7)
            .ToArray();
        var options = new GarchOptions<double>(MaxIterations: 20);

        var result = GarchOneOneEstimator<double>.Fit(returns, options);

        result.ConditionalVariances.Should().AllSatisfy(value =>
        {
            value.Should().BePositive();
            double.IsFinite(value).Should().BeTrue();
        });
        double.IsFinite(result.LogLikelihood).Should().BeTrue();
    }

    [Theory]
    [InlineData(0, 100)]
    [InlineData(1, 100)]
    [InlineData(20, 0)]
    [InlineData(20, -1)]
    [InlineData(20, 1)]
    public void Fit_InvalidIntegerOptions_ThrowArgumentOutOfRangeException(
        int minObservations,
        int maxIterations)
    {
        var returns = SimulateGarch(30, omega: 1e-5, alpha: 0.10, beta: 0.85);
        var options = new GarchOptions<double>(
            MinObservations: minObservations,
            MaxIterations: maxIterations);

        var act = () => GarchOneOneEstimator<double>.Fit(returns, options);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("options");
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1e-8)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void Fit_InvalidTolerance_ThrowsArgumentOutOfRangeException(double tolerance)
    {
        var returns = SimulateGarch(30, omega: 1e-5, alpha: 0.10, beta: 0.85);
        var options = new GarchOptions<double>(Tolerance: tolerance);

        var act = () => GarchOneOneEstimator<double>.Fit(returns, options);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("options");
    }
}
