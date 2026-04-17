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

using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.Tests.Verification;

/// <summary>
/// Python cross-checks for <see cref="LevenbergMarquardtSolver"/> against
/// ``scipy.optimize.least_squares(method='lm')``. Four smooth regimes
/// complement the NIST StRD Nonlinear suite that already covers Numerics'
/// LM solver — see <c>tests/Verification/generate_lm_vectors.py</c>.
/// </summary>
/// <remarks>
/// Regimes: exponential decay with Gaussian noise, sum-of-two-sinusoids,
/// Gompertz growth, logistic. scipy's LM delegates to MINPACK lmder (same
/// family as ours); parity at 1e-5 relative tolerance on parameters and
/// final cost is the spec's §2.1 bar.
/// </remarks>
public sealed class LevenbergMarquardtVerificationTests : CrossLanguageVerificationBase
{
    private const double ParameterTolerance = 1e-5;
    private const double FinalCostTolerance = 1e-5;

    [Fact]
    public void Solve_ExponentialDecayNoisy_MatchesScipy()
    {
        AssertRegime(
            "exponential_decay_noisy",
            (t, y) =>
            {
                double[] Residuals(double[] theta)
                {
                    var a = theta[0];
                    var b = theta[1];
                    var result = new double[t.Length];
                    for (var i = 0; i < t.Length; i++)
                    {
                        result[i] = (a * Math.Exp(-b * t[i])) - y[i];
                    }

                    return result;
                }

                return (Func<double[], double[]>)Residuals;
            });
    }

    [Fact]
    public void Solve_SumOfSinusoids_MatchesScipy()
    {
        AssertRegime(
            "sum_of_sinusoids",
            (t, y) =>
            {
                double[] Residuals(double[] theta)
                {
                    var a = theta[0];
                    var b = theta[1];
                    var omega1 = theta[2];
                    var omega2 = theta[3];
                    var result = new double[t.Length];
                    for (var i = 0; i < t.Length; i++)
                    {
                        result[i] = (a * Math.Sin(omega1 * t[i])) + (b * Math.Cos(omega2 * t[i])) - y[i];
                    }

                    return result;
                }

                return Residuals;
            });
    }

    [Fact]
    public void Solve_GompertzGrowth_MatchesScipy()
    {
        AssertRegime(
            "gompertz_growth",
            (t, y) =>
            {
                double[] Residuals(double[] theta)
                {
                    var a = theta[0];
                    var b = theta[1];
                    var c = theta[2];
                    var result = new double[t.Length];
                    for (var i = 0; i < t.Length; i++)
                    {
                        result[i] = (a * Math.Exp(-b * Math.Exp(-c * t[i]))) - y[i];
                    }

                    return result;
                }

                return Residuals;
            });
    }

    [Fact]
    public void Solve_Logistic_MatchesScipy()
    {
        AssertRegime(
            "logistic",
            (t, y) =>
            {
                double[] Residuals(double[] theta)
                {
                    var a = theta[0];
                    var b = theta[1];
                    var c = theta[2];
                    var result = new double[t.Length];
                    for (var i = 0; i < t.Length; i++)
                    {
                        result[i] = (a / (1.0 + Math.Exp(-b * (t[i] - c)))) - y[i];
                    }

                    return result;
                }

                return Residuals;
            });
    }

    /// <summary>
    /// Loads a named regime from <c>lm.json</c>, runs Numerics' LM solver
    /// against the caller-supplied residual factory, and asserts parity with
    /// the scipy reference parameters and final cost.
    /// </summary>
    private static void AssertRegime(
        string regimeName,
        Func<double[], double[], Func<double[], double[]>> residualFactory)
    {
        using var doc = LoadVector("lm");
        var regime = doc.RootElement.GetProperty("regimes").GetProperty(regimeName);

        var t = GetDoubleArray(regime.GetProperty("t"));
        var y = GetDoubleArray(regime.GetProperty("y"));
        var initialGuess = GetDoubleArray(regime.GetProperty("initial_guess"));
        var expectedParameters = GetDoubleArray(regime.GetProperty("parameters"));
        var expectedCost = regime.GetProperty("final_cost").GetDouble();

        var residuals = residualFactory(t, y);
        var solver = new LevenbergMarquardtSolver();
        var result = solver.Solve(residuals, initialGuess);

        Assert.True(result.Converged, $"{regimeName}: LM must converge (reason={result.TerminationReason}).");
        Assert.Equal(expectedParameters.Length, result.Parameters.Length);
        for (var i = 0; i < expectedParameters.Length; i++)
        {
            AssertRelative(
                result.Parameters[i],
                expectedParameters[i],
                ParameterTolerance,
                $"{regimeName}/parameter[{i}]");
        }

        AssertRelative(result.FinalCost, expectedCost, FinalCostTolerance, $"{regimeName}/final_cost");
    }

    /// <summary>
    /// ``max(|expected|, 1)``-relative assertion; same convention as the OLS
    /// verification tests.
    /// </summary>
    private static void AssertRelative(double actual, double expected, double tolerance, string label)
    {
        var denominator = Math.Max(Math.Abs(expected), 1.0);
        var relativeError = Math.Abs(actual - expected) / denominator;
        Assert.True(
            relativeError <= tolerance,
            $"{label}: expected {expected}, got {actual}, |Δ|/max(|expected|,1) = {relativeError} > tol = {tolerance}");
    }
}
