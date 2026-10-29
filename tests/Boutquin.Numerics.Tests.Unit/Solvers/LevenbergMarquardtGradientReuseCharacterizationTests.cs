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
using FluentAssertions;
using Xunit.Abstractions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// Independent stationary-point checks for the exponential least-squares problem.
/// </summary>
public sealed class LevenbergMarquardtGradientReuseCharacterizationTests
{
    private readonly ITestOutputHelper _output;

    public LevenbergMarquardtGradientReuseCharacterizationTests(ITestOutputHelper output) => _output = output;

    // Exponential model y = a·exp(b·x) fit to deterministic samples — a canonical LM problem
    // whose gradient/Jacobian are dense and whose convergence path spans several accepted and
    // rejected steps (exercising the reuse on both the initial check and every accepted step).
    private static (double[] xs, double[] ys) ExpData()
    {
        var xs = new double[20];
        var ys = new double[20];
        for (var i = 0; i < 20; i++)
        {
            var x = i * 0.1;
            xs[i] = x;
            // True params a=2.5, b=0.7 plus tiny deterministic perturbation.
            ys[i] = 2.5 * Math.Exp(0.7 * x) + Math.Sin(i) * 0.001;
        }

        return (xs, ys);
    }

    private static Func<double[], double[]> ExpResiduals(double[] xs, double[] ys)
        => theta =>
        {
            var r = new double[xs.Length];
            for (var i = 0; i < xs.Length; i++)
            {
                r[i] = ys[i] - theta[0] * Math.Exp(theta[1] * xs[i]);
            }

            return r;
        };

    private static double AnalyticGradientInfinityNorm(double[] xs, double[] ys, double[] theta)
    {
        var ga = 0.0;
        var gb = 0.0;
        for (var i = 0; i < xs.Length; i++)
        {
            var e = Math.Exp(theta[1] * xs[i]);
            var residual = ys[i] - theta[0] * e;
            ga -= e * residual;
            gb -= theta[0] * xs[i] * e * residual;
        }
        return Math.Max(Math.Abs(ga), Math.Abs(gb));
    }

    [Fact]
    public void ExpFit_FiniteDifference_MatchesIndependentStationaryPoint()
    {
        var (xs, ys) = ExpData();
        var solver = new LevenbergMarquardtSolver<double>(
            maxIterations: 200,
            functionTolerance: 1e-12,
            parameterTolerance: 1e-12,
            gradientTolerance: 1e-12,
            initialDamping: 1e-3);

        var r = solver.Solve(ExpResiduals(xs, ys), new[] { 1.0, 1.0 });
        _output.WriteLine($"fd: p=[{r.Parameters[0]:R},{r.Parameters[1]:R}] cost={r.FinalCost:R} it={r.Iterations} conv={r.Converged} reason={r.TerminationReason}");

        r.Converged.Should().BeTrue();
        // With the complete MINPACK FTOL ratio gate, this point satisfies the
        // independent analytic gradient criterion instead of the old two-clause FTOL.
        AnalyticGradientInfinityNorm(xs, ys, r.Parameters).Should().BeLessThanOrEqualTo(1e-12);
        r.TerminationReason.Should().Be(LmTerminationReason.GradientToleranceReached);
        // Independent 60-digit mpmath root of the two analytic normal equations
        // for these binary input samples; this is not a recorded solver output.
        r.Parameters[0].Should().BeApproximately(2.50013756101917086, 1e-10);
        r.Parameters[1].Should().BeApproximately(0.69995347971710163, 1e-10);
        r.FinalCost.Should().BeApproximately(4.54661308338381792e-6, 1e-16);
    }

    [Fact]
    public void ExpFit_AnalyticJacobian_MatchesIndependentStationaryPoint()
    {
        var (xs, ys) = ExpData();
        var solver = new LevenbergMarquardtSolver<double>(
            maxIterations: 200,
            functionTolerance: 1e-12,
            parameterTolerance: 1e-12,
            gradientTolerance: 1e-12,
            initialDamping: 1e-3);

        Func<double[], double[,]> jac = theta =>
        {
            var j = new double[xs.Length, 2];
            for (var i = 0; i < xs.Length; i++)
            {
                var e = Math.Exp(theta[1] * xs[i]);
                j[i, 0] = -e;                      // ∂r/∂a
                j[i, 1] = -theta[0] * xs[i] * e;   // ∂r/∂b
            }

            return j;
        };

        var r = solver.Solve(ExpResiduals(xs, ys), new[] { 1.0, 1.0 }, jac);
        _output.WriteLine($"an: p=[{r.Parameters[0]:R},{r.Parameters[1]:R}] cost={r.FinalCost:R} it={r.Iterations} conv={r.Converged} reason={r.TerminationReason}");

        r.Converged.Should().BeTrue();
        // With the complete MINPACK FTOL ratio gate, this point satisfies the
        // independent analytic gradient criterion instead of the old two-clause FTOL.
        AnalyticGradientInfinityNorm(xs, ys, r.Parameters).Should().BeLessThanOrEqualTo(1e-12);
        r.TerminationReason.Should().Be(LmTerminationReason.GradientToleranceReached);
        r.Parameters[0].Should().BeApproximately(2.50013756101917086, 1e-10);
        r.Parameters[1].Should().BeApproximately(0.69995347971710163, 1e-10);
        r.FinalCost.Should().BeApproximately(4.54661308338381792e-6, 1e-16);
    }
}
