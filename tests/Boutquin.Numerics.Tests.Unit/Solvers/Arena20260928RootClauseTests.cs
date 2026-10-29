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

namespace Boutquin.Numerics.Tests.Unit.Solvers;

[Trait("Tier", "Unit")]
public sealed class Arena20260928RootClauseTests
{
    [Fact]
    public void Secant_UsesTwoMostRecentPointsAndGoldenRatioLocalLaw()
    {
        var visited = new List<double>();
        double Objective(double x) { visited.Add(x); return x * x - 2; }
        new SecantSolver<double>(1e-28, 7).Solve(Objective, 1, 2);
        visited[2].Should().BeApproximately(4.0 / 3, 5e-16);
        visited[3].Should().BeApproximately(10.0 / 7, 5e-16);

        // NR9.2: for a smooth simple root, e_next ~ f''/(2f')*e_i*e_i-1.
        // This gives p²=p+1. Choose the pre-roundoff window, not the final rounded step.
        var errors = visited.Select(x => Math.Abs(x - Math.Sqrt(2))).ToArray();
        var ratio = errors[6] / (errors[5] * errors[4]);
        ratio.Should().BeApproximately(1 / (2 * Math.Sqrt(2)), 0.001);
        var phi = (1 + Math.Sqrt(5)) / 2;
        var orders = Enumerable.Range(4, 3)
            .Select(i => Math.Log(errors[i] / errors[i - 1]) / Math.Log(errors[i - 1] / errors[i - 2]))
            .ToArray();
        orders[0].Should().BeGreaterThan(phi);
        orders[1].Should().BeLessThan(phi);
        orders[2].Should().BeGreaterThan(phi);
        Math.Abs(orders[1] - phi).Should().BeLessThan(Math.Abs(orders[0] - phi));
        Math.Abs(orders[2] - phi).Should().BeLessThan(Math.Abs(orders[1] - phi));
    }

    [Theory]
    [InlineData(false, 0)]
    [InlineData(true, 0)]
    [InlineData(false, -80)]
    [InlineData(true, -80)]
    public void Brent_InverseQuadraticStepUsesIndependentLagrangeOracle(bool reject, int scaleExponent)
    {
        // A finite piecewise-linear objective prescribes the history without porting
        // the solver's P/Q formula. Brent1973 p58 and NR9.3.1 govern interpolation.
        var scale = Math.ScaleB(1, scaleExponent);
        double[] nodes = [0, 0.4, 8.0 / 15, 2.0 / 3, 0.8, 1, 4];
        double[] values = [-1, -1.0 / 16, -1.0 / 32, reject ? -1.0 / 64 : -1.0 / 128, 1.0 / 8, 1.0 / 4, 3];
        var visited = new List<double>();
        double Objective(double x)
        {
            x /= scale;
            visited.Add(x);
            for (var i = 1; i < nodes.Length; i++)
            {
                if (x <= nodes[i])
                {
                    return values[i - 1] + (values[i] - values[i - 1]) * ((x - nodes[i - 1]) / (nodes[i] - nodes[i - 1]));
                }
            }
            return values[^1];
        }
        new BrentSolver<double>(1e-12 * scale, 6).Solve(Objective, 0, 4 * scale);
        double[] history = [0, 4, 1, 0.8, 0.4, 8.0 / 15, 2.0 / 3];
        for (var i = 0; i < history.Length; i++) { visited[i].Should().BeApproximately(history[i], 2e-14); }

        var a = 8.0 / 15;
        var b = 2.0 / 3;
        var c = 4.0 / 5;
        var fa = -1.0 / 32;
        var fb = reject ? -1.0 / 64 : -1.0 / 128;
        var fc = 1.0 / 8;
        var lagrange = a * fb * fc / ((fa - fb) * (fa - fc))
            + b * fa * fc / ((fb - fa) * (fb - fc))
            + c * fa * fb / ((fc - fa) * (fc - fb));
        // Accepted case gives2692/3825. Rejected case gives524/675 inside [b,c], but its step exceeds1.5*m.
        lagrange.Should().BeApproximately(reject ? 524.0 / 675 : 2692.0 / 3825, 5e-15);
        visited[7].Should().BeApproximately(reject ? 11.0 / 15 : lagrange, 2e-14);
    }
}
