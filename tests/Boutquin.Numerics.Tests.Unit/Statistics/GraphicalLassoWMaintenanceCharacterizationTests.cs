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

/// <summary>Independent KKT and closed-form BIC checks under the all-entry penalty.</summary>
public sealed class GraphicalLassoWMaintenanceCharacterizationTests
{
    [Fact]
    public void Solve_FixedLambda_PinnedResult()
    {
        // W=[[2.5,.5],[.5,3.5]] from KKT; inverse via the 2x2 cofactor identity.
        var result = GraphicalLassoSolver<double>.Solve(new double[,] { { 2, 1 }, { 1, 3 } }, .5);
        result.Converged.Should().BeTrue();
        result.Precision[0, 0].Should().BeApproximately(7.0 / 17, 1e-12);
        result.Precision[0, 1].Should().BeApproximately(-1.0 / 17, 1e-12);
        result.Precision[1, 1].Should().BeApproximately(5.0 / 17, 1e-12);
        result.Covariance[0, 0].Should().BeApproximately(2.5, 1e-12);
    }

    [Fact]
    public void SolveWithBic_Grid_PinnedResult()
    {
        var grid = new[] { .01, .05, .1, .2, .4, 1.0 };
        // Analytic 2x2 solution at each lambda, no numerical optimization oracle.
        var scores = grid.Select(lambda =>
        {
            var a = 2 + lambda;
            var d = 3 + lambda;
            var b = Math.Max(1 - lambda, 0);
            var determinant = a * d - b * b;
            var trace = (2 * d + 3 * a - 2 * b) / determinant;
            return 100 * (Math.Log(determinant) + trace) + (b == 0 ? 0 : 2 * Math.Log(100));
        }).ToArray();
        var expected = grid[Array.IndexOf(scores, scores.Min())];
        var result = GraphicalLassoSolver<double>.SolveWithBic(new double[,] { { 2, 1 }, { 1, 3 } }, 100, grid);
        result.Lambda.Should().Be(expected);
        result.Converged.Should().BeTrue();
    }
}
