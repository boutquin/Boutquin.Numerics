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

[Trait("Tier", "Unit")]
public sealed class C4IndependentBoundaryControls
{
    [Fact]
    public void Forecast_ZeroOmegaIsOutsidePrimaryDomain()
    {
        var fit = new GarchFitResult<double>(0, 0, 0, .25, .5, [4], [2], 0, true);
        var act = () => fit.ForecastVariance(1);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Forecast_ZeroAlphaIsLegal()
    {
        // eps^2=16; omega + 0*16 + .5*4 = 4.
        var fit = new GarchFitResult<double>(0, 0, 2, 0, .5, [4], [2], 0, true);
        var act = () => fit.ForecastVariance(1);
        act.Should().NotThrow().Which.Should().Be(4);
    }

    [Fact]
    public void Forecast_ZeroBetaIsLegal()
    {
        // eps^2=16; omega + .5*16 + 0*4 = 10.
        var fit = new GarchFitResult<double>(0, 0, 2, .5, 0, [4], [2], 0, true);
        var act = () => fit.ForecastVariance(1);
        act.Should().NotThrow().Which.Should().Be(10);
    }

    [Fact]
    public void Glasso_EqualityDoesNotMeetStrictStop()
    {
        // Both offdiagonals move by .75; mean deltaW=.375 equals .5*.75.
        var result = GraphicalLassoSolver<double>.Solve(
            new double[,] { { 1, .75 }, { .75, 1.5625 } }, .75,
            new GraphicalLassoOptions<double>(MaxIterations: 1, Tolerance: .5));
        result.Converged.Should().BeFalse();
        result.Iterations.Should().Be(1);
    }
}
