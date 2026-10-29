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

namespace Boutquin.Numerics.Tests.Unit.Solvers;

[Trait("Tier", "Unit")]
public sealed class FidelityC1RootSignsAndOlsInferenceControls
{
    [Theory]
    [InlineData(1)]
    [InlineData(-1)]
    public void BrentRejectsBothStrictSameSignOrientations(double sign)
    {
        var error = Record.Exception(() => new BrentSolver<double>(1e-12, 1).Solve(x => sign * (x * x + 1), 1, 3));
        Assert.IsType<InvalidOperationException>(error);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(-1)]
    public void NewtonRejectsBothStrictSameSignOrientations(double sign)
    {
        var error = Record.Exception(() => new NewtonRaphsonSolver<double>(1e-12, 1e-12, 1e-12, 1, x => sign * 2 * x)
            .Solve(x => sign * (x * x + 1), 1, 3));
        Assert.IsType<InvalidOperationException>(error);
    }

    [Fact]
    public void OrthogonalInconsistentFitHasIndependentInferenceScale()
    {
        double[,] design = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
        // y=X*(2,2)+(1,1,2,2), X^T*r=0, X^T*X=2I.
        var result = OrdinaryLeastSquares<double>.Fit(design, new double[] { 3, -1, 4, 0 }, includeIntercept: false);
        Assert.Equal(2, result.DegreesOfFreedom);
        Assert.InRange(Math.Abs(result.ResidualSumOfSquares - 10), 0, 1e-12);
        for (var i = 0; i < 2; i++)
        {
            Assert.InRange(Math.Abs(result.Coefficients[i] - 2), 0, 1e-13);
            Assert.InRange(Math.Abs(result.StandardErrors[i] - Math.Sqrt(2.5)), 0, 1e-13);
        }
    }
    [Theory]
    [InlineData(0)]
    [InlineData(-80)]
    public void AcceptedInverseQuadraticInterpolation(int scaleExponent) =>
        new Arena20260928RootClauseTests().Brent_InverseQuadraticStepUsesIndependentLagrangeOracle(false, scaleExponent);
}
