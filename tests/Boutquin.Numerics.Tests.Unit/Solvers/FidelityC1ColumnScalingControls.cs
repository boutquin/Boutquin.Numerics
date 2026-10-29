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
public sealed class FidelityC1ColumnScalingControls
{
    [Theory]
    [InlineData(0.0625)]
    [InlineData(16.0)]
    public void PositiveDiagonalColumnScalingTransformsExactOrthogonalSolution(double scale)
    {
        // X=diag(2,3) with two zero rows; y=(6,-6,5,-7).
        // Multiplying columns by D=diag(s,1/s) transforms beta=(3,-2)
        // into D^-1 beta=(3/s,-2s), preserving residual(0,0,5,-7), RSS74.
        // Every entry and the exact solution are representable binary rationals.
        // This verifies equivariance, not the paper's optimal-scaling rate bound.
        double[,] x = { { 2 * scale, 0 }, { 0, 3 / scale }, { 0, 0 }, { 0, 0 } };
        var result = OrdinaryLeastSquares<double>.Fit(x, [6, -6, 5, -7], includeIntercept: false);
        Assert.Equal(new double[] { 3 / scale, -2 * scale }, result.Coefficients);
        Assert.Equal(new double[] { 0, 0, 5, -7 }, result.Residuals);
        Assert.Equal(74, result.ResidualSumOfSquares);
    }
}
