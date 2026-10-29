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
public sealed class FidelityC1QpGradientControls
{
    [Fact]
    public void MinVarianceGradientKeepsCorrectlyActiveLowerBound() => CheckCertificate(false);

    [Fact]
    public void MeanVarianceGradientKeepsCorrectlyActiveLowerBound() => CheckCertificate(true);

    private static void CheckCertificate(bool meanVariance)
    {
        // Leading principal minors39,1613,3317 establish SPD.
        // At w=(37/71,34/71,0), Cw=(1613/71,1613/71,1994/71).
        // The free gradients agree and the lower-bound gradient is larger:
        // feasibility plus convex KKT conditions certify the unique optimum.
        // Mean-variance with zero means and risk1 reverses every gradient sign.
        double[,] c = { { 39, 5, 30 }, { 5, 42, 26 }, { 30, 26, 37 } };
        double[]? weights = null;
        var error = Record.Exception(() => weights = meanVariance
            ? ActiveSetQpSolver<double>.SolveMeanVariance(c, [0, 0, 0], 1, 0, 0.6)
            : ActiveSetQpSolver<double>.SolveMinVariance(c, 0, 0.6));
        Assert.Null(error);
        Assert.NotNull(weights);
        Assert.InRange(Math.Abs(weights[0] - 37.0 / 71), 0, 2e-13);
        Assert.InRange(Math.Abs(weights[1] - 34.0 / 71), 0, 2e-13);
        Assert.Equal(0, weights[2]);
    }
}
