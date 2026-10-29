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
public sealed class FidelityC1QpCertificateTests
{
    private static double[,] Covariance() => new double[,]
    {
        { 67, -12, 21 }, { -12, 21, 12 }, { 21, 12, 22 },
    };

    [Fact]
    public void MinVariance_ReleasesWronglyFixedLowerBoundToCertifiedOptimum()
    {
        double[]? weights = null;
        var error = Record.Exception(() => weights = ActiveSetQpSolver<double>.SolveMinVariance(Covariance(), 0, 0.6));
        Assert.Null(error);
        Assert.NotNull(weights);
        CheckCertificate(weights);
    }

    [Fact]
    public void MeanVariance_ReleasesWronglyFixedLowerBoundToCertifiedOptimum()
    {
        double[]? weights = null;
        var error = Record.Exception(() => weights = ActiveSetQpSolver<double>.SolveMeanVariance(Covariance(), [0, 0, 0], 1, 0, 0.6));
        Assert.Null(error);
        Assert.NotNull(weights);
        CheckCertificate(weights);
    }

    private static void CheckCertificate(double[] weights)
    {
        // Sylvester principal minors67,1263,2829 prove positive definiteness.
        // Exact rational enumeration of the27 box faces gives w=(74/235,3/5,4/47).
        // Cw=(3686/235,2313/235,3686/235): free gradients equal nu, upper
        // gradient<nu. Feasibility and these KKT signs certify the unique optimum.
        // The unconstrained w=(249/364,493/364,-27/26) first fixes coordinate2
        // at zero. After fixing coordinate1 at3/5, w=(2/5,3/5,0) must release2:
        // its gradient78/5 is below free gradient98/5, giving feasible descent.
        double[] expected = [74.0 / 235, 3.0 / 5, 4.0 / 47];
        for (var i = 0; i < 3; i++) { Assert.InRange(Math.Abs(weights[i] - expected[i]), 0, 2e-13); }
        Assert.InRange(Math.Abs(weights.Sum() - 1), 0, 1e-14);
        var c = Covariance();
        var gradient = new double[3];
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++) { gradient[i] += c[i, j] * weights[j]; }
        }
        Assert.InRange(Math.Abs(gradient[0] - 3686.0 / 235), 0, 1e-11);
        Assert.InRange(Math.Abs(gradient[2] - gradient[0]), 0, 1e-11);
        Assert.True(gradient[1] < gradient[0]);
    }
}
