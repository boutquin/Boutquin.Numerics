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
using Boutquin.Numerics.Solvers.Internal;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

public sealed class FidelityC1JacobianTensorControls
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CentralColumnsDifferentiateEveryResidualCoordinate(bool writer)
    {
        const double h = 0.000006055454452393339060789892727936966935;
        static double[] Residual(double[] x) => [x[0] + x[0] * x[0] * x[0] + 2 * x[1], 3 * x[0] - x[1] + x[1] * x[1] * x[1], 2 * x[0] + x[1]];
        static void Write(ReadOnlySpan<double> x, Span<double> r)
        {
            r[0] = x[0] + x[0] * x[0] * x[0] + 2 * x[1];
            r[1] = 3 * x[0] - x[1] + x[1] * x[1] * x[1];
            r[2] = 2 * x[0] + x[1];
        }
        var pool = new LevenbergMarquardtBuffers<double>(2, 3);
        if (writer) { FiniteDifferenceJacobian<double>.EvaluateInto(Write, [0, 0], pool); }
        else { FiniteDifferenceJacobian<double>.EvaluateInto(Residual, [0, 0], pool); }
        // Exact central polynomial quotients: the only truncation terms are +h²
        // for x0³ in row0 and x1³ in row1. All cross derivatives are exact integers.
        double[,] expected = { { 1 + h * h, 2 }, { 3, -1 + h * h }, { 2, 1 } };
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 2; j++) { Assert.InRange(Math.Abs(pool.Jacobian[i, j] - expected[i, j]), 0, 2e-15); }
        }
    }
}
