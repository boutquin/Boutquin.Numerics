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

using Boutquin.Numerics.LinearAlgebra;
using Boutquin.Numerics.Statistics;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

public sealed class ArenaR3B05RegressionTests
{
    [Fact]
    public void Jacobi_DecimalMaximumDiagonal_ReturnsRepresentableEigenpairs()
    {
        EigenResult<decimal>? result = null;
        var error = Record.Exception(() => result = JacobiEigenDecomposition<decimal>.Decompose(new decimal[,] { { decimal.MaxValue, 0m }, { 0m, decimal.MaxValue } }));
        Assert.Null(error);
        Assert.Equal(new[] { decimal.MaxValue, decimal.MaxValue }, result!.Values);
    }

    [Fact]
    public void Jacobi_HalfUnequalDiagonals_PreservesNegativeEigenvalue()
    {
        var result = JacobiEigenDecomposition<Half>.Decompose(new Half[,] { { (Half)0, (Half)0.1 }, { (Half)0.1, (Half)100 } });
        Assert.True(result.Values[1] < (Half)0, "The representable negative eigenvalue must not be rounded to zero.");
    }

    [Fact]
    public void StandardizedPca_ProjectionHasFittedComponentVariance()
    {
        decimal[,] data = { { 1m, 10m }, { 2m, 40m }, { 3m, 20m }, { 4m, 80m } };
        var result = PrincipalComponentAnalysis.FromReturns(new ReturnsMatrix(data), standardize: true);
        var scores = result.Project(data, 2);
        for (var k = 0; k < 2; k++)
        {
            var variance = 0m;
            for (var i = 0; i < 4; i++)
            {
                variance += scores[i, k] * scores[i, k] / 3m;
            }

            Assert.True(Math.Abs(variance - result.Eigenvalues[k]) < 0.00000001m, $"Score variance {variance} differs from fitted eigenvalue {result.Eigenvalues[k]}.");
        }
    }
}
