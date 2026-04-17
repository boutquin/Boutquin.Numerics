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

namespace Boutquin.Numerics.Tests.Verification;

public sealed class LinearAlgebraVerificationTests : CrossLanguageVerificationBase
{
    [Fact]
    public void Cholesky_ReconstructsNumpyInputToPrecisionNumeric()
    {
        using var doc = LoadVector("linalg");
        var spd = GetDecimal2D(doc.RootElement.GetProperty("spd_matrix"));
        var lower = CholeskyDecomposition.Decompose(spd);

        // Reconstruct L · Lᵀ and compare to the original SPD matrix.
        var n = spd.GetLength(0);
        var reconstructed = new decimal[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var sum = 0m;
                for (var k = 0; k < n; k++)
                {
                    sum += lower[i, k] * lower[j, k];
                }

                reconstructed[i, j] = sum;
            }
        }

        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                AssertScalarWithin(reconstructed[i, j], spd[i, j], (decimal)PrecisionNumeric, $"L·Lᵀ[{i},{j}]");
            }
        }
    }

    [Fact]
    public void JacobiEigenvalues_MatchNumpyLapackToStatisticalPrecision()
    {
        using var doc = LoadVector("linalg");
        var spd = GetDecimal2D(doc.RootElement.GetProperty("spd_matrix"));
        var expectedEigvals = GetDoubleArray(doc.RootElement.GetProperty("eigenvalues_ascending"));

        var result = JacobiEigenDecomposition.Decompose(spd);
        var actualSorted = result.Values.Select(v => (double)v).OrderBy(v => v).ToArray();

        // Jacobi vs LAPACK can diverge at ~1e-5 on poorly conditioned matrices.
        for (var i = 0; i < actualSorted.Length; i++)
        {
            AssertScalarWithin(actualSorted[i], expectedEigvals[i], PrecisionStatistical, $"eigvals[{i}]");
        }
    }

    [Fact]
    public void GaussianElimination_SolvesLinearSystemToPrecisionNumeric()
    {
        using var doc = LoadVector("linalg");
        var spd = GetDecimal2D(doc.RootElement.GetProperty("spd_matrix"));
        var b = GetDecimalArray(doc.RootElement.GetProperty("linear_system").GetProperty("b"));
        var expected = GetDoubleArray(doc.RootElement.GetProperty("linear_system").GetProperty("x"));

        var x = GaussianElimination.Solve(spd, b);

        for (var i = 0; i < x.Length; i++)
        {
            AssertScalarWithin((double)x[i], expected[i], PrecisionNumeric, $"x[{i}]");
        }
    }
}
