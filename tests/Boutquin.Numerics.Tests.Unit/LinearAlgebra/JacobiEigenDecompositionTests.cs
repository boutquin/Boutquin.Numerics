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
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

public sealed class JacobiEigenDecompositionTests
{
    [Fact]
    public void Decompose_IdentityMatrix_ReturnsOnesAndIdentity()
    {
        var identity = new decimal[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        var result = JacobiEigenDecomposition.Decompose(identity);

        result.Values.Should().HaveCount(3);
        foreach (var v in result.Values)
        {
            ((double)v).Should().BeApproximately(1.0, 1e-10);
        }
    }

    [Fact]
    public void Decompose_3x3SymmetricMatrix_ReturnsCorrectEigenvalues()
    {
        // Known eigenvalues: 3, 1, -1 for this matrix.
        var matrix = new decimal[,]
        {
            { 1, 1, 1 },
            { 1, 1, 1 },
            { 1, 1, 1 },
        };

        var result = JacobiEigenDecomposition.Decompose(matrix);

        // Eigenvalues of rank-1 matrix [1,1,1]^T * [1,1,1] = 3, 0, 0
        result.Values.Should().HaveCount(3);
        ((double)result.Values[0]).Should().BeApproximately(3.0, 1e-8);
        ((double)result.Values[1]).Should().BeApproximately(0.0, 1e-8);
        ((double)result.Values[2]).Should().BeApproximately(0.0, 1e-8);
    }

    [Fact]
    public void Decompose_DiagonalMatrix_ReturnsDiagonalValues()
    {
        var matrix = new decimal[,] { { 5, 0, 0 }, { 0, 3, 0 }, { 0, 0, 1 } };
        var result = JacobiEigenDecomposition.Decompose(matrix);

        // Should return eigenvalues in descending order.
        ((double)result.Values[0]).Should().BeApproximately(5.0, 1e-10);
        ((double)result.Values[1]).Should().BeApproximately(3.0, 1e-10);
        ((double)result.Values[2]).Should().BeApproximately(1.0, 1e-10);
    }

    [Fact]
    public void Decompose_ReconstructsOriginalMatrix()
    {
        var matrix = new decimal[,]
        {
            { 4, 2, 1 },
            { 2, 5, 3 },
            { 1, 3, 6 },
        };

        var result = JacobiEigenDecomposition.Decompose(matrix);
        var n = 3;

        // Reconstruct: A = V * diag(λ) * V^T
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var sum = 0.0;
                for (var k = 0; k < n; k++)
                {
                    sum += (double)result.Vectors[i, k] * (double)result.Values[k] * (double)result.Vectors[j, k];
                }

                sum.Should().BeApproximately((double)matrix[i, j], 1e-8,
                    $"Reconstruction failed at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void Decompose_NonSquareMatrix_Throws()
    {
        var matrix = new decimal[,] { { 1, 2, 3 }, { 4, 5, 6 } };
        var act = () => JacobiEigenDecomposition.Decompose(matrix);
        act.Should().Throw<ArgumentException>();
    }
}
