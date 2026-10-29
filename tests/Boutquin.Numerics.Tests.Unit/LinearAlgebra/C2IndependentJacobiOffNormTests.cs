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

public sealed class C2IndependentJacobiOffNormTests
{
    [Fact]
    public void BothTriangles_RequireARealRotationRegardlessOfTiePermutation()
    {
        // Full off-norm sqrt(2)*8e-16 exceeds1e-15; upper-only8e-16 does not.
        // Equal diagonals have eigenvectors at45degrees. Checking every
        // component rejects either permutation of a falsely returned identity.
        var result = JacobiEigenDecomposition<double>.Decompose(new[,] { { 1d, 8e-16 }, { 8e-16, 1d } });
        var magnitude = 1 / Math.Sqrt(2);
        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++) { Math.Abs(result.Vectors[i, j]).Should().BeApproximately(magnitude, 2e-15); }
        }
    }
}
