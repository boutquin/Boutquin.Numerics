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

public sealed class C2IndependentPivotedCholeskyTests
{
    [Fact]
    public void TwoNontrivialPivots_PreserveBothMatrixAxesAndCompletedFactorRows()
    {
        double[,] a = { { 4, 2, 1 }, { 2, 5, 3 }, { 1, 3, 6 } };
        PivotedCholeskyResult<double>? result = null;
        FluentActions.Invoking(() => result = CholeskyDecomposition<double>.DecomposePivoted(a)).Should().NotThrow();
        // Exact leading Schur diagonals after pivot6 are23/6 and7/2,
        // so the next pivot is original index0. det(A)=67 gives final67/23.
        result!.Permutation.Should().Equal(2, 0, 1);
        result.Rank.Should().Be(3);
        var lower = result.Lower;
        lower[0, 0].Should().BeApproximately(Math.Sqrt(6), 1e-14);
        lower[1, 1].Should().BeApproximately(Math.Sqrt(23d / 6), 1e-14);
        lower[2, 2].Should().BeApproximately(Math.Sqrt(67d / 23), 1e-14);
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                var reconstructed = 0d;
                for (var k = 0; k < 3; k++) { reconstructed += lower[i, k] * lower[j, k]; }
                reconstructed.Should().BeApproximately(a[result.Permutation[i], result.Permutation[j]], 3e-14);
            }
        }
    }
}
