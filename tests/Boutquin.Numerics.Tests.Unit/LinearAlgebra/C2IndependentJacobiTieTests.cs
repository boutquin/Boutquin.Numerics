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

using System.Numerics;

using Boutquin.Numerics.LinearAlgebra;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

public sealed class C2IndependentJacobiTieTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RepeatedEigenvalues_SelectLastEqualMaximumAndSwapMatchingColumns(bool allEqual)
    {
        Check<double>(allEqual);
        Check<decimal>(allEqual);
    }

    private static void Check<T>(bool allEqual) where T : IFloatingPoint<T>
    {
        // NR3 p.576 eigsrt scans with >=. On [2,2,1], the initial identity
        // columns become [1,0,2]. On [3,3,3], successive last-equal swaps
        // give [2,0,1]. No numerical eigensolver supplies the expected vectors.
        var repeated = T.CreateChecked(allEqual ? 3 : 2);
        var last = allEqual ? repeated : T.One;
        T[,] matrix = { { repeated, T.Zero, T.Zero }, { T.Zero, repeated, T.Zero }, { T.Zero, T.Zero, last } };
        int[] expectedColumns = allEqual ? [2, 0, 1] : [1, 0, 2];
        var result = JacobiEigenDecomposition<T>.Decompose(matrix);
        result.Values.Should().Equal(repeated, repeated, last);
        for (var col = 0; col < 3; col++)
        {
            for (var row = 0; row < 3; row++)
            {
                result.Vectors[row, col].Should().Be(row == expectedColumns[col] ? T.One : T.Zero);
            }
        }
    }
}
