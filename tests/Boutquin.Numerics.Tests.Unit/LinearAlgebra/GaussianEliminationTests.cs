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

public sealed class GaussianEliminationTests
{
    [Fact]
    public void Solve_3x3System_ReturnsCorrectSolution()
    {
        // 2x + y - z = 8
        // -3x - y + 2z = -11
        // -2x + y + 2z = -3
        // Solution: x=2, y=3, z=-1
        var a = new decimal[,]
        {
            { 2, 1, -1 },
            { -3, -1, 2 },
            { -2, 1, 2 },
        };
        var b = new decimal[] { 8, -11, -3 };

        var x = GaussianElimination.Solve(a, b, seed: 42);

        ((double)x[0]).Should().BeApproximately(2.0, 1e-10);
        ((double)x[1]).Should().BeApproximately(3.0, 1e-10);
        ((double)x[2]).Should().BeApproximately(-1.0, 1e-10);
    }

    [Fact]
    public void Solve_SingularMatrix_Throws()
    {
        var a = new decimal[,] { { 1, 2 }, { 2, 4 } };
        var b = new decimal[] { 3, 6 };

        var act = () => GaussianElimination.Solve(a, b, seed: 42);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Solve_DeterministicWithSeed_ProducesSameResult()
    {
        var a = new decimal[,]
        {
            { 4, 3, 2 },
            { 3, 5, 1 },
            { 2, 1, 6 },
        };
        var b = new decimal[] { 1, 2, 3 };

        var x1 = GaussianElimination.Solve(a, b, seed: 123);
        var x2 = GaussianElimination.Solve(a, b, seed: 123);

        for (var i = 0; i < x1.Length; i++)
        {
            x1[i].Should().Be(x2[i]);
        }
    }

    [Fact]
    public void Solve_IdentityMatrix_ReturnsRhs()
    {
        var a = new decimal[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        var b = new decimal[] { 5, 10, 15 };

        var x = GaussianElimination.Solve(a, b, seed: 42);

        ((double)x[0]).Should().BeApproximately(5.0, 1e-10);
        ((double)x[1]).Should().BeApproximately(10.0, 1e-10);
        ((double)x[2]).Should().BeApproximately(15.0, 1e-10);
    }

    [Fact]
    public void Solve_DimensionMismatch_Throws()
    {
        var a = new decimal[,] { { 1, 2 }, { 3, 4 } };
        var b = new decimal[] { 1, 2, 3 };

        var act = () => GaussianElimination.Solve(a, b);
        act.Should().Throw<ArgumentException>();
    }
}
