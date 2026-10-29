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

public sealed class C2IndependentCholeskyBoundsTests
{
    [Fact]
    public void FactorResiduals_SatisfyExactPerEntryGammaBounds()
    {
        double[,] a = { { 6, 2, 1, -1 }, { 2, 5, -1, .5 }, { 1, -1, 4, .25 }, { -1, .5, .25, 3 } };
        var lower = CholeskyDecomposition<double>.Decompose(a);
        var scale = BigInteger.One << 1074;
        var inverseUnitRoundoff = BigInteger.One << 53;
        for (var row = 0; row < 4; row++)
        {
            for (var col = 0; col <= row; col++)
            {
                var product = BigInteger.Zero;
                var absoluteProduct = BigInteger.Zero;
                for (var k = 0; k <= col; k++)
                {
                    var term = Exact(lower[row, k]) * Exact(lower[col, k]);
                    product += term;
                    absoluteProduct += BigInteger.Abs(term);
                }
                var error = BigInteger.Abs(Exact(a[row, col]) * scale - product);
                // Higham eq10.4 has gamma_i for off-diagonal upper entry(i,j).
                // Its adjacent square-root analysis uses gamma_(j+1) on diagonal.
                var gammaIndex = col + (row == col ? 2 : 1);
                (error * (inverseUnitRoundoff - gammaIndex) <= gammaIndex * absoluteProduct)
                    .Should().BeTrue($"exact rational gamma bound at lower entry({row},{col})");
            }
        }
    }

    [Fact]
    public void Solve_SatisfiesExactNormwiseBackwardBoundOnKnownSpectrum()
    {
        // A has eigenvalues1,3, hence norm2(A)=3. The minimal unstructured
        // perturbation norm is ||b−Ax||2/||x||2. For n2 Higham's bound is168u.
        double[,] a = { { 2, 1 }, { 1, 2 } };
        var x = CholeskyDecomposition<double>.Solve(CholeskyDecomposition<double>.Decompose(a), [1, 0]);
        var scale = BigInteger.One << 1074;
        var x0 = Exact(x[0]);
        var x1 = Exact(x[1]);
        var r0 = scale - 2 * x0 - x1;
        var r1 = -x0 - 2 * x1;
        var residualSquared = r0 * r0 + r1 * r1;
        var solutionSquared = x0 * x0 + x1 * x1;
        (residualSquared * (BigInteger.One << 106) <= 168 * 168 * solutionSquared).Should().BeTrue();
    }

    [Fact]
    public void PivotedFactor_SatisfiesCompleteDiagonalPivotInequality()
    {
        // An independently specified Gram matrix; all stages are retained.
        double[,] a = { { 4, 2, 2 }, { 2, 10, 4 }, { 2, 4, 6 } };
        var result = CholeskyDecomposition<double>.DecomposePivoted(a, tolerance: 0, maxRank: null);
        result.Rank.Should().Be(3);
        var lower = result.Lower;
        for (var k = 0; k < 3; k++)
        {
            for (var j = k + 1; j < 3; j++)
            {
                var sum = 0d;
                for (var i = k; i <= j; i++) { sum += lower[j, i] * lower[j, i]; }
                sum.Should().BeLessThanOrEqualTo(lower[k, k] * lower[k, k] + 2e-14);
            }
        }
    }

    private static BigInteger Exact(double value)
    {
        var bits = BitConverter.DoubleToInt64Bits(value);
        var magnitude = bits & long.MaxValue;
        var exponent = (int)(magnitude >> 52);
        var significand = magnitude & ((1L << 52) - 1);
        if (exponent != 0) { significand |= 1L << 52; }
        var integer = new BigInteger(significand) << Math.Max(0, exponent - 1);
        return bits < 0 ? -integer : integer;
    }
}
