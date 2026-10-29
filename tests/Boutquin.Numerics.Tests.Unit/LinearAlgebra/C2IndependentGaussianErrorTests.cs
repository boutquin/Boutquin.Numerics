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
using System.Reflection;

using Boutquin.Numerics.LinearAlgebra;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

public sealed class C2IndependentGaussianErrorTests
{
    [Fact]
    public void ControlledElimination_SatisfiesFactorDependentComponentwiseBound()
    {
        double[,] input = { { 1, 1 }, { .5, 2 } };
        double[,] lower = { { 1, 0 }, { .5, 1 } };
        double[,] upper = { { 1, 1 }, { 0, 1.5 } };
        double[,] augmented = { { 1, 1, 1 }, { .5, 2, 1 } };
        var solution = Eliminate(augmented, 0);
        for (var i = 0; i < 2; i++)
        {
            for (var j = i; j < 2; j++) { augmented[i, j].Should().Be(upper[i, j]); }
        }
        // The exact dyadic LU has no factorization error. Check a stronger
        // finite-fixture version of GVL's leading n*u*(2|A|+4|L||U|) bound,
        // without using its unspecified O(u²) as an adjustable tolerance.
        var scale = BigInteger.One << 1074;
        for (var i = 0; i < 2; i++)
        {
            var residual = -scale * scale;
            var bound = BigInteger.Zero;
            for (var j = 0; j < 2; j++)
            {
                residual += Exact(input[i, j]) * Exact(solution[j]);
                var product = BigInteger.Zero;
                for (var k = 0; k < 2; k++) { product += BigInteger.Abs(Exact(lower[i, k]) * Exact(upper[k, j])); }
                bound += (2 * BigInteger.Abs(Exact(input[i, j])) * scale + 4 * product) * BigInteger.Abs(Exact(solution[j]));
            }
            (BigInteger.Abs(residual) * scale * (BigInteger.One << 53) <= 2 * bound).Should().BeTrue();
        }
    }

    [Fact]
    public void ControlledRowSwap_PreservesAugmentedRightHandSideAndTriangularSolve()
    {
        double[,] augmented = { { 1, 2, 5 }, { -2, 2, 2 } };
        var solution = Eliminate(augmented, 1);
        solution.Should().Equal(1, 2);
        augmented[0, 0].Should().Be(-2);
        augmented[0, 1].Should().Be(2);
        augmented[0, 2].Should().Be(2);
        augmented[1, 1].Should().Be(3);
        augmented[1, 2].Should().Be(6);
    }

    private static double[] Eliminate(double[,] augmented, int draw)
    {
        var method = typeof(GaussianElimination<double>).GetMethod("EliminateAndBackSubstitute", BindingFlags.Static | BindingFlags.NonPublic);
        method.Should().NotBeNull();
        object? result = null;
        FluentActions.Invoking(() => result = method!.Invoke(null, [augmented, 2, new FixedIndexRandom(draw), 0d])).Should().NotThrow();
        return (double[])result!;
    }

    private sealed class FixedIndexRandom(int index) : System.Random
    {
        public override int Next(int maxValue) => index;
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
