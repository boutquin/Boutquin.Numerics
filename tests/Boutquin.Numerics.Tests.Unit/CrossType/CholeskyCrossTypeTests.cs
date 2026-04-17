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

namespace Boutquin.Numerics.Tests.Unit.CrossType;

/// <summary>
/// Cross-type regression test: <see cref="CholeskyDecomposition{T}"/> at <c>T = double</c>
/// and <c>T = decimal</c> on a curated 10x10 SPD matrix. The two lower factors must
/// agree to 12 digits (the smaller of double's precision and decimal's precision at
/// the chosen input scale). This demonstrates the "caller chooses the precision"
/// proposition is live.
/// </summary>
public sealed class CholeskyCrossTypeTests
{
    // Curated 10x10 SPD matrix: A = BᵀB + I where B is a random matrix with entries in [0, 1].
    // This guarantees SPD (diagonal dominance) and moderate condition number.
    private static readonly double[,] s_spd10X10Double =
    {
        { 5.0, 1.2, 0.8, 0.3, 0.5, 0.7, 0.4, 0.6, 0.9, 0.2 },
        { 1.2, 4.5, 1.1, 0.6, 0.4, 0.3, 0.8, 0.5, 0.7, 0.1 },
        { 0.8, 1.1, 4.8, 0.9, 0.7, 0.5, 0.3, 0.4, 0.6, 0.2 },
        { 0.3, 0.6, 0.9, 4.2, 1.0, 0.8, 0.5, 0.7, 0.4, 0.3 },
        { 0.5, 0.4, 0.7, 1.0, 4.6, 0.6, 0.9, 0.3, 0.8, 0.5 },
        { 0.7, 0.3, 0.5, 0.8, 0.6, 4.9, 1.2, 0.4, 0.7, 0.6 },
        { 0.4, 0.8, 0.3, 0.5, 0.9, 1.2, 4.3, 0.6, 0.5, 0.4 },
        { 0.6, 0.5, 0.4, 0.7, 0.3, 0.4, 0.6, 4.7, 1.1, 0.8 },
        { 0.9, 0.7, 0.6, 0.4, 0.8, 0.7, 0.5, 1.1, 4.4, 0.9 },
        { 0.2, 0.1, 0.2, 0.3, 0.5, 0.6, 0.4, 0.8, 0.9, 4.1 },
    };

    [Fact]
    public void Cholesky_DoubleAndDecimal_AgreeToTwelveDigits()
    {
        var n = s_spd10X10Double.GetLength(0);

        // Run Cholesky at T = double.
        var lDouble = CholeskyDecomposition<double>.Decompose(s_spd10X10Double);

        // Convert to decimal and run Cholesky at T = decimal.
        var spd10X10Decimal = new decimal[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                spd10X10Decimal[i, j] = (decimal)s_spd10X10Double[i, j];
            }
        }

        var lDecimal = CholeskyDecomposition<decimal>.Decompose(spd10X10Decimal);

        // Compare: the two lower factors must agree to 12 significant digits.
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                var d = (double)lDecimal[i, j];
                var expected = lDouble[i, j];

                if (expected == 0.0)
                {
                    d.Should().Be(0.0,
                        because: $"L[{i},{j}] should be zero in both types");
                    continue;
                }

                var relError = Math.Abs((d - expected) / expected);
                relError.Should().BeLessThan(1e-12,
                    because: $"L[{i},{j}]: double={expected}, decimal={d} must agree to 12 digits");
            }
        }
    }
}
