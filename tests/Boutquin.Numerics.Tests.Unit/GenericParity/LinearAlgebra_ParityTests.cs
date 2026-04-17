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

namespace Boutquin.Numerics.Tests.Unit.GenericParity;

/// <summary>
/// Parity tests verifying that the generic LinearAlgebra types instantiated at
/// <c>T = decimal</c> produce identical results to the pre-migration legacy
/// concrete-typed implementations.
/// </summary>
public sealed class LinearAlgebra_ParityTests
{
    private static readonly decimal[,] s_spd3X3 = {
        { 4m, 2m, 1m },
        { 2m, 5m, 3m },
        { 1m, 3m, 6m },
    };

    [Fact]
    public void CholeskyDecomposition_GenericMatchesLegacy_AtDecimal()
    {
        var legacyL = CholeskyDecomposition.Decompose(s_spd3X3);
        var genericL = CholeskyDecomposition<decimal>.Decompose(s_spd3X3);

        var n = legacyL.GetLength(0);
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                genericL[i, j].Should().Be(legacyL[i, j],
                    because: $"generic Cholesky L[{i},{j}] must match legacy");
            }
        }
    }

    [Fact]
    public void GaussianElimination_GenericMatchesLegacy_AtDecimal()
    {
        var a = new decimal[,] { { 2m, 1m }, { 1m, 3m } };
        var b = new decimal[] { 5m, 7m };

        var legacyX = GaussianElimination.Solve(a, b);
        var genericX = GaussianElimination<decimal>.Solve(a, b);

        for (var i = 0; i < legacyX.Length; i++)
        {
            genericX[i].Should().Be(legacyX[i],
                because: $"generic GaussElim x[{i}] must match legacy");
        }
    }

    [Fact]
    public void NearestPsdProjection_IsPsd_GenericMatchesLegacy_AtDecimal()
    {
        NearestPsdProjection.IsPsd(s_spd3X3).Should().Be(
            NearestPsdProjection<decimal>.IsPsd(s_spd3X3));
    }
}
