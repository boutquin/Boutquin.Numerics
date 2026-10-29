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

public sealed class C2IndependentEigenpairTieConsumersTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Pca_TruncationThroughEqualBlock_UsesAuthorSelectedColumn(bool allEqual)
    {
        var repeated = allEqual ? 3m : 2m;
        var last = allEqual ? 3m : 1m;
        decimal[,] covariance = { { repeated, 0, 0 }, { 0, repeated, 0 }, { 0, 0, last } };
        var pca = PrincipalComponentAnalysis<decimal>.Decompose(covariance);
        decimal[,] input = { { 7, 11, 13 }, { -2, 3, 5 } };
        var scores = pca.Project(input, 1);
        // Last equal maximum selects e3 for a triple tie, e2 for a leading
        // double tie. Truncating at one component exposes that actual choice.
        scores[0, 0].Should().Be(allEqual ? 13m : 11m);
        scores[1, 0].Should().Be(allEqual ? 5m : 3m);
        pca.Eigenvalues.Should().Equal(repeated, repeated, last);
        var firstRatio = allEqual ? 1m / 3m : 2m / 5m;
        pca.ExplainedVarianceRatio[0].Should().BeApproximately(firstRatio, 1e-26m);
    }

    [Fact]
    public void FullEqualWeightReconstruction_IsPermutationInvariant()
    {
        decimal[,] covariance = { { 2, 0, 0 }, { 0, 2, 0 }, { 0, 0, 1 } };
        var pca = PrincipalComponentAnalysis<decimal>.Decompose(covariance);
        var clipped = NearestPsdProjection<decimal>.EigenClip(covariance);
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                var reconstructed = 0m;
                for (var k = 0; k < 3; k++) { reconstructed += pca.Eigenvalues[k] * pca.Eigenvectors[i, k] * pca.Eigenvectors[j, k]; }
                reconstructed.Should().Be(covariance[i, j]);
                clipped[i, j].Should().BeApproximately(covariance[i, j], 1e-26m);
            }
        }
    }
}
