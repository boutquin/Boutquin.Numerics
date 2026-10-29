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

namespace Boutquin.Numerics.Tests.Verification;

public sealed class PsdVerificationTests : CrossLanguageVerificationBase
{
    [Fact]
    public void Higham_MatchesReferenceNearestCorrelationMatrix()
    {
        using var doc = LoadVector("psd");
        var input = GetDecimal2D(doc.RootElement.GetProperty("input"));
        var expected = GetDouble2D(doc.RootElement.GetProperty("nearest_correlation"));

        Assert.True(input.GetLength(0) > 0, "PSD reference input must not be empty.");
        Assert.Equal(input.GetLength(0), input.GetLength(1));
        Assert.Equal(input.GetLength(0), expected.GetLength(0));
        Assert.Equal(input.GetLength(1), expected.GetLength(1));

        var nearest = NearestPsdProjection.Higham(input);

        // Alternating-projection iteration convergence depends on tolerance
        // and operation order; statistical tier is appropriate.
        AssertMatrixWithin(nearest, expected, (decimal)PrecisionStatistical, "higham");
    }

    [Fact]
    public void EigenClip_MatchesReferenceNearestPsdMatrix()
    {
        using var doc = LoadVector("psd");
        var input = GetDecimal2D(doc.RootElement.GetProperty("input"));
        var expected = GetDouble2D(doc.RootElement.GetProperty("eigen_clip"));

        Assert.True(input.GetLength(0) > 0, "PSD reference input must not be empty.");
        Assert.Equal(input.GetLength(0), input.GetLength(1));
        Assert.Equal(input.GetLength(0), expected.GetLength(0));
        Assert.Equal(input.GetLength(1), expected.GetLength(1));

        var projected = NearestPsdProjection.EigenClip(input);

        // NumPy eigh + eigenvalue clipping supplies an independent matrix oracle.
        // A PSD-only check would also accept the zero matrix for this indefinite input.
        AssertMatrixWithin(projected, expected, (decimal)PrecisionNumeric, "eigen clip");
        NearestPsdProjection.IsPsd(projected).Should().BeTrue();
    }
}
