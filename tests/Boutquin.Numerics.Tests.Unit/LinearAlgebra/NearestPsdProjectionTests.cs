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

public sealed class NearestPsdProjectionTests
{
    [Fact]
    public void EigenClip_PassesPsdInputThrough()
    {
        var psd = new decimal[,]
        {
            { 2m, 1m, 0m },
            { 1m, 2m, 1m },
            { 0m, 1m, 2m },
        };

        var projected = NearestPsdProjection.EigenClip(psd);
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                ((double)Math.Abs(projected[i, j] - psd[i, j])).Should().BeLessThan(1e-10);
            }
        }
    }

    [Fact]
    public void EigenClip_RemovesNegativeEigenvalues()
    {
        // Indefinite matrix: λ = {3, 1, −1} on a perturbation of identity.
        // Construct via eigendecomp: V·diag(3,1,-1)·V^T with random orthogonal V.
        var indefinite = new decimal[,]
        {
            { 1m, 2m, 0m },
            { 2m, 1m, 0m },
            { 0m, 0m, 1m },
        };

        var projected = NearestPsdProjection.EigenClip(indefinite);
        NearestPsdProjection.IsPsd(projected).Should().BeTrue();
    }

    [Fact]
    public void Higham_ConvergesToValidCorrelation()
    {
        // Invalid correlation: triangle inequality violated.
        var bad = new decimal[,]
        {
            { 1.0m, 0.9m, -0.9m },
            { 0.9m, 1.0m, 0.9m },
            { -0.9m, 0.9m, 1.0m },
        };

        var fixed_ = NearestPsdProjection.Higham(bad);
        NearestPsdProjection.IsPsd(fixed_).Should().BeTrue();

        for (var i = 0; i < 3; i++)
        {
            ((double)Math.Abs(fixed_[i, i] - 1m)).Should().BeLessThan(1e-10);
        }
    }

    [Fact]
    public void IsPsd_RejectsIndefinite()
    {
        var indefinite = new decimal[,]
        {
            { 1m, 2m },
            { 2m, 1m },
        };
        NearestPsdProjection.IsPsd(indefinite).Should().BeFalse();
    }
}
