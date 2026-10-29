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

public sealed class C2IndependentProjectionStoppingTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(10)]
    public void FirstIteration_UsesAbsoluteFullFrobeniusNormIncludingEquality(int scale)
    {
        var unit = 1m / scale;
        decimal[,] a = { { 1m + 3m * unit, 0m }, { 0m, 1m + 4m * unit } };
        // PSD projection leaves a unchanged; affine projection produces I.
        // Therefore delta²=(3/scale)²+(4/scale)² and ||delta||F=5/scale.
        decimal[,]? result = null;
        FluentActions.Invoking(() => result = NearestPsdProjection<decimal>.Higham(a, 1, 5m * unit)).Should().NotThrow();
        result![0, 0].Should().Be(1m);
        result[1, 1].Should().Be(1m);
        result[0, 1].Should().Be(0m);
        FluentActions.Invoking(() => NearestPsdProjection<decimal>.Higham(a, 1, 4.999m * unit)).Should().Throw<InvalidOperationException>();
    }
}
