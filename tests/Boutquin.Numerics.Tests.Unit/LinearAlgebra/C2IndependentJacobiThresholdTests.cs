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

public sealed class C2IndependentJacobiThresholdTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CappedRelativeThreshold_RotatesBothSidesOfUnitNorm(bool large)
    {
        var scale = large ? 1d : .1d;
        var off = large ? 1e-15 : 6e-16;
        double[,] matrix = { { 3 * scale, off }, { off, 4 * scale } };
        var result = JacobiEigenDecomposition<double>.Decompose(matrix);
        // Norm is approximately5*scale. Both full off-norms exceed
        // 1e-15*min(norm,1), but fail if the opposite threshold branch is used.
        // Analytic small rotation t=off/(aqq-app)+O(t^3); O(t^3)<1e-42.
        Math.Abs(result.Vectors[0, 0]).Should().BeApproximately(off / scale, 1e-28);
        Math.Abs(result.Vectors[1, 1]).Should().BeApproximately(off / scale, 1e-28);
    }
}
