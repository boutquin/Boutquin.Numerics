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

using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

[Trait("Tier", "Unit")]
public sealed class C4MpEdgeTests
{
    [Fact]
    public void UnitVarianceEdges_MatchExactSquareAspectRatios()
    {
        // Laloux unit-variance upper edge (1+1/sqrt(Q))².
        // Rational Q values make these expectations algebraic, independent
        // of a density fitter or covariance eigensolver.
        CovarianceHelpers<double>.MarcenkoPasturUpperBound(1).Should().Be(4);
        CovarianceHelpers<double>.MarcenkoPasturUpperBound(4).Should().Be(9.0 / 4);
        CovarianceHelpers<double>.MarcenkoPasturUpperBound(9).Should().BeApproximately(16.0 / 9, 1e-15);
    }
}
