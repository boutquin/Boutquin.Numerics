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

public sealed class C4TailBoundaryTests
{
    [Fact]
    public void TiedPseudoObservationAtThreshold_IncludesBothMarginalAndJointEquality()
    {
        // Average ranks of x are [2,2,2,4], divided by n+1=5.
        // At q=2/5 all three tied x values qualify; y ranks 1 and 2 qualify.
        // Thus joint=2, first marginal=3 and conditional lambda=2/3.
        // Discriminates inclusive-lower, empirical-inclusive and variant.ranks:
        // strict x drops all joint observations, strict y leaves one, rank/n drops x.
        var result = LowerTailDependence<decimal>.Estimate(
            [0m, 0m, 0m, 1m], [1m, 2m, 3m, 4m],
            new TailOptions<decimal>(Quantiles: [0.4m]));

        result.JointCount.Should().Equal(2);
        result.Lambda.Should().Be(2m / 3m);
    }
}
