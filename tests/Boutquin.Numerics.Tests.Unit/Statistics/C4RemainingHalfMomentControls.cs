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
public sealed class C4RemainingHalfMomentControls
{
    [Fact]
    public void Half_AsymmetricPopulationMoments_HaveExactNormalArgumentWithinRounding()
    {
        // Exact input moments give z=1.5, as in the double/decimal control.
        // Half rounds the four reduced moments and annualized Sharpe before Evaluate.
        var result = DeflatedSharpeRatio<Half>.ComputeFromReturns([(Half)0, (Half)0, (Half)0, (Half)4], 1, 1);
        ((double)result.PValue).Should().BeApproximately(0.066807201268858066, .001);
        ((double)result.DeflatedSharpe).Should().BeApproximately(0.5773502691896257645, .001);
    }
}
