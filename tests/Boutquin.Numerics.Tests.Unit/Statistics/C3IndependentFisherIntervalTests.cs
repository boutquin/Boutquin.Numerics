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

public sealed class C3IndependentFisherIntervalTests
{
    [Fact]
    public void ConfidenceInterval_UnitNormalCritical_MatchesIndependentExponentialBounds()
    {
        // P(|Z| <= 1) = 0.6826894921370859; n=7 gives standard error1/2.
        // r=3/5 gives z=ln(2), hence exp(2z +/- 1)=4*exp(+/-1).
        // Bounds (4*exp(+/-1)-1)/(4*exp(+/-1)+1) were computed with
        // Python decimal.exp at70digits, independently of this library's
        // atanh/tanh and inverse-normal approximation.
        var actual = FisherZTransform<double>.ConfidenceInterval(0.6, 7, 0.6826894921370859);
        actual.Lower.Should().BeApproximately(0.19078064961662067036, 2e-14);
        actual.Upper.Should().BeApproximately(0.83155238319820521972, 2e-14);
    }
}
