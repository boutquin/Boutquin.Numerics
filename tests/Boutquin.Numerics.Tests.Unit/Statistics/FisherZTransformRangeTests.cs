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

public sealed class FisherZTransformRangeTests
{
    [Fact]
    public void ConfidenceInterval_HalfCountBeyondFiniteRange_RetainsNonzeroWidth()
    {
        var interval = FisherZTransform<Half>.ConfidenceInterval((Half)0.5, 70_000);

        ((float)interval.Upper - (float)interval.Lower).Should().BeInRange(0.009f, 0.013f);
    }

    [Fact]
    public void ConfidenceInterval_FloatValidHighConfidence_RetainsFiniteCriticalValue()
    {
        Action act = () => _ = FisherZTransform<float>.ConfidenceInterval(0.5f, 100, 0.99999999);

        act.Should().NotThrow();

        var interval = FisherZTransform<float>.ConfidenceInterval(0.5f, 100, 0.99999999);
        interval.Lower.Should().BeApproximately(-0.03254971f, 1e-5f);
        interval.Upper.Should().BeApproximately(0.8114205f, 1e-5f);
    }
}
