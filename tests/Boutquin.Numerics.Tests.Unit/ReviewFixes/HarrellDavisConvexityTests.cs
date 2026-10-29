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

using Boutquin.Numerics.MonteCarlo;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

public sealed class HarrellDavisConvexityTests
{
    [Theory]
    [InlineData(10)]
    [InlineData(1000)]
    public void Compute_ConstantDecimalSample_PreservesItsValue(int count)
    {
        foreach (var value in new[] { 1m, decimal.MaxValue, decimal.MinValue })
        {
            var sorted = Enumerable.Repeat(value, count).ToArray();
            decimal actual = 0;
            FluentActions.Invoking(() => actual = HarrellDavisPercentile.Compute<decimal>(sorted, 0.5))
                .Should().NotThrow("a convex average of finite identical values is representable");
            actual.Should().Be(value, "a normalized weighted average preserves constant samples");
        }
    }
}
