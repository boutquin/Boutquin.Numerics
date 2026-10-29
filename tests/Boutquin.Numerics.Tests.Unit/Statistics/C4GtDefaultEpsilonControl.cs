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
public sealed class C4GtDefaultEpsilonControl
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PositiveSeries_DefaultSmoothingIsOneMillionth(bool explicitBenchmark)
    {
        // Returns 1..51: mean26, unbiased variance221, z^2=156, downside0.
        // Cumulative y_k=k(k+1)/2. Exact centered finite sums give r^2=795/844.
        // This is the declared GT-inspired loss; no claim of author stream parity.
        var returns = Enumerable.Range(1, 51).Select(i => (double)i).ToArray();
        var result = explicitBenchmark
            ? GtInspiredScore<double>.Compute(returns, 0)
            : GtInspiredScore<double>.Compute(returns);
        var expected = -26 * Math.Log(Math.Sqrt(156)) * (795.0 / 844) / 1e-6;
        result.Score.Should().BeApproximately(expected, 1e-6);
    }
}
