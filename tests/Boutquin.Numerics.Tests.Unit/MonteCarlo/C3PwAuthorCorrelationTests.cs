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

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class C3PwAuthorCorrelationTests
{
    [Fact]
    public void AuthorListing_CorrelationRows_DiscriminateFullSampleSelector()
    {
        // Author m=1; full-N correlations incorrectly select m=5.
        double[] series = [0, -5, 6, 2, -2, 5, -4, -2, -2, -1, 6, 4, -1, -5, 1, -1, -4, 0, -6, -4, 1, -2, 6, 1, -1, 6, -3, 4, -2, 0, 5, -5, 3, -2, -1, 1, -5, -1, 6, -6, -3, -3, -6, -2, 4, 6, 6, -5, 0, -1, -2, 3, -6, 4, -6, -5, 2, 6, -4, 5, -1, 1, 5, 0];
        PolitisWhiteBlockLength.EstimateDouble(series).Should().BeApproximately(2.4011920853695696, 1e-11);
    }
}
