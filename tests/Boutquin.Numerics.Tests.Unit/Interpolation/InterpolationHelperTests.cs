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

using Boutquin.Numerics.Interpolation;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Interpolation;

public sealed class InterpolationHelperTests
{
    [Fact]
    public void LinearBlend_AtSecondEndpoint_PreservesSmallDoubleOrdinate()
    {
        var actual = InterpolationHelper.LinearBlend(1e16, 1.0, 1.0);

        actual.Should().Be(1.0);
    }

    [Fact]
    public void LinearBlend_AtSecondEndpoint_PreservesSmallDecimalOrdinate()
    {
        var actual = InterpolationHelper.LinearBlend(1e28m, 0.1m, 1m);

        actual.Should().Be(0.1m);
    }
}
