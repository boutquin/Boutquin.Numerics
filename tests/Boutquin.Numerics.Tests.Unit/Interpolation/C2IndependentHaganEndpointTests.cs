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

public sealed class C2IndependentHaganEndpointTests
{
    [Fact]
    public void NonuniformDefaultEndpoints_HaveIndependentCubicIntegrals()
    {
        decimal[] xs = [0, 1, 3];
        decimal[] ys = [0, 1, 5];
        // F=(1,2), h=(1,2) gives f=(5/6,4/3,7/3). Both
        // intervals lie on sector-I boundary rays, with explicit primitives.
        for (var k = 1; k < 8; k++)
        {
            var u = k / 8m;
            MonotoneConvexInterpolator<decimal>.Instance.Interpolate(u, xs, ys)
                .Should().BeApproximately((5m * u + u * u * u) / 6m, 1e-25m);
            MonotoneConvexInterpolator<decimal>.Instance.Interpolate(1m + 2m * u, xs, ys)
                .Should().BeApproximately(1m + 2m * (4m * u / 3m + u * u - u * u * u / 3m), 1e-25m);
        }
    }

    [Fact]
    public void EndpointAndInteriorPositivityBounds_HaveExactQuadraticIntegrals()
    {
        decimal[] xs = [0, 1, 2];
        decimal[] rising = [0, 1, 11];
        decimal[] falling = [0, 10, 11];
        // Raw f=(−5/4,11/2,49/4). Left clamp gives0 and interior
        // clamp gives2, so first-interval forward is2u and NCR is u².
        // Reflection supplies the independent right-end case.
        for (var k = 1; k < 8; k++)
        {
            var u = k / 8m;
            MonotoneConvexInterpolator<decimal>.Instance.Interpolate(u, xs, rising)
                .Should().BeApproximately(u * u, 1e-25m);
            MonotoneConvexInterpolator<decimal>.Instance.Interpolate(2m - u, xs, falling)
                .Should().BeApproximately(11m - u * u, 1e-25m);
        }
    }

    [Fact]
    public void ConstantForward_HasExactLinearCumulativeReturn()
    {
        decimal[] xs = [0, 1, 3, 5];
        decimal[] ys = [0, 2, 6, 10];
        for (var k = 1; k < 20; k++)
        {
            var x = k / 4m;
            MonotoneConvexInterpolator<decimal>.Instance.Interpolate(x, xs, ys).Should().Be(2m * x);
        }
    }
}
