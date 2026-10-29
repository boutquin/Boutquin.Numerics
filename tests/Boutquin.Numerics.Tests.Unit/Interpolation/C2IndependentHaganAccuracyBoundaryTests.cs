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

public sealed class C2IndependentHaganAccuracyBoundaryTests
{
    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(16)]
    public void DefaultZeroSlopeEndpoint_HasDerivedQuadraticIntegralError(int intervals)
    {
        // True forward f(t)=1+t, NCR=t+t²/2, uniform mesh h.
        // Eq31 chooses f0=1+h/4, imposing fhat'(0)=0 unlike this true f.
        // Eq34 on first interval gives fhat(hu)=1+h/4+3*h*u²/4.
        // Integrating to u=1/2 gives h/2+5*h²/32, so error=h²/32.
        // This is a boundary-compatibility counterexample to an unconditional
        // global third-order integral-accuracy claim, not a production defect.
        var h = 1m / intervals;
        var xs = Enumerable.Range(0, intervals + 1).Select(i => i * h).ToArray();
        var ys = xs.Select(x => x + x * x / 2m).ToArray();
        var actual = MonotoneConvexInterpolator<decimal>.Instance.Interpolate(h / 2m, xs, ys);
        var trueIntegral = h / 2m + h * h / 8m;
        (actual - trueIntegral).Should().BeApproximately(h * h / 32m, 1e-25m);

        // At an interior interval both adjacent nodal forwards are exact;
        // the basic quadratic reduces to the true linear forward.
        var interior = 1.5m * h;
        MonotoneConvexInterpolator<decimal>.Instance.Interpolate(interior, xs, ys)
            .Should().BeApproximately(interior + interior * interior / 2m, 1e-25m);
    }
}
