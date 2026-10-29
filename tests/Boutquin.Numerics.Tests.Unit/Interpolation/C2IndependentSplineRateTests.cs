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

public sealed class C2IndependentSplineRateTests
{
    [Theory]
    [InlineData(4, 0.021205357142857142857142857142857142857)]
    [InlineData(8, 0.00533585695876288659793814432989690721649)]
    [InlineData(16, 0.00133400918218897805176170484136684912579)]
    public void NaturalQuadratic_EndpointErrorMatchesClosedFormMoments(int intervals, double expected)
    {
        // Uniform-grid recurrence: M_(i-1)+4M_i+M_(i+1)=12, M_0=M_n=0.
        // With r=sqrt(3)-2, M_i=2[1-(r^i+r^(n-i))/(1+r^n)].
        // Therefore S(h/2)=h²/2-h²*M_1/16. Values above were evaluated
        // independently at 80 decimal digits. Nonzero endpoint curvature
        // gives error/h² -> (sqrt(3)-1)/8; this fixture is not a universal bound.
        var xs = Enumerable.Range(0, intervals + 1).Select(i => (double)i / intervals).ToArray();
        var spline = new CubicSplineInterpolator<double>(xs, xs.Select(x => x * x).ToArray(), SplineBoundary<double>.Natural);
        var midpoint = 0.5 / intervals;
        spline.Interpolate(midpoint).Should().BeApproximately(expected, 1e-16);
        spline.Interpolate(1 - midpoint).Should().BeApproximately(1 - 2 * midpoint + expected, 5e-16);
    }

    [Theory]
    [InlineData(4, 0.00244140625)]
    [InlineData(8, 0.00016667292668269230769230769230769230769)]
    [InlineData(16, 0.000010421946615228951804839349464498214994)]
    public void NotAKnotQuartic_EndpointErrorMatchesClosedFormMoments(int intervals, double expected)
    {
        // Quartic data on h=1/n: particular moment 12*x_i²-2*h².
        // The homogeneous correction is C*(r^i+r^(n-i)), r=sqrt(3)-2,
        // C=-24*h²/[(1-r)²*(1+r^(n-2))], from equal first two third derivatives.
        // S(h/2)=h^4/2-h²*(M_0+M_1)/16. These independent closed-form
        // values exhibit fourth-order endpoint error on this smooth fixture.
        var xs = Enumerable.Range(0, intervals + 1).Select(i => (double)i / intervals).ToArray();
        var spline = new CubicSplineInterpolator<double>(xs, xs.Select(x => x * x * x * x).ToArray(), SplineBoundary<double>.NotAKnot);
        spline.Interpolate(0.5 / intervals).Should().BeApproximately(expected, 1e-16);
    }
}
