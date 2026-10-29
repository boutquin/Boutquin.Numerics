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

using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

[Trait("Tier", "Unit")]
public sealed class FidelityC1MullerDoubleRootControls
{
    [Fact]
    public void NondegenerateCubicDoubleRootShowsTheConditionalSquaredErrorLaw()
    {
        // f=x^2(1+x) has double root0, second derivative2 and third derivative6.
        // Its interpolating quadratic at errors a,b,c is exactly
        // (1+a+b+c)x^2-(ab+ac+bc)x+abc. Negative seeds keep its roots real.
        // Solving that polynomial independently in80-digit Decimal arithmetic gives
        // first negative root -0.00596475702399953910843924820547079447.
        var visits = new List<double>();
        double Objective(double x) { visits.Add(x); return x * x * (1 + x); }
        new MullerSolver<double>(1e-50, 9).Solve(Objective, -0.08, -0.04, -0.02);
        Assert.True(visits.Count >= 10);
        Assert.InRange(Math.Abs(visits[3] + 0.00596475702399953910843924820547079447), 0, 1e-14);
        Assert.All(visits.Take(10), x => Assert.True(x < 0));
        // Equation23 gives e_next^2/(-e_i*e_(i-1)*e_(i-2)) ->1.
        // Independent quadratic roots give ratios0.88219883 at index7 and
        // 0.96768230 at index9; the latter is already within5% of its limit.
        var earlier = visits[7] * visits[7] / (-visits[6] * visits[5] * visits[4]);
        var later = visits[9] * visits[9] / (-visits[8] * visits[7] * visits[6]);
        Assert.InRange(later, 0.95, 0.99);
        Assert.True(Math.Abs(1 - later) < Math.Abs(1 - earlier));
        // The positive root of2p^3=p^2+p+1 is1.2337519... .
        // The independent finite-window observed order is1.23752974; no
        // claim is made after rounding dominates or on complex continuations.
        var observedOrder = Math.Log(Math.Abs(visits[9] / visits[8]))
            / Math.Log(Math.Abs(visits[8] / visits[7]));
        Assert.InRange(observedOrder, 1.20, 1.27);
    }
}
