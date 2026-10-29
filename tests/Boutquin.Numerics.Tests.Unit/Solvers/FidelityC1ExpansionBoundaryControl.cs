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
public sealed class FidelityC1ExpansionBoundaryControl
{
    [Fact]
    public void UnitExpansionIsRejectedWhenReflectionIsSmaller()
    {
        // chi=1 violates chi>1 but satisfies chi>rho=.5, isolating the first guard.
        var calls = 0;
        var error = Record.Exception(() => NelderMeadMinimizer<double>.Minimize(
            x => { calls++; return x[0] * x[0]; }, [1],
            new SimplexOptions<double>(Reflection: 0.5, Expansion: 1)));
        Assert.IsType<ArgumentOutOfRangeException>(error);
        Assert.Equal(0, calls);
    }
}
