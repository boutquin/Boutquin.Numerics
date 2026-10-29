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
public sealed class FidelityC1SimplexStoppingMeaningControls
{
    [Fact]
    public void FunctionSpreadStopDoesNotCertifyStationarityEvenForStrictConvexQuadratic()
    {
        // f=1e-8*x^2 is strictly convex with unique minimizer0 and bounded level sets.
        // Initial vertices2,3 have diameter1 and function spread5e-8 below1e-6.
        // The selected OR stop therefore returns2, whose derivative4e-8 is nonzero.
        var calls = 0;
        var result = NelderMeadMinimizer<double>.Minimize(
            x => { calls++; return 1e-8 * x[0] * x[0]; }, [2],
            new SimplexOptions<double>(MaxIterations: 1, InitialStep: 1, XTolerance: 0, FTolerance: 1e-6));
        Assert.True(result.Converged);
        Assert.Equal(2, calls);
        Assert.Equal(2, result.Argmin[0]);
        Assert.Equal(4e-8, result.MinValue);
        Assert.NotEqual(0, 2e-8 * result.Argmin[0]);
    }
}
