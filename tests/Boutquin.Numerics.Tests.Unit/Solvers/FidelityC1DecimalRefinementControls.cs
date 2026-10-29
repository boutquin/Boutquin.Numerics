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

using System.Reflection;
using Boutquin.Numerics.LinearAlgebra.Internal;
using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

[Trait("Tier", "Unit")]
public sealed class FidelityC1DecimalRefinementControls
{
    private delegate decimal Refine(decimal[,] x, ReadOnlySpan<decimal> y,
        decimal[] coefficients, decimal[,] qr, decimal[] diagonal, int[] permutation,
        decimal[] residuals, bool includeIntercept, int p, int m, int designP);

    [Fact]
    public void CoefficientOnlyRefinementCorrectsIndependentPerturbedStart()
    {
        // X has orthogonal columns with exact norms2,3. The unique least-squares
        // solution is(2,3), residual(0,0,5), RSS25. Starting at(1,1) requires
        // correction(1,2). All these decimal operations are exactly representable;
        // no condition-dependent accuracy bar or observed solver output is used.
        decimal[,] x = { { 2m, 0m }, { 0m, 3m }, { 0m, 0m } };
        decimal[] y = [4m, 9m, 5m];
        var qr = (decimal[,])x.Clone();
        var diagonal = new decimal[2];
        var permutation = new int[2];
        Assert.True(HouseholderQr<decimal>.Factor(qr, new decimal[3], diagonal, permutation));
        decimal[] coefficients = [1m, 1m];
        var residuals = new decimal[3];
        var method = typeof(OrdinaryLeastSquares<decimal>).GetMethod("RefineAndComputeResiduals",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);
        var refine = method.CreateDelegate<Refine>();
        var rss = refine(x, y, coefficients, qr, diagonal, permutation, residuals, false, 2, 3, 2);
        Assert.Equal(new decimal[] { 2m, 3m }, coefficients);
        Assert.Equal(new decimal[] { 0m, 0m, 5m }, residuals);
        Assert.Equal(25m, rss);
    }
}
