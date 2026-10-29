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
public sealed class FidelityC1NonfiniteSimplexControls
{
    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void EveryNonfiniteValueRanksBehindFiniteBest(double invalid)
    {
        var calls = 0;
        var result = NelderMeadMinimizer<double>.Minimize(x => ++calls <= 2 ? x[0] * x[0] : invalid, [0],
            new SimplexOptions<double>(MaxIterations: 1, InitialStep: 1, XTolerance: 0, FTolerance: 0));
        Assert.Equal(0, result.MinValue);
        Assert.Equal(0, result.Argmin[0]);
        Assert.False(result.Converged);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void EntirelyInvalidInitialSimplexIsRejected(double invalid)
    {
        var error = Record.Exception(() => NelderMeadMinimizer<double>.Minimize(_ => invalid, [0],
            new SimplexOptions<double>(MaxIterations: 1, InitialStep: 1, XTolerance: 2, FTolerance: 0)));
        Assert.IsType<ArgumentException>(error);
    }
}
