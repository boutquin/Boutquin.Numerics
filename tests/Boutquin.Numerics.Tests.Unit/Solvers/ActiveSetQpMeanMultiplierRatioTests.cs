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
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// The reduced-QP multiplier is a ratio of two aggregates. It must be non-finite only when the
/// ratio itself is: an aggregate that leaves the range while the ratio stays representable must
/// not turn a representable optimum into a rejection.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpMeanMultiplierRatioTests
{
    private const double M = double.MaxValue;

    private static double[,] Diag(int n, double c)
    {
        var a = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            a[i, i] = c;
        }

        return a;
    }

    [Fact]
    public void QpMeanVariance_PinnedAssetLeavesFreeTermsWhoseTotalOverflowsButWhoseRatioDoesNot_ReturnsTheRepresentableOptimum()
    {
        double[] weights = [];
        var error = Record.Exception(() => weights = ActiveSetQpSolver<double>.SolveMeanVariance(
            Diag(3, 1e-10),
            [1.5460160959815915e298, -1.7167969437935113e298, -1.7185946369283737e298],
            1.6,
            -0.8664 * M,
            0.7233 * M));

        error.Should().BeNull("every optimum weight is representable and inside the box");
        weights[0].Should().BeApproximately(1.300271444445913e308, 1.300271444445913e308 * 1e-12);
        weights[1].Should().BeApproximately(-6.49573943118312e307, 6.49573943118312e307 * 1e-12);
        weights[2].Should().BeApproximately(-6.50697501327601e307, 6.50697501327601e307 * 1e-12);
    }
}
