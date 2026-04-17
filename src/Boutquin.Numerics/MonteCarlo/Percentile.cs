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

using System.Numerics;

namespace Boutquin.Numerics.MonteCarlo;

/// <summary>
/// Linear-interpolation percentile. The sample at index <c>p·(N−1)</c> is
/// returned, with the fractional part interpolated between the two bracketing
/// samples. Matches NumPy's default <c>linear</c> method and R's <c>type=7</c>.
/// </summary>
/// <remarks>
/// <para>
/// The input array must already be sorted ascending. Empty input returns 0;
/// single-element input returns that element. The percentile rank is
/// clamped to [0, 1].
/// </para>
/// <para>
/// Tier A: Arithmetic order-statistic operations on floating-point types.
/// </para>
/// </remarks>
public static class Percentile
{
    /// <summary>Computes the linear-interpolation percentile of a sorted array.</summary>
    /// <param name="sorted">Sorted (ascending) sample.</param>
    /// <param name="p">Percentile rank in [0, 1].</param>
    public static T Compute<T>(T[] sorted, T p)
        where T : IFloatingPoint<T>
    {
        ArgumentNullException.ThrowIfNull(sorted);
        if (sorted.Length == 0)
        {
            return T.Zero;
        }

        if (sorted.Length == 1)
        {
            return sorted[0];
        }

        p = T.Clamp(p, T.Zero, T.One);
        var index = double.CreateChecked(p) * (sorted.Length - 1);
        var lower = (int)Math.Floor(index);
        var upper = Math.Min(lower + 1, sorted.Length - 1);
        var fraction = T.CreateChecked(index - lower);
        return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
    }
}
