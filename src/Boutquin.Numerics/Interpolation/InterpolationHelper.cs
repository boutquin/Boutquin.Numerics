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

namespace Boutquin.Numerics.Interpolation;

/// <summary>
/// Shared helper methods used by all interpolator implementations.
/// </summary>
internal static class InterpolationHelper
{
    /// <summary>
    /// Finds the index i such that xs[i] &lt;= x &lt; xs[i+1] using binary search.
    /// The caller must ensure that xs[0] &lt; x &lt; xs[^1] (boundary cases handled before calling).
    /// </summary>
    internal static int FindInterval<T>(T x, ReadOnlySpan<T> xs)
        where T : IFloatingPoint<T>
    {
        int lo = 0, hi = xs.Length - 2;
        while (lo < hi)
        {
            var mid = (lo + hi + 1) / 2;
            if (x < xs[mid])
            {
                hi = mid - 1;
            }
            else
            {
                lo = mid;
            }
        }

        return lo;
    }
}
