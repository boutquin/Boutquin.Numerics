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
/// Hagan-West (2006) monotone-convex interpolation. Guarantees non-negative instantaneous
/// forward rates across the full maturity range when applied to normalized cumulative return
/// (NCR) node arrays derived from discount factors.
/// </summary>
/// <typeparam name="T">The floating-point type for interpolation.</typeparam>
/// <remarks>
/// Tier A: Requires only basic floating-point arithmetic.
/// </remarks>
public sealed class MonotoneConvexInterpolator<T> : IInterpolator<T> where T : IFloatingPoint<T>
{
    /// <inheritdoc cref="MonotoneConvexInterpolator.Instance"/>
    public static readonly MonotoneConvexInterpolator<T> Instance = new();

    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_three = T.CreateChecked(3);

    private MonotoneConvexInterpolator() { }

    /// <inheritdoc/>
    public string Name => "MonotoneConvex";

    /// <summary>
    /// Evaluates the monotone-convex interpolant at <paramref name="x"/> given an augmented
    /// node array that includes a virtual node at <c>xs[0] = 0, ys[0] = 0</c> followed by
    /// the actual curve nodes.
    /// </summary>
    /// <param name="x">Target x-coordinate (year fraction). Must be positive for meaningful results.</param>
    /// <param name="xs">
    /// Strictly-increasing node x-coordinates, with <c>xs[0] = 0</c> (the virtual origin node)
    /// and <c>xs[1..N-1]</c> as the actual node times in year fractions.
    /// </param>
    /// <param name="ys">
    /// Node normalized cumulative return (NCR) values: <c>ys[i] = −ln P(xs[i])</c>. Must satisfy
    /// <c>ys[0] = 0</c> (the virtual origin, corresponding to <c>P(0) = 1</c>) and be non-decreasing
    /// for a no-arbitrage yield curve.
    /// </param>
    /// <returns>
    /// The interpolated NCR value at <paramref name="x"/>. Convert back to a discount factor via
    /// <c>P(x) = exp(−result)</c>.
    /// </returns>
    /// <remarks>
    /// Algorithm summary (Hagan &amp; West 2006, §2):
    /// <list type="number">
    ///   <item>Compute per-segment discrete forward rates <c>F[j] = (ys[j] − ys[j−1]) / h[j]</c>,
    ///         where <c>h[j] = xs[j] − xs[j−1]</c>.</item>
    ///   <item>Estimate instantaneous forward tangents <c>g[i]</c> at each node as a time-weighted
    ///         average of adjacent segment forwards; extrapolate to boundary nodes.</item>
    ///   <item>Clamp each <c>g[i]</c> to <c>[0, min(2F[k], 2F[k+1])]</c> to enforce non-negativity
    ///         of instantaneous forward rates.</item>
    ///   <item>Integrate the resulting cubic forward polynomial within the bracketing segment to
    ///         obtain the NCR at <paramref name="x"/>.</item>
    /// </list>
    /// Flat extrapolation beyond the node range. Stateless; exposed as the singleton
    /// <see cref="Instance"/>. Every call recomputes tangents from the node arrays.
    /// Reference: Hagan, P. S. and West, G. (2006), "Interpolation Methods for Curve Construction",
    /// Applied Mathematical Finance 13(2), pp. 89–129.
    /// </remarks>
    public T Interpolate(T x, ReadOnlySpan<T> xs, ReadOnlySpan<T> ys)
    {
        if (xs.Length != ys.Length)
        {
            throw new ArgumentException("xs and ys must have the same length.");
        }

        if (xs.Length == 0)
        {
            throw new ArgumentException("At least one node is required.");
        }

        if (x <= xs[0])
        {
            return ys[0];
        }

        if (x >= xs[^1])
        {
            return ys[^1];
        }

        int N = xs.Length; // total nodes (includes virtual node at index 0)

        // --- Step 1: Segment widths h[j] and discrete forward rates F[j] for j = 1..N-1 ---
        T[] h = new T[N];
        T[] F = new T[N];
        for (int j = 1; j < N; j++)
        {
            h[j] = xs[j] - xs[j - 1];
            F[j] = (ys[j] - ys[j - 1]) / h[j];
        }

        // --- Step 2: Instantaneous forward tangents g[i] at each node ---
        T[] g = new T[N];

        // Interior nodes: time-weighted average of adjacent segment forwards
        for (int i = 1; i < N - 1; i++)
        {
            g[i] = (h[i] * F[i + 1] + h[i + 1] * F[i]) / (h[i] + h[i + 1]);
        }

        // Boundary extrapolation (Hagan-West §2.3)
        g[0] = N >= 3 ? (s_three * F[1] - g[1]) / s_two : F[1];
        g[N - 1] = N >= 3 ? (s_three * F[N - 1] - g[N - 2]) / s_two : F[N - 1];

        // --- Step 3: Enforce non-negativity (Hagan-West §2.4) ---
        g[0] = T.Max(T.Zero, T.Min(g[0], s_two * F[1]));

        for (int i = 1; i < N - 1; i++)
        {
            T limit = T.Min(s_two * F[i], s_two * F[i + 1]);
            g[i] = limit > T.Zero ? T.Max(T.Zero, T.Min(g[i], limit)) : T.Zero;
        }

        g[N - 1] = T.Max(T.Zero, T.Min(g[N - 1], s_two * F[N - 1]));

        // --- Step 4: Locate the segment containing x ---
        int k = 1;
        for (; k < N; k++)
        {
            if (x <= xs[k])
            {
                break;
            }
        }

        // --- Step 5: Integrate the cubic forward polynomial within segment k ---
        // u = relative position in [0, 1]; deltaR is the NCR increment over u
        T u = (x - xs[k - 1]) / h[k];
        T u2 = u * u;
        T u3 = u2 * u;

        T deltaR = h[k] * (
            F[k] * u
            + (g[k - 1] - F[k]) * (u - s_two * u2 + u3)
            + (g[k] - F[k]) * (-u2 + u3));

        return ys[k - 1] + deltaR;
    }
}

/// <summary>
/// Hagan-West (2006) monotone-convex interpolation. Guarantees non-negative instantaneous
/// forward rates across the full maturity range when applied to normalized cumulative return
/// (NCR) node arrays derived from discount factors.
/// </summary>
/// <remarks>
/// <para>
/// Domain-layer usage: callers supply an augmented node array whose first element is the virtual
/// origin (<c>xs[0] = 0, ys[0] = 0</c>), followed by the actual curve nodes expressed as year
/// fractions and NCR values (<c>ys[i] = −ln P(xs[i])</c>). The method returns the interpolated
/// NCR; convert back to a discount factor via <c>P = exp(−result)</c>.
/// </para>
/// <para>
/// Algorithm: Hagan, P. S. and West, G. (2006), "Interpolation Methods for Curve Construction",
/// Applied Mathematical Finance 13(2), pp. 89–129.
/// </para>
/// <para>
/// Unlike the monotone cubic Hermite method (<see cref="MonotoneCubicInterpolator"/>), which
/// preserves monotonicity of the y-values, the Hagan-West method preserves <em>non-negativity
/// of the derivative</em> — equivalently, non-negative instantaneous forward rates — which is
/// the relevant no-arbitrage constraint for discount-factor curves.
/// </para>
/// <para>
/// Flat extrapolation beyond the node range. Stateless; exposed as the singleton
/// <see cref="Instance"/>. Every call recomputes tangents from the node arrays.
/// For repeated evaluations on the same node set consider caching the node arrays
/// rather than this instance, as there is no stateful precomputed form of this algorithm.
/// </para>
/// </remarks>
public sealed class MonotoneConvexInterpolator : IInterpolator
{
    /// <summary>
    /// Shared singleton instance.
    /// </summary>
    public static readonly MonotoneConvexInterpolator Instance = new();

    private MonotoneConvexInterpolator() { }

    /// <inheritdoc />
    public string Name => "MonotoneConvex";

    /// <inheritdoc />
    public double Interpolate(double x, ReadOnlySpan<double> xs, ReadOnlySpan<double> ys)
    {
        return MonotoneConvexInterpolator<double>.Instance.Interpolate(x, xs, ys);
    }
}
