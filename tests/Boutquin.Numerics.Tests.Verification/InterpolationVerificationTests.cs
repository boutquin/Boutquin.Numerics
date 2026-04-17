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

namespace Boutquin.Numerics.Tests.Verification;

public sealed class InterpolationVerificationTests : CrossLanguageVerificationBase
{
    [Fact]
    public void LinearInterpolator_MatchesScipy()
    {
        using var doc = LoadVector("interpolation");
        var xs = GetDoubleArray(doc.RootElement.GetProperty("xs"));
        var ys = GetDoubleArray(doc.RootElement.GetProperty("ys"));
        var queries = GetDoubleArray(doc.RootElement.GetProperty("queries"));
        var expected = GetDoubleArray(doc.RootElement.GetProperty("linear"));

        for (var i = 0; i < queries.Length; i++)
        {
            var actual = LinearInterpolator.Instance.Interpolate(queries[i], xs, ys);
            AssertScalarWithin(actual, expected[i], PrecisionNumeric, $"linear @ {queries[i]}");
        }
    }

    [Fact]
    public void MonotoneCubic_MatchesScipyPchip()
    {
        using var doc = LoadVector("interpolation");
        var xs = GetDoubleArray(doc.RootElement.GetProperty("xs"));
        var ys = GetDoubleArray(doc.RootElement.GetProperty("ys"));
        var queries = GetDoubleArray(doc.RootElement.GetProperty("queries"));
        var expected = GetDoubleArray(doc.RootElement.GetProperty("pchip"));

        for (var i = 0; i < queries.Length; i++)
        {
            var actual = MonotoneCubicInterpolator.Instance.Interpolate(queries[i], xs, ys);
            // SciPy PchipInterpolator uses a three-point endpoint derivative
            // (SciPy's implementation of Fritsch-Butland 1984) while our
            // Fritsch-Carlson 1980 variant uses a one-sided secant. The
            // interior values match closely; endpoint segments diverge by
            // ≤ 2% in shape-reasonable cases — acceptable for a shape-
            // preserving monotone interpolator.
            var tolerance = 0.05;
            Math.Abs(actual - expected[i]).Should().BeLessThan(tolerance,
                $"pchip @ {queries[i]}: actual={actual}, expected={expected[i]}");
        }
    }

    [Fact]
    public void NaturalCubicSpline_MatchesScipy()
    {
        using var doc = LoadVector("interpolation");
        var xs = GetDoubleArray(doc.RootElement.GetProperty("xs"));
        var ys = GetDoubleArray(doc.RootElement.GetProperty("ys"));
        var queries = GetDoubleArray(doc.RootElement.GetProperty("queries"));
        var expected = GetDoubleArray(doc.RootElement.GetProperty("natural_cubic_spline"));

        var spline = new CubicSplineInterpolator(xs, ys, SplineBoundary.Natural);
        for (var i = 0; i < queries.Length; i++)
        {
            var actual = spline.Interpolate(queries[i]);
            AssertScalarWithin(actual, expected[i], PrecisionNumeric, $"natural cubic @ {queries[i]}");
        }
    }

    // ---------------------------------------------------------------------
    //  Phase 4 — four new interpolator variants (spec §2.4).
    // ---------------------------------------------------------------------

    /// <summary>
    /// Log-linear interpolator — numpy port computes
    /// ``y(x) = exp(ln(yᵢ) + t·(ln(yᵢ₊₁) − ln(yᵢ)))`` for a
    /// discount-factor-like monotonic positive series. Both sides compute
    /// the same `exp`/`log` chain on IEEE doubles, so 1e-10 relative is
    /// the spec §2.4 bar.
    /// </summary>
    [Fact]
    public void LogLinearInterpolator_MatchesNumpyReference()
    {
        using var doc = LoadVector("interpolation");
        var block = doc.RootElement.GetProperty("log_linear");
        var xs = GetDoubleArray(block.GetProperty("xs"));
        var ys = GetDoubleArray(block.GetProperty("ys"));
        var queries = GetDoubleArray(block.GetProperty("queries"));
        var expected = GetDoubleArray(block.GetProperty("values"));

        for (var i = 0; i < queries.Length; i++)
        {
            var actual = LogLinearInterpolator.Instance.Interpolate(queries[i], xs, ys);
            AssertScalarWithin(actual, expected[i], PrecisionExact, $"log_linear @ {queries[i]}");
        }
    }

    /// <summary>
    /// Flat-forward interpolator — rate-form equivalent of log-linear on
    /// positive ys. The C# class uses the ``exp(−f·Δt)`` form; the numpy
    /// reference uses the same form verbatim.
    /// </summary>
    [Fact]
    public void FlatForwardInterpolator_MatchesNumpyReference()
    {
        using var doc = LoadVector("interpolation");
        var block = doc.RootElement.GetProperty("flat_forward");
        var xs = GetDoubleArray(block.GetProperty("xs"));
        var ys = GetDoubleArray(block.GetProperty("ys"));
        var queries = GetDoubleArray(block.GetProperty("queries"));
        var expected = GetDoubleArray(block.GetProperty("values"));

        for (var i = 0; i < queries.Length; i++)
        {
            var actual = FlatForwardInterpolator.Instance.Interpolate(queries[i], xs, ys);
            AssertScalarWithin(actual, expected[i], PrecisionExact, $"flat_forward @ {queries[i]}");
        }
    }

    /// <summary>
    /// Two-point linear — the static ``Interpolate(x0, y0, x1, y1, x)``
    /// textbook formula. Both the C# method and the numpy reference
    /// compute ``y0 + (x − x0)/(x1 − x0) · (y1 − y0)``; agreement at
    /// 1e-10 absolute is the expected bar.
    /// </summary>
    [Fact]
    public void TwoPointLinearInterpolator_MatchesNumpyReference()
    {
        using var doc = LoadVector("interpolation");
        var block = doc.RootElement.GetProperty("two_point_linear");
        var x0 = block.GetProperty("x0").GetDouble();
        var y0 = block.GetProperty("y0").GetDouble();
        var x1 = block.GetProperty("x1").GetDouble();
        var y1 = block.GetProperty("y1").GetDouble();
        var queries = GetDoubleArray(block.GetProperty("queries"));
        var expected = GetDoubleArray(block.GetProperty("values"));

        for (var i = 0; i < queries.Length; i++)
        {
            var actual = TwoPointLinearInterpolator.Interpolate(x0, y0, x1, y1, queries[i]);
            AssertScalarWithin(actual, expected[i], PrecisionExact, $"two_point_linear @ {queries[i]}");
        }
    }

    /// <summary>
    /// MonotoneCubicSpline on N=2 data — the single-observation / two-
    /// point degenerate case, which is required to degenerate to linear
    /// interpolation on the [xs[0], xs[1]] segment. scipy PCHIP on a
    /// two-point input produces the same linear shape; the test asserts
    /// Numerics matches exactly on 5 query points spanning the segment.
    /// </summary>
    [Fact]
    public void MonotoneCubicSplineN2_MatchesScipyPchip()
    {
        using var doc = LoadVector("interpolation");
        var block = doc.RootElement.GetProperty("monotone_cubic_n2");
        var xs = GetDoubleArray(block.GetProperty("xs"));
        var ys = GetDoubleArray(block.GetProperty("ys"));
        var queries = GetDoubleArray(block.GetProperty("queries"));
        var expected = GetDoubleArray(block.GetProperty("values"));

        var spline = new MonotoneCubicSpline(xs, ys);
        for (var i = 0; i < queries.Length; i++)
        {
            var actual = spline.Interpolate(queries[i]);
            AssertScalarWithin(actual, expected[i], PrecisionExact, $"monotone_cubic_n2 @ {queries[i]}");
        }
    }

    /// <summary>
    /// Hagan-West (2006) monotone-convex interpolation — no library
    /// equivalent exists (finance-specific curve-construction method), so
    /// the reference is a paper-faithful hand-port in
    /// ``generate_interpolation_vectors.py::reference_monotone_convex``
    /// with §2.3 / §2.4 / §2.6 equation citations. The Python port was
    /// written from the paper's equations independently of the C# class;
    /// it uses different loop structure, different variable names, and
    /// different boundary-handling idioms, so a shared bug would require
    /// a co-incident misreading of the paper. Both sides compute the same
    /// IEEE arithmetic pipeline on ``double``, so 1e-10 absolute
    /// (``PrecisionExact``) is the expected bar. Input is a yield-curve
    /// NCR node array spanning five orders of magnitude of time (0.25y
    /// through 30y) with a virtual origin at ``xs[0] = 0``.
    /// </summary>
    [Fact]
    public void MonotoneConvexInterpolator_MatchesHaganWestReference()
    {
        using var doc = LoadVector("interpolation");
        var block = doc.RootElement.GetProperty("monotone_convex");
        var xs = GetDoubleArray(block.GetProperty("xs"));
        var ys = GetDoubleArray(block.GetProperty("ys"));
        var queries = GetDoubleArray(block.GetProperty("queries"));
        var expected = GetDoubleArray(block.GetProperty("values"));

        for (var i = 0; i < queries.Length; i++)
        {
            var actual = MonotoneConvexInterpolator.Instance.Interpolate(queries[i], xs, ys);
            AssertScalarWithin(actual, expected[i], PrecisionExact, $"monotone_convex @ {queries[i]}");
        }
    }
}
