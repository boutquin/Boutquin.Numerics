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
using Boutquin.Numerics.LinearAlgebra;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Argument-guard regression tests from the 2026-06-20 code review. Each asserts that a
/// documented <see cref="ArgumentNullException"/>/<see cref="ArgumentException"/> contract is
/// honored rather than degrading to a raw <see cref="NullReferenceException"/> or silently
/// producing wrong results.
/// </summary>
public sealed class GuardTests
{
    // --- LinearAlgebra null guards (review findings #5, #6) ---

    [Fact]
    public void GaussianElimination_Solve_NullMatrix_ThrowsArgumentNull()
    {
        var b = new double[] { 1.0, 2.0 };
        var act = () => GaussianElimination<double>.Solve(null!, b);
        act.Should().Throw<ArgumentNullException>().WithParameterName("a");
    }

    [Fact]
    public void GaussianElimination_Solve_NullRhs_ThrowsArgumentNull()
    {
        var a = new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } };
        var act = () => GaussianElimination<double>.Solve(a, null!);
        act.Should().Throw<ArgumentNullException>().WithParameterName("b");
    }

    [Fact]
    public void JacobiEigenDecomposition_Decompose_NullMatrix_ThrowsArgumentNull()
    {
        var act = () => JacobiEigenDecomposition<double>.Decompose(null!);
        act.Should().Throw<ArgumentNullException>().WithParameterName("matrix");
    }

    // --- Interpolation null guards (review findings #7, #8) ---

    [Fact]
    public void CubicSplineInterpolator_Ctor_NullXs_ThrowsArgumentNull()
    {
        var ys = new double[] { 1.0, 2.0, 3.0 };
        var act = () => new CubicSplineInterpolator<double>(null!, ys, SplineBoundary.Natural);
        act.Should().Throw<ArgumentNullException>().WithParameterName("xs");
    }

    [Fact]
    public void CubicSplineInterpolator_Ctor_NullYs_ThrowsArgumentNull()
    {
        var xs = new double[] { 1.0, 2.0, 3.0 };
        var act = () => new CubicSplineInterpolator<double>(xs, null!, SplineBoundary.Natural);
        act.Should().Throw<ArgumentNullException>().WithParameterName("ys");
    }

    [Fact]
    public void MonotoneCubicSpline_Ctor_NullXs_ThrowsArgumentNull()
    {
        var ys = new double[] { 1.0, 2.0, 3.0 };
        var act = () => new MonotoneCubicSpline<double>(null!, ys);
        act.Should().Throw<ArgumentNullException>().WithParameterName("xs");
    }

    [Fact]
    public void MonotoneCubicSpline_Ctor_NullYs_ThrowsArgumentNull()
    {
        var xs = new double[] { 1.0, 2.0, 3.0 };
        var act = () => new MonotoneCubicSpline<double>(xs, null!);
        act.Should().Throw<ArgumentNullException>().WithParameterName("ys");
    }

    // --- MonotoneCubicInterpolator unsorted-xs validation (review finding #9) ---

    [Fact]
    public void MonotoneCubicInterpolator_UnsortedXs_ThrowsArgumentException()
    {
        var xs = new double[] { 0.0, 2.0, 1.0, 3.0 };
        var ys = new double[] { 0.0, 1.0, 2.0, 3.0 };
        var act = () => MonotoneCubicInterpolator<double>.Instance.Interpolate(1.5, xs, ys);
        act.Should().Throw<ArgumentException>();
    }
}
