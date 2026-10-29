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

namespace Boutquin.Numerics.Solvers;

/// <summary>
/// Fills <paramref name="residuals"/> in-place given the current parameter vector
/// <paramref name="parameters"/>. Used with the buffer-pooled
/// <see cref="LevenbergMarquardtSolver{T}.Solve(LmResidualWriter{T}, ReadOnlySpan{T}, LevenbergMarquardtBuffers{T}, LmJacobianWriter{T}?, ReadOnlySpan{T}, ReadOnlySpan{T})"/>
/// overload to eliminate per-iteration heap allocation from the callback itself.
/// </summary>
/// <typeparam name="T">IEEE 754 floating-point type.</typeparam>
/// <param name="parameters">Current parameter vector (read-only).</param>
/// <param name="residuals">
/// Pre-allocated output buffer. The solver prefills it with NaN before every call.
/// The writer must replace every element with a finite residual; an unwritten or
/// non-finite entry causes <see cref="InvalidOperationException"/>. Span bounds checks
/// reject writes beyond the buffer.
/// </param>
public delegate void LmResidualWriter<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
    where T : IFloatingPointIeee754<T>;

/// <summary>
/// Fills <paramref name="jacobian"/> (m rows × n columns, row-major flat span,
/// stride = parameterCount) in-place given the current parameter vector.
/// Row <c>i</c>, column <c>j</c> is at <c>jacobian[i * parameterCount + j]</c>.
/// </summary>
/// <typeparam name="T">IEEE 754 floating-point type.</typeparam>
/// <param name="parameters">Current parameter vector (read-only).</param>
/// <param name="jacobian">
/// Pre-allocated flat output buffer of length <c>residualCount × parameterCount</c>.
/// Layout is row-major: element (i, j) is at index <c>i * parameterCount + j</c>.
/// The solver prefills the span with NaN. Every entry must be overwritten with a finite
/// derivative; otherwise the solver throws <see cref="InvalidOperationException"/>.
/// </param>
public delegate void LmJacobianWriter<T>(ReadOnlySpan<T> parameters, Span<T> jacobian)
    where T : IFloatingPointIeee754<T>;
