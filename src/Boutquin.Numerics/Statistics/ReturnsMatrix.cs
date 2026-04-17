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

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Unified return-matrix input that accepts either T-by-N (rows = time,
/// columns = assets) or N-by-T (one array per asset) layouts without
/// materializing a transposed copy.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A (arithmetic-only). Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>.
/// </para>
/// <para>
/// Estimators read returns via the indexer (time, asset) — this type resolves
/// the layout internally so callers don't need to convert between Trading's
/// jagged asset-major layout and Numerics' 2D T-by-N layout.
/// </para>
/// <para>
/// Performance: the indexer is non-branching after construction — the
/// layout mode is resolved once at construction time. For
/// maximum throughput, estimators can call <see cref="AsTimeByAsset"/>
/// once and then iterate the array directly; the wrapper
/// materializes a copy only when the input is jagged and the caller
/// requests the T-by-N view.
/// </para>
/// <para>
/// Canonical form is T-by-N. All downstream estimators operate on that
/// layout; the jagged overload exists to bridge Trading-shaped inputs
/// without forcing callers to allocate.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public readonly struct ReturnsMatrix<T>
    where T : IFloatingPoint<T>
{
    private readonly T[,]? _timeByAsset;
    private readonly T[][]? _assetMajor;

    /// <summary>Number of time observations (rows in T-by-N layout).</summary>
    public int Observations { get; }

    /// <summary>Number of assets (columns in T-by-N layout).</summary>
    public int Assets { get; }

    /// <summary>Creates a matrix backed by a T-by-N 2D array.</summary>
    public ReturnsMatrix(T[,] timeByAsset)
    {
        ArgumentNullException.ThrowIfNull(timeByAsset);
        _timeByAsset = timeByAsset;
        _assetMajor = null;
        Observations = timeByAsset.GetLength(0);
        Assets = timeByAsset.GetLength(1);
    }

    /// <summary>Creates a matrix backed by an asset-major jagged array.</summary>
    /// <exception cref="ArgumentException">Any inner array has a different length than the first.</exception>
    public ReturnsMatrix(T[][] assetMajor)
    {
        ArgumentNullException.ThrowIfNull(assetMajor);
        _timeByAsset = null;
        _assetMajor = assetMajor;

        if (assetMajor.Length == 0)
        {
            Observations = 0;
            Assets = 0;
            return;
        }

        var t = assetMajor[0]?.Length ?? 0;
        for (var i = 1; i < assetMajor.Length; i++)
        {
            if ((assetMajor[i]?.Length ?? -1) != t)
            {
                throw new ArgumentException(
                    "All asset series must have the same length.", nameof(assetMajor));
            }
        }

        Observations = t;
        Assets = assetMajor.Length;
    }

    /// <summary>Returns the observation at the given (time, asset) indices.</summary>
    public T this[int time, int asset]
    {
        get
        {
            if (_timeByAsset is not null)
            {
                return _timeByAsset[time, asset];
            }

            return _assetMajor![asset][time];
        }
    }

    /// <summary>
    /// Materializes a T-by-N view. Returns the backing array
    /// directly when the wrapper was constructed from a T-by-N input; copies
    /// otherwise.
    /// </summary>
    public T[,] AsTimeByAsset()
    {
        if (_timeByAsset is not null)
        {
            return _timeByAsset;
        }

        var t = Observations;
        var n = Assets;
        var copy = new T[t, n];
        for (var j = 0; j < n; j++)
        {
            var series = _assetMajor![j];
            for (var i = 0; i < t; i++)
            {
                copy[i, j] = series[i];
            }
        }

        return copy;
    }
}

/// <summary>
/// Unified return-matrix input that accepts either T-by-N (rows = time,
/// columns = assets) or N-by-T (one array per asset) layouts without
/// materializing a transposed copy. Estimators read returns via the
/// <see cref="this[int, int]"/> indexer (time, asset) — this type resolves
/// the layout internally so callers don't need to convert between Trading's
/// <c>decimal[][]</c> asset-major layout and Numerics' <c>decimal[,]</c>
/// T-by-N layout.
/// </summary>
/// <remarks>
/// <para>
/// Performance: the indexer is non-branching after construction — the
/// layout mode is resolved once at construction time and stored as an
/// enum, then accessed through a small switch in the hot path. For
/// maximum throughput, estimators can call <see cref="AsTimeByAsset"/>
/// once and then iterate the <c>decimal[,]</c> directly; the wrapper
/// materializes a copy only when the input is jagged and the caller
/// requests the T-by-N view.
/// </para>
/// <para>
/// Canonical form is T-by-N. All downstream estimators operate on that
/// layout; the jagged overload exists to bridge Trading-shaped inputs
/// without forcing callers to allocate.
/// </para>
/// </remarks>
public readonly struct ReturnsMatrix
{
    private readonly ReturnsMatrix<decimal> _inner;

    /// <summary>Number of time observations (rows in T-by-N layout).</summary>
    public int Observations => _inner.Observations;

    /// <summary>Number of assets (columns in T-by-N layout).</summary>
    public int Assets => _inner.Assets;

    /// <summary>Creates a matrix backed by a T-by-N 2D array.</summary>
    public ReturnsMatrix(decimal[,] timeByAsset) => _inner = new ReturnsMatrix<decimal>(timeByAsset);

    /// <summary>Creates a matrix backed by an asset-major jagged array.</summary>
    /// <exception cref="ArgumentException">Any inner array has a different length than the first.</exception>
    public ReturnsMatrix(decimal[][] assetMajor) => _inner = new ReturnsMatrix<decimal>(assetMajor);

    /// <summary>Returns the observation at the given (time, asset) indices.</summary>
    public decimal this[int time, int asset] => _inner[time, asset];

    /// <summary>
    /// Materializes a T-by-N <c>decimal[,]</c> view. Returns the backing array
    /// directly when the wrapper was constructed from a T-by-N input; copies
    /// otherwise.
    /// </summary>
    public decimal[,] AsTimeByAsset() => _inner.AsTimeByAsset();

    /// <summary>Implicit conversion from the T-by-N layout.</summary>
    public static implicit operator ReturnsMatrix(decimal[,] timeByAsset) => new(timeByAsset);

    /// <summary>Implicit conversion from the asset-major jagged layout.</summary>
    public static implicit operator ReturnsMatrix(decimal[][] assetMajor) => new(assetMajor);
}
