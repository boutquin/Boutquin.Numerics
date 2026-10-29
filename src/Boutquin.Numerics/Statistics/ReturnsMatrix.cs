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
/// exposing caller-owned arrays.
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
/// Performance: the indexer performs a layout null check on each access. For
/// maximum throughput, estimators can call <see cref="AsTimeByAsset"/>
/// once and then iterate the array directly; the wrapper
/// materializes a copy for construction and for the requested T-by-N view.
/// </para>
/// <para>
/// Canonical form is T-by-N. All downstream estimators operate on that
/// layout; the jagged overload exists to bridge Trading-shaped inputs.
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

        if (timeByAsset.GetLength(1) == 0)
        {
            throw new ArgumentException(
                "Returns matrix must contain at least one asset series.", nameof(timeByAsset));
        }

        _timeByAsset = (T[,])timeByAsset.Clone();
        _assetMajor = null;
        Observations = timeByAsset.GetLength(0);
        Assets = timeByAsset.GetLength(1);
    }

    /// <summary>Creates a matrix backed by an asset-major jagged array.</summary>
    /// <exception cref="ArgumentException">Any inner array has a different length than the first.</exception>
    public ReturnsMatrix(T[][] assetMajor)
    {
        ArgumentNullException.ThrowIfNull(assetMajor);

        if (assetMajor.Length == 0)
        {
            throw new ArgumentException(
                "Returns matrix must contain at least one asset series.", nameof(assetMajor));
        }

        if (assetMajor[0] is null)
        {
            throw new ArgumentException("assetMajor[0] is null.", nameof(assetMajor));
        }

        var t = assetMajor[0].Length;
        for (var i = 1; i < assetMajor.Length; i++)
        {
            if (assetMajor[i] is null)
            {
                throw new ArgumentException($"assetMajor[{i}] is null.", nameof(assetMajor));
            }

            if (assetMajor[i].Length != t)
            {
                throw new ArgumentException(
                    "All asset series must have the same length.", nameof(assetMajor));
            }
        }

        _timeByAsset = null;
        _assetMajor = new T[assetMajor.Length][];
        for (var i = 0; i < assetMajor.Length; i++)
        {
            _assetMajor[i] = (T[])assetMajor[i].Clone();
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
    /// Materializes a T-by-N copy.
    /// </summary>
    public T[,] AsTimeByAsset()
    {
        if (_timeByAsset is not null)
        {
            return (T[,])_timeByAsset.Clone();
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
/// exposing caller-owned arrays. Estimators read returns via the
/// <see cref="this[int, int]"/> indexer (time, asset) — this type resolves
/// the layout internally so callers don't need to convert between Trading's
/// <c>decimal[][]</c> asset-major layout and Numerics' <c>decimal[,]</c>
/// T-by-N layout.
/// </summary>
/// <remarks>
/// <para>
/// Performance: the indexer dispatches via a null-check on the backing
/// <c>decimal[,]</c> field — the layout is resolved once at construction time
/// and accessed in the hot path without a switch or enum. For
/// maximum throughput, estimators can call <see cref="AsTimeByAsset"/>
/// once and then iterate the <c>decimal[,]</c> directly; the wrapper
/// materializes a copy for construction and for the requested T-by-N view.
/// </para>
/// <para>
/// Canonical form is T-by-N. All downstream estimators operate on that
/// layout; the jagged overload exists to bridge Trading-shaped inputs.
/// </para>
/// </remarks>
[Obsolete(
    "Use ReturnsMatrix<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
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
    /// Materializes a T-by-N <c>decimal[,]</c> copy.
    /// </summary>
    public decimal[,] AsTimeByAsset() => _inner.AsTimeByAsset();

    /// <summary>Implicit conversion from the T-by-N layout.</summary>
    public static implicit operator ReturnsMatrix(decimal[,] timeByAsset) => new(timeByAsset);

    /// <summary>Implicit conversion from the asset-major jagged layout.</summary>
    public static implicit operator ReturnsMatrix(decimal[][] assetMajor) => new(assetMajor);
}
