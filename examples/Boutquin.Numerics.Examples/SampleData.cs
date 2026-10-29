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

namespace Boutquin.Numerics.Examples;

internal static class SampleData
{
    // ── Single-asset returns ──────────────────────────────────────────────────

    internal static readonly decimal[] s_returns = GenerateReturns(60, seed: 1u);
    internal static readonly double[] s_returnsDouble = Array.ConvertAll(s_returns, v => (double)v);
    internal static readonly decimal[] s_residuals = GenerateReturns(60, seed: 7u);

    // ── Multi-asset ───────────────────────────────────────────────────────────

    internal static readonly decimal[,] s_assetReturns = GenerateAssetReturns(60, 4, seed: 2u);
    internal static readonly decimal[][] s_assetMajorReturns = ToAssetMajor(s_assetReturns);

    // ── Portfolio covariance ──────────────────────────────────────────────────

    internal static readonly decimal[,] s_covariance4 = new decimal[4, 4]
    {
        { 0.0004m, 0.0002m, 0.0001m, 0.0001m },
        { 0.0002m, 0.0003m, 0.0001m, 0.0000m },
        { 0.0001m, 0.0001m, 0.0005m, 0.0002m },
        { 0.0001m, 0.0000m, 0.0002m, 0.0006m },
    };

    internal static readonly decimal[] s_expectedReturns4 = [0.0005m, 0.0004m, 0.0006m, 0.0003m];

    internal static readonly double[,] s_covariance4D = Convert2d(s_covariance4);
    internal static readonly double[] s_expectedReturns4D = Array.ConvertAll(s_expectedReturns4, v => (double)v);

    // ── Correlation series ────────────────────────────────────────────────────

    internal static readonly decimal[] s_seriesA = GenerateReturns(50, seed: 3u);
    internal static readonly decimal[] s_seriesB = GenerateCorrelated(s_seriesA, rho: 0.7m, seed: 5u);

    // ── PBO strategy panel ────────────────────────────────────────────────────

    internal static readonly decimal[,] s_strategyPanel = GenerateStrategyPanel(8, 128, seed: 6u);

    // ── NearestPsdProjection inputs ───────────────────────────────────────────

    // Symmetric matrix with negative determinant — not PSD.
    internal static readonly decimal[,] s_maybeIndefinite = new decimal[3, 3]
    {
        { 1.00m, 0.90m, 0.70m },
        { 0.90m, 1.00m, 0.30m },
        { 0.70m, 0.30m, 1.00m },
    };

    // Near-correlation matrix — positive definite but not unit-diagonal.
    internal static readonly decimal[,] s_messyCorr = new decimal[3, 3]
    {
        { 1.00m, 0.85m, 0.60m },
        { 0.85m, 1.00m, 0.45m },
        { 0.60m, 0.45m, 1.03m },
    };

    // ── NIST Longley dataset ──────────────────────────────────────────────────

    internal static readonly double[,] s_longleyX = new double[16, 6]
    {
        {  83.0, 234289, 2356, 1590, 107608, 1947 },
        {  88.5, 259426, 2325, 1456, 108632, 1948 },
        {  88.2, 258054, 3682, 1616, 109773, 1949 },
        {  89.5, 284599, 3351, 1650, 110929, 1950 },
        {  96.2, 328975, 2099, 3099, 112075, 1951 },
        {  98.1, 346999, 1932, 3594, 113270, 1952 },
        {  99.0, 365385, 1870, 3547, 115094, 1953 },
        { 100.0, 363112, 3578, 3350, 116219, 1954 },
        { 101.2, 397469, 2904, 3048, 117388, 1955 },
        { 104.6, 419180, 2822, 2857, 118734, 1956 },
        { 108.4, 442769, 2936, 2798, 120445, 1957 },
        { 110.8, 444546, 4681, 2637, 121950, 1958 },
        { 112.6, 482704, 3813, 2552, 123366, 1959 },
        { 114.2, 502601, 3931, 2514, 125368, 1960 },
        { 115.7, 518173, 4806, 2572, 127852, 1961 },
        { 116.9, 554594, 4007, 2827, 130081, 1962 },
    };

    internal static readonly double[] s_longleyY =
        [60323, 61122, 60171, 61187, 63221, 63639, 64989, 63761,
          66019, 67857, 68169, 66513, 68655, 69564, 69331, 70551];

    // ── Levenberg-Marquardt calibration data ──────────────────────────────────

    // y = 2.5 * exp(0.3 * x), x = 0..9
    internal static readonly double[] s_lmXs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    internal static readonly double[] s_lmYs = BuildLmYs(s_lmXs, a: 2.5, b: 0.3);

    // ── Wampler-5 polynomial design ───────────────────────────────────────────

    // y = 1 + 2x + 3x² + 4x³ + 5x⁴ + 6x⁵, x = 0..20
    // X = [x, x², x³, x⁴, x⁵] — intercept handled by includeIntercept = true.
    internal static readonly decimal[,] s_wampler5X = BuildWampler5X();
    internal static readonly decimal[] s_wampler5Y = BuildWampler5Y();

    // ── Private helpers ───────────────────────────────────────────────────────

    private static decimal[] GenerateReturns(int n, uint seed)
    {
        var r = new decimal[n];
        uint s = seed;
        for (var i = 0; i < n; i++)
        {
            s = unchecked(s * 1664525u + 1013904223u);
            double d = (s >> 1) / (double)(1u << 31) - 0.5;
            r[i] = Math.Round((decimal)(d * 0.02), 6);
        }
        return r;
    }

    private static decimal[,] GenerateAssetReturns(int periods, int assets, uint seed)
    {
        var mat = new decimal[periods, assets];
        uint s = seed;
        for (var t = 0; t < periods; t++)
        {
            for (var a = 0; a < assets; a++)
            {
                s = unchecked(s * 1664525u + 1013904223u);
                double d = (s >> 1) / (double)(1u << 31) - 0.5;
                mat[t, a] = Math.Round((decimal)(d * 0.02), 6);
            }
        }
        return mat;
    }

    private static decimal[][] ToAssetMajor(decimal[,] timeByAsset)
    {
        int periods = timeByAsset.GetLength(0);
        int assets = timeByAsset.GetLength(1);
        var result = new decimal[assets][];
        for (var a = 0; a < assets; a++)
        {
            result[a] = new decimal[periods];
            for (var t = 0; t < periods; t++)
            {
                result[a][t] = timeByAsset[t, a];
            }
        }
        return result;
    }

    private static decimal[] GenerateCorrelated(decimal[] x, decimal rho, uint seed)
    {
        var r = new decimal[x.Length];
        uint s = seed;
        decimal sqrtOneMinusRhoSq = (decimal)Math.Sqrt(1.0 - (double)(rho * rho));
        for (var i = 0; i < x.Length; i++)
        {
            s = unchecked(s * 1664525u + 1013904223u);
            double d = (s >> 1) / (double)(1u << 31) - 0.5;
            decimal noise = Math.Round((decimal)(d * 0.02), 6);
            r[i] = rho * x[i] + sqrtOneMinusRhoSq * noise;
        }
        return r;
    }

    private static decimal[,] GenerateStrategyPanel(int strategies, int periods, uint seed)
    {
        var panel = new decimal[strategies, periods];
        uint s = seed;
        for (var strat = 0; strat < strategies; strat++)
        {
            decimal drift = (decimal)(strat - strategies / 2.0) * 0.0001m;
            for (var t = 0; t < periods; t++)
            {
                s = unchecked(s * 1664525u + 1013904223u);
                double d = (s >> 1) / (double)(1u << 31) - 0.5;
                panel[strat, t] = drift + Math.Round((decimal)(d * 0.015), 6);
            }
        }
        return panel;
    }

    private static double[,] Convert2d(decimal[,] src)
    {
        int rows = src.GetLength(0), cols = src.GetLength(1);
        var dst = new double[rows, cols];
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < cols; j++)
            {
                dst[i, j] = (double)src[i, j];
            }
        }

        return dst;
    }

    private static double[] BuildLmYs(double[] xs, double a, double b)
    {
        var ys = new double[xs.Length];
        for (var i = 0; i < xs.Length; i++)
        {
            ys[i] = a * Math.Exp(b * xs[i]);
        }

        return ys;
    }

    private static decimal[,] BuildWampler5X()
    {
        var mat = new decimal[21, 5];
        for (var i = 0; i <= 20; i++)
        {
            decimal xi = i;
            mat[i, 0] = xi;
            mat[i, 1] = xi * xi;
            mat[i, 2] = xi * xi * xi;
            mat[i, 3] = xi * xi * xi * xi;
            mat[i, 4] = xi * xi * xi * xi * xi;
        }
        return mat;
    }

    private static decimal[] BuildWampler5Y()
    {
        var y = new decimal[21];
        for (var i = 0; i <= 20; i++)
        {
            decimal xi = i;
            y[i] = 1 + 2 * xi + 3 * xi * xi + 4 * xi * xi * xi
                   + 5 * xi * xi * xi * xi + 6 * xi * xi * xi * xi * xi;
        }
        return y;
    }
}
