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

using Boutquin.Numerics.LinearAlgebra;

namespace Boutquin.Numerics.Solvers;

/// <summary>
/// Generic active-set QP solver for two standard portfolio optimization problems:
/// <list type="bullet">
///   <item><description><b>MinVar:</b>  min w′Σw  s.t. 1′w=1, lb ≤ w ≤ ub</description></item>
///   <item><description><b>MeanVar:</b> max w′μ − (λ/2)w′Σw  s.t. 1′w=1, lb ≤ w ≤ ub</description></item>
/// </list>
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+√. Works for any <typeparamref name="T"/> implementing
/// <see cref="IFloatingPoint{TSelf}"/>. Uses <see cref="CholeskyDecomposition{T}"/>
/// internally which calls <see cref="Boutquin.Numerics.Internal.NumericPrecision{T}.Sqrt"/>.
/// </para>
/// <para>
/// The active-set method iteratively:
/// <list type="number">
///   <item><description>Solves the unconstrained (sum=1 only) reduced problem via Cholesky.</description></item>
///   <item><description>Fixes the most-violated bound constraint (one per iteration).</description></item>
///   <item><description>Checks KKT conditions to release constraints that are no longer active.</description></item>
///   <item><description>Terminates in at most 2N iterations for MinVar and 3N+3 for MeanVar.</description></item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type (e.g. <c>double</c>, <c>decimal</c>).</typeparam>
public static class ActiveSetQpSolver<T>
    where T : IFloatingPoint<T>
{
    // Degeneracy tolerance: Σ⁻¹·1 sums to zero guard.
    private static readonly T s_epsilon20 = T.CreateChecked(1e-20);

    // KKT tolerance: threshold for releasing a fixed variable.
    private static readonly T s_epsilon10 = T.CreateChecked(1e-10);

    // ─── Public QP solvers ─────────────────────────────────────────────

    /// <summary>
    /// Solves the minimum-variance QP: min w′Σw s.t. 1′w=1, minWeight ≤ w_i ≤ maxWeight.
    /// </summary>
    /// <param name="covariance">Symmetric positive-definite N×N covariance matrix.</param>
    /// <param name="minWeight">Lower weight bound (per asset).</param>
    /// <param name="maxWeight">Upper weight bound (per asset).</param>
    /// <returns>Optimal weight vector summing to 1.</returns>
    /// <exception cref="ArgumentException">Matrix is not square or has zero dimension.</exception>
    /// <exception cref="InvalidOperationException">Covariance is degenerate (Σ⁻¹·1 sums to zero).</exception>
    public static T[] SolveMinVariance(T[,] covariance, T minWeight, T maxWeight)
    {
        var n = covariance.GetLength(0);
        if (covariance.GetLength(1) != n)
        {
            throw new ArgumentException("Covariance matrix must be square.", nameof(covariance));
        }

        if (n == 1)
        {
            return [T.One];
        }

        var invN = T.One / T.CreateChecked(n);
        maxWeight = T.Max(maxWeight, invN);
        minWeight = T.Min(minWeight, invN);

        // 0 = free, -1 = fixed at lower bound, +1 = fixed at upper bound.
        var status = new int[n];

        for (var iter = 0; iter < 2 * n; iter++)
        {
            var (freeIndices, fixedSum) = CollectFreeIndices(status, n, minWeight, maxWeight);

            var nFree = freeIndices.Count;
            if (nFree == 0)
            {
                return EqualWeightFallback(n);
            }

            var remainingSum = T.One - fixedSum;
            var covFree = ExtractSubmatrix(covariance, freeIndices);
            var choleskyL = CholeskyDecomposition<T>.Decompose(covFree);

            var ones = FilledArray(nFree, T.One);
            var z = CholeskyDecomposition<T>.Solve(choleskyL, ones);

            var sumZ = Sum(z);
            if (T.Abs(sumZ) < s_epsilon20)
            {
                throw new InvalidOperationException("Degenerate covariance: Σ⁻¹·1 sums to zero.");
            }

            // Adjust for cross-covariance linear terms from fixed variables.
            // Full KKT: Σ_sub·w_free = (ν/2)·1 − Σ_cross^T·w_fixed
            // Solve Σ_sub·zc = crossTerms, then w_free = cScale·z − zc.
            var crossTerms = new T[nFree];
            for (var fi = 0; fi < nFree; fi++)
            {
                var ii = freeIndices[fi];
                for (var j = 0; j < n; j++)
                {
                    if (status[j] == -1)
                    {
                        crossTerms[fi] += covariance[ii, j] * minWeight;
                    }
                    else if (status[j] == 1)
                    {
                        crossTerms[fi] += covariance[ii, j] * maxWeight;
                    }
                }
            }

            var zc = CholeskyDecomposition<T>.Solve(choleskyL, crossTerms);
            var sumZc = Sum(zc);
            var cScale = (remainingSum + sumZc) / sumZ;
            var wFree = new T[nFree];
            for (var i = 0; i < nFree; i++)
            {
                wFree[i] = cScale * z[i] - zc[i];
            }

            var (worstIdx, worstDir) = FindWorstViolation(wFree, freeIndices, minWeight, maxWeight);
            if (worstIdx >= 0)
            {
                status[worstIdx] = worstDir;
                continue;
            }

            // All free weights feasible — build full solution.
            var w = BuildFullWeights(status, n, wFree, freeIndices, minWeight, maxWeight);

            // KKT check: release any over-constrained variable.
            if (!TryReleaseMinVar(covariance, w, n, status))
            {
                return w;
            }
        }

        return EqualWeightFallback(n);
    }

    /// <summary>
    /// Solves the mean-variance QP: max w′μ − (λ/2)w′Σw s.t. 1′w=1, minWeight ≤ w_i ≤ maxWeight.
    /// </summary>
    /// <param name="covariance">Symmetric positive-definite N×N covariance matrix.</param>
    /// <param name="means">Expected return vector of length N.</param>
    /// <param name="riskAversion">Risk-aversion parameter λ (λ = 0 reduces to pure LP).</param>
    /// <param name="minWeight">Lower weight bound (per asset).</param>
    /// <param name="maxWeight">Upper weight bound (per asset).</param>
    /// <returns>Optimal weight vector summing to 1.</returns>
    /// <exception cref="ArgumentException">Matrix is not square or dimensions mismatch.</exception>
    /// <exception cref="InvalidOperationException">Problem did not converge or covariance is degenerate.</exception>
    public static T[] SolveMeanVariance(
        T[,] covariance, T[] means, T riskAversion, T minWeight, T maxWeight)
    {
        var n = covariance.GetLength(0);
        if (covariance.GetLength(1) != n)
        {
            throw new ArgumentException("Covariance matrix must be square.", nameof(covariance));
        }

        if (means.Length != n)
        {
            throw new ArgumentException("Means vector length must match covariance dimension.", nameof(means));
        }

        if (n == 1)
        {
            return [T.One];
        }

        if (riskAversion == T.Zero)
        {
            return SolveMaxReturnLP(means, n, minWeight, maxWeight);
        }

        var invN = T.One / T.CreateChecked(n);
        maxWeight = T.Max(maxWeight, invN);
        minWeight = T.Min(minWeight, invN);

        var status = new int[n];

        for (var iter = 0; iter < 3 * n + 3; iter++)
        {
            var (freeIndices, fixedSum) = CollectFreeIndices(status, n, minWeight, maxWeight);

            var nFree = freeIndices.Count;
            if (nFree == 0)
            {
                // All variables fixed: normalise to sum=1.
                var fixedW = new T[n];
                var fixedTotal = T.Zero;
                for (var i = 0; i < n; i++)
                {
                    fixedW[i] = status[i] == -1 ? minWeight : maxWeight;
                    fixedTotal += fixedW[i];
                }

                if (fixedTotal > T.Zero)
                {
                    for (var i = 0; i < n; i++)
                    {
                        fixedW[i] /= fixedTotal;
                    }
                }

                return fixedW;
            }

            var remainingSum = T.One - fixedSum;
            var covFree = ExtractSubmatrix(covariance, freeIndices);

            // Adjust means for cross-covariance with fixed variables.
            var meansFree = new T[nFree];
            for (var fi = 0; fi < nFree; fi++)
            {
                meansFree[fi] = means[freeIndices[fi]];
                var i = freeIndices[fi];
                for (var j = 0; j < n; j++)
                {
                    if (status[j] == -1)
                    {
                        meansFree[fi] -= riskAversion * covariance[i, j] * minWeight;
                    }
                    else if (status[j] == 1)
                    {
                        meansFree[fi] -= riskAversion * covariance[i, j] * maxWeight;
                    }
                }
            }

            var choleskyL = CholeskyDecomposition<T>.Decompose(covFree);
            var onesFree = FilledArray(nFree, T.One);

            var a = CholeskyDecomposition<T>.Solve(choleskyL, onesFree);
            var b = CholeskyDecomposition<T>.Solve(choleskyL, meansFree);

            var sumA = Sum(a);
            var sumB = Sum(b);

            if (T.Abs(sumA) < s_epsilon20)
            {
                throw new InvalidOperationException("Degenerate covariance: Σ⁻¹·1 sums to zero.");
            }

            var nu = (sumB / riskAversion - remainingSum) / (sumA / riskAversion);

            var wFree = new T[nFree];
            for (var i = 0; i < nFree; i++)
            {
                wFree[i] = (b[i] - nu * a[i]) / riskAversion;
            }

            var (worstIdx, worstDir) = FindWorstViolation(wFree, freeIndices, minWeight, maxWeight);
            if (worstIdx >= 0)
            {
                status[worstIdx] = worstDir;
                continue;
            }

            var w = BuildFullWeights(status, n, wFree, freeIndices, minWeight, maxWeight);

            if (!TryReleaseMeanVar(covariance, means, w, n, riskAversion, status))
            {
                return w;
            }
        }

        throw new InvalidOperationException(
            "MeanVariance active-set did not converge within iteration limit.");
    }

    // ─── Private helpers ───────────────────────────────────────────────

    /// <summary>
    /// Solves the pure max-return LP: maximize w′μ s.t. 1′w=1, minWeight ≤ w_i ≤ maxWeight.
    /// Greedy: assign maxWeight to assets in descending return order until budget exhausted.
    /// </summary>
    private static T[] SolveMaxReturnLP(T[] means, int n, T minWeight, T maxWeight)
    {
        var weights = new T[n];
        var remaining = T.One - T.CreateChecked(n) * minWeight;
        for (var i = 0; i < n; i++)
        {
            weights[i] = minWeight;
        }

        var indices = Enumerable.Range(0, n)
            .OrderByDescending(i => means[i])
            .ToArray();

        foreach (var i in indices)
        {
            if (remaining <= T.Zero)
            {
                break;
            }

            var add = T.Min(maxWeight - minWeight, remaining);
            weights[i] += add;
            remaining -= add;
        }

        return weights;
    }

    private static (List<int> FreeIndices, T FixedSum) CollectFreeIndices(
        int[] status, int n, T minWeight, T maxWeight)
    {
        var freeIndices = new List<int>(n);
        var fixedSum = T.Zero;
        for (var i = 0; i < n; i++)
        {
            switch (status[i])
            {
                case -1:
                    fixedSum += minWeight;
                    break;
                case 1:
                    fixedSum += maxWeight;
                    break;
                default:
                    freeIndices.Add(i);
                    break;
            }
        }

        return (freeIndices, fixedSum);
    }

    private static T[,] ExtractSubmatrix(T[,] full, List<int> indices)
    {
        var m = indices.Count;
        var sub = new T[m, m];
        for (var i = 0; i < m; i++)
        {
            for (var j = 0; j < m; j++)
            {
                sub[i, j] = full[indices[i], indices[j]];
            }
        }

        return sub;
    }

    private static T[] FilledArray(int length, T value)
    {
        var arr = new T[length];
        Array.Fill(arr, value);
        return arr;
    }

    private static T Sum(T[] arr)
    {
        var s = T.Zero;
        foreach (var v in arr)
        {
            s += v;
        }

        return s;
    }

    private static (int WorstIdx, int WorstDir) FindWorstViolation(
        T[] wFree, List<int> freeIndices, T minWeight, T maxWeight)
    {
        var worstIdx = -1;
        var worstViolation = T.Zero;
        var worstDir = 0;

        for (var fi = 0; fi < wFree.Length; fi++)
        {
            if (wFree[fi] < minWeight)
            {
                var violation = minWeight - wFree[fi];
                if (violation > worstViolation)
                {
                    worstViolation = violation;
                    worstIdx = freeIndices[fi];
                    worstDir = -1;
                }
            }
            else if (wFree[fi] > maxWeight)
            {
                var violation = wFree[fi] - maxWeight;
                if (violation > worstViolation)
                {
                    worstViolation = violation;
                    worstIdx = freeIndices[fi];
                    worstDir = 1;
                }
            }
        }

        return (worstIdx, worstDir);
    }

    private static T[] BuildFullWeights(
        int[] status, int n, T[] wFree, List<int> freeIndices, T minWeight, T maxWeight)
    {
        var w = new T[n];
        for (var i = 0; i < n; i++)
        {
            w[i] = status[i] switch
            {
                -1 => minWeight,
                1 => maxWeight,
                _ => T.Zero,
            };
        }

        for (var fi = 0; fi < freeIndices.Count; fi++)
        {
            w[freeIndices[fi]] = wFree[fi];
        }

        return w;
    }

    private static T[] EqualWeightFallback(int n)
    {
        var w = new T[n];
        var invN = T.One / T.CreateChecked(n);
        Array.Fill(w, invN);
        return w;
    }

    /// <summary>
    /// KKT check for MinVar: tries to release one fixed variable.
    /// At optimality for free variables: (Σw)_i = ν for all free i.
    /// At lower bound: (Σw)_i ≥ ν (releasing would NOT reduce variance).
    /// At upper bound: (Σw)_i ≤ ν (releasing would NOT reduce variance).
    /// </summary>
    private static bool TryReleaseMinVar(T[,] cov, T[] w, int n, int[] status)
    {
        var grad = new T[n];
        for (var i = 0; i < n; i++)
        {
            grad[i] = T.Zero;
            for (var j = 0; j < n; j++)
            {
                grad[i] += cov[i, j] * w[j];
            }
        }

        var nu = T.Zero;
        var nFree = 0;
        for (var i = 0; i < n; i++)
        {
            if (status[i] == 0)
            {
                nu += grad[i];
                nFree++;
            }
        }

        if (nFree == 0)
        {
            return false;
        }

        nu /= T.CreateChecked(nFree);

        var worstIdx = -1;
        var worstViolation = T.Zero;

        for (var i = 0; i < n; i++)
        {
            if (status[i] == -1 && grad[i] < nu - s_epsilon10)
            {
                var violation = nu - grad[i];
                if (violation > worstViolation)
                {
                    worstViolation = violation;
                    worstIdx = i;
                }
            }
            else if (status[i] == 1 && grad[i] > nu + s_epsilon10)
            {
                var violation = grad[i] - nu;
                if (violation > worstViolation)
                {
                    worstViolation = violation;
                    worstIdx = i;
                }
            }
        }

        if (worstIdx < 0)
        {
            return false;
        }

        status[worstIdx] = 0;
        return true;
    }

    /// <summary>
    /// KKT check for MeanVar: tries to release one fixed variable.
    /// Objective gradient: grad_i = μ_i − λ(Σw)_i.
    /// At optimality for free variables: grad_i = ν for all free i.
    /// At lower bound: grad_i ≤ ν (can't improve by increasing).
    /// At upper bound: grad_i ≥ ν (can't improve by decreasing).
    /// </summary>
    private static bool TryReleaseMeanVar(
        T[,] cov, T[] means, T[] w, int n, T riskAversion, int[] status)
    {
        var grad = new T[n];
        for (var i = 0; i < n; i++)
        {
            var covW = T.Zero;
            for (var j = 0; j < n; j++)
            {
                covW += cov[i, j] * w[j];
            }

            grad[i] = means[i] - riskAversion * covW;
        }

        var nu = T.Zero;
        var nFree = 0;
        for (var i = 0; i < n; i++)
        {
            if (status[i] == 0)
            {
                nu += grad[i];
                nFree++;
            }
        }

        if (nFree == 0)
        {
            return false;
        }

        nu /= T.CreateChecked(nFree);

        var worstIdx = -1;
        var worstViolation = T.Zero;

        for (var i = 0; i < n; i++)
        {
            if (status[i] == -1 && grad[i] > nu + s_epsilon10)
            {
                var violation = grad[i] - nu;
                if (violation > worstViolation)
                {
                    worstViolation = violation;
                    worstIdx = i;
                }
            }
            else if (status[i] == 1 && grad[i] < nu - s_epsilon10)
            {
                var violation = nu - grad[i];
                if (violation > worstViolation)
                {
                    worstViolation = violation;
                    worstIdx = i;
                }
            }
        }

        if (worstIdx < 0)
        {
            return false;
        }

        status[worstIdx] = 0;
        return true;
    }
}

/// <summary>
/// Decimal facade for <see cref="ActiveSetQpSolver{T}"/>, forwarding to
/// <see cref="ActiveSetQpSolver{T}"/> instantiated at <c>T = decimal</c>.
/// </summary>
public static class ActiveSetQpSolver
{
    /// <inheritdoc cref="ActiveSetQpSolver{T}.SolveMinVariance"/>
    public static decimal[] SolveMinVariance(
        decimal[,] covariance, decimal minWeight, decimal maxWeight)
        => ActiveSetQpSolver<decimal>.SolveMinVariance(covariance, minWeight, maxWeight);

    /// <inheritdoc cref="ActiveSetQpSolver{T}.SolveMeanVariance"/>
    public static decimal[] SolveMeanVariance(
        decimal[,] covariance, decimal[] means, decimal riskAversion,
        decimal minWeight, decimal maxWeight)
        => ActiveSetQpSolver<decimal>.SolveMeanVariance(
            covariance, means, riskAversion, minWeight, maxWeight);
}
