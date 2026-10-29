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

using System.Globalization;
using System.Numerics;

using Boutquin.Numerics.Internal;

using Corpus = Boutquin.Numerics.Tests.Unit.Internal.QpKktCertificateTests.Corpus;
using Rational = Boutquin.Numerics.Tests.Unit.Internal.QpKktCertificateTests.Rational;

namespace Boutquin.Numerics.Tests.Unit.Internal;

/// <summary>
/// The covariance equilibration of the active-set QP rewrite (<c>Internal/QpScaling&lt;T&gt;.EquilibrateCovariance</c>,
/// AC-2.3 as amended by amendment #11): every factor <c>D_i</c> is a power of the type's radix, placed just above
/// <c>√Σ_ii</c> unless exactness forbids it, <c>D⁻¹ΣD⁻¹</c> is exact, and <c>D·Σ̃·D</c> reproduces <c>Σ</c> (as a value,
/// and bit for bit at <see cref="double"/>). Every check is made against exact rationals, never against the unit's own
/// arithmetic.
/// </summary>
public sealed class QpScalingTests
{
    // ------------------------------------------------------------------------------------------
    // AC-2.3  the factors are radix powers of the diagonal root and the scaled diagonal is unit scale
    // ------------------------------------------------------------------------------------------

    // Diagonals spread over many binades/decades and off-diagonals with few significant digits, so that no exponent
    // needs limiting for exactness.
    [Fact]
    public void Factors_AreRadixPowersOfTheDiagonalRoot_AndTheScaledDiagonalIsUnitScale_Double()
    {
        Assert.False(typeof(QpScaling<>).IsPublic, "QpScaling<T> must be internal (PublicAPI unchanged)");
        double[,] cov = { { 1e-12, 3e-13, 0 }, { 3e-13, 4e-6, 1e-4 }, { 0, 1e-4, 250 } };
        var (exponents, scaled) = QpScaling<double>.EquilibrateCovariance(cov);
        AssertRadixPowersNearTheDiagonalRoot(cov, exponents, scaled, radix: 2);
        AssertExactAndReproduced(cov, exponents, scaled, radix: 2, "hand-built");
    }

    [Fact]
    public void Factors_AreRadixPowersOfTheDiagonalRoot_AndTheScaledDiagonalIsUnitScale_Decimal()
    {
        decimal[,] cov = { { 0.000000000001m, 0.0000000000003m, 0m }, { 0.0000000000003m, 0.000004m, 0.0001m }, { 0m, 0.0001m, 250m } };
        var (exponents, scaled) = QpScaling<decimal>.EquilibrateCovariance(cov);
        AssertRadixPowersNearTheDiagonalRoot(cov, exponents, scaled, radix: 10);
        AssertExactAndReproduced(cov, exponents, scaled, radix: 10, "hand-built");
    }

    // ------------------------------------------------------------------------------------------
    // AC-2.3  the covariance scaling is exact, placed and reproduced on every corpus problem
    // ------------------------------------------------------------------------------------------

    [Fact]
    public void Scaling_IsExact_OnEveryCorpusProblem_Double() => AssertExactAndPlacedOnCorpus<double>("double", radix: 2);

    [Fact]
    public void Scaling_IsExact_OnEveryCorpusProblem_Decimal() => AssertExactAndPlacedOnCorpus<decimal>("decimal", radix: 10);

    // The covariance-only form, which the active-set core uses, takes no bound and no linear term: on diag(1e14, 1e-14)
    // (whose problem may carry 28-place bounds ±2/3 and a 28-place mean 1e-28, none of which can hold an exponent at 0)
    // it picks the radix powers just above √Σ_ii and scales the covariance exactly (the exactness of the covariance
    // still limits it).
    [Fact]
    public void EquilibrateCovariance_IgnoresBoundsAndLinearTerm_AndScalesTheCovarianceExactly_Decimal()
    {
        var cov = new decimal[,] { { 100000000000000m, 0m }, { 0m, 0.00000000000001m } };
        var (exponents, scaled) = QpScaling<decimal>.EquilibrateCovariance(cov);
        Assert.Equal([8, -6], exponents);
        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                var restored = Rational.From(scaled[i, j]) * Pow(10, exponents[i] + exponents[j]);
                Assert.True(restored.CompareTo(Rational.From(cov[i, j])) == 0, "scaled covariance [" + i + "," + j + "] = " + scaled[i, j] + " is not exactly Σ_ij / (D_i D_j)");
            }
        }

        foreach (var c in Corpus.Load("decimal"))
        {
            var original = c.Cov<decimal>();
            var (e, s) = QpScaling<decimal>.EquilibrateCovariance(original);
            for (var i = 0; i < c.N; i++)
            {
                for (var j = 0; j < c.N; j++)
                {
                    var restored = Rational.From(s[i, j]) * Pow(10, e[i] + e[j]);
                    Assert.True(restored.CompareTo(Rational.From(original[i, j])) == 0, c.Id + ": covariance-only scaling is not exact at [" + i + "," + j + "]");
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // Independent checks (exact rationals; nothing computed by QpScaling is trusted)
    // ------------------------------------------------------------------------------------------

    private static void AssertExactAndPlacedOnCorpus<T>(string type, int radix)
        where T : IFloatingPoint<T>
    {
        var cases = Corpus.Load(type);
        Assert.True(cases.Count >= 700, "the " + type + " corpus must be present, found " + cases.Count);
        var unitDiagonal = 0;
        var scaledRows = 0;
        foreach (var c in cases)
        {
            var cov = c.Cov<T>();
            var (exponents, scaled) = QpScaling<T>.EquilibrateCovariance(cov);
            AssertExactAndReproduced(cov, exponents, scaled, radix, c.Id);
            AssertPlaced(cov, exponents, scaled, radix, c.Id);
            scaledRows += exponents.Count(e => e != 0);
            var low = Pow(radix, -2);
            if (Enumerable.Range(0, c.N).All(i => Rational.From(scaled[i, i]).CompareTo(low) >= 0 && Rational.From(scaled[i, i]).CompareTo(Pow(radix, 0)) < 0))
            {
                unitDiagonal++;
            }
        }

        // Exponents are only limited where exactness demands it; most corpus problems end with a unit-scale diagonal,
        // and a scaling that never moves an exponent fails the placement clause above on many rows.
        Assert.True(unitDiagonal >= cases.Count / 2, "only " + unitDiagonal + " of " + cases.Count + " scaled diagonals are unit scale");
        Assert.True(scaledRows > 0, "the corpus must exercise a non-zero exponent");
    }

    private static void AssertRadixPowersNearTheDiagonalRoot<T>(T[,] cov, int[] exponents, T[,] scaled, int radix)
        where T : IFloatingPoint<T>
    {
        var n = cov.GetLength(0);
        Assert.Equal(n, exponents.Length);
        for (var i = 0; i < n; i++)
        {
            // D_i = radix^e_i with radix^(2(e_i - 1)) <= Σ_ii < radix^(2 e_i): D_i is the radix power just above √Σ_ii.
            var sigma = Rational.From(cov[i, i]);
            var e = exponents[i];
            Assert.True(Pow(radix, 2 * (e - 1)).CompareTo(sigma) <= 0 && sigma.CompareTo(Pow(radix, 2 * e)) < 0, "row " + i + ": exponent " + e + " is not the radix power just above sqrt(" + cov[i, i] + ")");
            var s = Rational.From(scaled[i, i]);
            Assert.True(s.CompareTo(Pow(radix, -2)) >= 0 && s.CompareTo(Pow(radix, 0)) < 0, "row " + i + ": scaled diagonal " + scaled[i, i] + " is not unit scale");
        }
    }

    // D⁻¹ΣD⁻¹ is exact (Σ̃_ij · D_i · D_j == Σ_ij as values), and D·Σ̃·D reproduces Σ bit for bit at double.
    private static void AssertExactAndReproduced<T>(T[,] cov, int[] exponents, T[,] scaled, int radix, string id)
        where T : IFloatingPoint<T>
    {
        var n = cov.GetLength(0);
        Assert.Equal(n, exponents.Length);
        Assert.Equal(n, scaled.GetLength(0));
        Assert.Equal(n, scaled.GetLength(1));
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var restored = Rational.From(scaled[i, j]) * Pow(radix, exponents[i] + exponents[j]);
                Assert.True(restored.CompareTo(Rational.From(cov[i, j])) == 0, id + ": scaled covariance [" + i + "," + j + "] = " + scaled[i, j] + " is not exactly Σ_ij / (D_i D_j)");
                if (typeof(T) == typeof(double))
                {
                    var back = Math.ScaleB((double)(object)scaled[i, j], exponents[i] + exponents[j]);
                    Assert.True(
                        BitConverter.DoubleToInt64Bits(back) == BitConverter.DoubleToInt64Bits((double)(object)cov[i, j]),
                        id + ": D·Σ̃·D at [" + i + "," + j + "] is " + back.ToString("R", CultureInfo.InvariantCulture) + ", not Σ_ij bit for bit");
                }
            }
        }
    }

    // Placement (AC-2.3): each exponent is the radix power just above √Σ_ii (so Σ̃_ii lies in [1/radix², 1)), unless the
    // exactness limit moved it toward 0, accepted only where the unlimited exponents leave some entry of that row of
    // D⁻¹ΣD⁻¹ inexact, or unless the root of a diagonal within rounding of a power of the radix put it one step off.
    private static void AssertPlaced<T>(T[,] cov, int[] exponents, T[,] scaled, int radix, string id)
        where T : IFloatingPoint<T>
    {
        var n = cov.GetLength(0);
        var unlimited = new int[n];
        for (var i = 0; i < n; i++)
        {
            unlimited[i] = UnlimitedExponent(Rational.From(cov[i, i]), radix);
        }

        for (var i = 0; i < n; i++)
        {
            var e = exponents[i];
            if (e == unlimited[i])
            {
                var s = Rational.From(scaled[i, i]);
                Assert.True(
                    Rational.From(cov[i, i]).Sign <= 0 || (s.CompareTo(Pow(radix, -2)) >= 0 && s.CompareTo(Pow(radix, 0)) < 0),
                    id + ": row " + i + " keeps its unlimited exponent " + e + " but the scaled diagonal " + scaled[i, i] + " is not unit scale");
                continue;
            }

            var towardZero = Math.Abs(e) < Math.Abs(unlimited[i]) && (e == 0 || Math.Sign(e) == Math.Sign(unlimited[i]));
            var rowInexact = Enumerable.Range(0, n).Any(j => !Representable<T>(Rational.From(cov[i, j]) / Pow(radix, unlimited[i] + unlimited[j])));
            var nearPower = Math.Abs(e - unlimited[i]) == 1 && NearEvenPowerOfRadix(Rational.From(cov[i, i]), radix, Math.Max(e, unlimited[i]) - 1);
            Assert.True(
                (towardZero && rowInexact) || nearPower,
                id + ": row " + i + " has exponent " + e + " instead of " + unlimited[i] + " although the unlimited scaling of that row is exact and its diagonal is not at a power of the radix");
        }
    }

    // k with radix^(2(k-1)) <= x < radix^(2k): the radix power just above √x (0 for x <= 0).
    private static int UnlimitedExponent(Rational x, int radix)
    {
        if (x.Sign <= 0)
        {
            return 0;
        }

        var k = 0;
        while (x.CompareTo(Pow(radix, 2 * k)) >= 0)
        {
            k++;
        }

        while (x.CompareTo(Pow(radix, 2 * (k - 1))) < 0)
        {
            k--;
        }

        return k;
    }

    // Whether x lies within a few roundings of radix^(2k), where a rounded square root can land on the power itself.
    private static bool NearEvenPowerOfRadix(Rational x, int radix, int k)
    {
        var power = Pow(radix, 2 * k);
        var tolerance = radix == 2 ? Pow(2, -50) : Pow(10, -25);
        return (x - power).Abs().CompareTo(power * tolerance) <= 0;
    }

    // Whether the rational value r has a representation in T: a finite double equal to it, or a decimal with at most 28
    // places and a 96-bit mantissa.
    private static bool Representable<T>(Rational r)
        where T : IFloatingPoint<T>
    {
        if (r.Sign == 0)
        {
            return true;
        }

        if (typeof(T) == typeof(double))
        {
            var candidate = (double)r.Num / (double)r.Den;
            if (!double.IsFinite(candidate) || candidate == 0.0)
            {
                // Outside the range of a direct quotient: decide by the binary exponent of the reduced value.
                return BinaryRepresentable(r);
            }

            foreach (var d in new[] { candidate, Math.BitIncrement(candidate), Math.BitDecrement(candidate) })
            {
                if (double.IsFinite(d) && Rational.FromDouble(d).CompareTo(r) == 0)
                {
                    return true;
                }
            }

            return BinaryRepresentable(r);
        }

        for (var places = 0; places <= 28; places++)
        {
            var scaled = r * Pow(10, places);
            if (scaled.Den.IsOne)
            {
                return BigInteger.Abs(scaled.Num) < BigInteger.One << 96;
            }
        }

        return false;
    }

    // r = m·2^q with m odd: a double when |m| < 2^53 and m·2^q lies within the normal or subnormal range.
    private static bool BinaryRepresentable(Rational r)
    {
        if (!(r.Den & (r.Den - 1)).IsZero)
        {
            return false;
        }

        var m = BigInteger.Abs(r.Num);
        var q = -(int)(r.Den.GetBitLength() - 1);
        while (m.IsEven)
        {
            m >>= 1;
            q++;
        }

        var top = q + (int)m.GetBitLength() - 1;
        return m.GetBitLength() <= 53 && top <= 1023 && q >= -1074;
    }

    private static Rational Pow(int radix, int exponent) => exponent >= 0
        ? Rational.Of(BigInteger.Pow(radix, exponent), BigInteger.One)
        : Rational.Of(BigInteger.One, BigInteger.Pow(radix, -exponent));
}
