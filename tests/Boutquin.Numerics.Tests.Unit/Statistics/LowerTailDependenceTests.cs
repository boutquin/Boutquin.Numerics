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

using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

public sealed class LowerTailDependenceTests
{
    [Theory]
    [InlineData(LowerTailDependenceEstimator.Conditional)]
    [InlineData(LowerTailDependenceEstimator.EmpiricalCopula)]
    public void Estimate_HalfLargeExceedanceCount_ReturnsFiniteCoefficient(LowerTailDependenceEstimator estimator)
    {
        var values = Enumerable.Range(0, 70_000).Select(i => (Half)(i / 70_000.0)).ToArray();
        var quantile = (Half)0.95;
        var result = LowerTailDependence<Half>.Estimate(values, values,
            new TailOptions<Half>(Quantiles: [quantile], Estimator: estimator));

        Half.IsFinite(result.Lambda).Should().BeTrue("the count ratio is finite even when counts exceed Half.MaxValue");
        var expected = estimator == LowerTailDependenceEstimator.Conditional
            ? 1.0
            : result.JointCount[0] / (values.Length * (double)quantile);
        ((double)result.Lambda).Should().BeApproximately(expected, 0.001);
    }

    [Fact]
    public void Estimate_LargeQuantileGridDoesNotClonePerQuantile()
    {
        var quantiles = Enumerable.Range(1, 1024).Select(i => i / 1025.0).ToArray();
        var options = new TailOptions<double>(Quantiles: quantiles);
        double[] values = [1.0, 2.0, 3.0, 4.0];

        LowerTailDependence<double>.Estimate(values, values, options);
        var before = GC.GetAllocatedBytesForCurrentThread();
        var result = LowerTailDependence<double>.Estimate(values, values, options);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        result.Threshold.Should().HaveCount(quantiles.Length);
        allocated.Should().BeLessThan(1_000_000);
    }

    [Fact]
    public void TailOptions_SnapshotsQuantilesOnBothEdges()
    {
        double[] input = [0.1, 0.2];
        var options = new TailOptions<double>(input);
        input[0] = 0.9;
        options.Quantiles![0].Should().Be(0.1);
        options.Quantiles[0] = 0.8;
        options.Quantiles[0].Should().Be(0.1);

        var initialized = options with { Quantiles = input };
        input[0] = 0.7;
        initialized.Quantiles![0].Should().Be(0.9);
    }

    private static readonly double[] s_coMonotone = Enumerable.Range(1, 100).Select(i => (double)i).ToArray();

    // F-L1-03 / F-L5-02 (shared root cause): Estimate forms nPlusOne =
    // T.CreateChecked(n + 1) and ComputePseudoObservations forms avgRank the
    // same way. For T = Half, both silently become +Infinity once n exceeds
    // 65,504 — every rank below that collapses to u = Inf-quotient 0 (a false
    // lower-tail exceedance at every quantile) and every rank above it becomes
    // Inf/Inf = NaN. The oracle is LowerTailDependence<double> on the SAME
    // Half-rounded pairs — an independent, already-correct sibling
    // instantiation, not a re-derivation of the fix.
    [Fact]
    public void Estimate_HalfSeriesLongerThanHalfRange_LambdaAgreesWithDoubleOracle()
    {
        const int n = 70_000;
        var x = new Half[n];
        var y = new Half[n];
        var xd = new double[n];
        var yd = new double[n];
        for (var i = 0; i < n; i++)
        {
            var nx = Math.Sin(i * 12.9898) * 43758.5453;
            nx -= Math.Floor(nx);
            var ny = Math.Sin(i * 78.233) * 12543.123;
            ny -= Math.Floor(ny);
            x[i] = (Half)nx;
            y[i] = (Half)ny;
            xd[i] = (double)x[i];
            yd[i] = (double)y[i];
        }

        var expected = LowerTailDependence<double>.Estimate(xd, yd);
        var result = LowerTailDependence<Half>.Estimate(x, y);

        ((double)result.Lambda).Should().BeApproximately(expected.Lambda, 0.02);
    }

    private static double[] IndepSeries(int n, int seed)
    {
        var rng = new System.Random(seed);
        return Enumerable.Range(0, n).Select(_ => rng.NextDouble()).ToArray();
    }

    [Fact]
    public void Estimate_PerfectComonotone_LambdaEqualsOne()
    {
        var result = LowerTailDependence<double>.Estimate(s_coMonotone, s_coMonotone);
        result.Lambda.Should().BeApproximately(1.0, 1e-9);
    }

    [Fact]
    public void Estimate_Independent_LambdaLessThanThreshold()
    {
        var x = IndepSeries(2000, 1);
        var y = IndepSeries(2000, 2);
        var result = LowerTailDependence<double>.Estimate(x, y);
        result.Lambda.Should().BeLessThan(0.20);
    }

    [Fact]
    public void Estimate_AntiComonotone_LambdaApproximatelyZero()
    {
        var n = 200;
        var x = Enumerable.Range(1, n).Select(i => (double)i).ToArray();
        var y = Enumerable.Range(1, n).Select(i => (double)(n + 1 - i)).ToArray();
        var result = LowerTailDependence<double>.Estimate(x, y);
        result.Lambda.Should().BeApproximately(0.0, 1e-9);
    }

    [Fact]
    public void Estimate_ReturnsCorrectArrayLengths()
    {
        var x = IndepSeries(100, 10);
        var y = IndepSeries(100, 11);
        var opts = new TailOptions<double>(Quantiles: [0.05, 0.10, 0.20]);
        var result = LowerTailDependence<double>.Estimate(x, y, opts);
        result.Threshold.Length.Should().Be(3);
        result.JointCount.Length.Should().Be(3);
    }

    [Fact]
    public void Estimate_DifferentLengths_Throws()
    {
        var act = () => LowerTailDependence<double>.Estimate([1.0, 2.0], [1.0, 2.0, 3.0]);
        act.Should().Throw<ArgumentException>().WithMessage("*same length*");
    }

    [Fact]
    public void Estimate_TooFewObservations_Throws()
    {
        var act = () => LowerTailDependence<double>.Estimate([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]);
        act.Should().Throw<ArgumentException>().WithMessage("*Insufficient*");
    }

    [Fact]
    public void Estimate_EmptyQuantiles_ThrowsArgumentException()
    {
        // F-L5-1: a user-supplied empty Quantiles array must be rejected with a
        // documented ArgumentException, not an IndexOutOfRangeException from the
        // unguarded quantiles[0] access.
        // F-VS3-03: ParamName must name the offending "Quantiles" property, not the
        // whole "options" object, so a caller diagnosing the failure is pointed at
        // the actual defect.
        var opts = new TailOptions<double>(Quantiles: Array.Empty<double>());
        var act = () => LowerTailDependence<double>.Estimate(s_coMonotone, s_coMonotone, opts);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*Quantiles*")
            .Where(ex => ex.ParamName == "Quantiles");
    }

    [Fact]
    public void Estimate_NaNObservation_ThrowsArgumentException()
    {
        // F-L1-001: Comparer<T>.Default orders NaN below every value, so a NaN
        // would be assigned the smallest pseudo-observation and counted as an
        // extreme lower-tail co-mover, inflating λ̂. A NaN input must be rejected.
        var x = new double[] { 1.0, 2.0, double.NaN, 4.0, 5.0, 6.0 };
        var y = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        var act = () => LowerTailDependence<double>.Estimate(x, y);
        act.Should().Throw<ArgumentException>().WithMessage("*finite*");
    }

    [Fact]
    public void Estimate_TiedObservations_UsesAverageRanks()
    {
        // F-L1-001: a tied block must receive the AVERAGE rank (sibling convention
        // RankCorrelation.AverageRanks), not arbitrary distinct ascending ranks.
        // x has a 5-value tied block at the bottom; with n=10 their average 1-based
        // rank is 3, so each tied member's pseudo-obs is 3/11 ≈ 0.2727. At q=0.30 all
        // five tied members fall in the tail (marginal = 5). y co-moves on three of
        // them, so the correct λ̂ = 3/5 = 0.6.
        //
        // The old distinct-rank logic assigned ranks 1..5 to the tied block, placing
        // only ranks 1,2,3 (u ≤ 0.2727 < 0.30) in the tail (marginal = 3) — an
        // arbitrary straddle that yielded λ̂ = 3/3 = 1.0. The average-rank fix changes
        // the marginal from an arbitrary 3 to the correct 5.
        var x = new double[] { 5, 5, 5, 5, 5, 6, 7, 8, 9, 10 };
        var y = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var opts = new TailOptions<double>(Quantiles: [0.30]);
        var result = LowerTailDependence<double>.Estimate(x, y, opts);
        result.Lambda.Should().BeApproximately(0.6, 1e-12);
    }

    [Fact]
    public void Estimate_TiedObservations_DeterministicAcrossPartnerArrangement()
    {
        // F-L1-001: the estimate must not depend on the arbitrary ordering of tied
        // members' partner values. Two y-arrangements that are identical in their
        // rank structure over the tied group must produce identical λ̂.
        var x = new double[] { 5, 5, 5, 5, 5, 6, 7, 8, 9, 10 };
        var yForward = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var yReversed = new double[] { 5, 4, 3, 2, 1, 6, 7, 8, 9, 10 };
        var opts = new TailOptions<double>(Quantiles: [0.30]);
        var fwd = LowerTailDependence<double>.Estimate(x, yForward, opts).Lambda;
        var rev = LowerTailDependence<double>.Estimate(x, yReversed, opts).Lambda;
        rev.Should().Be(fwd);
    }

    [Fact]
    public void Estimate_DefaultEstimator_IsConditionalCht()
    {
        // F-L7-03: the DEFAULT estimator must be the Coles-Heffernan-Tawn (CHT)
        // conditional form λ̂ = joint / marginal, NOT the Joe empirical-copula form
        // λ̂ = joint / (n·q). Use a fixture where the two denominators differ:
        // n=10, q=0.19 -> u = avg-rank/(n+1) = r/11. u ≤ 0.19 -> r ≤ 2.09 -> ranks
        // {1,2} so marginal = 2; n·q = 1.9. y co-moves on exactly the smallest x,
        // so joint = 1. CHT = 1/2 = 0.5; Joe = 1/1.9 ≈ 0.5263. The default must be
        // 0.5 (CHT), proving the default did not silently switch to Joe.
        var x = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var y = new double[] { 1, 9, 2, 3, 4, 5, 6, 7, 8, 10 };
        var opts = new TailOptions<double>(Quantiles: [0.19]);
        var result = LowerTailDependence<double>.Estimate(x, y, opts);
        result.Lambda.Should().BeApproximately(0.5, 1e-12);
    }

    [Fact]
    public void Estimate_EmpiricalCopulaEstimator_UsesJointOverNTimesU()
    {
        // F-L7-03: the EmpiricalCopula option selects the Joe (1997) / Nelsen (2006)
        // form λ̂ = joint / (n·q). Same fixture as the default test, where the two
        // denominators differ: marginal = 2 but n·q = 1.9. joint = 1, so the
        // EmpiricalCopula estimate is 1 / 1.9 = 0.526315789..., distinct from the
        // CHT 0.5. Expected value computed independently from the copula definition
        // (n·q = 10·0.19 = 1.9; 1/1.9).
        var x = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var y = new double[] { 1, 9, 2, 3, 4, 5, 6, 7, 8, 10 };
        var opts = new TailOptions<double>(
            Quantiles: [0.19],
            Estimator: LowerTailDependenceEstimator.EmpiricalCopula);
        var result = LowerTailDependence<double>.Estimate(x, y, opts);
        result.Lambda.Should().BeApproximately(1.0 / 1.9, 1e-12);
    }

    [Fact]
    public void Estimate_BothEstimators_AgreeWhenMarginalEqualsNTimesU()
    {
        // F-L7-03: both estimators are consistent estimators of the same limit λ_L;
        // they coincide at finite n exactly when the realized marginal count equals
        // n·q. n=10, q=0.30, strictly increasing x and identical y -> u = r/11,
        // u ≤ 0.30 -> r ≤ 3.3 -> ranks {1,2,3} so marginal = 3 = n·q = 3.0, and the
        // bottom three co-move so joint = 3. CHT = 3/3 = 1.0; Joe = 3/3.0 = 1.0.
        var x = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var y = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var optsCht = new TailOptions<double>(
            Quantiles: [0.30],
            Estimator: LowerTailDependenceEstimator.Conditional);
        var optsJoe = new TailOptions<double>(
            Quantiles: [0.30],
            Estimator: LowerTailDependenceEstimator.EmpiricalCopula);
        var cht = LowerTailDependence<double>.Estimate(x, y, optsCht).Lambda;
        var joe = LowerTailDependence<double>.Estimate(x, y, optsJoe).Lambda;
        cht.Should().BeApproximately(1.0, 1e-12);
        joe.Should().BeApproximately(1.0, 1e-12);
        joe.Should().Be(cht);
    }

    [Fact]
    public void EstimateMatrix_UntiedColumns_AreSymmetric()
    {
        var cols = new double[][] {
            IndepSeries(200, 1),
            IndepSeries(200, 2),
            IndepSeries(200, 3)
        };
        var mat = LowerTailDependence<double>.EstimateMatrix(cols);
        mat[0, 1].Should().BeApproximately(mat[1, 0], 1e-10);
        mat[0, 2].Should().BeApproximately(mat[2, 0], 1e-10);
        mat[1, 2].Should().BeApproximately(mat[2, 1], 1e-10);
    }

    [Fact]
    public void EstimateMatrix_NullColumns_ThrowsArgumentNullException()
    {
        // F-LQ-6: a null columns array must be rejected with ArgumentNullException,
        // not a NullReferenceException from the unguarded columns.Length access.
        var act = () => LowerTailDependence<double>.EstimateMatrix(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void EstimateMatrix_EmptyColumns_ThrowsArgumentException()
    {
        // F-LQ-6: an empty columns array must be rejected, not silently return a 0×0
        // matrix with no diagnostic.
        var act = () => LowerTailDependence<double>.EstimateMatrix(Array.Empty<double[]>());
        act.Should().Throw<ArgumentException>().WithMessage("*at least*");
    }

    // ---------------------------------------------------------------------
    // De-trivialize F-VS3-02: the diagonal is unconditionally set to T.One by
    // EstimateMatrix (result[i,i] = T.One), so a diagonal-only assertion can
    // never fail regardless of any regression in the PAIRWISE estimator. This
    // test replaces that with meaningful off-diagonal content: symmetry for
    // these untied columns (both marginal counts are equal) AND known values derived
    // independently from the CHT estimator's definition
    // (/tmp/ltd_diagonal.py, implementing lambda_hat(q) = joint/marginal
    // directly from Coles-Heffernan-Tawn 1999 — not ported from the C# under
    // test), using the library's DEFAULT quantiles (primary q = 0.05, n = 20
    // so n*q = 1 -> the tail is exactly the single smallest rank of the
    // "primary" series, making the joint/marginal counts hand-verifiable).
    //
    // Fixture (independent reference derivation):
    //   col0 = 1..20 ascending
    //   col1 = a fixed permutation of 1..20 constructed so the index holding
    //          col0's single smallest value (index 0, value 1) holds a
    //          MID-RANGE value in col1 (11) — i.e. NOT also in col1's own
    //          lower tail — so no pairing with col0 lands in the joint tail
    //          at q=0.05: joint=0, marginal=1 -> lambda(col0,col1) = 0.0
    //   col2 = identical to col0 (perfectly co-monotone) -> at q=0.05 the
    //          single index in the tail co-moves trivially with itself:
    //          joint=1, marginal=1 -> lambda(col0,col2) = 1.0
    //   lambda(col1,col2): col2 = col0, so this reduces to lambda(col1,col0)
    //   using col1's own marginal count (1, since q=0.05 also isolates a
    //   single rank in col1) — the value at that index in col0 is 11, not
    //   in col0's tail -> joint=0, marginal=1 -> lambda(col1,col2) = 0.0
    //
    // A regression that broke the pairwise estimator (e.g. an off-by-one in
    // rank computation, a wrong quantile/threshold comparison, or a joint-
    // count miscount) would flip one or more of these off-diagonal values
    // away from the hand-derived 0.0/1.0/0.0 pattern, or break the expected
    // symmetry for this fixture — either of which fails this test.
    // ---------------------------------------------------------------------
    [Fact]
    public void Estimate_UseGarchFilterTrue_ThrowsNotSupportedException()
    {
        // M-4 AC-2: UseGarchFilter was documented+public but silently ignored
        // (the option was never read by Estimate). That is a doc/behavior
        // mismatch — a caller opting into pre-whitening got the unfiltered
        // estimate with no signal. The corrected contract makes the request
        // fail loudly instead of silently degrading to unfiltered.
        var x = IndepSeries(50, 21);
        var y = IndepSeries(50, 22);
        var opts = new TailOptions<double>(UseGarchFilter: true);
        var act = () => LowerTailDependence<double>.Estimate(x, y, opts);
        act.Should().Throw<NotSupportedException>().WithMessage("*UseGarchFilter*");
    }

    [Fact]
    public void EstimateMatrix_UseGarchFilterTrue_ThrowsNotSupportedException()
    {
        // M-4 AC-2: EstimateMatrix threads options through to the pairwise
        // Estimate call, so the same guard must be reachable via the matrix
        // entry point, not just the pairwise one.
        var cols = new double[][] { IndepSeries(50, 23), IndepSeries(50, 24) };
        var opts = new TailOptions<double>(UseGarchFilter: true);
        var act = () => LowerTailDependence<double>.EstimateMatrix(cols, opts);
        act.Should().Throw<NotSupportedException>().WithMessage("*UseGarchFilter*");
    }

    [Fact]
    public void Estimate_UseGarchFilterFalse_MatchesDefaultBehavior()
    {
        // Characterization: the default (false) path is byte-identical to
        // omitting the option entirely — the guard must not perturb the
        // unfiltered path in any way.
        var x = IndepSeries(200, 31);
        var y = IndepSeries(200, 32);
        var explicitFalse = new TailOptions<double>(UseGarchFilter: false);
        var withDefault = LowerTailDependence<double>.Estimate(x, y);
        var withExplicitFalse = LowerTailDependence<double>.Estimate(x, y, explicitFalse);
        withExplicitFalse.Lambda.Should().Be(withDefault.Lambda);
        withExplicitFalse.Threshold.Should().BeEquivalentTo(withDefault.Threshold);
        withExplicitFalse.JointCount.Should().BeEquivalentTo(withDefault.JointCount);
    }

    [Fact]
    public void EstimateMatrix_DiagonalIsOne()
    {
        double[] col0 = Enumerable.Range(1, 20).Select(i => (double)i).ToArray();
        double[] col1 =
        [
            11, 15, 3, 18, 7, 20, 1, 13, 9, 4,
            16, 2, 19, 6, 12, 8, 17, 5, 14, 10,
        ];
        double[] col2 = (double[])col0.Clone();

        var mat = LowerTailDependence<double>.EstimateMatrix([col0, col1, col2]);

        // Diagonal is still asserted (retained as documented), but is no
        // longer the test's only content.
        mat[0, 0].Should().BeApproximately(1.0, 1e-9);
        mat[1, 1].Should().BeApproximately(1.0, 1e-9);
        mat[2, 2].Should().BeApproximately(1.0, 1e-9);

        // These untied columns have equal realized marginal counts, so the
        // conditional estimates agree in both directions.
        mat[0, 1].Should().Be(mat[1, 0]);
        mat[0, 2].Should().Be(mat[2, 0]);
        mat[1, 2].Should().Be(mat[2, 1]);

        // Known-value off-diagonal assertions, independently derived from the
        // CHT estimator definition (see comment block above).
        mat[0, 1].Should().BeApproximately(0.0, 1e-9); // independent-ish pair
        mat[0, 2].Should().BeApproximately(1.0, 1e-9); // co-monotone pair (col2 == col0)
        mat[1, 2].Should().BeApproximately(0.0, 1e-9); // independent-ish pair
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void Estimate_InvalidQuantile_ThrowsArgumentOutOfRangeException(double quantile)
    {
        var options = new TailOptions<double>(Quantiles: [quantile]);
        var act = () => LowerTailDependence<double>.Estimate(
            s_coMonotone, s_coMonotone, options);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithMessage("*Quantile at index 0*")
            .WithParameterName("options");
    }

    [Fact]
    public void Estimate_UndefinedEstimator_ThrowsArgumentOutOfRangeException()
    {
        var options = new TailOptions<double>(
            Estimator: (LowerTailDependenceEstimator)99);
        var act = () => LowerTailDependence<double>.Estimate(
            s_coMonotone, s_coMonotone, options);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("options");
    }

    [Fact]
    public void EstimateMatrix_SingleColumnStillRejectsUnsupportedGarchFilter()
    {
        var options = new TailOptions<double>(UseGarchFilter: true);
        var act = () => LowerTailDependence<double>.EstimateMatrix(
            [s_coMonotone], options);

        act.Should().Throw<NotSupportedException>()
            .WithMessage("*UseGarchFilter*");
    }

    [Fact]
    public void EstimateMatrix_SingleColumnStillValidatesSeries()
    {
        var values = (double[])s_coMonotone.Clone();
        values[7] = double.NaN;
        var act = () => LowerTailDependence<double>.EstimateMatrix([values]);

        act.Should().Throw<ArgumentException>()
            .WithMessage("*Column 0*index 7*finite*")
            .WithParameterName("columns");
    }

    [Fact]
    public void EstimateMatrix_DifferentColumnLengths_ThrowsArgumentException()
    {
        var act = () => LowerTailDependence<double>.EstimateMatrix(
            [s_coMonotone, s_coMonotone[..^1]]);

        act.Should().Throw<ArgumentException>()
            .WithMessage("*same length*")
            .WithParameterName("columns");
    }
}
