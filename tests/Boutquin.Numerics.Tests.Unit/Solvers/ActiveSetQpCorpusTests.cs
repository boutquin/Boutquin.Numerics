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

using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;

using Boutquin.Numerics.LinearAlgebra;
using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// The committed ActiveSetQpSolver regression corpus. Fixtures under <c>tests/Verification/qp-corpus/</c>
/// hold seeded, deterministic problems; each carries its exact optimum (as an active-set pattern plus the
/// optimum rounded to 60 digits), a qualification flag and a one-ulp input sensitivity. This class
/// recomputes every exact optimum with a <see cref="BigInteger"/> rational oracle that shares no code
/// with the solver, certifies it with the full KKT conditions, drives the PRODUCTION
/// <see cref="ActiveSetQpSolver{T}.SolveMinVariance"/> / <see cref="ActiveSetQpSolver{T}.SolveMeanVariance"/>
/// on every case, and retains calibrated performance measurements separately from numerical correctness.
/// </summary>
[Collection(ActiveSetQpPerformanceCollection.Name)]
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpCorpusTests
{
    private static readonly Lazy<CorpusRun> s_run = new(CorpusRun.Execute);

    private static readonly string[] s_requiredFamilies =
    [
        "dbl-corr", "dbl-minvar", "dec-corr", "dec-diag-count", "r5-dbl-disp-mv",
        "r5-dec-rng-mean", "r5-dec-rng-budget", "r5-dec-rng-budget-min", "r5-dec-rng-zc", "r5-dec-zc2",
        "r5-rng-dbl-mean", "r5-rng-dbl-budget", "r5-rng-dbl-budget-min", "r5-rng-dbl-zc",
        "nearq-dbl", "nearq-dec", "nearo-dbl", "nearo-dec", "degenb-dbl", "degenb-dec", "nearmo-dbl", "nearmo-dec",
        "heldout-dbl", "heldout-dec",
    ];

    // The fixed case set on which the M-1 calibrated per-type timing ratios were recorded.
    private static readonly string[] s_m1Families =
    [
        "dbl-corr", "dbl-minvar", "dec-corr", "dec-diag-count", "r5-dbl-disp-mv",
        "r5-dec-rng-mean", "r5-dec-rng-budget", "r5-dec-rng-budget-min", "r5-dec-rng-zc", "r5-dec-zc2",
        "r5-rng-dbl-mean", "r5-rng-dbl-budget", "r5-rng-dbl-budget-min", "r5-rng-dbl-zc", "hd-hand",
    ];

    // The M-1 named witnesses, then the held-out witnesses: explicit matrices of wrong vectors and refusals of
    // qualified cases that held-out controls found in earlier cores (the zero-correlation a1Y9 shape, extreme
    // diagonal disparity, repeated structure, the nine qualified double corr multiplier-sign refusals, the
    // degenerate-at-bound decimal refusals and wrong vectors).
    private static readonly string[] s_requiredWitnesses =
    [
        "dbl-corr-6-28", "dbl-corr-8-49", "hd-diag4-12", "r5-dec-rng-zc-37", "r5-dec-rng-zc-47",
        "r5-dec-rng-mean-0", "r5-dbl-disp-mv-10",
        "a1Y9. /1 0#1", "v3probe-188#2", "degen-120#1",
        "tAg1450#0", "tAg412#0", "tAg473#0", "tAg4367#2", "tAg3269#0", "tAg2217#1", "tBg1440#1", "tBg3395#2", "tBg2214#1",
        ".0#3", "v3r2degenb-66#3", "v3r2degenb-129#2", "vr2dgb-35#1", "tBd219#0", "tBg113#0",
        "m3d1-175#0", "m3d1-156#2", "m3d1-258#1", "m3d1-8#2", "m3d1-21#1", "m3d1-38#0", "m3d1-106#1", "m3d1-159#0",
        "v4p-302#1", "v4s-149#0", "tightub-10#0",
    ];

    // ------------------------------------------------------------------------------------------
    // AC-1.1  deterministic corpus; every case carries exact optimum, qualification flag, sensitivity
    // ------------------------------------------------------------------------------------------

    [Fact]
    public void Corpus_IsCommittedSeededAndWithinTheCap()
    {
        var run = s_run.Value;
        Assert.True(run.Loaded, "the corpus fixtures must be committed and copied to the output directory: " + run.LoadReport);
        Assert.Empty(run.LoadErrors);

        var cases = run.Files.SelectMany(f => f.Cases).ToList();
        Assert.True(cases.Count <= 2300, "the corpus is capped at 2,300 cases, found " + cases.Count);
        Assert.True(cases.Count >= 1000, "the corpus must be substantial, found " + cases.Count);
        Assert.Equal(cases.Count, cases.Select(c => c.Id).Distinct(StringComparer.Ordinal).Count());

        var families = run.Files.SelectMany(f => f.Families).ToList();
        foreach (var required in s_requiredFamilies)
        {
            var spec = families.SingleOrDefault(f => f.Family == required);
            Assert.True(spec is not null, "missing generator family " + required);
            Assert.True(spec!.Seeds.Count > 0, "family " + required + " must declare its fixed seeds");
            Assert.Equal(spec.Count, cases.Count(c => c.Family == required));
        }

        foreach (var spec in families)
        {
            Assert.False(string.IsNullOrEmpty(spec.Generator), "family " + spec.Family + " must name its generator");
            Assert.Equal(spec.Count, cases.Count(c => c.Family == spec.Family));
        }

        var ids = cases.Select(c => c.Id).ToHashSet(StringComparer.Ordinal);
        foreach (var witness in s_requiredWitnesses)
        {
            Assert.True(ids.Contains(witness), "missing named witness " + witness);
        }

        Assert.Equal(cases.Count, families.Sum(f => f.Count));
        Assert.Contains(cases, c => c.Qualified);
        Assert.Contains(cases, c => !c.Qualified);
    }

    [Fact]
    public void Corpus_EveryExactOptimumIsRecomputedByTheIndependentRationalOracle()
    {
        var run = s_run.Value;
        Assert.True(run.Loaded, "the corpus fixtures must be committed and copied to the output directory: " + run.LoadReport);
        Assert.True(run.Files.Sum(f => f.Cases.Count) >= 1000, "the corpus must be present before its optima can be verified");

        Assert.Empty(run.OptimumFailures);
        Assert.Equal(run.Files.Sum(f => f.Cases.Count), run.ExactOptima.Count);
    }

    [Fact]
    public void Corpus_QualificationFlagAndConditionAreRecomputedFromTheExactDefinition()
    {
        var run = s_run.Value;
        Assert.True(run.Loaded, "the corpus fixtures must be committed and copied to the output directory: " + run.LoadReport);
        Assert.True(run.Files.Sum(f => f.Cases.Count) >= 1000, "the corpus must be present before its flags can be verified");

        Assert.Empty(run.FlagFailures);
    }

    [Fact]
    public void Corpus_OneUlpSensitivityIsRecomputedOnTheWitnessesAndASample()
    {
        var run = s_run.Value;
        Assert.True(run.Loaded, "the corpus fixtures must be committed and copied to the output directory: " + run.LoadReport);

        foreach (var witness in s_requiredWitnesses)
        {
            Assert.True(run.SensitivityChecked.Contains(witness), "the sensitivity of witness " + witness + " was not recomputed");
        }

        Assert.True(run.SensitivityChecked.Count >= 100, "the sampled sensitivity check covered " + run.SensitivityChecked.Count + " cases");
        Assert.Empty(run.SensitivityFailures);
    }

    // ------------------------------------------------------------------------------------------
    // Calibrated performance baseline metadata
    // ------------------------------------------------------------------------------------------

    [Fact]
    public void PerformanceBaseline_RecordsTheCommitShaAndPerTypeCalibrationRatios()
    {
        var run = s_run.Value;
        Assert.True(run.Loaded, "the corpus fixtures must be committed and copied to the output directory: " + run.LoadReport);

        var shaPath = Path.Combine(CorpusRun.FixtureDirectory, "baseline-commit.txt");
        Assert.True(File.Exists(shaPath), "baseline-commit.txt is missing: " + shaPath);
        var sha = File.ReadAllText(shaPath).Trim();
        Assert.Matches("^[0-9a-f]{40}$", sha);

        var path = Path.Combine(CorpusRun.FixtureDirectory, "baseline.json");
        Assert.True(File.Exists(path), "the committed baseline is missing: " + path);
        var committed = PerformanceBaseline.Load(path);
        Assert.Empty(committed.Errors);
        Assert.Equal("200x200 CholeskyDecomposition<double>.Decompose", committed.CalibrationWorkload);
        Assert.True(committed.CalibrationMs > 0, "calibration time must be recorded");

        foreach (var type in new[] { "double", "decimal" })
        {
            Assert.True(committed.Ratios.TryGetValue(type, out var ratios), "missing solve-time ratios for " + type);
            Assert.True(ratios![0] > 0 && double.IsFinite(ratios[0]), type + " median ratio must be a positive finite number");
            Assert.True(ratios[1] > 0 && double.IsFinite(ratios[1]), type + " p99 ratio must be a positive finite number");
            Assert.True(ratios[1] >= ratios[0], type + " p99 ratio must be at least the median ratio");

            // The ratios were recorded on the M-1 case set; cases added later are not part of them.
            Assert.Equal(run.Files.Single(f => f.Type == type).Cases.Count(c => s_m1Families.Contains(c.Family)), (int)ratios[2]);
        }

        Assert.Equal(run.Files.Sum(f => f.Cases.Count(c => s_m1Families.Contains(c.Family))), committed.M1CorpusCases);
    }

    // ------------------------------------------------------------------------------------------
    // AC-1.3  the corpus runs in the unit suite in under 60 s; the measured runtime is recorded
    // ------------------------------------------------------------------------------------------

    [Fact]
    public void Corpus_RunsInUnderSixtySecondsAndTheMeasuredRuntimeIsRecorded()
    {
        var run = s_run.Value;
        Assert.True(run.Loaded, "the corpus fixtures must be committed and copied to the output directory: " + run.LoadReport);
        var solved = s_newCore.Value;
        Assert.True(solved.Table.Values.Sum(v => v[0]) >= 1000, "the full corpus must have been driven through the production solver");
        // The bound is on the work of the pass, as process CPU seconds: wall time on a loaded host runs to many times the
        // CPU time of the same unchanged code (a pass of 25 s of CPU took 524 s of wall time at the lowest priority).
        var cpu = run.CpuSeconds + solved.CpuSeconds;
        var wall = run.ElapsedSeconds + solved.ElapsedSeconds;
        Assert.True(
            cpu < 60,
            "the whole corpus pass (oracle verification and solver) took " + cpu.ToString("F1", CultureInfo.InvariantCulture) + " s of process CPU time (" + wall.ToString("F1", CultureInfo.InvariantCulture) + " s of wall time, not asserted)");

        var path = Path.Combine(CorpusRun.FixtureDirectory, "baseline.json");
        Assert.True(File.Exists(path), "the committed baseline is missing: " + path);
        var committed = PerformanceBaseline.Load(path);
        Assert.True(committed.MeasuredRuntimeSeconds is not null, "baseline.json must record measured_runtime_seconds");
        Assert.True(committed.MeasuredRuntimeSeconds > 0 && committed.MeasuredRuntimeSeconds < 60,
            "the recorded corpus runtime must be positive and under 60 s, recorded " + committed.MeasuredRuntimeSeconds);

        // The recorded runtime is a measurement of the corpus as committed, not of an earlier, smaller one.
        Assert.True(
            committed.CorpusCases == run.Files.Sum(f => f.Cases.Count),
            "the recorded runtime was measured on " + committed.CorpusCases + " cases; the committed corpus has " + run.Files.Sum(f => f.Cases.Count));
    }

    // ------------------------------------------------------------------------------------------
    // AC-3.1  the rewritten SolveMinVariance on every min-variance corpus case, through the
    //         production entry point: qualified cases 0 wrong / 0 refused; refusal-permitted 0 wrong
    // ------------------------------------------------------------------------------------------

    [Theory]
    [InlineData("double")]
    [InlineData("decimal")]
    public void NewCore_QualifiedMinVarianceCases_ReturnTheExactOptimumWithinTheBound_NoWrongAndNoRefusal(string type)
    {
        var run = s_newCore.Value;
        Assert.True(run.Loaded, "the corpus fixtures must be committed and copied to the output directory");
        var stats = run.Stats(type, qualified: true);
        Assert.True(stats.Cases >= 60, "the qualified " + type + " min-variance cases must be present, found " + stats.Cases);
        Assert.True(
            stats.Wrong.Count == 0 && stats.Refused.Count == 0,
            "qualified " + type + " min-variance cases through ActiveSetQpSolver<" + type + ">.SolveMinVariance: " + stats.Describe());
    }

    [Theory]
    [InlineData("double")]
    [InlineData("decimal")]
    public void NewCore_RefusalPermittedMinVarianceCases_ReturnTheOptimumOrTheDocumentedRefusal_NeverAWrongVector(string type)
    {
        var run = s_newCore.Value;
        Assert.True(run.Loaded, "the corpus fixtures must be committed and copied to the output directory");
        var stats = run.Stats(type, qualified: false);

        // Floors re-derived from the regenerated corpus (220 refusal-permitted double and 84 decimal min-variance cases
        // under 48·n·u·κ_eq on the block of variables strictly inside the box): the equilibration qualifies most of the
        // former range families.
        var floor = type == "double" ? 200 : 70;
        Assert.True(stats.Cases >= floor, "the refusal-permitted " + type + " min-variance cases must be present, found " + stats.Cases);
        Assert.True(stats.Wrong.Count == 0, "refusal-permitted " + type + " min-variance cases returned wrong vectors: " + stats.Describe());
        Assert.True(
            stats.Refused.All(r => r.StartsWith(nameof(InvalidOperationException) + ":", StringComparison.Ordinal)),
            "a refusal must be the documented InvalidOperationException: " + stats.Describe());
    }

    /// <summary>
    /// The reference-variable retry changes the basis of the null-space step, never the acceptance measure: with the
    /// retry disabled and each variable in turn forced as the only budget-elimination reference whenever it is free
    /// (the default reference otherwise), every qualified min-variance corpus case returns a vector within the AC-3.1
    /// bound of the exact optimum and is never refused, and every refusal-permitted case returns a vector within the
    /// bound or the documented refusal. A working set that the forced basis leaves unusable (no positive-definite
    /// reduced Hessian, or a block past the conditioning margin) is handed to the exact route from the rounded route's
    /// working set and iterate (design step 3, <i>Unusable working sets</i>), not rolled back or refused, so a forced
    /// reference cannot turn a qualified case into a refusal (AC-3.1: 0 wrong, 0 refusals).
    /// </summary>
    [Theory]
    [InlineData("double")]
    [InlineData("decimal")]
    public void ActiveSetQpMinVariance_ForcedReference_EveryVariableAsTheOnlyReference_ReturnsTheOptimumWithinTheBound_NeverRefusingAQualifiedCase(string type)
    {
        var run = s_run.Value;
        Assert.True(run.Loaded, "the corpus fixtures must be committed and copied to the output directory: " + run.LoadReport);
        var solves = 0;
        var refusals = 0;
        var wrong = new List<string>();
        var qualifiedRefusals = new List<string>();
        foreach (var c in run.Files.Single(f => f.Type == type).Cases.Where(c => c.Mode == "minvar"))
        {
            if (!run.ExactOptima.TryGetValue(c.Id, out var w0))
            {
                continue;
            }

            for (var reference = 0; reference < c.N; reference++)
            {
                solves++;
                Rational[] output;
                try
                {
                    output = CorpusRun.Solve(c, forcedReference: reference);
                }
                catch (InvalidOperationException ex)
                {
                    // Only a refusal-permitted case may raise the documented refusal.
                    refusals++;
                    if (c.Qualified)
                    {
                        qualifiedRefusals.Add(c.Id + " (reference " + reference + "): " + ex.Message);
                    }

                    continue;
                }

                if (CorpusRun.Classify(c, w0, output) != Kind.Ok)
                {
                    wrong.Add(c.Id + " (reference " + reference + (c.Qualified ? ", qualified)" : ", refusal-permitted)"));
                }
            }
        }

        Assert.True(solves >= 1000, "the forced-reference pass must cover the corpus, solved " + solves);
        Assert.True(
            qualifiedRefusals.Count == 0,
            qualifiedRefusals.Count + " forced-reference solves of qualified cases (of " + solves + ", " + refusals + " refused in all) were refused: "
            + string.Join("; ", qualifiedRefusals.Take(10)));
        Assert.True(wrong.Count == 0, wrong.Count + " forced-reference solves of " + solves + " (" + refusals + " refused) returned a vector outside the bound: " + string.Join(", ", wrong.Take(20)));
    }

    /// <summary>
    /// The exit check's tolerance scale <c>κ</c>, calibrated on the rewritten solver's own corpus outputs: for every
    /// min-variance output within the AC-3.1 bound, the smallest power-of-two <c>κ</c> at which the independent
    /// certificate accepts it is measured, and the stated <c>κ = 8</c> must cover the largest of them with a
    /// fourfold margin. (The degenerate-problem property in <see cref="ActiveSetQpMinVarianceTests"/> applies the
    /// same margin.) A <c>κ</c> too loose to matter is bounded by the AC-3.1 assertions themselves, which compare
    /// each returned vector with the exact optimum rather than with the certificate.
    /// </summary>
    [Theory]
    [InlineData("double")]
    [InlineData("decimal")]
    public void NewCore_ExitCertificateToleranceScale_CoversEveryCorrectCorpusOutputWithAFourfoldMargin(string type)
    {
        var run = s_newCore.Value;
        Assert.True(run.Loaded, "the corpus fixtures must be committed and copied to the output directory");
        var kappa = type == "double"
            ? ActiveSetQpSolver<double>.CertificateToleranceScale
            : (double)ActiveSetQpSolver<decimal>.CertificateToleranceScale;
        Assert.Equal(8.0, kappa);
        foreach (var qualified in new[] { true, false })
        {
            var stats = run.Stats(type, qualified);

            // Every refusal-permitted decimal min-variance case of the corpus is refused with the documented refusal, so
            // that set may have no correct output; it is then empty only because every case in it was refused. The
            // qualified set must always have outputs to calibrate on.
            Assert.True(
                stats.Ok > 0 || (!qualified && stats.Cases > 0 && stats.Refused.Count == stats.Cases),
                "no correct " + type + (qualified ? " qualified" : " refusal-permitted") + " outputs to calibrate on: " + stats.Describe());
            Assert.True(
                stats.MaxNeededKappa * 4 <= kappa,
                type + (qualified ? " qualified" : " refusal-permitted") + " outputs need κ = " + stats.MaxNeededKappa
                + " (case " + stats.MaxNeededKappaCase + "), more than κ/4 = " + (kappa / 4));
        }
    }

    // ------------------------------------------------------------------------------------------
    // AC-4.1  the rewritten SolveMeanVariance on every mean-variance corpus case, through the production
    //         entry point: qualified cases 0 wrong / 0 refused; refusal-permitted 0 wrong. A mean-variance
    //         case is qualified by the corpus-membership definition applied to the free block of its own KKT
    //         system: the free block λ·Σ̃_FF has the condition number of Σ̃_FF, with Σ̃ equilibrated by the
    //         covariance-only exponents of QpScaling<T>.EquilibrateCovariance (amendment #11; generate_corpus.py).
    // ------------------------------------------------------------------------------------------

    [Theory]
    [InlineData("double")]
    [InlineData("decimal")]
    public void ActiveSetQpMeanVariance_QualifiedCorpusCases_ReturnTheExactOptimumWithinTheBound_NoWrongAndNoRefusal(string type)
    {
        var run = s_newCore.Value;
        Assert.True(run.Loaded, "the corpus fixtures must be committed and copied to the output directory");
        var stats = run.Stats(type, qualified: true, mode: "meanvar");

        // 570 qualified double and 352 qualified decimal mean-variance cases in the committed corpus.
        var floor = type == "double" ? 560 : 340;
        Assert.True(stats.Cases >= floor, "the qualified " + type + " mean-variance cases must be present, found " + stats.Cases);
        Assert.True(
            stats.Wrong.Count == 0 && stats.Refused.Count == 0,
            "qualified " + type + " mean-variance cases through ActiveSetQpSolver<" + type + ">.SolveMeanVariance: " + stats.Describe());
    }

    [Theory]
    [InlineData("double")]
    [InlineData("decimal")]
    public void ActiveSetQpMeanVariance_RefusalPermittedCorpusCases_ReturnTheOptimumOrTheDocumentedRefusal_NeverAWrongVector(string type)
    {
        var run = s_newCore.Value;
        Assert.True(run.Loaded, "the corpus fixtures must be committed and copied to the output directory");
        var stats = run.Stats(type, qualified: false, mode: "meanvar");

        // 10 refusal-permitted double and 50 decimal mean-variance cases in the committed corpus.
        var floor = type == "double" ? 10 : 50;
        Assert.True(stats.Cases >= floor, "the refusal-permitted " + type + " mean-variance cases must be present, found " + stats.Cases);
        Assert.True(stats.Wrong.Count == 0, "refusal-permitted " + type + " mean-variance cases returned wrong vectors: " + stats.Describe());
        Assert.True(
            stats.Refused.All(r => r.StartsWith(nameof(InvalidOperationException) + ":", StringComparison.Ordinal)),
            "a refusal must be the documented InvalidOperationException: " + stats.Describe());
    }

    /// <summary>
    /// The shared exit check's tolerance scale <c>κ = 8</c> on the mean-variance outputs: for every output within the
    /// bound, the smallest power-of-two <c>κ</c> at which the independent certificate (with the case's means and risk
    /// aversion) accepts it must be covered by <c>κ = 8</c> with a fourfold margin, as for the min-variance outputs.
    /// </summary>
    [Theory]
    [InlineData("double")]
    [InlineData("decimal")]
    public void ActiveSetQpMeanVariance_ExitCertificateToleranceScale_CoversEveryCorrectCorpusOutputWithAFourfoldMargin(string type)
    {
        var run = s_newCore.Value;
        Assert.True(run.Loaded, "the corpus fixtures must be committed and copied to the output directory");
        var kappa = type == "double"
            ? ActiveSetQpSolver<double>.CertificateToleranceScale
            : (double)ActiveSetQpSolver<decimal>.CertificateToleranceScale;
        foreach (var qualified in new[] { true, false })
        {
            var stats = run.Stats(type, qualified, mode: "meanvar");
            Assert.True(stats.Ok > 0, "no correct " + type + " mean-variance outputs to calibrate on");
            Assert.True(
                stats.MaxNeededKappa * 4 <= kappa,
                type + (qualified ? " qualified" : " refusal-permitted") + " mean-variance outputs need κ = " + stats.MaxNeededKappa
                + " (case " + stats.MaxNeededKappaCase + "), more than κ/4 = " + (kappa / 4));
        }
    }

    // ------------------------------------------------------------------------------------------
    // AC-3.1 and AC-3.3 beyond the committed corpus: the witnesses a held-out control found, and seeded
    // censuses of their two shapes at double and decimal (n = 2..12), each output classified against the
    // exact optimum of a rational active-set oracle with the AC-3.1 bound
    // ------------------------------------------------------------------------------------------

    /// <summary>
    /// Near-singular repeated-structure covariances <c>Σ = c·G + d·I</c> (assets of one group covary at <c>c</c>,
    /// across groups at <c>c/2</c>) with a ridge <c>d/c ≈ 1e-8</c>. Their free blocks have an exact condition
    /// number near 1e9, far below <c>1/(4u)</c>, so they are qualified and the solver must return the exact
    /// optimum within the bound, never a refusal.
    /// </summary>
    [Theory]
    [InlineData("degen-120#1")]
    [InlineData("v3probe-239#1")]
    public void NewCore_HeldOutRepeatedStructureWitness_IsQualifiedAndReturnsTheExactOptimumWithinTheBound(string witness)
    {
        var result = SolveHeldOut(HeldOutWitness(witness));
        Assert.True(result.Qualified, witness + " must be a qualified case: " + result.Detail);
        Assert.True(result.Outcome == HeldOutOutcome.Ok, witness + ": " + result.Outcome + ": " + result.Detail);
    }

    /// <summary>
    /// Diagonal covariances whose variances differ by up to 1e176, with <c>lb = 0</c> and <c>ub = 1</c>: the
    /// optimum <c>w ∝ 1/Σᵢᵢ</c> puts weights far below the resolution of the dominant weight on the high-variance
    /// assets. The solve must terminate (never the iteration budget) and return the optimum within the bound.
    /// </summary>
    [Theory]
    [InlineData("v3probe-188#2")]
    [InlineData("diag-59#2")]
    public void NewCore_HeldOutExtremeDiagonalDisparityWitness_TerminatesWithTheExactOptimumWithinTheBound(string witness)
    {
        var result = SolveHeldOut(HeldOutWitness(witness));
        Assert.True(result.Outcome == HeldOutOutcome.Ok, witness + ": " + result.Outcome + ": " + result.Detail);
    }

    /// <summary>
    /// A zero-correlation-shape problem a held-out control found (input <c>'a1Y9. /1 0'</c>, sub-problem 1, double,
    /// <c>n = 5</c>): one asset of variance 2.7e123 coupled at 1e-20 to a block of four assets of variance 2.1e-148
    /// whose correlations ±1/3 make <c>(1, −1, −1, 1)</c> a near-null direction orthogonal to the budget row, inside
    /// a box of ±1e112..1e137. The exact condition number of its free block is about 3.7e286, so it is
    /// refusal-permitted: the solver must return the exact optimum within the bound or the documented refusal naming
    /// stationarity, never a vector that moved along the near-null direction (the objective excess of such a move
    /// is below the resolution of <c>w′Σw</c>, so no residual certificate can see it). The other seven are the wrong
    /// vectors a seeded census of the same shape and of its two-block variant found (prefixes <c>b2zc-</c>,
    /// <c>b2zc2-</c>, <c>b2all-</c>), all double and refusal-permitted: some moved along the near-null direction,
    /// others stopped at a vertex of the box because the anti-cycling rule excluded a release whose single step could
    /// not make progress on the near-singular block.
    /// </summary>
    [Theory]
    [InlineData("a1Y9. /1 0#1")]
    [InlineData("b2zc-57#3")]
    [InlineData("b2zc-115#3")]
    [InlineData("b2zc-302#0")]
    [InlineData("b2zc2-133#3")]
    [InlineData("b2zc2-138#2")]
    [InlineData("b2all-58#2")]
    [InlineData("b2all-256#2")]
    public void NewCore_HeldOutNearNullBudgetOrthogonalWitness_ReturnsTheOptimumOrTheDocumentedRefusal_NeverAWrongVector(string witness)
    {
        var result = SolveHeldOut(HeldOutWitness(witness));
        Assert.False(result.Qualified, witness + " must be a refusal-permitted case: " + result.Detail);
        Assert.True(
            result.Outcome is HeldOutOutcome.Ok or HeldOutOutcome.Refused,
            witness + ": " + result.Outcome + ": " + result.Detail);
        if (result.Outcome == HeldOutOutcome.Refused)
        {
            Assert.Contains("stationarity", result.Detail, StringComparison.Ordinal);
        }
    }

    /// <summary>
    /// Both directions of the conditioning refusal on one shape. <c>Σ = I − (1−ε)/4·ssᵀ</c> with
    /// <c>s = (1, −1, −1, 1)</c> and <c>ε = k·2⁻⁵⁰</c> (every entry exact in double) has the budget-orthogonal
    /// near-null direction <c>s</c> of eigenvalue <c>ε</c>, exact optimum <c>(¼, ¼, ¼, ¼)</c> and a free-block condition
    /// number of about <c>1.5/ε</c> (the equilibration leaves this unit-scale diagonal unchanged). At <c>k = 64</c> and
    /// <c>k = 256</c>, <c>48·n·u·κ_eq &lt; 1</c>: the problem is qualified and must be solved, not refused, so a refusal
    /// would mean the guard tested more than the margin. At <c>k = 2</c>, <c>k = 4</c> and <c>k = 35</c>,
    /// <c>48·n·u·κ_eq</c> is 18, 9 and 1.03: the problem is refusal-permitted, and since its optimum lies strictly inside
    /// the box, the block that decides the result is its whole free block (<c>B = F*</c>), so the conditioning guard
    /// of design step 4 must refuse it naming stationarity. <c>k = 35</c> sits 3% past the margin, so a margin
    /// constant below about 46.7 (the documented one is 48) stops refusing it, and a margin of 1 refuses none of the
    /// three.
    /// </summary>
    [Theory]
    [InlineData(64)]
    [InlineData(256)]
    public void NewCore_QualifiedNearSingularBudgetOrthogonalBlock_IsSolvedNotRefused(int k)
    {
        var result = SolveHeldOut(NearSingularBudgetOrthogonalBlock(k));
        Assert.True(result.Qualified, "the block must be qualified: " + result.Detail);
        Assert.True(result.Outcome == HeldOutOutcome.Ok, result.Outcome + ": " + result.Detail);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(35)]
    public void NewCore_NearSingularBudgetOrthogonalBlockPastTheMargin_IsRefusedNamingStationarity(int k)
    {
        var result = SolveHeldOut(NearSingularBudgetOrthogonalBlock(k));
        Assert.False(result.Qualified, "the block must be refusal-permitted: " + result.Detail);
        Assert.True(
            result.Outcome == HeldOutOutcome.Refused,
            "48·n·u·κ_eq ≥ 1 on the deciding block must be refused by the conditioning guard, got " + result.Outcome + ": " + result.Detail);
        Assert.Contains("stationarity", result.Detail, StringComparison.Ordinal);
    }

    /// <summary>
    /// Degenerate-at-bound problems a fresh-seed census found (prefixes <c>r4a-</c>, <c>r4d-</c>): the exact optimum puts
    /// a variable exactly on its bound with a zero multiplier, so it has two valid free blocks. The block of the variables
    /// strictly inside the box meets the margin <c>48·n·u·κ_eq &lt; 1</c>; the block that also holds the variable on its
    /// bound is numerically singular. The exact optimum is determined by the interior block, so the solver must return it
    /// within the bound, never a refusal (an earlier conditioning guard counted the variable on its bound in the deciding
    /// block and refused).
    /// </summary>
    [Theory]
    [InlineData("r4d-396#2")]
    [InlineData("r4d-628#0")]
    [InlineData("r4d-777#1")]
    [InlineData("r4a-435#1")]
    public void NewCore_DegenerateAtBoundWitness_WhoseInteriorBlockMeetsTheMargin_IsSolvedNotRefused(string witness)
    {
        var problem = DegenerateAtBoundWitness(witness);
        var c = HeldOutCase(problem);
        var q = QpInstance.From(c);
        var (w0, _) = ExactOracle.ActiveSet(q);
        var lb = HeldOutParse(problem.Type, problem.Lb);
        var ub = HeldOutParse(problem.Type, problem.Ub);
        var interior = Enumerable.Range(0, q.N).Where(i => w0[i] > lb && w0[i] < ub).ToArray();
        var u = problem.Type == "double"
            ? Rational.Of(BigInteger.One, BigInteger.One << 53)
            : Rational.Of(BigInteger.One, BigInteger.Pow(10, 28));
        var scaled = EquilibratedCovariance(c, q);
        Assert.True(interior.Length >= 2 && interior.Length < q.N, witness + ": the exact optimum must have variables on a bound");
        Assert.True(
            QualifiesAtTheMargin(q.N, u, ExactOracle.Cond1(scaled, interior)),
            witness + ": the interior block of the exact optimum must meet the margin");
        Assert.Contains(Enumerable.Range(0, q.N), i => (w0[i] == lb || w0[i] == ub) && !QualifiesAtTheMargin(q.N, u, ExactOracle.Cond1(scaled, [.. interior, i])));

        var result = SolveHeldOut(problem);
        Assert.True(result.Outcome == HeldOutOutcome.Ok, witness + ": " + result.Outcome + ": " + result.Detail);
    }

    private static HeldOutProblem DegenerateAtBoundWitness(string name) => name switch
    {
        "r4d-396#2" => new HeldOutProblem(
            name,
            "decimal",
            [
                ["9.000000000000000000000000000", "-0.2275390625000000000000000000", "-1.0000000000000000000000000000", "0", "1.0000000000000000000000000000", "0", "-1.0000000000000000000000000000"],
                ["-0.2275390625000000000000000000", "7.0083780288696289062500000001", "5.3691406250000000000000000000", "-1.2949218750000000000000000000", "-2.4931640625000000000000000000", "6.9072265625000000000000000000", "3.1123046875000000000000000000"],
                ["-1.0000000000000000000000000000", "5.3691406250000000000000000000", "11.000000000000000000000000000", "0", "-1.0000000000000000000000000000", "0", "0"],
                ["0", "-1.2949218750000000000000000000", "0", "11.000000000000000000000000000", "1.0000000000000000000000000000", "0", "0"],
                ["1.0000000000000000000000000000", "-2.4931640625000000000000000000", "-1.0000000000000000000000000000", "1.0000000000000000000000000000", "10.000000000000000000000000000", "-1.0000000000000000000000000000", "1.0000000000000000000000000000"],
                ["0", "6.9072265625000000000000000000", "0", "0", "-1.0000000000000000000000000000", "16.000000000000000000000000000", "0"],
                ["-1.0000000000000000000000000000", "3.1123046875000000000000000000", "0", "0", "1.0000000000000000000000000000", "0", "11.000000000000000000000000000"],
            ],
            "0",
            "2"),
        "r4d-628#0" => new HeldOutProblem(
            name,
            "double",
            [
                ["2.75", "-0.0712890625", "-0.25", "-0.25"],
                ["-0.0712890625", "2.088701009750575", "0.29638671875", "2.279052734375"],
                ["-0.25", "0.29638671875", "3.25", "0.25"],
                ["-0.25", "2.279052734375", "0.25", "2.5"],
            ],
            "0",
            "2"),
        "r4d-777#1" => new HeldOutProblem(
            name,
            "double",
            [
                ["4194304", "-315392", "-524288", "524288"],
                ["-315392", "4243296.000000005", "3497984", "4210688"],
                ["-524288", "3497984", "6291456", "524288"],
                ["524288", "4210688", "524288", "6815744"],
            ],
            "0",
            "2"),
        "r4a-435#1" => new HeldOutProblem(
            name,
            "double",
            [
                ["12582912", "-151552", "-1048576", "0", "1048576"],
                ["-151552", "5492952.0000005495", "2166784", "6928384", "2449408"],
                ["-1048576", "2166784", "14680064", "0", "0"],
                ["0", "6928384", "0", "10485760", "-1048576"],
                ["1048576", "2449408", "0", "-1048576", "16777216"],
            ],
            "0",
            "2"),
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unknown witness"),
    };

    /// <summary>
    /// Near-margin budget-orthogonal problems a fresh-seed held-out control found (family <c>nearmo</c>, prefixes
    /// <c>v4p-</c>, <c>v4s-</c>): <c>Σ = v·(I − (1−ε)/k·ssᵀ)</c> rounded once into decimal, with a sign vector <c>s</c>
    /// of equal halves, so the near-null direction <c>s</c> is orthogonal to the budget row and the exact optimum is
    /// <c>1/n</c> in every weight. The free block sits near the top of the qualified band (<c>48·n·u·κ_eq</c> of 0.66 and
    /// 0.82). The rounded reduced Hessian resolves that direction only coarsely (each exact-residual refinement step
    /// shrinks the error along it by about a third or a half), so a refinement stopped at its step limit left the
    /// vector 1.06 and 2.02 times the bound away from the optimum along <c>s</c>; the solver must return the optimum
    /// within the bound.
    /// </summary>
    [Theory]
    [InlineData("v4p-302#1")]
    [InlineData("v4s-149#0")]
    public void NewCore_NearMarginBudgetOrthogonalWitness_IsQualifiedAndReturnsTheExactOptimumWithinTheBound(string witness)
    {
        var result = SolveHeldOut(NearMarginWitness(witness));
        Assert.True(result.Qualified, witness + " must be a qualified case: " + result.Detail);
        Assert.True(result.Outcome == HeldOutOutcome.Ok, witness + ": " + result.Outcome + ": " + result.Detail);
    }

    /// <summary>
    /// The same near-margin shape (<c>n = 10</c>, <c>v = 0.74</c>, <c>ε = 1.1e-25</c>, signs alternating, rounded once
    /// into decimal; <c>48·n·u·κ_eq</c> about 0.79) with the upper bound <c>0.1003</c> three sensitivity bounds above the
    /// interior optimum <c>0.1</c>. The first solve of the working set lands outside the box along the near-null
    /// direction, so the blocking bound is decided on the refined minimiser; the refinement steps allowed shrink that
    /// error only to about 0.45⁵ of the first solve's, which still crosses the bound, and a solver that took that
    /// blocking bound stopped on a vertex 3 times the bound away from the optimum. The solver must return the interior
    /// optimum within the bound.
    /// </summary>
    [Theory]
    [InlineData("tightub-10#0")]
    public void NewCore_NearMarginBlockWithABoundJustAboveTheOptimum_IsQualifiedAndReturnsTheExactOptimumWithinTheBound(string witness)
    {
        var result = SolveHeldOut(NearMarginWitness(witness));
        Assert.True(result.Qualified, witness + " must be a qualified case: " + result.Detail);
        Assert.True(result.Outcome == HeldOutOutcome.Ok, witness + ": " + result.Outcome + ": " + result.Detail);
    }

    private static HeldOutProblem NearMarginWitness(string name) => name switch
    {
        "tightub-10#0" => new HeldOutProblem(
            name,
            "decimal",
            [
                ["0.6660000000000000000000000081", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919"],
                ["0.0739999999999999999999999919", "0.6660000000000000000000000081", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919"],
                ["-0.0739999999999999999999999919", "0.0739999999999999999999999919", "0.6660000000000000000000000081", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919"],
                ["0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "0.6660000000000000000000000081", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919"],
                ["-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "0.6660000000000000000000000081", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919"],
                ["0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "0.6660000000000000000000000081", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919"],
                ["-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "0.6660000000000000000000000081", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919"],
                ["0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "0.6660000000000000000000000081", "0.0739999999999999999999999919", "-0.0739999999999999999999999919"],
                ["-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "0.6660000000000000000000000081", "0.0739999999999999999999999919"],
                ["0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "-0.0739999999999999999999999919", "0.0739999999999999999999999919", "0.6660000000000000000000000081"],
            ],
            "-1",
            "0.1003"),
        "v4p-302#1" => new HeldOutProblem(
            name,
            "decimal",
            [
                ["0.8066666666666666666666666784", "-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216"],
                ["-0.0733333333333333333333333216", "0.8066666666666666666666666784", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216"],
                ["-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.8066666666666666666666666784", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216"],
                ["0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.8066666666666666666666666784", "-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216"],
                ["0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.8066666666666666666666666784", "-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216"],
                ["0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.8066666666666666666666666784", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216"],
                ["0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.8066666666666666666666666784", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216"],
                ["-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.8066666666666666666666666784", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216"],
                ["0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.8066666666666666666666666784", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216"],
                ["-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.8066666666666666666666666784", "0.0733333333333333333333333216", "-0.0733333333333333333333333216"],
                ["0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.8066666666666666666666666784", "0.0733333333333333333333333216"],
                ["-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "-0.0733333333333333333333333216", "0.0733333333333333333333333216", "0.8066666666666666666666666784"],
            ],
            "0",
            "1"),
        "v4s-149#0" => new HeldOutProblem(
            name,
            "decimal",
            [
                ["0.6660000000000000000000000078", "0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922"],
                ["0.0739999999999999999999999922", "0.6660000000000000000000000078", "0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922", "-0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922"],
                ["-0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.6660000000000000000000000078", "0.0739999999999999999999999922", "-0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922"],
                ["0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.6660000000000000000000000078", "0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922", "-0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922"],
                ["-0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.6660000000000000000000000078", "-0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922"],
                ["-0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.6660000000000000000000000078", "0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922"],
                ["0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.6660000000000000000000000078", "-0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922"],
                ["0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.6660000000000000000000000078", "-0.0739999999999999999999999922", "0.0739999999999999999999999922"],
                ["0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.6660000000000000000000000078", "0.0739999999999999999999999922"],
                ["-0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922", "-0.0739999999999999999999999922", "-0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.0739999999999999999999999922", "0.6660000000000000000000000078"],
            ],
            "-1",
            "2"),
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unknown witness"),
    };

    private static HeldOutProblem NearSingularBudgetOrthogonalBlock(int k)
    {
        var quarterEps = Math.ScaleB(k, -52);
        var diagonal = (0.75 + quarterEps).ToString("R", CultureInfo.InvariantCulture);
        var off = (0.25 - quarterEps).ToString("R", CultureInfo.InvariantCulture);
        int[] s = [1, -1, -1, 1];
        var cov = new string[4][];
        for (var i = 0; i < 4; i++)
        {
            cov[i] = new string[4];
            for (var j = 0; j < 4; j++)
            {
                cov[i][j] = i == j ? diagonal : (s[i] * s[j] > 0 ? "-" : "") + off;
            }
        }

        return new HeldOutProblem("near-singular-" + k, "double", cov, "-1", "2");
    }

    // ------------------------------------------------------------------------------------------
    // AC-3.1 / AC-4.1  qualification and the conditioning guard measure one block (amendment #11). Corpus membership
    //                  takes κ_eq after the covariance-only equilibration QpScaling<T>.EquilibrateCovariance, the block
    //                  the solver's guard measures. The witnesses are decimal problems (optimum strictly inside the
    //                  box, so B = F*) whose 28-place bound or mean held an exponent at 0 under the definition before
    //                  the amendment (exponents chosen with the bounds and means), which qualified them, while the
    //                  guard measured the covariance-only block past the margin and refused them. Each witness asserts
    //                  its exponents and its classification under both definitions (48·n·u·κ_eq in exact rationals)
    //                  and that the shared oracle's classification and the solver agree: a qualified problem is solved
    //                  within the AC-3.1 bound; a refusal-permitted one is solved within it or refused naming
    //                  stationarity.
    // ------------------------------------------------------------------------------------------

    [Theory]
    [InlineData("r2-L7-K2-b78-box0")]
    [InlineData("r2-L7-K2-b78-boxpm")]
    [InlineData("r2-L7-K2-b62-box0")]
    [InlineData("r2-L7-K3-lb1e-28")]
    public void ActiveSetQpMinVariance_BoundLimitedExponentWitness_IsClassifiedOnTheCovarianceOnlyScaling_AndTheSolverAgrees(string witness)
        => AssertQualificationAgreementWitness(witness);

    [Theory]
    [InlineData("r2-L1-K1-mu1e-28")]
    [InlineData("r2-L5-K1-mu28")]
    [InlineData("r2-L7-K1-s771")]
    [InlineData("r2-L7-K1-s720")]
    [InlineData("r2-L7-K1-s796")]
    [InlineData("r2-L7-K1-s670")]
    [InlineData("r2-L7-K2-meanclamp")]
    [InlineData("r2-L7-K3-lb1e-28-mv")]
    public void ActiveSetQpMeanVariance_BoundOrMeanLimitedExponentWitness_IsClassifiedOnTheCovarianceOnlyScaling_AndTheSolverAgrees(string witness)
        => AssertQualificationAgreementWitness(witness);

    private static void AssertQualificationAgreementWitness(string witness)
    {
        var (problem, earlierExponents, covarianceOnlyExponents) = QualificationAgreementWitness(witness);
        var margins = MarginsUnderBothDefinitions(problem);

        // The exponents of both definitions, from the in-test port, and the shared function's covariance-only exponents.
        Assert.Equal(earlierExponents, margins.EarlierExponents);
        Assert.Equal(covarianceOnlyExponents, margins.CovarianceOnlyExponents);
        Assert.Equal(covarianceOnlyExponents, Boutquin.Numerics.Internal.QpScaling<decimal>.EquilibrateCovariance(DecimalMatrix(problem.Cov)).Exponents);

        // The classification under both definitions: qualified before the amendment, refusal-permitted after it.
        Assert.True(margins.Interior, witness + ": the exact optimum must lie strictly inside the box (B = F*)");
        Assert.True(margins.EarlierQualified, witness + ": the earlier definition qualifies it, 48·n·u·κ_eq = " + margins.EarlierMargin);
        Assert.False(margins.CovarianceOnlyQualified, witness + ": the covariance-only definition makes it refusal-permitted, 48·n·u·κ_eq = " + margins.CovarianceOnlyMargin);

        var result = SolveHeldOut(problem);
        AssertOracleAndSolverAgree(result);
        Assert.True(
            result.Qualified == margins.CovarianceOnlyQualified,
            witness + ": the shared oracle must qualify on the covariance-only scaling the guard measures (oracle qualified=" + result.Qualified + ")");
    }

    /// <summary>
    /// The pin of amendment #11: a problem the shared oracle qualifies is solved within the AC-3.1 bound; a
    /// refusal-permitted one is solved within it or refused naming stationarity, never a wrong vector.
    /// </summary>
    internal static void AssertOracleAndSolverAgree(HeldOutResult result)
    {
        if (result.Qualified)
        {
            Assert.True(result.Outcome == HeldOutOutcome.Ok, "a qualified problem must be solved within the AC-3.1 bound, got " + result.Outcome + ": " + result.Detail);
            return;
        }

        Assert.True(
            result.Outcome == HeldOutOutcome.Ok || (result.Outcome == HeldOutOutcome.Refused && result.Detail.Contains("stationarity", StringComparison.Ordinal)),
            "a refusal-permitted problem must be solved within the AC-3.1 bound or refused naming stationarity, got " + result.Outcome + ": " + result.Detail);
    }

    private static (HeldOutProblem Problem, int[] Earlier, int[] CovarianceOnly) QualificationAgreementWitness(string name)
    {
        const string TwoThirds = "0.6666666666666666666666666667";
        const string MinimumPositive = "0.0000000000000000000000000001";
        string[][] Pair(string a, string b, string d) => [[a, b], [b, d]];
        return name switch
        {
            // L1-K1 F-L1-01: the 28-place mean 1e-28 held e₀ at 0; the guard's exponents are (1, 0).
            "r2-L1-K1-mu1e-28" => (
                new HeldOutProblem(name, "decimal", Pair("1.01", "0.999949998749937496093476492", "0.99"), "-1000", "1000", [MinimumPositive, "0"], "1"),
                [0, 0], [1, 0]),

            // L5-K1 F-L5-01: the 28-place mean 0.0100000000000000000000000001 held e₁ at 0.
            "r2-L5-K1-mu28" => (
                new HeldOutProblem(name, "decimal", Pair("0.99", "0.999949998749937496093476501", "1.01"), "0", "1", ["0", "0.0100000000000000000000000001"], "1"),
                [0, 0], [0, 1]),

            // L7-K1 F-L7-01: the 28-place second mean held e₁ at 0 (four off-diagonals around the margin).
            "r2-L7-K1-s771" => (L7K1("9.94987437106619954734479771"), [1, 0], [1, 1]),
            "r2-L7-K1-s720" => (L7K1("9.94987437106619954734479720"), [1, 0], [1, 1]),
            "r2-L7-K1-s796" => (L7K1("9.94987437106619954734479796"), [1, 0], [1, 1]),
            "r2-L7-K1-s670" => (L7K1("9.94987437106619954734479670"), [1, 0], [1, 1]),

            // L7-K2 F-L7-01: the 28-place bound 2/3 held e₀ at 0 (box [0, 2/3] and [−2/3, 2/3]); and a mean-clamped analogue.
            "r2-L7-K2-b78-box0" => (
                new HeldOutProblem(name, "decimal", Pair("0.0099", "-0.0099498743710661995473447978", "0.01"), "0", TwoThirds),
                [0, 0], [-1, 0]),
            "r2-L7-K2-b78-boxpm" => (
                new HeldOutProblem(name, "decimal", Pair("0.0099", "-0.0099498743710661995473447978", "0.01"), "-" + TwoThirds, TwoThirds),
                [0, 0], [-1, 0]),
            "r2-L7-K2-b62-box0" => (
                new HeldOutProblem(name, "decimal", Pair("0.0099", "-0.0099498743710661995473447962", "0.01"), "0", TwoThirds),
                [0, 0], [-1, 0]),
            "r2-L7-K2-meanclamp" => (
                new HeldOutProblem(name, "decimal", Pair("99", "-99.498743710661995473447978", "100"), "0", "1", ["0.1234567890123456789012345678", "0.0987654321098765432109876543"], "1"),
                [0, 0], [1, 2]),

            // L7-K3 F-L7-01: the 28-place lower bound 1e-28 held e₀ at 0, through both entry points.
            "r2-L7-K3-lb1e-28" => (
                new HeldOutProblem(name, "decimal", Pair("0.0099", "-0.0099994999874993749609347614", "0.0101"), MinimumPositive, "1"),
                [0, 0], [-1, 0]),
            "r2-L7-K3-lb1e-28-mv" => (
                new HeldOutProblem(name, "decimal", Pair("0.0099", "-0.0099994999874993749609347614", "0.0101"), MinimumPositive, "1", ["0", "0"], "2"),
                [0, 0], [-1, 0]),
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unknown witness"),
        };

        HeldOutProblem L7K1(string s) => new(
            "r2-L7-K1-s" + s[^3..], "decimal", Pair("99", s, "1"), "-1", "2", ["0.5", "0.1234567890123456789012345671"], "1");
    }

    /// <summary>The margins of a decimal problem under both exponent definitions (see <see cref="MarginsUnderBothDefinitions"/>).</summary>
    internal sealed record QualificationMargins(
        int[] EarlierExponents,
        int[] CovarianceOnlyExponents,
        double EarlierMargin,
        double CovarianceOnlyMargin,
        bool EarlierQualified,
        bool CovarianceOnlyQualified,
        bool Interior);

    /// <summary>
    /// A decimal problem's <c>48·n·u·κ_eq</c> on the free block of its exact optimum, in exact rationals, under the
    /// covariance-only exponents of amendment #11 and under the definition before it (exponents also limited where a
    /// scaled bound or mean would round). The exponents come from an in-test port (<see cref="DecimalScalingExponents"/>),
    /// not from <c>QpScaling&lt;T&gt;</c>; <c>Interior</c> says whether every variable lies strictly inside the
    /// box (<c>B = F*</c>).
    /// </summary>
    internal static QualificationMargins MarginsUnderBothDefinitions(HeldOutProblem p)
    {
        Assert.Equal("decimal", p.Type);
        var c = HeldOutCase(p);
        var q = QpInstance.From(c);
        var (w0, _) = ExactOracle.ActiveSet(q);
        var free = DecidingBlock(w0, q.Lb, q.Ub);
        var unit = Rational.Of(BigInteger.One, BigInteger.Pow(10, 28));
        var admissible = q.InDomain(p.Type) && w0.All(x => HeldOutRepresentable(p.Type, x));
        var cov = DecimalMatrix(p.Cov);
        var earlier = DecimalScalingExponents(cov, (ParseDecimal(p.Lb), ParseDecimal(p.Ub), p.Mu?.Select(ParseDecimal).ToArray()));
        var covarianceOnly = DecimalScalingExponents(cov, limits: null);
        var earlierMargin = Rational.FromInt(48L * q.N) * unit * ExactOracle.Cond1(ScaledByPowersOfTen(q.Cov, earlier), free);
        var covarianceOnlyMargin = Rational.FromInt(48L * q.N) * unit * ExactOracle.Cond1(ScaledByPowersOfTen(q.Cov, covarianceOnly), free);
        return new QualificationMargins(
            earlier,
            covarianceOnly,
            double.Parse(Scientific(earlierMargin, 17), NumberStyles.Float, CultureInfo.InvariantCulture),
            double.Parse(Scientific(covarianceOnlyMargin, 17), NumberStyles.Float, CultureInfo.InvariantCulture),
            admissible && earlierMargin < Rational.One,
            admissible && covarianceOnlyMargin < Rational.One,
            free.Length == q.N);
    }

    private static decimal[,] DecimalMatrix(string[][] rows)
    {
        var n = rows.Length;
        var m = new decimal[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                m[i, j] = ParseDecimal(rows[i][j]);
            }
        }

        return m;
    }

    private static Rational[,] ScaledByPowersOfTen(Rational[,] cov, int[] exponents)
    {
        var n = exponents.Length;
        var scaled = new Rational[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                scaled[i, j] = cov[i, j] / PowerOfTen(exponents[i] + exponents[j]);
            }
        }

        return scaled;
    }

    /// <summary>
    /// Port of the decimal exponent choice, decided on the decimal's own mantissa and scale: the radix power just above
    /// <c>√Σᵢᵢ</c> (in exact rationals), then each exponent moved toward 0 while an entry of <c>D⁻¹ΣD⁻¹</c> would round
    /// (with <paramref name="limits"/> null: the covariance-only scaling of amendment #11), and, with the bounds and
    /// the linear term given, also while a scaled bound <c>Dᵢ·lb</c>, <c>Dᵢ·ub</c> or a scaled mean <c>μᵢ/Dᵢ</c> would
    /// round (the definition before amendment #11). Same loop and order as the scaling it ports.
    /// </summary>
    private static int[] DecimalScalingExponents(decimal[,] cov, (decimal Lb, decimal Ub, decimal[]? Mu)? limits)
    {
        var n = cov.GetLength(0);
        var e = new int[n];
        for (var i = 0; i < n; i++)
        {
            if (cov[i, i] > 0m)
            {
                var x = Rational.FromDecimal(cov[i, i]);
                while (x >= PowerOfTen(2 * e[i]))
                {
                    e[i]++;
                }

                while (x < PowerOfTen(2 * (e[i] - 1)))
                {
                    e[i]--;
                }
            }
        }

        bool changed;
        do
        {
            changed = false;
            for (var i = 0; i < n; i++)
            {
                for (var j = i; j < n; j++)
                {
                    var k = -(e[i] + e[j]);
                    if (k != 0 && !(DecimalShiftIsExact(cov[i, j], k) && DecimalShiftIsExact(cov[j, i], k)))
                    {
                        var m = k < 0 ? (e[i] >= e[j] ? i : j) : (e[i] <= e[j] ? i : j);
                        e[m] += k < 0 ? -1 : 1;
                        changed = true;
                    }
                }

                var ei = e[i];
                if (ei != 0 && limits is (var lb, var ub, var mu)
                    && (!DecimalShiftIsExact(lb, ei) || !DecimalShiftIsExact(ub, ei) || (mu is not null && !DecimalShiftIsExact(mu[i], -ei))))
                {
                    e[i] += ei > 0 ? -1 : 1;
                    changed = true;
                }
            }
        }
        while (changed);
        return e;
    }

    // Whether x·10^k has a decimal representation: a scale of 0 to 28 places after the shift (trailing zeros of the
    // mantissa may be dropped), and a mantissa that fits 96 bits when the shift grows it.
    private static bool DecimalShiftIsExact(decimal x, int k)
    {
        if (k == 0 || x == 0m)
        {
            return true;
        }

        var bits = decimal.GetBits(x);
        var target = ((bits[3] >> 16) & 0xFF) - k;
        if (target is >= 0 and <= 28)
        {
            return true;
        }

        var mantissa = (new BigInteger((uint)bits[2]) << 64) | (new BigInteger((uint)bits[1]) << 32) | new BigInteger((uint)bits[0]);
        if (target < 0)
        {
            return mantissa * BigInteger.Pow(10, -target) < BigInteger.One << 96;
        }

        while (target > 28 && (mantissa % 10).IsZero)
        {
            mantissa /= 10;
            target--;
        }

        return target <= 28;
    }

    /// <summary>
    /// The unconstrained inverse-variance weights for diag(1,2,3,4) lie inside [0,0.5].
    /// Verify that the independent active-set oracle agrees with their closed form
    /// and that the production solver returns each exact optimum rounded to double.
    /// </summary>
    [Fact]
    public void DiagonalCovariance_ReturnsTheCorrectlyRoundedIndependentOptimum()
    {
        var problem = Diagonal("closed-form-diag-1-2-3-4", "double", ["1", "2", "3", "4"], "0", "0.5");
        var (w0, pattern) = ExactOracle.ActiveSet(QpInstance.From(HeldOutCase(problem)));

        // The oracle's optimum agrees with the closed form w ∝ 1/Σᵢᵢ (all four weights free inside the box).
        Rational[] closedForm = [Rational.Of(12, 25), Rational.Of(6, 25), Rational.Of(4, 25), Rational.Of(3, 25)];
        Assert.Equal(closedForm, w0);
        Assert.All(pattern, p => Assert.Equal(0, p));

        // The production entry point returns every weight as the correctly rounded exact optimum.
        var cov = new double[4, 4] { { 1, 0, 0, 0 }, { 0, 2, 0, 0 }, { 0, 0, 3, 0 }, { 0, 0, 0, 4 } };
        var w = ActiveSetQpSolver<double>.SolveMinVariance(cov, 0.0, 0.5);
        for (var i = 0; i < 4; i++)
        {
            var e = (Rational.FromDouble(w[i]) - w0[i]).Abs();
            var u = Rational.FromDouble(Math.BitIncrement(w[i])) - Rational.FromDouble(w[i]);
            Assert.True(Rational.FromInt(2) * e < u, "w[" + i + "] = " + w[i].ToString("R", CultureInfo.InvariantCulture) + " is not the correctly rounded " + w0[i]);
        }

    }

    /// <summary>
    /// Seeded census of the repeated-structure shape (one to three groups, group covariance
    /// <c>c ∈ {0.5, 0.95, 1.5, 2, 3}</c>, ridge <c>d ∈ {1e-8, 1e-6, 1e-4, 0.05}</c>, bounds at and around the
    /// equal-weight frontier, a long-short box) at both types: no qualified case refused, no wrong vector, no budget
    /// exhaustion.
    /// </summary>
    [Fact]
    public void NewCore_RepeatedStructureCensus_NoQualifiedRefusalNoWrongVectorNoBudgetExhaustion()
    {
        var rng = new System.Random(20261009);
        var results = new List<HeldOutResult>();
        for (var trial = 0; trial < 240; trial++)
        {
            var problem = RepeatedStructureProblem(rng, trial % 2 == 0 ? "double" : "decimal", trial);
            if (problem is not null)
            {
                results.Add(SolveHeldOut(problem));
            }
        }

        AssertCensus(results, minimumQualified: 120);
    }

    /// <summary>
    /// Seeded census of diagonal and near-diagonal covariances across variance ratios from 1e4 to 1e298 (double)
    /// and 1e4 to 1e27 (decimal), with the box <c>[0, 1]</c> and three other boxes, at both types: no budget
    /// exhaustion, no wrong vector, no qualified case refused.
    /// </summary>
    [Fact]
    public void NewCore_DiagonalDisparityCensus_NoBudgetExhaustionNoWrongVectorNoQualifiedRefusal()
    {
        var rng = new System.Random(20261010);
        var results = new List<HeldOutResult>();
        for (var trial = 0; trial < 240; trial++)
        {
            var problem = DiagonalProblem(rng, trial % 2 == 0 ? "double" : "decimal", trial);
            if (problem is not null)
            {
                results.Add(SolveHeldOut(problem));
            }
        }

        AssertCensus(results, minimumQualified: 40);
    }

    /// <summary>
    /// Seeded census of the zero-correlation shapes at both types, <c>n = 2..12</c>: one dominant asset coupled to a
    /// block <c>v·(I − (1−ε)/k·ssᵀ)</c> whose near-null direction <c>s</c> has relative curvature <c>ε ∈ [1e-16, 1e-10]</c>,
    /// and the two-block variant, inside wide long-short or long-only boxes. Both directions of the conditioning
    /// refusal are pinned: no wrong vector on any case, and no refusal of a qualified case.
    /// </summary>
    [Fact]
    public void NewCore_ZeroCorrelationCensus_NoWrongVectorNoQualifiedRefusal()
    {
        var rng = new System.Random(20261011);
        var results = new List<HeldOutResult>();
        for (var trial = 0; trial < 240; trial++)
        {
            var problem = ZeroCorrelationProblem(rng, trial % 2 == 0 ? "double" : "decimal", trial);
            if (problem is not null)
            {
                results.Add(SolveHeldOut(problem));
            }
        }

        AssertCensus(results, minimumQualified: 40);
    }

    // ------------------------------------------------------------------------------------------
    // AC-4.1 beyond the committed corpus: seeded mean-variance censuses of the shapes the held-out controls attack
    // (near-margin blocks, degenerate at a bound, the D1 cancellation and D4 masking regimes), at double and decimal,
    // through the production SolveMeanVariance, each output classified against the exact rational optimum of
    // max μ′w − (λ/2)·w′Σw with the AC-3.1 bound; the qualification uses the covariance-only equilibration
    // QpScaling<T>.EquilibrateCovariance (amendment #11), the block the solver's conditioning guard measures.
    // ------------------------------------------------------------------------------------------

    /// <summary>
    /// Blocks <c>v·(I − (1−ε)/k·ssᵀ)</c> built exactly and rounded once, with the near-null direction <c>s</c> either
    /// orthogonal to the budget row (half the signs negative) or generic, <c>ε</c> straddling the margin
    /// <c>48·n·u·κ_eq = 1</c>, random means and risk aversion from 1e-40 (double) or 1e-6 (decimal) to 1e40 or 1e6:
    /// no qualified case refused, no wrong vector, no budget exhaustion, and every member whose deciding block crosses
    /// the margin refused naming stationarity (the conditioning guard of design step 4 on the mean-variance entry point).
    /// </summary>
    [Fact]
    public void ActiveSetQpMeanVariance_NearMarginBlockCensus_NoWrongVectorNoQualifiedRefusal()
    {
        var rng = new System.Random(20261013);
        var results = new List<HeldOutResult>();
        for (var trial = 0; trial < 160; trial++)
        {
            var problem = MeanVarianceNearMarginProblem(rng, trial % 2 == 0 ? "double" : "decimal", trial);
            if (problem is not null)
            {
                results.Add(SolveHeldOut(problem));
            }
        }

        // 160 problems at the committed seed, 137 qualified (23 refusal-permitted, refused by the guard).
        AssertCensus(results, minimumQualified: 110);
        AssertMarginCrossersAreRefusedNamingStationarity(results, expectedCrossers: 23);
    }

    /// <summary>
    /// Degenerate at a bound: a seeded problem is solved exactly, then one variable at a bound has its own mean moved
    /// so that its exact multiplier is zero or a tiny fraction of its former value (the optimum is unchanged, since a
    /// bound variable's mean enters only its own multiplier), rounded once into the working type: no qualified case
    /// refused, no wrong vector, no budget exhaustion.
    /// </summary>
    [Fact]
    public void ActiveSetQpMeanVariance_DegenerateAtBoundCensus_NoWrongVectorNoQualifiedRefusal()
    {
        var rng = new System.Random(20261014);
        var results = new List<HeldOutResult>();
        for (var trial = 0; trial < 160; trial++)
        {
            var problem = MeanVarianceDegenerateAtBoundProblem(rng, trial % 2 == 0 ? "double" : "decimal", trial);
            if (problem is not null)
            {
                results.Add(SolveHeldOut(problem));
            }
        }

        // 149 admissible problems at the committed seed, all qualified.
        AssertCensus(results, minimumQualified: 120);
    }

    /// <summary>
    /// The two mean-variance defect regimes of the pre-rewrite core. D1: a dominant asset beside a correlated block,
    /// with means <c>μ = λΣ(1/n·1 + δz)</c> for a budget-orthogonal <c>z</c> and <c>δ</c> below the rounding of the
    /// budget share, so the budget-orthogonal part of <c>Σ⁻¹μ</c> is below that rounding. D4: one or two high-variance
    /// assets beside a low-variance block, with one dominant mean, so that every gradient is dominated by one asset
    /// while a low-variance asset must still be released. No qualified case refused, no wrong vector.
    /// </summary>
    [Fact]
    public void ActiveSetQpMeanVariance_CancellationAndMaskingCensus_NoWrongVectorNoQualifiedRefusal()
    {
        var rng = new System.Random(20261015);
        var results = new List<HeldOutResult>();
        for (var trial = 0; trial < 160; trial++)
        {
            var problem = MeanVarianceCancellationOrMaskingProblem(rng, (trial / 2) % 2 == 0 ? "double" : "decimal", trial);
            if (problem is not null)
            {
                results.Add(SolveHeldOut(problem));
            }
        }

        // 151 admissible problems at the committed seed, all qualified.
        AssertCensus(results, minimumQualified: 120);
    }

    /// <summary>
    /// Mean-variance problems at the edge of the decimal operating domain (every input at most 1e14) whose exact optimum
    /// is representable and well inside the box while single products <c>λ·Σᵢⱼ·wⱼ</c> of the gradient exceed the
    /// decimal range: a near-null block along a budget-orthogonal direction <c>s</c>, means along <c>s</c>, and a risk
    /// aversion that leverages the optimum along <c>s</c> to about 1e12–1e13, so each gradient row is a difference of
    /// terms beyond 7.9e28 although the row itself is representable. <c>range-witness-1</c> is
    /// <c>Σ = 1e8·[[1, 1 − 1e-15], [1 − 1e-15, 1]]</c>, <c>μ = (1e14, −1e14)</c>, <c>λ = 1e8</c>, box <c>±1e14</c>,
    /// exact optimum <c>(1e13 + 0.5, −1e13 + 0.5)</c>; the other five are the inputs two held-out controls replayed
    /// (kernel seeds 2026101501 and 2026101502, decimal sub-problems). Each is qualified and must return the exact
    /// optimum within the bound through the production entry point, never a refusal.
    /// </summary>
    [Theory]
    [InlineData("range-witness-1")]
    [InlineData("_c0b9c _9-.YXYc #2")]
    [InlineData("/a-_Y!X/cbX1!1#3")]
    [InlineData("/1XX11Y 1-#1")]
    [InlineData("_c/X _1-b_c Y_ccbZZb-#2")]
    [InlineData("1bcb-bZ! ///__-9YX#2")]
    public void ActiveSetQpMeanVariance_DomainEdgeRangeWitness_IsQualifiedAndReturnsTheExactOptimumWithinTheBound(string witness)
    {
        var problem = DomainEdgeRangeWitness(witness);
        Assert.True(MaxQuadraticTerm(problem) > s_decimalMax, witness + " must have a gradient product λ·Σᵢⱼ·wⱼ beyond the decimal range");
        var result = SolveHeldOut(problem);
        Assert.True(result.Qualified, witness + " must be a qualified case: " + result.Detail);
        Assert.True(result.Outcome == HeldOutOutcome.Ok, witness + ": " + result.Outcome + ": " + result.Detail);
    }

    /// <summary>
    /// The range class of the witnesses above as a seeded census at both types: blocks <c>c·(I − (1−ε)/k·ssᵀ)</c> with the
    /// scale <c>c</c> up to the edge of the operating domain, <c>s</c> budget-orthogonal (two thirds of the trials) or
    /// generic, means <c>m·s</c> with a 0.1% perturbation, the risk aversion chosen so that the optimum moves by up to
    /// 3e13 (decimal) or 1e140 (double) along <c>s</c>, and a box either well outside the optimum or cutting it (a third
    /// of the trials, so that variables sit at a bound with a gradient row far beyond the decimal range). No qualified
    /// case refused, no wrong vector, no budget exhaustion; the census must reach the class: decimal problems whose
    /// exact optimum has a gradient product <c>λ·Σᵢⱼ·wⱼ</c> beyond the decimal range.
    /// </summary>
    [Fact]
    public void ActiveSetQpMeanVariance_DomainEdgeRangeCensus_NoWrongVectorNoQualifiedRefusal()
    {
        var rng = new System.Random(20261017);
        var results = new List<HeldOutResult>();
        var beyondRange = 0;
        for (var trial = 0; trial < 320; trial++)
        {
            var type = trial % 2 == 0 ? "double" : "decimal";
            var problem = MeanVarianceDomainEdgeRangeProblem(rng, type, trial);
            if (problem is null)
            {
                continue;
            }

            if (type == "decimal" && MaxQuadraticTerm(problem) > s_decimalMax)
            {
                beyondRange++;
            }

            results.Add(SolveHeldOut(problem));
        }

        Assert.True(beyondRange >= 30, "the census must reach the range class: " + beyondRange + " decimal problems beyond the range of " + results.Count + " admissible");
        // 242 admissible problems at the committed seed, all qualified, 38 decimal ones beyond the decimal range.
        AssertCensus(results, minimumQualified: 200);
    }

    /// <summary>
    /// Qualified double mean-variance problems with extreme or mixed scales that an earlier core refused with "the
    /// result failed the KKT certificate (multiplier sign ...)": the exit certificate rejected the final iterate for a
    /// multiplier of the wrong sign, orders of magnitude beyond its tolerance, and the release rule offered no
    /// candidate. The covariances mix variances from about 1e-150 to 1e150 (random correlations or diagonal), the boxes
    /// are <c>[0, 1]</c> or as wide as <c>±1e150</c>, and the risk aversion is matched to the smallest, median or largest
    /// variance or sits at a domain edge. At weights of magnitude 1e50 to 1e150 the rounded iterate cannot carry the
    /// budget to the unit (a weight's own rounding is far above 1), so the trajectory of the rounded core is not that of
    /// the exact method. The inputs are those of held-out controls (kernel seeds 2026101603, 2026101611 and 2026101612
    /// and seeded batches), dumped as explicit matrices; each is qualified and must return the exact optimum within the
    /// bound through the production entry point, never a refusal.
    /// </summary>
    [Theory]
    [InlineData("cZZ!Y1. #3")]
    [InlineData("x9b-167#3")]
    [InlineData("x9c-98#2")]
    [InlineData("x9d-47#0")]
    [InlineData("x9d-241#1")]
    [InlineData("x9e-98#1")]
    [InlineData("x9e-165#1")]
    [InlineData("_/c!a/1X Xa9c9Xb#3")]
    [InlineData("9!-a -b-  a.c9b0_199_9XZ#1")]
    [InlineData("X91/Y/ c /.#3")]
    [InlineData("_9 .!b_.b_..-cbc 1X1./#0")]
    [InlineData("._c!bcZ cc 0a_X a0/YZ#1")]
    [InlineData("9YX XYc1#0")]
    [InlineData("aY!a-_cX!-!Z/.#1")]
    [InlineData("0a-!0aZbYb1aY0.0a/ab/! Y#0")]
    public void ActiveSetQpMeanVariance_ExtremeScaleCertificateWitness_IsQualifiedAndReturnsTheExactOptimumWithinTheBound(string witness)
    {
        var problem = ExtremeScaleWitness(witness);
        Assert.NotNull(problem.Mu);
        var result = SolveHeldOut(problem);
        Assert.True(result.Qualified, witness + " must be a qualified case: " + result.Detail);
        Assert.True(result.Outcome == HeldOutOutcome.Ok, witness + ": " + result.Outcome + ": " + result.Detail);
    }

    /// <summary>
    /// The same mechanism reached through minimum variance: a qualified double problem with mixed scales (n = 15,
    /// variances spanning the double operating domain) that the shared core refused with a multiplier-sign certificate
    /// failure. It must return the exact optimum within the bound.
    /// </summary>
    [Theory]
    [InlineData("x9y-24#1")]
    public void ActiveSetQpMinVariance_ExtremeScaleCertificateWitness_IsQualifiedAndReturnsTheExactOptimumWithinTheBound(string witness)
    {
        var problem = ExtremeScaleWitness(witness);
        Assert.Null(problem.Mu);
        var result = SolveHeldOut(problem);
        Assert.True(result.Qualified, witness + " must be a qualified case: " + result.Detail);
        Assert.True(result.Outcome == HeldOutOutcome.Ok, witness + ": " + result.Outcome + ": " + result.Detail);
    }

    /// <summary>
    /// The class of the witnesses above as a seeded census of mean-variance problems at both types: covariances
    /// <c>Σᵢⱼ = ρᵢⱼ·√(sᵢ·sⱼ)</c> with <c>sᵢ = 10^U(−149, 149)</c> (decimal <c>10^U(−12, 12)</c>), drawn over the whole range,
    /// a band of 14 decades or a random sub-range, random correlations or a diagonal; boxes <c>[0, 1]</c> or the widest
    /// in the domain; risk aversion <c>1/median s</c>, <c>1/max s</c>, <c>1/min s</c>, the domain maximum or 1; means
    /// matched to <c>λ·sᵢ</c>, of independent extreme scale, zero, or matched to <c>√sᵢ</c>. No qualified case refused, no
    /// wrong vector, no budget exhaustion.
    /// </summary>
    [Fact]
    public void ActiveSetQpMeanVariance_ExtremeScaleCensus_NoWrongVectorNoQualifiedRefusal()
    {
        var rng = new System.Random(20261018);
        var results = new List<HeldOutResult>();
        for (var trial = 0; trial < ExtremeScaleTrials; trial++)
        {
            var problem = ExtremeScaleProblem(rng, trial % 3 == 2 ? "decimal" : "double", trial, meanVariance: true);
            if (problem is not null)
            {
                results.Add(SolveHeldOut(problem));
            }
        }

        AssertCensus(results, minimumQualified: ExtremeScaleTrials / 2);
    }

    /// <summary>
    /// The same census through minimum variance (the covariances and boxes of the mean-variance census, no means): the
    /// release rule and the exit certificate are shared by both entry points.
    /// </summary>
    [Fact]
    public void ActiveSetQpMinVariance_ExtremeScaleCensus_NoWrongVectorNoQualifiedRefusal()
    {
        var rng = new System.Random(20261019);
        var results = new List<HeldOutResult>();
        for (var trial = 0; trial < ExtremeScaleTrials; trial++)
        {
            var problem = ExtremeScaleProblem(rng, trial % 3 == 2 ? "decimal" : "double", trial, meanVariance: false);
            if (problem is not null)
            {
                results.Add(SolveHeldOut(problem));
            }
        }

        AssertCensus(results, minimumQualified: ExtremeScaleTrials / 2);
    }

    private const int ExtremeScaleTrials = 1500;

    // ------------------------------------------------------------------------------------------
    // Degenerate structure: identical assets, repeated blocks, tied means, coinciding bounds
    // ------------------------------------------------------------------------------------------

    /// <summary>
    /// Qualified double mean-variance problems on which the core exhausted its iteration budget: seven identical 2×2
    /// near-perfect-correlation blocks at scale 1e127 beside two tiny-variance assets that take a long-short position
    /// at the box limit ±1.04e89, under 16 means tied to within a unit of their last place (<c>'1_-/XYX-1Z.YY.b'#3</c>,
    /// <c>n = 16</c>, λ = 1; the pre-rewrite core solved it), and three identical blocks with the same hedge legs at
    /// ±1.14e49 (<c>y1ub-35#0</c>, <c>n = 8</c>, λ about 1e-95). Each must return the exact optimum within the bound.
    /// </summary>
    [Theory]
    [InlineData("1_-/XYX-1Z.YY.b#3")]
    [InlineData("y1ub-35#0")]
    public void ActiveSetQpMeanVariance_BudgetExhaustionWitness_IsQualifiedAndReturnsTheExactOptimumWithinTheBound(string witness)
    {
        var problem = DegenerateStructureWitness(witness);
        Assert.NotNull(problem.Mu);
        var result = SolveHeldOut(problem);
        Assert.True(result.Qualified, witness + " must be a qualified case: " + result.Detail);
        Assert.True(result.Outcome == HeldOutOutcome.Ok, witness + ": " + result.Outcome + ": " + result.Detail);
        var route = RouteBudgetFailure(problem);
        Assert.True(route is null, route);
    }

    /// <summary>
    /// The exact route must be a complete active-set iteration: on a mixed-scale min-variance problem
    /// (<c>y1uh-38#0</c>, <c>n = 16</c>) with the box <c>[0, 0.25]</c>, the feasible start fills four weights to the upper
    /// bound, so the start is a vertex whose one free variable sits on its bound and every release first blocks at step
    /// length zero. Five symmetric permutations of it were refused (four by the exit certificate after the exact route
    /// rolled those zero-length steps back, one by budget exhaustion); the unpermuted problem is the natural trigger of
    /// the exact route with the box <c>[0, 1]</c>. Each must return the exact optimum within the bound.
    /// </summary>
    [Theory]
    [InlineData("y1uh-38#0")]
    [InlineData("y1uh-38#0/perm20")]
    [InlineData("y1uh-38#0/perm29")]
    [InlineData("y1uh-38#0/perm30")]
    [InlineData("y1uh-38#0/perm46")]
    [InlineData("y1uh-38#0/perm54")]
    public void ActiveSetQpMinVariance_ZeroLengthStepWitness_IsQualifiedAndReturnsTheExactOptimumWithinTheBound(string witness)
    {
        var problem = DegenerateStructureWitness(witness);
        Assert.Null(problem.Mu);
        var result = SolveHeldOut(problem);
        Assert.True(result.Qualified, witness + " must be a qualified case: " + result.Detail);
        Assert.True(result.Outcome == HeldOutOutcome.Ok, witness + ": " + result.Outcome + ": " + result.Detail);
        var route = RouteBudgetFailure(problem);
        Assert.True(route is null, route);
    }

    /// <summary>
    /// The <c>y1uh-38#0</c> family as a seeded census: symmetric permutations of its covariance under the boxes
    /// <c>[0, 1/k]</c> (k = 1, 2, 4, 8, 16), so that the feasible start is a vertex with free variables on their bounds
    /// in every position. No qualified case refused, no wrong vector, no budget exhaustion.
    /// </summary>
    [Fact]
    public void ActiveSetQpMinVariance_ZeroLengthStepPermutationCensus_NoBudgetExhaustionNoWrongVectorNoQualifiedRefusal()
    {
        var rng = new System.Random(20261021);
        var results = new List<HeldOutResult>();
        var routeFailures = new List<string?>();
        for (var trial = 0; trial < 40; trial++)
        {
            var permutation = Enumerable.Range(0, 16).ToArray();
            if (trial > 0)
            {
                permutation = [.. permutation.OrderBy(_ => rng.Next())];
            }

            foreach (var k in new[] { 1, 2, 4, 8, 16 })
            {
                var problem = PermutedY1uh38("y1uh-38#0/p" + trial + "/k" + k, permutation, Reciprocal("double", k));
                results.Add(SolveHeldOut(problem));
                routeFailures.Add(RouteBudgetFailure(problem));
            }
        }

        AssertCensus(results, minimumQualified: 150);
        AssertRouteBudgets(routeFailures);
    }

    /// <summary>
    /// Termination of both routes: whether either route of the primal core reaches its iteration budget on the problem.
    /// The production solve must not spend the rounded route's budget (a spent budget is handed to the exact route, so
    /// only the route diagnostics show it), and the exact route run on its own, from the exact feasible start, must not
    /// spend its own budget. Returns the failure, or <see langword="null"/>; refusals are judged by the census itself.
    /// </summary>
    private static string? RouteBudgetFailure(HeldOutProblem p) => p.Type == "double"
        ? RouteBudgetFailure<double>(p, ParseDouble)
        : RouteBudgetFailure<decimal>(p, ParseDecimal);

    private static string? RouteBudgetFailure<T>(HeldOutProblem p, Func<string, T> parse)
        where T : IFloatingPoint<T>
    {
        var n = p.Cov.Length;
        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                cov[i, j] = parse(p.Cov[i][j]);
            }
        }

        var lb = parse(p.Lb);
        var ub = parse(p.Ub);
        foreach (var force in new[] { false, true })
        {
            var trace = new ActiveSetQpSolver<T>.RouteTrace { ForceExactRoute = force };
            var route = force ? "exact route" : "rounded route";
            try
            {
                _ = p.Mu is null
                    ? ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(cov, lb, ub, forcedReference: -1, trace: trace)
                    : ActiveSetQpSolver<T>.SolveMeanVarianceUsingCore(cov, p.Mu.Select(parse).ToArray(), parse(p.Lambda!), lb, ub, trace: trace);
            }
            catch (InvalidOperationException ex) when (ex.Message.Contains("iteration budget", StringComparison.Ordinal))
            {
                return p.Name + " (" + p.Type + ", " + route + "): " + ex.Message;
            }
            catch (InvalidOperationException)
            {
                // A documented refusal: the census decides whether it is permitted.
            }

            if (!force && trace.RoundedBudgetReached)
            {
                return p.Name + " (" + p.Type + "): the rounded route spent its budget of " + ActiveSetQpSolver<T>.IterationBudget(n)
                    + " working-set changes and handed the problem to the exact route";
            }

            // The exact route is a complete active-set iteration: a release on a strictly negative exact multiplier
            // lowers the exact objective, so it never undoes one (in particular never a release whose step had length 0).
            if (trace.ExactRollbacks > 0)
            {
                return p.Name + " (" + p.Type + ", " + route + "): the exact route rolled back " + trace.ExactRollbacks + " release(s)";
            }
        }

        return null;
    }

    private static void AssertRouteBudgets(List<string?> failures)
    {
        var reached = failures.Where(f => f is not null).ToList();
        Assert.True(reached.Count == 0, reached.Count + " of " + failures.Count + " problems reached an iteration budget or saw the exact route roll a release back: " + string.Join(" | ", reached.Take(6)));
    }

    /// <summary>
    /// Degenerate structure as a seeded census of mean-variance problems at both types (<see cref="DegenerateStructureProblem"/>):
    /// repeated 2×2 blocks, groups of identical assets, a whole matrix of identical assets, repeated 3×3 blocks and tied
    /// factor models, each at a scale drawn over the operating domain, beside up to two tiny- or huge-variance hedge legs;
    /// means tied, tied within groups, a common offset with differences at its last place, random, matched to λ·Σᵢᵢ, or
    /// zero; boxes <c>[0, 1]</c>, <c>[0, 1/k]</c>, <c>[1/(2n), 1]</c> and <c>±B</c> with B up to 1e89 (decimal 1e14);
    /// n = 2..16. No qualified case refused, no wrong vector, no budget exhaustion.
    /// </summary>
    [Fact]
    public void ActiveSetQpMeanVariance_DegenerateStructureCensus_NoBudgetExhaustionNoWrongVectorNoQualifiedRefusal()
    {
        var rng = new System.Random(20261022);
        var results = new List<HeldOutResult>();
        var routeFailures = new List<string?>();
        for (var trial = 0; trial < DegenerateStructureTrials; trial++)
        {
            var problem = DegenerateStructureProblem(rng, trial % 2 == 0 ? "double" : "decimal", trial, meanVariance: true);
            if (problem is not null)
            {
                results.Add(SolveHeldOut(problem));
                routeFailures.Add(RouteBudgetFailure(problem));
            }
        }

        AssertCensus(results, minimumQualified: DegenerateStructureTrials / 3);
        AssertRouteBudgets(routeFailures);
    }

    /// <summary>The same census through minimum variance (the covariances and boxes of the mean-variance census, no means).</summary>
    [Fact]
    public void ActiveSetQpMinVariance_DegenerateStructureCensus_NoBudgetExhaustionNoWrongVectorNoQualifiedRefusal()
    {
        var rng = new System.Random(20261023);
        var results = new List<HeldOutResult>();
        var routeFailures = new List<string?>();
        for (var trial = 0; trial < DegenerateStructureTrials; trial++)
        {
            var problem = DegenerateStructureProblem(rng, trial % 2 == 0 ? "double" : "decimal", trial, meanVariance: false);
            if (problem is not null)
            {
                results.Add(SolveHeldOut(problem));
                routeFailures.Add(RouteBudgetFailure(problem));
            }
        }

        AssertCensus(results, minimumQualified: DegenerateStructureTrials / 3);
        AssertRouteBudgets(routeFailures);
    }

    private const int DegenerateStructureTrials = 600;

    // ------------------------------------------------------------------------------------------
    // Termination at n = 20: repeated 2×2 blocks, the exact route's budget on the production path
    // ------------------------------------------------------------------------------------------

    /// <summary>
    /// Termination on the production path, as a seeded property over repeated 2×2 blocks up to <c>n = 20</c>
    /// (<see cref="RepeatedTwoByTwoBlockProblem"/>): radix-scaled copies of one random positive-definite 2×2 block with a
    /// rank-one coupling, randomly permuted, at box limits from <c>[0, 1]</c> to <c>±1e89</c> and <c>±1e150</c> (decimal:
    /// <c>[0, 1]</c> and <c>±1e14</c>, the edge of its operating domain). Each problem runs the body of the public entry
    /// point with a route trace. The rounded route reaching its budget is the documented handoff to the exact route and is
    /// permitted; the exact route must never spend its own budget of <c>12n + 12</c> working-set changes, and every result
    /// must be within the bound of the exact optimum (or the documented refusal of a refusal-permitted problem).
    /// </summary>
    [Theory]
    [InlineData("double")]
    [InlineData("decimal")]
    public void ActiveSetQpMinVariance_RepeatedTwoByTwoBlockProperty_ExactRouteNeverSpendsItsBudget_ResultWithinTheBound(string type)
        => AssertRepeatedBlockProperty(type, meanVariance: false, seed: type == "double" ? 2026101916 : 2026101917);

    /// <summary>The same property through mean-variance: means zero, tied, identical per copy after equilibration, or random.</summary>
    [Theory]
    [InlineData("double")]
    [InlineData("decimal")]
    public void ActiveSetQpMeanVariance_RepeatedTwoByTwoBlockProperty_ExactRouteNeverSpendsItsBudget_ResultWithinTheBound(string type)
        => AssertRepeatedBlockProperty(type, meanVariance: true, seed: type == "double" ? 2026101918 : 2026101919);

    /// <summary>
    /// Problems of the family on which the production rounded route spent its budget (169 of 168 changes at <c>n = 20</c>)
    /// and handed the problem to the exact route, found by the held-out census at double with the box <c>±1e89</c> or
    /// <c>±1e150</c>. The exact route must finish within its own budget and the result must be within the bound.
    /// </summary>
    [Theory]
    [InlineData("z2i-mi0-13#0")]
    [InlineData("z2g-mi0-146#0")]
    [InlineData("z2i-mi2-54#3")]
    public void ActiveSetQpMinVariance_RoundedBudgetHandoffWitness_ExactRouteStaysWithinItsBudget_ResultWithinTheBound(string witness)
        => AssertRepeatedBlockWitness(witness);

    /// <summary>The mean-variance problems of the family on which the production rounded route spent its budget.</summary>
    [Theory]
    [InlineData("z2g-mv2-174#2")]
    [InlineData("z2i-mv2-76#3")]
    public void ActiveSetQpMeanVariance_RoundedBudgetHandoffWitness_ExactRouteStaysWithinItsBudget_ResultWithinTheBound(string witness)
        => AssertRepeatedBlockWitness(witness);

    /// <summary>
    /// A qualified mean-variance problem of the family (<c>n = 20</c>, box <c>±1e89</c>, random means) on which the rounded
    /// route returned a certified vector with a pair of near-identical tiny-variance assets on the wrong bounds (error
    /// 2e89 against a bound of 1.7e74): their gradients differ by about 1.3e-8, an exact difference far below the rounding
    /// of the reference row that sets the budget multiplier, so neither multiplier counted as negative and the certificate
    /// accepted. Found by the held-out census (<c>k7e-rblk-935007-135</c>). It must return the exact optimum within the
    /// bound, with the exact route within its budget.
    /// </summary>
    [Theory]
    [InlineData("k7e-935007-135")]
    public void ActiveSetQpMeanVariance_BoundPairBelowTheReferenceRounding_IsQualifiedAndReturnsTheExactOptimumWithinTheBound(string witness)
        => AssertRepeatedBlockWitness(witness);

    /// <summary>
    /// A qualified min-variance problem of the family (<c>n = 19</c>, box <c>±1e89</c>, every variable free at the exact
    /// optimum) on which the production route hands over to the exact route (refinement not converged): from the far
    /// start the exact route's released variables cross the box from bound to bound, and it needs 169 working-set changes
    /// (174 from the exact feasible start), more than the rounded route's <c>8n + 8 = 160</c> and within the exact
    /// route's own <c>12n + 12 = 240</c>. Found by the held-out census (<c>k7e-rblk-935004-236</c>). It must return the
    /// exact optimum within the bound, with the exact route within its budget; the test also pins that the exact route is
    /// entered and needs more than <c>8n + 8</c> changes, so that it keeps witnessing the size of the exact budget.
    /// </summary>
    [Theory]
    [InlineData("k7e-935004-236")]
    public void ActiveSetQpMinVariance_FarStartInteriorOptimum_ExactRouteStaysWithinItsBudget_ResultWithinTheBound(string witness)
    {
        AssertRepeatedBlockWitness(witness);
        var observation = ObserveRoute(RepeatedBlockWitness(witness), force: false);
        var n = RepeatedBlockWitness(witness).Cov.Length;
        Assert.True(
            observation.ExactChanges > ActiveSetQpSolver<double>.IterationBudget(n) && observation.ExactChanges <= ActiveSetQpSolver<double>.ExactRouteBudget(n),
            witness + ": the exact route (entered for " + observation.Reason + ") made " + observation.ExactChanges + " changes; expected more than "
                + ActiveSetQpSolver<double>.IterationBudget(n) + " and at most " + ActiveSetQpSolver<double>.ExactRouteBudget(n));
    }

    /// <summary>
    /// Diagnostic, not part of the termination property: the exact route forced from the exact feasible start
    /// (<c>RouteTrace.ForceExactRoute</c>) on two qualified problems of the family (<c>n = 20</c>, box <c>±1e89</c>).
    /// Started there, far from the optimum, the exact route can spend its budget of 252 working-set changes; its documented
    /// behaviour is then the refusal that names the iteration budget, and it never returns a wrong vector. The production
    /// route solves both problems within the bound and within its budgets.
    /// </summary>
    [Theory]
    [InlineData("z2f-hmido1-22#0")]
    [InlineData("z2f-hmido1-84#3")]
    public void ActiveSetQpMinVariance_ForcedExactRouteDiagnostic_RefusesByItsBudgetOrReturnsWithinTheBound_ProductionSolvesIt(string witness)
    {
        var problem = RepeatedBlockWitness(witness);
        var production = ObserveRoute(problem, force: false);
        Assert.True(production.Result.Qualified, witness + " must be a qualified case: " + production.Result.Detail);
        Assert.True(production.Result.Outcome == HeldOutOutcome.Ok, witness + ": " + production.Result.Outcome + ": " + production.Result.Detail);
        Assert.True(production.Failure is null, production.Failure);

        var forced = ObserveRoute(problem, force: true);
        Assert.Equal("Forced", forced.Reason);
        var refusedByBudget = forced.Result.Outcome == HeldOutOutcome.Exhausted
            && forced.ExactChanges == forced.Budget + 1
            && forced.Result.Detail.Contains("iteration budget of " + forced.Budget.ToString(CultureInfo.InvariantCulture), StringComparison.Ordinal);
        Assert.True(
            forced.Result.Outcome == HeldOutOutcome.Ok || refusedByBudget,
            witness + " (exact route forced, " + forced.ExactChanges + " of " + forced.Budget + " changes): " + forced.Result.Outcome + ": " + forced.Result.Detail);
    }

    private static void AssertRepeatedBlockWitness(string witness)
    {
        var problem = RepeatedBlockWitness(witness);
        var observation = ObserveRoute(problem, force: false);
        Assert.True(observation.Result.Qualified, witness + " must be a qualified case: " + observation.Result.Detail);
        Assert.True(observation.Result.Outcome == HeldOutOutcome.Ok, witness + ": " + observation.Result.Outcome + ": " + observation.Result.Detail);
        Assert.True(observation.Failure is null, observation.Failure);
    }

    private static void AssertRepeatedBlockProperty(string type, bool meanVariance, int seed)
    {
        var rng = new System.Random(seed);
        var results = new List<HeldOutResult>();
        var failures = new List<string>();
        int handoffs = 0, exactRoutes = 0, maxExact = 0, atTwenty = 0;
        var slowest = 0.0;
        for (var trial = 0; trial < RepeatedBlockTrials; trial++)
        {
            var problem = RepeatedTwoByTwoBlockProblem(rng, type, trial, meanVariance);
            if (problem is null)
            {
                continue;
            }

            var observation = ObserveRoute(problem, force: false);
            results.Add(observation.Result);
            if (observation.Failure is not null)
            {
                failures.Add(observation.Failure);
            }

            handoffs += observation.RoundedBudgetReached ? 1 : 0;
            exactRoutes += observation.Reason == "None" ? 0 : 1;
            maxExact = Math.Max(maxExact, observation.ExactChanges);
            atTwenty += problem.Cov.Length == 20 ? 1 : 0;
            slowest = Math.Max(slowest, observation.Milliseconds);
        }

        var summary = results.Count + " problems (" + atTwenty + " at n = 20), " + exactRoutes + " reached the exact route, "
            + handoffs + " rounded-budget handoffs, at most " + maxExact + " exact-route changes, slowest solve "
            + slowest.ToString("F0", CultureInfo.InvariantCulture) + " ms";
        Assert.True(failures.Count == 0, failures.Count + " exact-route budget exhaustion(s); " + summary + " | " + string.Join(" | ", failures.Take(4)));
        AssertCensus(results, minimumQualified: results.Count / 2);
        Assert.True(atTwenty >= RepeatedBlockTrials / 2, "the population must reach n = 20: " + summary);

        // At double the family drives the rounded route into the exact route (unusable working sets, unconverged
        // refinements, budget handoffs), so the budget assertion is exercised; at decimal, whose 28 digits resolve these
        // blocks at the scales of its operating domain, the rounded route certifies them and the exact route is rarely
        // entered, so only the bound and the absence of exhaustion are asserted there.
        Assert.True(type == "decimal" || exactRoutes > 0, "the population must reach the exact route: " + summary);
    }

    private const int RepeatedBlockTrials = 30;

    /// <summary>
    /// One production solve with a route trace (<paramref name="force"/>: the exact route forced from the exact feasible
    /// start, the diagnostic hook): the result against the exact optimum, the exact route's budget failure if any, and the
    /// route diagnostics. The rounded route spending its budget is reported, not a failure: it hands the problem over.
    /// </summary>
    private static RouteObservation ObserveRoute(HeldOutProblem p, bool force) => p.Type == "double"
        ? ObserveRoute<double>(p, ParseDouble, x => double.IsFinite(x) ? Rational.FromDouble(x) : throw new ArithmeticException("non-finite weight " + x.ToString("R", CultureInfo.InvariantCulture)), force)
        : ObserveRoute<decimal>(p, ParseDecimal, Rational.FromDecimal, force);

    private static RouteObservation ObserveRoute<T>(HeldOutProblem p, Func<string, T> parse, Func<T, Rational> exact, bool force)
        where T : IFloatingPoint<T>
    {
        var n = p.Cov.Length;
        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                cov[i, j] = parse(p.Cov[i][j]);
            }
        }

        var trace = new ActiveSetQpSolver<T>.RouteTrace { ForceExactRoute = force };
        var budget = ActiveSetQpSolver<T>.ExactRouteBudget(n);
        var milliseconds = 0.0;
        var result = ClassifyHeldOut(p, () =>
        {
            var t0 = Stopwatch.GetTimestamp();
            try
            {
                var weights = p.Mu is null
                    ? ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(cov, parse(p.Lb), parse(p.Ub), forcedReference: -1, trace: trace)
                    : ActiveSetQpSolver<T>.SolveMeanVarianceUsingCore(cov, p.Mu.Select(parse).ToArray(), parse(p.Lambda!), parse(p.Lb), parse(p.Ub), trace: trace);
                return weights.Select(exact).ToArray();
            }
            finally
            {
                milliseconds = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
            }
        });

        // The rounded route never refuses for its budget (it hands the problem to the exact route), so a refusal that
        // names the iteration budget is the exact route's.
        var failure = result.Outcome == HeldOutOutcome.Exhausted || trace.ExactChanges > budget
            ? p.Name + " (" + p.Type + (force ? ", exact route forced" : "") + "): the exact route spent its budget of " + budget
                + " working-set changes (" + trace.ExactChanges + " changes, entered for " + trace.ExactRouteReason + "): " + result.Detail
            : null;
        return new RouteObservation(result, failure, trace.RoundedBudgetReached, trace.ExactRouteReason.ToString(), trace.ExactChanges, budget, milliseconds);
    }

    private sealed record RouteObservation(HeldOutResult Result, string? Failure, bool RoundedBudgetReached, string Reason, int ExactChanges, int Budget, double Milliseconds);

    /// <summary>
    /// A seeded problem of the repeated-2×2-block family: one random positive-definite 2×2 block <c>B = AA′/2 + δI</c>
    /// (entries of <c>A</c> in quarters, <c>δ</c> from 1 down to the last bits of the type; one block in three is instead a
    /// pair of near-identical assets <c>[[1, 1 − δ], [1 − δ, 1]]</c>), copied <c>⌊n/2⌋</c> times with
    /// copy scales <c>s_c</c> drawn as radix powers (double <c>2^U(−230, 230)</c>; decimal <c>10^U(−6, 6)</c>, inside its
    /// operating domain), each copy <c>s_c²·B</c>, plus the rank-one coupling <c>(1/8)·s·s′</c> across all copies; an odd
    /// <c>n</c> adds one asset of radix-power variance. The assets are randomly permuted. Two problems in three have
    /// <c>n = 20</c> (ten copies), the others <c>n = 4..19</c>.
    /// </summary>
    private static HeldOutProblem? RepeatedTwoByTwoBlockProblem(System.Random rng, string type, int trial, bool meanVariance)
    {
        var dec = type == "decimal";
        var n = trial % 3 == 2 ? rng.Next(4, 20) : 20;
        var copies = n / 2;
        Rational Radix(int e) => dec
            ? (e >= 0 ? Rational.Of(BigInteger.Pow(10, e), BigInteger.One) : Rational.Of(BigInteger.One, BigInteger.Pow(10, -e)))
            : (e >= 0 ? Rational.Of(BigInteger.One << e, BigInteger.One) : Rational.Of(BigInteger.One, BigInteger.One << -e));

        var a = new Rational[2, 2];
        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                a[i, j] = Rational.Of(rng.Next(-9, 10), 4);
            }
        }

        var delta = rng.Next(4) switch
        {
            0 => Rational.One,
            1 => Rational.Of(1, 1000),
            2 => Radix(dec ? -rng.Next(6, 14) : -rng.Next(20, 44)),
            _ => Radix(dec ? -rng.Next(14, 24) : -rng.Next(44, 52)),
        };
        var block = new Rational[2, 2];
        var twins = rng.Next(3) == 0;
        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                block[i, j] = twins
                    ? (i == j ? Rational.One : Rational.One - delta)
                    : (((a[i, 0] * a[j, 0]) + (a[i, 1] * a[j, 1])) / Rational.FromInt(2)) + (i == j ? delta : Rational.Zero);
            }
        }

        var scale = new Rational[copies];
        for (var c = 0; c < copies; c++)
        {
            scale[c] = Radix(dec ? rng.Next(-6, 7) : rng.Next(-230, 231));
        }

        var m = new Rational[n, n];
        var coupling = Rational.Of(1, 8);
        for (var i = 0; i < 2 * copies; i++)
        {
            for (var j = 0; j < 2 * copies; j++)
            {
                m[i, j] = (i / 2 == j / 2 ? scale[i / 2] * scale[i / 2] * block[i % 2, j % 2] : Rational.Zero) + (coupling * scale[i / 2] * scale[j / 2]);
            }
        }

        if (n % 2 == 1)
        {
            m[n - 1, n - 1] = Radix(dec ? rng.Next(-10, 11) : rng.Next(-300, 301));
        }

        var permutation = Enumerable.Range(0, n).ToArray();
        for (var i = n - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (permutation[i], permutation[j]) = (permutation[j], permutation[i]);
        }

        var cs = new string[n][];
        for (var i = 0; i < n; i++)
        {
            cs[i] = new string[n];
        }

        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                cs[permutation[i]][permutation[j]] = m[i, j].Sign == 0 ? "0" : InType(type, m[i, j]);
            }
        }

        var (lb, ub) = dec
            ? (rng.Next(3) == 0 ? ("0", "1") : ("-100000000000000", "100000000000000"))
            : rng.Next(4) switch
            {
                0 => ("0", "1"),
                1 or 2 => ("-1E+89", "1E+89"),
                _ => ("-1E+150", "1E+150"),
            };
        var name = "rblk-" + trial + "-n" + n + "-b" + ub;
        if (!meanVariance)
        {
            return Admissible(new HeldOutProblem(name, type, cs, lb, ub));
        }

        var max = dec ? 1e14 : 1e150;
        var min = dec ? 1e-14 : 1e-150;
        double Clip(double x) => Math.Abs(x) > max ? Math.Sign(x) * max : Math.Abs(x) < min ? 0.0 : x;
        double Uniform(double lo, double hi) => lo + ((hi - lo) * rng.NextDouble());
        var variances = Enumerable.Range(0, n).Select(i => ParseDouble(InType("double", m[i, i]))).OrderBy(v => v).ToArray();
        var lambda = Clip(rng.Next(3) switch
        {
            0 => 1.0,
            1 => 1 / variances[n / 2],
            _ => Math.Pow(10, Uniform(dec ? -6 : -100, 0)),
        });
        lambda = lambda == 0 ? min : lambda;
        var regime = rng.Next(4);
        var offset = Clip(Uniform(-1, 1) * Math.Pow(10, Uniform(0, dec ? 13 : 140)));
        var blockMean = new[] { Uniform(-1, 1), Uniform(-1, 1) };
        var factor = Math.Pow(10, Uniform(dec ? -3 : -60, dec ? 3 : 60));
        var means = new string[n];
        for (var i = 0; i < n; i++)
        {
            var mean = regime switch
            {
                0 => 0.0,
                1 => offset,
                2 => i < 2 * copies ? ParseDouble(InType("double", scale[i / 2])) * blockMean[i % 2] * factor : offset,
                _ => Uniform(-1, 1) * Math.Pow(10, Uniform(dec ? -6 : -74, dec ? 6 : 74)),
            };
            means[permutation[i]] = InType(type, Clip(mean));
        }

        return Admissible(new HeldOutProblem(name + "-m" + regime, type, cs, lb, ub, means, InType(type, lambda)));
    }

    // A covariance written as rows separated by ';' and entries by ','.
    private static string[][] ParseMatrix(string rows) => [.. rows.Split(';').Select(r => r.Split(','))];

    /// <summary>
    /// Explicit inputs of the repeated-2×2-block family at <c>n = 20</c> (double), dumped by the held-out census: the two
    /// problems on which the exact route forced from the feasible start spent its budget (<c>z2f-*</c>), and the problems on
    /// which the production rounded route spent its budget and handed over (<c>z2g-*</c>, <c>z2i-*</c>).
    /// </summary>
    private static HeldOutProblem RepeatedBlockWitness(string name) => name switch
    {
        "z2f-hmido1-22#0" => new HeldOutProblem(
            name,
            "double",
            ParseMatrix("4.132743924856186E-09,-1.7171259969472885E-09,3.2311742677852644E-27,2.980232238769531E-08,1.8367099231598242E-40,3.2311742677852644E-27,1.0097419586828951E-28,1.0195788231247695E-56,8192,7.482888383134223E+50,1.9721522630525295E-31,7.24454326306137E-71,1.0195788231247695E-56,1.0097419586828951E-28,1.8367099231598242E-40,7.482888383134223E+50,7.24454326306137E-71,1.9721522630525295E-31,2.980232238769531E-08,8192;-1.7171259969472885E-09,3.521563485264778E-09,3.2311742677852644E-27,2.980232238769531E-08,1.8367099231598242E-40,3.2311742677852644E-27,1.0097419586828951E-28,1.0195788231247695E-56,8192,7.482888383134223E+50,1.9721522630525295E-31,7.24454326306137E-71,1.0195788231247695E-56,1.0097419586828951E-28,1.8367099231598242E-40,7.482888383134223E+50,7.24454326306137E-71,1.9721522630525295E-31,2.980232238769531E-08,8192;3.2311742677852644E-27,3.2311742677852644E-27,3.1837501109459844E-42,8.271806125530277E-25,5.0978941156238473E-57,-1.3228257503226273E-42,2.802596928649634E-45,2.8298997121333476E-73,2.2737367544323206E-13,2.076918743413931E+34,5.473822126268817E-48,2.010764683385949E-87,2.8298997121333476E-73,2.802596928649634E-45,5.0978941156238473E-57,2.076918743413931E+34,2.010764683385949E-87,5.473822126268817E-48,8.271806125530277E-25,2.2737367544323206E-13;2.980232238769531E-08,2.980232238769531E-08,8.271806125530277E-25,0.000270843505859375,4.70197740328915E-38,8.271806125530277E-25,2.5849394142282115E-26,2.61012178719941E-54,2097152,1.915619426082361E+53,5.048709793414476E-29,1.8546030753437107E-68,2.61012178719941E-54,2.5849394142282115E-26,4.70197740328915E-38,1.915619426082361E+53,1.8546030753437107E-68,5.048709793414476E-29,-0.0001125335693359375,2097152;1.8367099231598242E-40,1.8367099231598242E-40,5.0978941156238473E-57,4.70197740328915E-38,8.765897348304258E-69,5.0978941156238473E-57,1.5930919111324523E-58,1.608611746708759E-86,1.2924697071141057E-26,1.1805916207174113E+21,3.111507638930571E-61,1.142987391282275E-100,1.608611746708759E-86,1.5930919111324523E-58,-4.274280525206208E-69,1.1805916207174113E+21,1.142987391282275E-100,3.111507638930571E-61,4.70197740328915E-38,1.2924697071141057E-26;3.2311742677852644E-27,3.2311742677852644E-27,-1.3228257503226273E-42,8.271806125530277E-25,5.0978941156238473E-57,2.712913826932846E-42,2.802596928649634E-45,2.8298997121333476E-73,2.2737367544323206E-13,2.076918743413931E+34,5.473822126268817E-48,2.010764683385949E-87,2.8298997121333476E-73,2.802596928649634E-45,5.0978941156238473E-57,2.076918743413931E+34,2.010764683385949E-87,5.473822126268817E-48,8.271806125530277E-25,2.2737367544323206E-13;1.0097419586828951E-28,1.0097419586828951E-28,2.802596928649634E-45,2.5849394142282115E-26,1.5930919111324523E-58,2.802596928649634E-45,2.6493299091141073E-45,8.843436600416711E-75,7.105427357601002E-15,6.490371073168535E+32,1.7105694144590052E-49,6.28363963558109E-89,8.843436600416711E-75,-1.2918220217994407E-45,1.5930919111324523E-58,6.490371073168535E+32,6.28363963558109E-89,1.7105694144590052E-49,2.5849394142282115E-26,7.105427357601002E-15;1.0195788231247695E-56,1.0195788231247695E-56,2.8298997121333476E-73,2.61012178719941E-54,1.608611746708759E-86,2.8298997121333476E-73,8.843436600416711E-75,2.701200670803814E-101,7.174648137343064E-43,65536,1.727233711018889E-77,6.344854593289123E-117,-1.317114376672934E-101,8.843436600416711E-75,1.608611746708759E-86,65536,6.344854593289123E-117,1.727233711018889E-77,2.61012178719941E-54,7.174648137343064E-43;8192,8192,2.2737367544323206E-13,2097152,1.2924697071141057E-26,2.2737367544323206E-13,7.105427357601002E-15,7.174648137343064E-43,2.0464356706771534E+19,5.2656145834278593E+64,1.3877787807814457E-17,5.0978941156238473E-57,7.174648137343064E-43,7.105427357601002E-15,1.2924697071141057E-26,5.2656145834278593E+64,5.0978941156238473E-57,1.3877787807814457E-17,2097152,-8.502796096475496E+18;7.482888383134223E+50,7.482888383134223E+50,2.076918743413931E+34,1.915619426082361E+53,1.1805916207174113E+21,2.076918743413931E+34,6.490371073168535E+32,65536,5.2656145834278593E+64,1.4549691008800452E+113,1.2676506002282294E+30,4.656612873077393E-10,65536,6.490371073168535E+32,1.1805916207174113E+21,-7.094477434043195E+112,4.656612873077393E-10,1.2676506002282294E+30,1.915619426082361E+53,5.2656145834278593E+64;1.9721522630525295E-31,1.9721522630525295E-31,5.473822126268817E-48,5.048709793414476E-29,3.111507638930571E-61,5.473822126268817E-48,1.7105694144590052E-49,1.727233711018889E-77,1.3877787807814457E-17,1.2676506002282294E+30,1.0106391560036115E-50,1.2272733663244316E-91,1.727233711018889E-77,1.7105694144590052E-49,3.111507638930571E-61,1.2676506002282294E+30,1.2272733663244316E-91,-4.927909934232486E-51,5.048709793414476E-29,1.3877787807814457E-17;7.24454326306137E-71,7.24454326306137E-71,2.010764683385949E-87,1.8546030753437107E-68,1.142987391282275E-100,2.010764683385949E-87,6.28363963558109E-89,6.344854593289123E-117,5.0978941156238473E-57,4.656612873077393E-10,1.2272733663244316E-91,1.6004430709540704E-129,6.344854593289123E-117,6.28363963558109E-89,1.142987391282275E-100,4.656612873077393E-10,-6.649728252555645E-130,1.2272733663244316E-91,1.8546030753437107E-68,5.0978941156238473E-57;1.0195788231247695E-56,1.0195788231247695E-56,2.8298997121333476E-73,2.61012178719941E-54,1.608611746708759E-86,2.8298997121333476E-73,8.843436600416711E-75,-1.317114376672934E-101,7.174648137343064E-43,65536,1.727233711018889E-77,6.344854593289123E-117,3.1700040930094345E-101,8.843436600416711E-75,1.608611746708759E-86,65536,6.344854593289123E-117,1.727233711018889E-77,2.61012178719941E-54,7.174648137343064E-43;1.0097419586828951E-28,1.0097419586828951E-28,2.802596928649634E-45,2.5849394142282115E-26,1.5930919111324523E-58,2.802596928649634E-45,-1.2918220217994407E-45,8.843436600416711E-75,7.105427357601002E-15,6.490371073168535E+32,1.7105694144590052E-49,6.28363963558109E-89,8.843436600416711E-75,3.109130967720688E-45,1.5930919111324523E-58,6.490371073168535E+32,6.28363963558109E-89,1.7105694144590052E-49,2.5849394142282115E-26,7.105427357601002E-15;1.8367099231598242E-40,1.8367099231598242E-40,5.0978941156238473E-57,4.70197740328915E-38,-4.274280525206208E-69,5.0978941156238473E-57,1.5930919111324523E-58,1.608611746708759E-86,1.2924697071141057E-26,1.1805916207174113E+21,3.111507638930571E-61,1.142987391282275E-100,1.608611746708759E-86,1.5930919111324523E-58,1.0287251433547145E-68,1.1805916207174113E+21,1.142987391282275E-100,3.111507638930571E-61,4.70197740328915E-38,1.2924697071141057E-26;7.482888383134223E+50,7.482888383134223E+50,2.076918743413931E+34,1.915619426082361E+53,1.1805916207174113E+21,2.076918743413931E+34,6.490371073168535E+32,65536,5.2656145834278593E+64,-7.094477434043195E+112,1.2676506002282294E+30,4.656612873077393E-10,65536,6.490371073168535E+32,1.1805916207174113E+21,1.7074843993798877E+113,4.656612873077393E-10,1.2676506002282294E+30,1.915619426082361E+53,5.2656145834278593E+64;7.24454326306137E-71,7.24454326306137E-71,2.010764683385949E-87,1.8546030753437107E-68,1.142987391282275E-100,2.010764683385949E-87,6.28363963558109E-89,6.344854593289123E-117,5.0978941156238473E-57,4.656612873077393E-10,1.2272733663244316E-91,-6.649728252555645E-130,6.344854593289123E-117,6.28363963558109E-89,1.142987391282275E-100,4.656612873077393E-10,1.3637578280664966E-129,1.2272733663244316E-91,1.8546030753437107E-68,5.0978941156238473E-57;1.9721522630525295E-31,1.9721522630525295E-31,5.473822126268817E-48,5.048709793414476E-29,3.111507638930571E-61,5.473822126268817E-48,1.7105694144590052E-49,1.727233711018889E-77,1.3877787807814457E-17,1.2676506002282294E+30,-4.927909934232486E-51,1.2272733663244316E-91,1.727233711018889E-77,1.7105694144590052E-49,3.111507638930571E-61,1.2676506002282294E+30,1.2272733663244316E-91,1.1860393401034118E-50,5.048709793414476E-29,1.3877787807814457E-17;2.980232238769531E-08,2.980232238769531E-08,8.271806125530277E-25,-0.0001125335693359375,4.70197740328915E-38,8.271806125530277E-25,2.5849394142282115E-26,2.61012178719941E-54,2097152,1.915619426082361E+53,5.048709793414476E-29,1.8546030753437107E-68,2.61012178719941E-54,2.5849394142282115E-26,4.70197740328915E-38,1.915619426082361E+53,1.8546030753437107E-68,5.048709793414476E-29,0.0002307891845703125,2097152;8192,8192,2.2737367544323206E-13,2097152,1.2924697071141057E-26,2.2737367544323206E-13,7.105427357601002E-15,7.174648137343064E-43,-8.502796096475496E+18,5.2656145834278593E+64,1.3877787807814457E-17,5.0978941156238473E-57,7.174648137343064E-43,7.105427357601002E-15,1.2924697071141057E-26,5.2656145834278593E+64,5.0978941156238473E-57,1.3877787807814457E-17,2097152,1.743793775717856E+19"),
            "-1E+89",
            "1E+89"),
        "z2f-hmido1-84#3" => new HeldOutProblem(
            name,
            "double",
            ParseMatrix("4.09221594583401E-30,64,6.525304467998525E-55,1.88079096131566E-37,8.112963841460668E+31,1.888946593147858E+22,1.0097419586828951E-28,3.42661455705377E-30,1.0842021724855044E-19,1.393796574908164E+42,1.0842021724855044E-19,6.525304467998525E-55,1.6313261169996311E-55,64,1.393796574908164E+42,1.6313261169996311E-55,8.112963841460668E+31,1.88079096131566E-37,1.0097419586828951E-28,1.888946593147858E+22;64,1.7238425570335675E+36,4.235164736271502E-22,0.0001220703125,5.2656145834278593E+64,1.2259964326927111E+55,65536,64,70368744177664,9.046256971665328E+74,70368744177664,4.235164736271502E-22,1.0587911840678754E-22,1.443458526672682E+36,9.046256971665328E+74,1.0587911840678754E-22,5.2656145834278593E+64,0.0001220703125,65536,1.2259964326927111E+55;6.525304467998525E-55,4.235164736271502E-22,1.7920049751821022E-76,1.2446030555722283E-60,536870912,0.125,6.681911775230489E-52,6.525304467998525E-55,7.174648137343064E-43,9.223372036854776E+18,7.174648137343064E-43,1.5005342864476598E-76,1.0795210693868056E-78,4.235164736271502E-22,9.223372036854776E+18,1.0795210693868056E-78,536870912,1.2446030555722283E-60,6.681911775230489E-52,0.125;1.88079096131566E-37,0.0001220703125,1.2446030555722283E-60,1.4887394884986897E-41,1.5474250491067253E+26,36028797018963970,1.925929944387236E-34,1.88079096131566E-37,2.0679515313825692E-25,2.658455991569832E+36,2.0679515313825692E-25,1.2446030555722283E-60,3.111507638930571E-61,0.0001220703125,2.658455991569832E+36,3.111507638930571E-61,1.5474250491067253E+26,1.2465951138633573E-41,1.925929944387236E-34,36028797018963970;8.112963841460668E+31,5.2656145834278593E+64,536870912,1.5474250491067253E+26,1.9524256500214645E+96,1.5541351137805833E+85,8.307674973655724E+34,8.112963841460668E+31,8.92029807941225E+43,1.1467493079950358E+105,8.92029807941225E+43,536870912,134217728,5.2656145834278593E+64,1.1467493079950358E+105,134217728,2.3195484218203633E+96,1.5474250491067253E+26,8.307674973655724E+34,1.5541351137805833E+85;1.888946593147858E+22,1.2259964326927111E+55,0.125,36028797018963970,1.5541351137805833E+85,1.5016786572964485E+77,1.9342813113834067E+25,1.888946593147858E+22,2.076918743413931E+34,2.6699837949011376E+95,2.076918743413931E+34,0.125,0.03125,1.2259964326927111E+55,2.6699837949011376E+95,0.03125,1.5541351137805833E+85,36028797018963970,1.9342813113834067E+25,1.2574297190614806E+77;1.0097419586828951E-28,65536,6.681911775230489E-52,1.925929944387236E-34,8.307674973655724E+34,1.9342813113834067E+25,3.0243791146470192E-24,1.0097419586828951E-28,1.1102230246251565E-16,1.42724769270596E+45,1.1102230246251565E-16,6.681911775230489E-52,1.6704779438076223E-52,65536,1.42724769270596E+45,1.6704779438076223E-52,8.307674973655724E+34,1.925929944387236E-34,3.593065785777214E-24,1.9342813113834067E+25;3.42661455705377E-30,64,6.525304467998525E-55,1.88079096131566E-37,8.112963841460668E+31,1.888946593147858E+22,1.0097419586828951E-28,2.8842726847143356E-30,1.0842021724855044E-19,1.393796574908164E+42,1.0842021724855044E-19,6.525304467998525E-55,1.6313261169996311E-55,64,1.393796574908164E+42,1.6313261169996311E-55,8.112963841460668E+31,1.88079096131566E-37,1.0097419586828951E-28,1.888946593147858E+22;1.0842021724855044E-19,70368744177664,7.174648137343064E-43,2.0679515313825692E-25,8.92029807941225E+43,2.076918743413931E+34,1.1102230246251565E-16,1.0842021724855044E-19,4.9471855163574354E-06,1.532495540865889E+54,4.1425228118896484E-06,7.174648137343064E-43,1.793662034335766E-43,70368744177664,1.532495540865889E+54,1.793662034335766E-43,8.92029807941225E+43,2.0679515313825692E-25,1.1102230246251565E-16,2.076918743413931E+34;1.393796574908164E+42,9.046256971665328E+74,9.223372036854776E+18,2.658455991569832E+36,1.1467493079950358E+105,2.6699837949011376E+95,1.42724769270596E+45,1.393796574908164E+42,1.532495540865889E+54,5.762543406222715E+116,1.532495540865889E+54,9.223372036854776E+18,2.305843009213694E+18,9.046256971665328E+74,6.846098576623541E+116,2.305843009213694E+18,1.1467493079950358E+105,2.658455991569832E+36,1.42724769270596E+45,2.6699837949011376E+95;1.0842021724855044E-19,70368744177664,7.174648137343064E-43,2.0679515313825692E-25,8.92029807941225E+43,2.076918743413931E+34,1.1102230246251565E-16,1.0842021724855044E-19,4.1425228118896484E-06,1.532495540865889E+54,3.486871719360365E-06,7.174648137343064E-43,1.793662034335766E-43,70368744177664,1.532495540865889E+54,1.793662034335766E-43,8.92029807941225E+43,2.0679515313825692E-25,1.1102230246251565E-16,2.076918743413931E+34;6.525304467998525E-55,4.235164736271502E-22,1.5005342864476598E-76,1.2446030555722283E-60,536870912,0.125,6.681911775230489E-52,6.525304467998525E-55,7.174648137343064E-43,9.223372036854776E+18,7.174648137343064E-43,1.2630396511825674E-76,1.0795210693868056E-78,4.235164736271502E-22,9.223372036854776E+18,1.0795210693868056E-78,536870912,1.2446030555722283E-60,6.681911775230489E-52,0.125;1.6313261169996311E-55,1.0587911840678754E-22,1.0795210693868056E-78,3.111507638930571E-61,134217728,0.03125,1.6704779438076223E-52,1.6313261169996311E-55,1.793662034335766E-43,2.305843009213694E+18,1.793662034335766E-43,1.0795210693868056E-78,7.893997819891046E-78,1.0587911840678754E-22,2.305843009213694E+18,9.378339290297873E-78,134217728,3.111507638930571E-61,1.6704779438076223E-52,0.03125;64,1.443458526672682E+36,4.235164736271502E-22,0.0001220703125,5.2656145834278593E+64,1.2259964326927111E+55,65536,64,70368744177664,9.046256971665328E+74,70368744177664,4.235164736271502E-22,1.0587911840678754E-22,1.2149974648971544E+36,9.046256971665328E+74,1.0587911840678754E-22,5.2656145834278593E+64,0.0001220703125,65536,1.2259964326927111E+55;1.393796574908164E+42,9.046256971665328E+74,9.223372036854776E+18,2.658455991569832E+36,1.1467493079950358E+105,2.6699837949011376E+95,1.42724769270596E+45,1.393796574908164E+42,1.532495540865889E+54,6.846098576623541E+116,1.532495540865889E+54,9.223372036854776E+18,2.305843009213694E+18,9.046256971665328E+74,8.175916285751877E+116,2.305843009213694E+18,1.1467493079950358E+105,2.658455991569832E+36,1.42724769270596E+45,2.6699837949011376E+95;1.6313261169996311E-55,1.0587911840678754E-22,1.0795210693868056E-78,3.111507638930571E-61,134217728,0.03125,1.6704779438076223E-52,1.6313261169996311E-55,1.793662034335766E-43,2.305843009213694E+18,1.793662034335766E-43,1.0795210693868056E-78,9.378339290297873E-78,1.0587911840678754E-22,2.305843009213694E+18,1.1200031094888139E-77,134217728,3.111507638930571E-61,1.6704779438076223E-52,0.03125;8.112963841460668E+31,5.2656145834278593E+64,536870912,1.5474250491067253E+26,2.3195484218203633E+96,1.5541351137805833E+85,8.307674973655724E+34,8.112963841460668E+31,8.92029807941225E+43,1.1467493079950358E+105,8.92029807941225E+43,536870912,134217728,5.2656145834278593E+64,1.1467493079950358E+105,134217728,2.770108187209938E+96,1.5474250491067253E+26,8.307674973655724E+34,1.5541351137805833E+85;1.88079096131566E-37,0.0001220703125,1.2446030555722283E-60,1.2465951138633573E-41,1.5474250491067253E+26,36028797018963970,1.925929944387236E-34,1.88079096131566E-37,2.0679515313825692E-25,2.658455991569832E+36,2.0679515313825692E-25,1.2446030555722283E-60,3.111507638930571E-61,0.0001220703125,2.658455991569832E+36,3.111507638930571E-61,1.5474250491067253E+26,1.0492922900864271E-41,1.925929944387236E-34,36028797018963970;1.0097419586828951E-28,65536,6.681911775230489E-52,1.925929944387236E-34,8.307674973655724E+34,1.9342813113834067E+25,3.593065785777214E-24,1.0097419586828951E-28,1.1102230246251565E-16,1.42724769270596E+45,1.1102230246251565E-16,6.681911775230489E-52,1.6704779438076223E-52,65536,1.42724769270596E+45,1.6704779438076223E-52,8.307674973655724E+34,1.925929944387236E-34,4.290999427618843E-24,1.9342813113834067E+25;1.888946593147858E+22,1.2259964326927111E+55,0.125,36028797018963970,1.5541351137805833E+85,1.2574297190614806E+77,1.9342813113834067E+25,1.888946593147858E+22,2.076918743413931E+34,2.6699837949011376E+95,2.076918743413931E+34,0.125,0.03125,1.2259964326927111E+55,2.6699837949011376E+95,0.03125,1.5541351137805833E+85,36028797018963970,1.9342813113834067E+25,1.0584120656848475E+77"),
            "-1E+89",
            "1E+89"),
        "z2i-mi0-13#0" => new HeldOutProblem(
            name,
            "double",
            ParseMatrix("1.5754451408030182E-107,268435456,1.809632932003464E-107,7.105427357601002E-15,3.308722450212111E-24,7.105427357601002E-15,1.0842021724855044E-19,1.1985091468012028E-94,4.930380657631324E-32,3.308722450212111E-24,5.397605346934028E-79,1.1985091468012028E-94,4.930380657631324E-32,1.1479437019748901E-41,1.1479437019748901E-41,0.0625,5.397605346934028E-79,0.0625,268435456,1.0842021724855044E-19;268435456,1.3353330569563037E+125,268435456,5.599361855444511E+101,2.6074060497081422E+92,5.599361855444511E+101,8.54394814368364E+96,9.44473296573929E+21,3.885337784451458E+84,2.6074060497081422E+92,4.253529586511731E+37,9.44473296573929E+21,3.885337784451458E+84,9.046256971665328E+74,9.046256971665328E+74,4.92525077454931E+114,4.253529586511731E+37,4.92525077454931E+114,1.1237951469434226E+125,8.54394814368364E+96;1.809632932003464E-107,268435456,2.1502697192041185E-107,7.105427357601002E-15,3.308722450212111E-24,7.105427357601002E-15,1.0842021724855044E-19,1.1985091468012028E-94,4.930380657631324E-32,3.308722450212111E-24,5.397605346934028E-79,1.1985091468012028E-94,4.930380657631324E-32,1.1479437019748901E-41,1.1479437019748901E-41,0.0625,5.397605346934028E-79,0.0625,268435456,1.0842021724855044E-19;7.105427357601002E-15,5.599361855444511E+101,7.105427357601002E-15,9.356000810375159E+79,6.901746346790564E+69,7.873862068137501E+79,2.261564242916332E+74,0.25,1.0284403483257538E+62,6.901746346790564E+69,1125899906842624,0.25,1.0284403483257538E+62,2.3945242826029513E+52,2.3945242826029513E+52,1.3037030248540711E+92,1125899906842624,1.3037030248540711E+92,5.599361855444511E+101,2.261564242916332E+74;3.308722450212111E-24,2.6074060497081422E+92,3.308722450212111E-24,6.901746346790564E+69,2.0287592808769775E+61,6.901746346790564E+69,1.0531229166855719E+65,1.1641532182693481E-10,4.789048565205903E+52,1.7073716720251772E+61,524288,1.1641532182693481E-10,4.789048565205903E+52,1.1150372599265312E+43,1.1150372599265312E+43,6.070840288205404E+82,524288,6.070840288205404E+82,2.6074060497081422E+92,1.0531229166855719E+65;7.105427357601002E-15,5.599361855444511E+101,7.105427357601002E-15,7.873862068137501E+79,6.901746346790564E+69,6.854891682849129E+79,2.261564242916332E+74,0.25,1.0284403483257538E+62,6.901746346790564E+69,1125899906842624,0.25,1.0284403483257538E+62,2.3945242826029513E+52,2.3945242826029513E+52,1.3037030248540711E+92,1125899906842624,1.3037030248540711E+92,5.599361855444511E+101,2.261564242916332E+74;1.0842021724855044E-19,8.54394814368364E+96,1.0842021724855044E-19,2.261564242916332E+74,1.0531229166855719E+65,2.261564242916332E+74,1.5960288426953203E+70,3.814697265625E-06,1.5692754338466702E+57,1.0531229166855719E+65,17179869184,3.814697265625E-06,1.5692754338466702E+57,3.6537540933272573E+47,3.6537540933272573E+47,1.9892929456391466E+87,17179869184,1.9892929456391466E+87,8.54394814368364E+96,1.8332763733662435E+70;1.1985091468012028E-94,9.44473296573929E+21,1.1985091468012028E-94,0.25,1.1641532182693481E-10,0.25,3.814697265625E-06,1.9503066194976498E-80,1.734723475976807E-18,1.1641532182693481E-10,1.8991135491519597E-65,2.240217062936486E-80,1.734723475976807E-18,4.0389678347315804E-28,4.0389678347315804E-28,2199023255552,1.8991135491519597E-65,2199023255552,9.44473296573929E+21,3.814697265625E-06;4.930380657631324E-32,3.885337784451458E+84,4.930380657631324E-32,1.0284403483257538E+62,4.789048565205903E+52,1.0284403483257538E+62,1.5692754338466702E+57,1.734723475976807E-18,3.300510289382537E+45,4.789048565205903E+52,0.0078125,1.734723475976807E-18,3.791126683750206E+45,1.661534994731145E+35,1.661534994731145E+35,9.046256971665328E+74,0.0078125,9.046256971665328E+74,3.885337784451458E+84,1.5692754338466702E+57;3.308722450212111E-24,2.6074060497081422E+92,3.308722450212111E-24,6.901746346790564E+69,1.7073716720251772E+61,6.901746346790564E+69,1.0531229166855719E+65,1.1641532182693481E-10,4.789048565205903E+52,1.4864176909395683E+61,524288,1.1641532182693481E-10,4.789048565205903E+52,1.1150372599265312E+43,1.1150372599265312E+43,6.070840288205404E+82,524288,6.070840288205404E+82,2.6074060497081422E+92,1.0531229166855719E+65;5.397605346934028E-79,4.253529586511731E+37,5.397605346934028E-79,1125899906842624,524288,1125899906842624,17179869184,1.8991135491519597E-65,0.0078125,524288,3.955691770936456E-49,1.8991135491519597E-65,0.0078125,1.8189894035458565E-12,1.8189894035458565E-12,9.903520314283042E+27,4.5437000071567326E-49,9.903520314283042E+27,4.253529586511731E+37,17179869184;1.1985091468012028E-94,9.44473296573929E+21,1.1985091468012028E-94,0.25,1.1641532182693481E-10,0.25,3.814697265625E-06,2.240217062936486E-80,1.734723475976807E-18,1.1641532182693481E-10,1.8991135491519597E-65,2.66190498066571E-80,1.734723475976807E-18,4.0389678347315804E-28,4.0389678347315804E-28,2199023255552,1.8991135491519597E-65,2199023255552,9.44473296573929E+21,3.814697265625E-06;4.930380657631324E-32,3.885337784451458E+84,4.930380657631324E-32,1.0284403483257538E+62,4.789048565205903E+52,1.0284403483257538E+62,1.5692754338466702E+57,1.734723475976807E-18,3.791126683750206E+45,4.789048565205903E+52,0.0078125,1.734723475976807E-18,4.504750530103191E+45,1.661534994731145E+35,1.661534994731145E+35,9.046256971665328E+74,0.0078125,9.046256971665328E+74,3.885337784451458E+84,1.5692754338466702E+57;1.1479437019748901E-41,9.046256971665328E+74,1.1479437019748901E-41,2.3945242826029513E+52,1.1150372599265312E+43,2.3945242826029513E+52,3.6537540933272573E+47,4.0389678347315804E-28,1.661534994731145E+35,1.1150372599265312E+43,1.8189894035458565E-12,4.0389678347315804E-28,1.661534994731145E+35,2.4420301556215537E+26,2.0551738933448696E+26,2.1062458333711437E+65,1.8189894035458565E-12,2.1062458333711437E+65,9.046256971665328E+74,3.6537540933272573E+47;1.1479437019748901E-41,9.046256971665328E+74,1.1479437019748901E-41,2.3945242826029513E+52,1.1150372599265312E+43,2.3945242826029513E+52,3.6537540933272573E+47,4.0389678347315804E-28,1.661534994731145E+35,1.1150372599265312E+43,1.8189894035458565E-12,4.0389678347315804E-28,1.661534994731145E+35,2.0551738933448696E+26,1.789210213029654E+26,2.1062458333711437E+65,1.8189894035458565E-12,2.1062458333711437E+65,9.046256971665328E+74,3.6537540933272573E+47;0.0625,4.92525077454931E+114,0.0625,1.3037030248540711E+92,6.070840288205404E+82,1.3037030248540711E+92,1.9892929456391466E+87,2199023255552,9.046256971665328E+74,6.070840288205404E+82,9.903520314283042E+27,2199023255552,9.046256971665328E+74,2.1062458333711437E+65,2.1062458333711437E+65,5.303715549477049E+105,9.903520314283042E+27,6.092105698723627E+105,4.92525077454931E+114,1.9892929456391466E+87;5.397605346934028E-79,4.253529586511731E+37,5.397605346934028E-79,1125899906842624,524288,1125899906842624,17179869184,1.8991135491519597E-65,0.0078125,524288,4.5437000071567326E-49,1.8991135491519597E-65,0.0078125,1.8189894035458565E-12,1.8189894035458565E-12,9.903520314283042E+27,5.398984714386241E-49,9.903520314283042E+27,4.253529586511731E+37,17179869184;0.0625,4.92525077454931E+114,0.0625,1.3037030248540711E+92,6.070840288205404E+82,1.3037030248540711E+92,1.9892929456391466E+87,2199023255552,9.046256971665328E+74,6.070840288205404E+82,9.903520314283042E+27,2199023255552,9.046256971665328E+74,2.1062458333711437E+65,2.1062458333711437E+65,6.092105698723627E+105,9.903520314283042E+27,7.238855006718671E+105,4.92525077454931E+114,1.9892929456391466E+87;268435456,1.1237951469434226E+125,268435456,5.599361855444511E+101,2.6074060497081422E+92,5.599361855444511E+101,8.54394814368364E+96,9.44473296573929E+21,3.885337784451458E+84,2.6074060497081422E+92,4.253529586511731E+37,9.44473296573929E+21,3.885337784451458E+84,9.046256971665328E+74,9.046256971665328E+74,4.92525077454931E+114,4.253529586511731E+37,4.92525077454931E+114,9.783628338095694E+124,8.54394814368364E+96;1.0842021724855044E-19,8.54394814368364E+96,1.0842021724855044E-19,2.261564242916332E+74,1.0531229166855719E+65,2.261564242916332E+74,1.8332763733662435E+70,3.814697265625E-06,1.5692754338466702E+57,1.0531229166855719E+65,17179869184,3.814697265625E-06,1.5692754338466702E+57,3.6537540933272573E+47,3.6537540933272573E+47,1.9892929456391466E+87,17179869184,1.9892929456391466E+87,8.54394814368364E+96,2.178363690705774E+70"),
            "-1E+150",
            "1E+150"),
        "z2g-mi0-146#0" => new HeldOutProblem(
            name,
            "double",
            ParseMatrix("1.5791236812556444E-52,7.596454196607839E-65,7.703719777548943E-34,7.703719777548943E-34,7.596454196607839E-65,9.273015376718553E-69,1.8367099231598242E-40,3.2311742677852644E-27,9.723461371658034E-63,1.0309981059437669E-52,1.7763568394002505E-15,1.7763568394002505E-15,9.5367431640625E-07,9.273015376718553E-69,3.2311742677852644E-27,9.5367431640625E-07,256,256,1.8367099231598242E-40,9.723461371658034E-63;7.596454196607839E-65,3.343924464532572E-74,1.1210387714598537E-44,1.1210387714598537E-44,2.1832234107278756E-74,1.349401336733507E-79,2.6727647100921956E-51,4.70197740328915E-38,1.4149498560666738E-73,7.596454196607839E-65,2.5849394142282115E-26,2.5849394142282115E-26,1.3877787807814457E-17,1.349401336733507E-79,4.70197740328915E-38,1.3877787807814457E-17,3.725290298461914E-09,3.725290298461914E-09,2.6727647100921956E-51,1.4149498560666738E-73;7.703719777548943E-34,1.1210387714598537E-44,3.439026841078888E-12,2.2453150450019166E-12,1.1210387714598537E-44,1.3684555315672042E-48,2.710505431213761E-20,4.76837158203125E-07,1.4349296274686127E-42,7.703719777548943E-34,262144,262144,140737488355328,1.3684555315672042E-48,4.76837158203125E-07,140737488355328,3.777893186295716E+22,3.777893186295716E+22,2.710505431213761E-20,1.4349296274686127E-42;7.703719777548943E-34,1.1210387714598537E-44,2.2453150450019166E-12,1.6200374375330317E-12,1.1210387714598537E-44,1.3684555315672042E-48,2.710505431213761E-20,4.76837158203125E-07,1.4349296274686127E-42,7.703719777548943E-34,262144,262144,140737488355328,1.3684555315672042E-48,4.76837158203125E-07,140737488355328,3.777893186295716E+22,3.777893186295716E+22,2.710505431213761E-20,1.4349296274686127E-42;7.596454196607839E-65,2.1832234107278756E-74,1.1210387714598537E-44,1.1210387714598537E-44,1.5752371444492298E-74,1.349401336733507E-79,2.6727647100921956E-51,4.70197740328915E-38,1.4149498560666738E-73,7.596454196607839E-65,2.5849394142282115E-26,2.5849394142282115E-26,1.3877787807814457E-17,1.349401336733507E-79,4.70197740328915E-38,1.3877787807814457E-17,3.725290298461914E-09,3.725290298461914E-09,2.6727647100921956E-51,1.4149498560666738E-73;9.273015376718553E-69,1.349401336733507E-79,1.3684555315672042E-48,1.3684555315672042E-48,1.349401336733507E-79,4.9828357466050567E-82,3.2626522339992623E-55,5.739718509874451E-42,1.727233711018889E-77,9.273015376718553E-69,3.1554436208840472E-30,3.1554436208840472E-30,1.6940658945086007E-21,3.253256396543794E-82,5.739718509874451E-42,1.6940658945086007E-21,4.547473508864641E-13,4.547473508864641E-13,3.2626522339992623E-55,1.727233711018889E-77;1.8367099231598242E-40,2.6727647100921956E-51,2.710505431213761E-20,2.710505431213761E-20,2.6727647100921956E-51,3.2626522339992623E-55,9.208846663188022E-26,1.1368683772161603E-13,3.4211388289180104E-49,1.8367099231598242E-40,0.0625,0.0625,33554432,3.2626522339992623E-55,1.1368683772161603E-13,33554432,9007199254740992,9007199254740992,1.2763138357751794E-25,3.4211388289180104E-49;3.2311742677852644E-27,4.70197740328915E-38,4.76837158203125E-07,4.76837158203125E-07,4.70197740328915E-38,5.739718509874451E-42,1.1368683772161603E-13,60.50000000000006,6.018531076210112E-36,3.2311742677852644E-27,1099511627776,1099511627776,5.902958103587057E+20,5.739718509874451E-42,39.5,5.902958103587057E+20,1.5845632502852868E+29,1.5845632502852868E+29,1.1368683772161603E-13,6.018531076210112E-36;9.723461371658034E-63,1.4149498560666738E-73,1.4349296274686127E-42,1.4349296274686127E-42,1.4149498560666738E-73,1.727233711018889E-77,3.4211388289180104E-49,6.018531076210112E-36,5.478685842690166E-70,9.723461371658034E-63,3.308722450212111E-24,3.308722450212111E-24,1.7763568394002505E-15,1.727233711018889E-77,6.018531076210112E-36,1.7763568394002505E-15,4.76837158203125E-07,4.76837158203125E-07,3.4211388289180104E-49,3.5769932361365514E-70;1.0309981059437669E-52,7.596454196607839E-65,7.703719777548943E-34,7.703719777548943E-34,7.596454196607839E-65,9.273015376718553E-69,1.8367099231598242E-40,3.2311742677852644E-27,9.723461371658034E-63,7.438847093518333E-53,1.7763568394002505E-15,1.7763568394002505E-15,9.5367431640625E-07,9.273015376718553E-69,3.2311742677852644E-27,9.5367431640625E-07,256,256,1.8367099231598242E-40,9.723461371658034E-63;1.7763568394002505E-15,2.5849394142282115E-26,262144,262144,2.5849394142282115E-26,3.1554436208840472E-30,0.0625,1099511627776,3.308722450212111E-24,1.7763568394002505E-15,1.8285003021671283E+25,1.1938142468694463E+25,3.2451855365842673E+32,3.1554436208840472E-30,1099511627776,3.2451855365842673E+32,8.711228593176025E+40,8.711228593176025E+40,0.0625,3.308722450212111E-24;1.7763568394002505E-15,2.5849394142282115E-26,262144,262144,2.5849394142282115E-26,3.1554436208840472E-30,0.0625,1099511627776,3.308722450212111E-24,1.7763568394002505E-15,1.1938142468694463E+25,8.61359646475425E+24,3.2451855365842673E+32,3.1554436208840472E-30,1099511627776,3.2451855365842673E+32,8.711228593176025E+40,8.711228593176025E+40,0.0625,3.308722450212111E-24;9.5367431640625E-07,1.3877787807814457E-17,140737488355328,140737488355328,1.3877787807814457E-17,1.6940658945086007E-21,33554432,5.902958103587057E+20,1.7763568394002505E-15,9.5367431640625E-07,3.2451855365842673E+32,3.2451855365842673E+32,5.2702932988715E+42,1.6940658945086007E-21,5.902958103587057E+20,3.44093529430453E+42,4.6768052394588893E+49,4.6768052394588893E+49,33554432,1.7763568394002505E-15;9.273015376718553E-69,1.349401336733507E-79,1.3684555315672042E-48,1.3684555315672042E-48,1.349401336733507E-79,3.253256396543794E-82,3.2626522339992623E-55,5.739718509874451E-42,1.727233711018889E-77,9.273015376718553E-69,3.1554436208840472E-30,3.1554436208840472E-30,1.6940658945086007E-21,2.347286260797426E-82,5.739718509874451E-42,1.6940658945086007E-21,4.547473508864641E-13,4.547473508864641E-13,3.2626522339992623E-55,1.727233711018889E-77;3.2311742677852644E-27,4.70197740328915E-38,4.76837158203125E-07,4.76837158203125E-07,4.70197740328915E-38,5.739718509874451E-42,1.1368683772161603E-13,39.5,6.018531076210112E-36,3.2311742677852644E-27,1099511627776,1099511627776,5.902958103587057E+20,5.739718509874451E-42,28.500000000000057,5.902958103587057E+20,1.5845632502852868E+29,1.5845632502852868E+29,1.1368683772161603E-13,6.018531076210112E-36;9.5367431640625E-07,1.3877787807814457E-17,140737488355328,140737488355328,1.3877787807814457E-17,1.6940658945086007E-21,33554432,5.902958103587057E+20,1.7763568394002505E-15,9.5367431640625E-07,3.2451855365842673E+32,3.2451855365842673E+32,3.44093529430453E+42,1.6940658945086007E-21,5.902958103587057E+20,2.482700149055172E+42,4.6768052394588893E+49,4.6768052394588893E+49,33554432,1.7763568394002505E-15;256,3.725290298461914E-09,3.777893186295716E+22,3.777893186295716E+22,3.725290298461914E-09,4.547473508864641E-13,9007199254740992,1.5845632502852868E+29,4.76837158203125E-07,256,8.711228593176025E+40,8.711228593176025E+40,4.6768052394588893E+49,4.547473508864641E-13,1.5845632502852868E+29,4.6768052394588893E+49,3.7976465499089454E+59,2.479455185477739E+59,9007199254740992,4.76837158203125E-07;256,3.725290298461914E-09,3.777893186295716E+22,3.777893186295716E+22,3.725290298461914E-09,4.547473508864641E-13,9007199254740992,1.5845632502852868E+29,4.76837158203125E-07,256,8.711228593176025E+40,8.711228593176025E+40,4.6768052394588893E+49,4.547473508864641E-13,1.5845632502852868E+29,4.6768052394588893E+49,2.479455185477739E+59,1.7889739945852076E+59,9007199254740992,4.76837158203125E-07;1.8367099231598242E-40,2.6727647100921956E-51,2.710505431213761E-20,2.710505431213761E-20,2.6727647100921956E-51,3.2626522339992623E-55,1.2763138357751794E-25,1.1368683772161603E-13,3.4211388289180104E-49,1.8367099231598242E-40,0.0625,0.0625,33554432,3.2626522339992623E-55,1.1368683772161603E-13,33554432,9007199254740992,9007199254740992,1.9548604320100868E-25,3.4211388289180104E-49;9.723461371658034E-63,1.4149498560666738E-73,1.4349296274686127E-42,1.4349296274686127E-42,1.4149498560666738E-73,1.727233711018889E-77,3.4211388289180104E-49,6.018531076210112E-36,3.5769932361365514E-70,9.723461371658034E-63,3.308722450212111E-24,3.308722450212111E-24,1.7763568394002505E-15,1.727233711018889E-77,6.018531076210112E-36,1.7763568394002505E-15,4.76837158203125E-07,4.76837158203125E-07,3.4211388289180104E-49,2.5808685374656182E-70"),
            "-1E+89",
            "1E+89"),
        "z2i-mi2-54#3" => new HeldOutProblem(
            name,
            "double",
            ParseMatrix("2.9937857748407735E-17,1.141798154164768E+46,3.4211388289180104E-49,1.305060893599705E-54,134217728,5.444517870735016E+39,3.4211388289180104E-49,3.599551212651875E-17,70368744177664,6.64613997892458E+35,0.0001220703125,1.305060893599705E-54,0.0001220703125,134217728,5.444517870735016E+39,70368744177664,9.860761315262648E-32,9.860761315262648E-32,1.141798154164768E+46,6.64613997892458E+35;1.141798154164768E+46,1.8982236337824473E+111,2251799813685248,8589934592,8.834235323891922E+71,3.583591587484487E+103,2251799813685248,1.141798154164768E+46,4.631683569492648E+77,4.374501449566024E+99,8.034690221294951E+59,8589934592,8.034690221294951E+59,8.834235323891922E+71,3.583591587484487E+103,4.631683569492648E+77,6.490371073168535E+32,6.490371073168535E+32,1.5594322749618253E+111,4.374501449566024E+99;3.4211388289180104E-49,2251799813685248,1.164398413467343E-78,2.5737787947340145E-85,2.6469779601696886E-23,1073741824,1.4000038868610135E-78,3.4211388289180104E-49,1.3877787807814457E-17,131072,2.407412430484045E-35,2.5737787947340145E-85,2.407412430484045E-35,2.6469779601696886E-23,1073741824,1.3877787807814457E-17,1.9446922743316068E-62,1.9446922743316068E-62,2251799813685248,131072;1.305060893599705E-54,8589934592,2.5737787947340145E-85,2.4798776549297995E-89,1.0097419586828951E-28,4096,2.5737787947340145E-85,1.305060893599705E-54,5.293955920339377E-23,0.5,9.183549615799121E-41,2.0372737880985565E-89,9.183549615799121E-41,1.0097419586828951E-28,4096,5.293955920339377E-23,7.418412301374843E-68,7.418412301374843E-68,8589934592,0.5;134217728,8.834235323891922E+71,2.6469779601696886E-23,1.0097419586828951E-28,1.792173183691881E+35,4.2124916667422875E+65,2.6469779601696886E-23,134217728,5.444517870735016E+39,5.142201741628769E+61,9.44473296573929E+21,1.0097419586828951E-28,9.44473296573929E+21,2.1548031962919535E+35,4.2124916667422875E+65,5.444517870735016E+39,7.62939453125E-06,7.62939453125E-06,8.834235323891922E+71,5.142201741628769E+61;5.444517870735016E+39,3.583591587484487E+103,1073741824,4096,4.2124916667422875E+65,2.949029141273845E+98,1073741824,5.444517870735016E+39,2.2085588309729804E+71,2.0859248397665138E+93,3.831238852164722E+53,4096,3.831238852164722E+53,4.2124916667422875E+65,3.545738479628711E+98,2.2085588309729804E+71,3.094850098213451E+26,3.094850098213451E+26,3.583591587484487E+103,2.0859248397665138E+93;3.4211388289180104E-49,2251799813685248,1.4000038868610135E-78,2.5737787947340145E-85,2.6469779601696886E-23,1073741824,1.704158948160746E-78,3.4211388289180104E-49,1.3877787807814457E-17,131072,2.407412430484045E-35,2.5737787947340145E-85,2.407412430484045E-35,2.6469779601696886E-23,1073741824,1.3877787807814457E-17,1.9446922743316068E-62,1.9446922743316068E-62,2251799813685248,131072;3.599551212651875E-17,1.141798154164768E+46,3.4211388289180104E-49,1.305060893599705E-54,134217728,5.444517870735016E+39,3.4211388289180104E-49,4.381564555622219E-17,70368744177664,6.64613997892458E+35,0.0001220703125,1.305060893599705E-54,0.0001220703125,134217728,5.444517870735016E+39,70368744177664,9.860761315262648E-32,9.860761315262648E-32,1.141798154164768E+46,6.64613997892458E+35;70368744177664,4.631683569492648E+77,1.3877787807814457E-17,5.293955920339377E-23,5.444517870735016E+39,2.2085588309729804E+71,1.3877787807814457E-17,70368744177664,4.926288136143891E+46,2.695994666715064E+67,4.951760157141521E+27,5.293955920339377E-23,4.951760157141521E+27,5.444517870735016E+39,2.2085588309729804E+71,5.923077924729734E+46,4,4,4.631683569492648E+77,2.695994666715064E+67;6.64613997892458E+35,4.374501449566024E+99,131072,0.5,5.142201741628769E+61,2.0859248397665138E+93,131072,6.64613997892458E+35,2.695994666715064E+67,4.39439585994757E+90,4.6768052394588893E+49,0.5,4.6768052394588893E+49,5.142201741628769E+61,2.0859248397665138E+93,2.695994666715064E+67,3.777893186295716E+22,3.777893186295716E+22,4.374501449566024E+99,5.283562063617573E+90;0.0001220703125,8.034690221294951E+59,2.407412430484045E-35,9.183549615799121E-41,9.44473296573929E+21,3.831238852164722E+53,2.407412430484045E-35,0.0001220703125,4.951760157141521E+27,4.6768052394588893E+49,216964567924.736,9.183549615799121E-41,178241142784,9.44473296573929E+21,3.831238852164722E+53,4.951760157141521E+27,6.938893903907228E-18,6.938893903907228E-18,8.034690221294951E+59,4.6768052394588893E+49;1.305060893599705E-54,8589934592,2.5737787947340145E-85,2.0372737880985565E-89,1.0097419586828951E-28,4096,2.5737787947340145E-85,1.305060893599705E-54,5.293955920339377E-23,0.5,9.183549615799121E-41,1.6944227004821632E-89,9.183549615799121E-41,1.0097419586828951E-28,4096,5.293955920339377E-23,7.418412301374843E-68,7.418412301374843E-68,8589934592,0.5;0.0001220703125,8.034690221294951E+59,2.407412430484045E-35,9.183549615799121E-41,9.44473296573929E+21,3.831238852164722E+53,2.407412430484045E-35,0.0001220703125,4.951760157141521E+27,4.6768052394588893E+49,178241142784,9.183549615799121E-41,148245091188.736,9.44473296573929E+21,3.831238852164722E+53,4.951760157141521E+27,6.938893903907228E-18,6.938893903907228E-18,8.034690221294951E+59,4.6768052394588893E+49;134217728,8.834235323891922E+71,2.6469779601696886E-23,1.0097419586828951E-28,2.1548031962919535E+35,4.2124916667422875E+65,2.6469779601696886E-23,134217728,5.444517870735016E+39,5.142201741628769E+61,9.44473296573929E+21,1.0097419586828951E-28,9.44473296573929E+21,2.6229406810574534E+35,4.2124916667422875E+65,5.444517870735016E+39,7.62939453125E-06,7.62939453125E-06,8.834235323891922E+71,5.142201741628769E+61;5.444517870735016E+39,3.583591587484487E+103,1073741824,4096,4.2124916667422875E+65,3.545738479628711E+98,1073741824,5.444517870735016E+39,2.2085588309729804E+71,2.0859248397665138E+93,3.831238852164722E+53,4096,3.831238852164722E+53,4.2124916667422875E+65,4.3160608442632276E+98,2.2085588309729804E+71,3.094850098213451E+26,3.094850098213451E+26,3.583591587484487E+103,2.0859248397665138E+93;70368744177664,4.631683569492648E+77,1.3877787807814457E-17,5.293955920339377E-23,5.444517870735016E+39,2.2085588309729804E+71,1.3877787807814457E-17,70368744177664,5.923077924729734E+46,2.695994666715064E+67,4.951760157141521E+27,5.293955920339377E-23,4.951760157141521E+27,5.444517870735016E+39,2.2085588309729804E+71,7.209884444473427E+46,4,4,4.631683569492648E+77,2.695994666715064E+67;9.860761315262648E-32,6.490371073168535E+32,1.9446922743316068E-62,7.418412301374843E-68,7.62939453125E-06,3.094850098213451E+26,1.9446922743316068E-62,9.860761315262648E-32,4,3.777893186295716E+22,6.938893903907228E-18,7.418412301374843E-68,6.938893903907228E-18,7.62939453125E-06,3.094850098213451E+26,4,9.673443558927077E-44,1.1630777253895982E-43,6.490371073168535E+32,3.777893186295716E+22;9.860761315262648E-32,6.490371073168535E+32,1.9446922743316068E-62,7.418412301374843E-68,7.62939453125E-06,3.094850098213451E+26,1.9446922743316068E-62,9.860761315262648E-32,4,3.777893186295716E+22,6.938893903907228E-18,7.418412301374843E-68,6.938893903907228E-18,7.62939453125E-06,3.094850098213451E+26,4,1.1630777253895982E-43,1.4157598644766491E-43,6.490371073168535E+32,3.777893186295716E+22;1.141798154164768E+46,1.5594322749618253E+111,2251799813685248,8589934592,8.834235323891922E+71,3.583591587484487E+103,2251799813685248,1.141798154164768E+46,4.631683569492648E+77,4.374501449566024E+99,8.034690221294951E+59,8589934592,8.034690221294951E+59,8.834235323891922E+71,3.583591587484487E+103,4.631683569492648E+77,6.490371073168535E+32,6.490371073168535E+32,1.296996732592346E+111,4.374501449566024E+99;6.64613997892458E+35,4.374501449566024E+99,131072,0.5,5.142201741628769E+61,2.0859248397665138E+93,131072,6.64613997892458E+35,2.695994666715064E+67,5.283562063617573E+90,4.6768052394588893E+49,0.5,4.6768052394588893E+49,5.142201741628769E+61,2.0859248397665138E+93,2.695994666715064E+67,3.777893186295716E+22,3.777893186295716E+22,4.374501449566024E+99,6.431431836282056E+90"),
            "-1E+89",
            "1E+89"),
        "z2g-mv2-174#2" => new HeldOutProblem(
            name,
            "double",
            ParseMatrix("149603950121713.66,1.6225927682921336E+32,0.03125,1.2855504354071922E+61,2.9514790517935283E+20,1.5407439555097887E-33,1.2855504354071922E+61,1.5407439555097887E-33,2.9514790517935283E+20,3.2138760885179806E+60,3.3409558876152446E-52,3.2138760885179806E+60,1.6225927682921336E+32,7.662477704329444E+53,6.681911775230489E-52,6.681911775230489E-52,0.03125,3.3409558876152446E-52,7.662477704329444E+53,167125767421952;1.6225927682921336E+32,5.090758624813874E+52,5.764607523034235E+17,2.3714219875802357E+80,5.444517870735016E+39,2.842170943040401E-14,2.3714219875802357E+80,2.842170943040401E-14,5.444517870735016E+39,5.928554968950589E+79,6.162975822039155E-33,5.928554968950589E+79,5.686995171182009E+52,1.4134776518227075E+73,1.232595164407831E-32,1.232595164407831E-32,5.764607523034235E+17,6.162975822039155E-33,1.4134776518227075E+73,1.6225927682921336E+32;0.03125,5.764607523034235E+17,2.3878676813637866E-15,4.567192616659072E+46,1048576,5.473822126268817E-48,4.567192616659072E+46,5.473822126268817E-48,1048576,1.141798154164768E+46,1.1869459682199748E-66,1.141798154164768E+46,5.764607523034235E+17,2.722258935367508E+39,2.3738919364399497E-66,2.3738919364399497E-66,2.1094237467877974E-15,1.1869459682199748E-66,2.722258935367508E+39,0.03125;1.2855504354071922E+61,2.3714219875802357E+80,4.567192616659072E+46,3.195520979825388E+110,4.3135914667441024E+68,2251799813685248,3.5697847258162265E+110,2251799813685248,4.3135914667441024E+68,4.6970851655476665E+108,0.00048828125,4.6970851655476665E+108,2.3714219875802357E+80,1.1198723710889021E+102,0.0009765625,0.0009765625,4.567192616659072E+46,0.00048828125,1.1198723710889021E+102,1.2855504354071922E+61;2.9514790517935283E+20,5.444517870735016E+39,1048576,4.3135914667441024E+68,1.6843907350532597E+29,5.169878828456423E-26,4.3135914667441024E+68,5.169878828456423E-26,1.881668859713778E+29,1.0783978666860256E+68,1.1210387714598537E-44,1.0783978666860256E+68,5.444517870735016E+39,2.5711008708143844E+61,2.2420775429197073E-44,2.2420775429197073E-44,1048576,1.1210387714598537E-44,2.5711008708143844E+61,2.9514790517935283E+20;1.5407439555097887E-33,2.842170943040401E-14,5.473822126268817E-48,2251799813685248,5.169878828456423E-26,5.804584790092853E-78,2251799813685248,5.1277250795873265E-78,5.169878828456423E-26,562949953421312,5.852095443365248E-98,562949953421312,2.842170943040401E-14,134217728,1.1704190886730496E-97,1.1704190886730496E-97,5.473822126268817E-48,5.852095443365248E-98,134217728,1.5407439555097887E-33;1.2855504354071922E+61,2.3714219875802357E+80,4.567192616659072E+46,3.5697847258162265E+110,4.3135914667441024E+68,2251799813685248,4.040996309623968E+110,2251799813685248,4.3135914667441024E+68,4.6970851655476665E+108,0.00048828125,4.6970851655476665E+108,2.3714219875802357E+80,1.1198723710889021E+102,0.0009765625,0.0009765625,4.567192616659072E+46,0.00048828125,1.1198723710889021E+102,1.2855504354071922E+61;1.5407439555097887E-33,2.842170943040401E-14,5.473822126268817E-48,2251799813685248,5.169878828456423E-26,5.1277250795873265E-78,2251799813685248,4.590123587032697E-78,5.169878828456423E-26,562949953421312,5.852095443365248E-98,562949953421312,2.842170943040401E-14,134217728,1.1704190886730496E-97,1.1704190886730496E-97,5.473822126268817E-48,5.852095443365248E-98,134217728,1.5407439555097887E-33;2.9514790517935283E+20,5.444517870735016E+39,1048576,4.3135914667441024E+68,1.881668859713778E+29,5.169878828456423E-26,4.3135914667441024E+68,5.169878828456423E-26,2.1300491491959966E+29,1.0783978666860256E+68,1.1210387714598537E-44,1.0783978666860256E+68,5.444517870735016E+39,2.5711008708143844E+61,2.2420775429197073E-44,2.2420775429197073E-44,1048576,1.1210387714598537E-44,2.5711008708143844E+61,2.9514790517935283E+20;3.2138760885179806E+60,5.928554968950589E+79,1.141798154164768E+46,4.6970851655476665E+108,1.0783978666860256E+68,562949953421312,4.6970851655476665E+108,562949953421312,1.0783978666860256E+68,2.52562269351498E+109,0.0001220703125,2.2311154536351416E+109,5.928554968950589E+79,2.7996809277222553E+101,0.000244140625,0.000244140625,1.141798154164768E+46,0.0001220703125,2.7996809277222553E+101,3.2138760885179806E+60;3.3409558876152446E-52,6.162975822039155E-33,1.1869459682199748E-66,0.00048828125,1.1210387714598537E-44,5.852095443365248E-98,0.00048828125,5.852095443365248E-98,1.1210387714598537E-44,0.0001220703125,2.729302651849249E-115,0.0001220703125,6.162975822039155E-33,2.9103830456733704E-11,2.5379418373156492E-116,2.5379418373156492E-116,1.1869459682199748E-66,2.4110447454498668E-115,2.9103830456733704E-11,3.3409558876152446E-52;3.2138760885179806E+60,5.928554968950589E+79,1.141798154164768E+46,4.6970851655476665E+108,1.0783978666860256E+68,562949953421312,4.6970851655476665E+108,562949953421312,1.0783978666860256E+68,2.2311154536351416E+109,0.0001220703125,1.9972006123908677E+109,5.928554968950589E+79,2.7996809277222553E+101,0.000244140625,0.000244140625,1.141798154164768E+46,0.0001220703125,2.7996809277222553E+101,3.2138760885179806E+60;1.6225927682921336E+32,5.686995171182009E+52,5.764607523034235E+17,2.3714219875802357E+80,5.444517870735016E+39,2.842170943040401E-14,2.3714219875802357E+80,2.842170943040401E-14,5.444517870735016E+39,5.928554968950589E+79,6.162975822039155E-33,5.928554968950589E+79,6.437678533778034E+52,1.4134776518227075E+73,1.232595164407831E-32,1.232595164407831E-32,5.764607523034235E+17,6.162975822039155E-33,1.4134776518227075E+73,1.6225927682921336E+32;7.662477704329444E+53,1.4134776518227075E+73,2.722258935367508E+39,1.1198723710889021E+102,2.5711008708143844E+61,134217728,1.1198723710889021E+102,134217728,2.5711008708143844E+61,2.7996809277222553E+101,2.9103830456733704E-11,2.7996809277222553E+101,1.4134776518227075E+73,1.1352771095919636E+96,5.820766091346741E-11,5.820766091346741E-11,2.722258935367508E+39,2.9103830456733704E-11,1.2682423025780404E+96,7.662477704329444E+53;6.681911775230489E-52,1.232595164407831E-32,2.3738919364399497E-66,0.0009765625,2.2420775429197073E-44,1.1704190886730496E-97,0.0009765625,1.1704190886730496E-97,2.2420775429197073E-44,0.000244140625,2.5379418373156492E-116,0.000244140625,1.232595164407831E-32,5.820766091346741E-11,1.0917210607396996E-114,9.644178981799467E-115,2.3738919364399497E-66,2.5379418373156492E-116,5.820766091346741E-11,6.681911775230489E-52;6.681911775230489E-52,1.232595164407831E-32,2.3738919364399497E-66,0.0009765625,2.2420775429197073E-44,1.1704190886730496E-97,0.0009765625,1.1704190886730496E-97,2.2420775429197073E-44,0.000244140625,2.5379418373156492E-116,0.000244140625,1.232595164407831E-32,5.820766091346741E-11,9.644178981799467E-115,8.633062953812912E-115,2.3738919364399497E-66,2.5379418373156492E-116,5.820766091346741E-11,6.681911775230489E-52;0.03125,5.764607523034235E+17,2.1094237467877974E-15,4.567192616659072E+46,1048576,5.473822126268817E-48,4.567192616659072E+46,5.473822126268817E-48,1048576,1.141798154164768E+46,1.1869459682199748E-66,1.141798154164768E+46,5.764607523034235E+17,2.722258935367508E+39,2.3738919364399497E-66,2.3738919364399497E-66,1.888267320282466E-15,1.1869459682199748E-66,2.722258935367508E+39,0.03125;3.3409558876152446E-52,6.162975822039155E-33,1.1869459682199748E-66,0.00048828125,1.1210387714598537E-44,5.852095443365248E-98,0.00048828125,5.852095443365248E-98,1.1210387714598537E-44,0.0001220703125,2.4110447454498668E-115,0.0001220703125,6.162975822039155E-33,2.9103830456733704E-11,2.5379418373156492E-116,2.5379418373156492E-116,1.1869459682199748E-66,2.158265738453228E-115,2.9103830456733704E-11,3.3409558876152446E-52;7.662477704329444E+53,1.4134776518227075E+73,2.722258935367508E+39,1.1198723710889021E+102,2.5711008708143844E+61,134217728,1.1198723710889021E+102,134217728,2.5711008708143844E+61,2.7996809277222553E+101,2.9103830456733704E-11,2.7996809277222553E+101,1.4134776518227075E+73,1.2682423025780404E+96,5.820766091346741E-11,5.820766091346741E-11,2.722258935367508E+39,2.9103830456733704E-11,1.4356502865183416E+96,7.662477704329444E+53;167125767421952,1.6225927682921336E+32,0.03125,1.2855504354071922E+61,2.9514790517935283E+20,1.5407439555097887E-33,1.2855504354071922E+61,1.5407439555097887E-33,2.9514790517935283E+20,3.2138760885179806E+60,3.3409558876152446E-52,3.2138760885179806E+60,1.6225927682921336E+32,7.662477704329444E+53,6.681911775230489E-52,6.681911775230489E-52,0.03125,3.3409558876152446E-52,7.662477704329444E+53,189186368721649.66"),
            "-1E+150",
            "1E+150",
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0".Split(','),
            "1"),
        "z2i-mv2-76#3" => new HeldOutProblem(
            name,
            "double",
            ParseMatrix("7.164719992461544E-49,1.6225927682921336E+32,2.710505431213761E-20,4.0091470651382935E-49,8.507059173023462E+37,3.6734198463196485E-40,4.861730685829017E-63,1.6225927682921336E+32,2.384185791015625E-07,5.764607523034235E+17,2.710505431213761E-20,5.764607523034235E+17,3.6734198463196485E-40,4.547473508864641E-13,6.103515625E-05,8.507059173023462E+37,4.547473508864641E-13,6.103515625E-05,2.384185791015625E-07,4.861730685829017E-63;1.6225927682921336E+32,4.125882573839957E+115,2.0568806966515076E+62,1.6225927682921336E+32,6.455624695217272E+119,2.787593149816328E+42,3.6893488147419103E+19,2.308711300569989E+115,1.8092513943330656E+75,4.374501449566024E+99,2.0568806966515076E+62,4.374501449566024E+99,2.787593149816328E+42,3.450873173395282E+69,4.631683569492648E+77,6.455624695217272E+119,3.450873173395282E+69,4.631683569492648E+77,1.8092513943330656E+75,3.6893488147419103E+19;2.710505431213761E-20,2.0568806966515076E+62,386821934546.944,2.710505431213761E-20,1.0783978666860256E+68,4.656612873077393E-10,6.162975822039155E-33,2.0568806966515076E+62,3.022314549036573E+23,7.307508186654515E+47,644245094400,7.307508186654515E+47,4.656612873077393E-10,5.764607523034235E+17,7.737125245533627E+25,1.0783978666860256E+68,5.764607523034235E+17,7.737125245533627E+25,3.022314549036573E+23,6.162975822039155E-33;4.0091470651382935E-49,1.6225927682921336E+32,2.710505431213761E-20,2.407198808497435E-49,8.507059173023462E+37,3.6734198463196485E-40,4.861730685829017E-63,1.6225927682921336E+32,2.384185791015625E-07,5.764607523034235E+17,2.710505431213761E-20,5.764607523034235E+17,3.6734198463196485E-40,4.547473508864641E-13,6.103515625E-05,8.507059173023462E+37,4.547473508864641E-13,6.103515625E-05,2.384185791015625E-07,4.861730685829017E-63;8.507059173023462E+37,6.455624695217272E+119,1.0783978666860256E+68,8.507059173023462E+37,1.134113966193851E+127,1.461501637330903E+48,1.9342813113834067E+25,6.455624695217272E+119,9.485687950320943E+80,2.2934986159900715E+105,1.0783978666860256E+68,2.2934986159900715E+105,1.461501637330903E+48,1.8092513943330656E+75,2.4283361152821613E+83,6.346137300386387E+126,1.8092513943330656E+75,2.4283361152821613E+83,9.485687950320943E+80,1.9342813113834067E+25;3.6734198463196485E-40,2.787593149816328E+42,4.656612873077393E-10,3.6734198463196485E-40,1.461501637330903E+48,2.1146520969716533E-28,8.352389719038111E-53,2.787593149816328E+42,4096,9.903520314283042E+27,4.656612873077393E-10,9.903520314283042E+27,1.1832913578315177E-28,0.0078125,1048576,1.461501637330903E+48,0.0078125,1048576,4096,8.352389719038111E-53;4.861730685829017E-63,3.6893488147419103E+19,6.162975822039155E-33,4.861730685829017E-63,1.9342813113834067E+25,8.352389719038111E-53,1.2444926155936416E-74,3.6893488147419103E+19,5.421010862427522E-20,131072,6.162975822039155E-33,131072,8.352389719038111E-53,1.0339757656912846E-25,1.3877787807814457E-17,1.9342813113834067E+25,1.0339757656912846E-25,1.3877787807814457E-17,5.421010862427522E-20,2.0726804532226667E-74;1.6225927682921336E+32,2.308711300569989E+115,2.0568806966515076E+62,1.6225927682921336E+32,6.455624695217272E+119,2.787593149816328E+42,3.6893488147419103E+19,1.3862118304969032E+115,1.8092513943330656E+75,4.374501449566024E+99,2.0568806966515076E+62,4.374501449566024E+99,2.787593149816328E+42,3.450873173395282E+69,4.631683569492648E+77,6.455624695217272E+119,3.450873173395282E+69,4.631683569492648E+77,1.8092513943330656E+75,3.6893488147419103E+19;2.384185791015625E-07,1.8092513943330656E+75,3.022314549036573E+23,2.384185791015625E-07,9.485687950320943E+80,4096,5.421010862427522E-20,1.8092513943330656E+75,8.907954336552193E+37,6.427752177035961E+60,3.022314549036573E+23,6.427752177035961E+60,4096,5.070602400912918E+30,6.80564733841877E+38,9.485687950320943E+80,5.070602400912918E+30,6.80564733841877E+38,4.9846049841934345E+37,5.421010862427522E-20;5.764607523034235E+17,4.374501449566024E+99,7.307508186654515E+47,5.764607523034235E+17,2.2934986159900715E+105,9.903520314283042E+27,131072,4.374501449566024E+99,6.427752177035961E+60,5.207595939255979E+86,7.307508186654515E+47,2.9140033383385936E+86,9.903520314283042E+27,1.2259964326927111E+55,1.645504557321206E+63,2.2934986159900715E+105,1.2259964326927111E+55,1.645504557321206E+63,6.427752177035961E+60,131072;2.710505431213761E-20,2.0568806966515076E+62,644245094400,2.710505431213761E-20,1.0783978666860256E+68,4.656612873077393E-10,6.162975822039155E-33,2.0568806966515076E+62,3.022314549036573E+23,7.307508186654515E+47,1151326113234.944,7.307508186654515E+47,4.656612873077393E-10,5.764607523034235E+17,7.737125245533627E+25,1.0783978666860256E+68,5.764607523034235E+17,7.737125245533627E+25,3.022314549036573E+23,6.162975822039155E-33;5.764607523034235E+17,4.374501449566024E+99,7.307508186654515E+47,5.764607523034235E+17,2.2934986159900715E+105,9.903520314283042E+27,131072,4.374501449566024E+99,6.427752177035961E+60,2.9140033383385936E+86,7.307508186654515E+47,1.7496453110941805E+86,9.903520314283042E+27,1.2259964326927111E+55,1.645504557321206E+63,2.2934986159900715E+105,1.2259964326927111E+55,1.645504557321206E+63,6.427752177035961E+60,131072;3.6734198463196485E-40,2.787593149816328E+42,4.656612873077393E-10,3.6734198463196485E-40,1.461501637330903E+48,1.1832913578315177E-28,8.352389719038111E-53,2.787593149816328E+42,4096,9.903520314283042E+27,4.656612873077393E-10,9.903520314283042E+27,7.10479685678252E-29,0.0078125,1048576,1.461501637330903E+48,0.0078125,1048576,4096,8.352389719038111E-53;4.547473508864641E-13,3.450873173395282E+69,5.764607523034235E+17,4.547473508864641E-13,1.8092513943330656E+75,0.0078125,1.0339757656912846E-25,3.450873173395282E+69,5.070602400912918E+30,1.2259964326927111E+55,5.764607523034235E+17,1.2259964326927111E+55,0.0078125,3.24069490909176E+26,1.298074214633707E+33,1.8092513943330656E+75,1.8133887294219438E+26,1.298074214633707E+33,5.070602400912918E+30,1.0339757656912846E-25;6.103515625E-05,4.631683569492648E+77,7.737125245533627E+25,6.103515625E-05,2.4283361152821613E+83,1048576,1.3877787807814457E-17,4.631683569492648E+77,6.80564733841877E+38,1.645504557321206E+63,7.737125245533627E+25,1.645504557321206E+63,1048576,1.298074214633707E+33,1.9614202300395136E+42,2.4283361152821613E+83,1.298074214633707E+33,3.266710722441009E+42,6.80564733841877E+38,1.3877787807814457E-17;8.507059173023462E+37,6.455624695217272E+119,1.0783978666860256E+68,8.507059173023462E+37,6.346137300386387E+126,1.461501637330903E+48,1.9342813113834067E+25,6.455624695217272E+119,9.485687950320943E+80,2.2934986159900715E+105,1.0783978666860256E+68,2.2934986159900715E+105,1.461501637330903E+48,1.8092513943330656E+75,2.4283361152821613E+83,3.8103900654799965E+126,1.8092513943330656E+75,2.4283361152821613E+83,9.485687950320943E+80,1.9342813113834067E+25;4.547473508864641E-13,3.450873173395282E+69,5.764607523034235E+17,4.547473508864641E-13,1.8092513943330656E+75,0.0078125,1.0339757656912846E-25,3.450873173395282E+69,5.070602400912918E+30,1.2259964326927111E+55,5.764607523034235E+17,1.2259964326927111E+55,0.0078125,1.8133887294219438E+26,1.298074214633707E+33,1.8092513943330656E+75,1.0888069501777195E+26,1.298074214633707E+33,5.070602400912918E+30,1.0339757656912846E-25;6.103515625E-05,4.631683569492648E+77,7.737125245533627E+25,6.103515625E-05,2.4283361152821613E+83,1048576,1.3877787807814457E-17,4.631683569492648E+77,6.80564733841877E+38,1.645504557321206E+63,7.737125245533627E+25,1.645504557321206E+63,1048576,1.298074214633707E+33,3.266710722441009E+42,2.4283361152821613E+83,1.298074214633707E+33,5.837916954002845E+42,6.80564733841877E+38,1.3877787807814457E-17;2.384185791015625E-07,1.8092513943330656E+75,3.022314549036573E+23,2.384185791015625E-07,9.485687950320943E+80,4096,5.421010862427522E-20,1.8092513943330656E+75,4.9846049841934345E+37,6.427752177035961E+60,3.022314549036573E+23,6.427752177035961E+60,4096,5.070602400912918E+30,6.80564733841877E+38,9.485687950320943E+80,5.070602400912918E+30,6.80564733841877E+38,2.9928897553093163E+37,5.421010862427522E-20;4.861730685829017E-63,3.6893488147419103E+19,6.162975822039155E-33,4.861730685829017E-63,1.9342813113834067E+25,8.352389719038111E-53,2.0726804532226667E-74,3.6893488147419103E+19,5.421010862427522E-20,131072,6.162975822039155E-33,131072,8.352389719038111E-53,1.0339757656912846E-25,1.3877787807814457E-17,1.9342813113834067E+25,1.0339757656912846E-25,1.3877787807814457E-17,5.421010862427522E-20,3.70407342008454E-74"),
            "-1E+150",
            "1E+150",
            "9.074416161091648E+99,1E+150,-5.938284859480767E+129,-4.684480769710253E+99,1E+150,1.558972825687299E+110,-1.0651276101521532E+87,-1E+150,1.0118312131696557E+143,1E+150,1.150318909332851E+130,-1E+150,-8.04787668185857E+109,1.929914881076156E+137,-1.337182874018866E+145,-1E+150,-9.962788775703797E+136,2.5902879057143203E+145,-5.223370601636193E+142,2.0632833550488572E+87".Split(','),
            "0.08307818935664357"),
        "k7e-935007-135" => new HeldOutProblem(
            name,
            "double",
            ParseMatrix("1.8273181228672674E-114,2.3283064365386963E-10,5.345529420184391E-51,8.156630584998156E-56,7.275957614183426E-12,5.345529420184391E-51,8.156630584998156E-56,7.275957614183426E-12,1.5340917079055395E-92,3.111507638930571E-61,7.965459555662261E-59,2.3283064365386963E-10,4.930380657631324E-32,1.8909140209225187E-124,3.111507638930571E-61,2.0303534698525194E-115,4.930380657631324E-32,1.8909140209225187E-124,7.965459555662261E-59,1.5340917079055395E-92;2.3283064365386963E-10,2.402985415411024E+96,6.129982163463556E+54,9.353610478917779E+49,8.343699359066055E+93,6.129982163463556E+54,9.353610478917779E+49,8.343699359066055E+93,17592186044416,3.5681192317649E+44,9.134385233318143E+46,2.6699837949011376E+95,5.65391060729083E+73,2.168404344971009E-19,3.5681192317649E+44,2.3283064365386963E-10,5.65391060729083E+73,2.168404344971009E-19,9.134385233318143E+46,17592186044416;5.345529420184391E-51,6.129982163463556E+54,1266637395197952,2147483648,1.915619426082361E+53,140737488355328,2147483648,1.915619426082361E+53,4.0389678347315804E-28,8192,2097152,6.129982163463556E+54,1.298074214633707E+33,4.9784122222889134E-60,8192,5.345529420184391E-51,1.298074214633707E+33,4.9784122222889134E-60,2097152,4.0389678347315804E-28;8.156630584998156E-56,9.353610478917779E+49,2147483648,294912,2.923003274661806E+48,2147483648,32768,2.923003274661806E+48,6.162975822039155E-33,0.125,32,9.353610478917779E+49,1.9807040628566084E+28,7.596454196607839E-65,0.125,8.156630584998156E-56,1.9807040628566084E+28,7.596454196607839E-65,32,6.162975822039155E-33;7.275957614183426E-12,8.343699359066055E+93,1.915619426082361E+53,2.923003274661806E+48,2.346665444737328E+93,1.915619426082361E+53,2.923003274661806E+48,2.6074060497081422E+92,549755813888,1.1150372599265312E+43,2.85449538541192E+45,8.343699359066055E+93,1.7668470647783843E+72,6.776263578034403E-21,1.1150372599265312E+43,7.275957614183426E-12,1.7668470647783843E+72,6.776263578034403E-21,2.85449538541192E+45,549755813888;5.345529420184391E-51,6.129982163463556E+54,140737488355328,2147483648,1.915619426082361E+53,1266637395197952,2147483648,1.915619426082361E+53,4.0389678347315804E-28,8192,2097152,6.129982163463556E+54,1.298074214633707E+33,4.9784122222889134E-60,8192,5.345529420184391E-51,1.298074214633707E+33,4.9784122222889134E-60,2097152,4.0389678347315804E-28;8.156630584998156E-56,9.353610478917779E+49,2147483648,32768,2.923003274661806E+48,2147483648,294912,2.923003274661806E+48,6.162975822039155E-33,0.125,32,9.353610478917779E+49,1.9807040628566084E+28,7.596454196607839E-65,0.125,8.156630584998156E-56,1.9807040628566084E+28,7.596454196607839E-65,32,6.162975822039155E-33;7.275957614183426E-12,8.343699359066055E+93,1.915619426082361E+53,2.923003274661806E+48,2.6074060497081422E+92,1.915619426082361E+53,2.923003274661806E+48,2.346665444737328E+93,549755813888,1.1150372599265312E+43,2.85449538541192E+45,8.343699359066055E+93,1.7668470647783843E+72,6.776263578034403E-21,1.1150372599265312E+43,7.275957614183426E-12,1.7668470647783843E+72,6.776263578034403E-21,2.85449538541192E+45,549755813888;1.5340917079055395E-92,17592186044416,4.0389678347315804E-28,6.162975822039155E-33,549755813888,4.0389678347315804E-28,6.162975822039155E-33,549755813888,1.0432142298808373E-68,2.350988701644575E-38,6.018531076210112E-36,17592186044416,3.725290298461914E-09,1.4287342391028437E-101,2.350988701644575E-38,1.5340917079055395E-92,3.725290298461914E-09,1.4287342391028437E-101,6.018531076210112E-36,1.1591269220898192E-69;3.111507638930571E-61,3.5681192317649E+44,8192,0.125,1.1150372599265312E+43,8192,0.125,1.1150372599265312E+43,2.350988701644575E-38,4.291534423828125E-06,0.0001220703125,3.5681192317649E+44,7.555786372591432E+22,2.897817305224548E-70,4.76837158203125E-07,3.111507638930571E-61,7.555786372591432E+22,2.897817305224548E-70,0.0001220703125,2.350988701644575E-38;7.965459555662261E-59,9.134385233318143E+46,2097152,32,2.85449538541192E+45,2097152,32,2.85449538541192E+45,6.018531076210112E-36,0.0001220703125,0.28125,9.134385233318143E+46,1.9342813113834067E+25,7.418412301374843E-68,0.0001220703125,7.965459555662261E-59,1.9342813113834067E+25,7.418412301374843E-68,0.03125,6.018531076210112E-36;2.3283064365386963E-10,2.6699837949011376E+95,6.129982163463556E+54,9.353610478917779E+49,8.343699359066055E+93,6.129982163463556E+54,9.353610478917779E+49,8.343699359066055E+93,17592186044416,3.5681192317649E+44,9.134385233318143E+46,2.402985415411024E+96,5.65391060729083E+73,2.168404344971009E-19,3.5681192317649E+44,2.3283064365386963E-10,5.65391060729083E+73,2.168404344971009E-19,9.134385233318143E+46,17592186044416;4.930380657631324E-32,5.65391060729083E+73,1.298074214633707E+33,1.9807040628566084E+28,1.7668470647783843E+72,1.298074214633707E+33,1.9807040628566084E+28,1.7668470647783843E+72,3.725290298461914E-09,7.555786372591432E+22,1.9342813113834067E+25,5.65391060729083E+73,1.077535927171328E+53,4.591774807899561E-41,7.555786372591432E+22,4.930380657631324E-32,1.1972621413014757E+52,4.591774807899561E-41,1.9342813113834067E+25,3.725290298461914E-09;1.8909140209225187E-124,2.168404344971009E-19,4.9784122222889134E-60,7.596454196607839E-65,6.776263578034403E-21,4.9784122222889134E-60,7.596454196607839E-65,6.776263578034403E-21,1.4287342391028437E-101,2.897817305224548E-70,7.418412301374843E-68,2.168404344971009E-19,4.591774807899561E-41,1.5849458229078602E-132,2.897817305224548E-70,1.8909140209225187E-124,4.591774807899561E-41,1.761050914342067E-133,7.418412301374843E-68,1.4287342391028437E-101;3.111507638930571E-61,3.5681192317649E+44,8192,0.125,1.1150372599265312E+43,8192,0.125,1.1150372599265312E+43,2.350988701644575E-38,4.76837158203125E-07,0.0001220703125,3.5681192317649E+44,7.555786372591432E+22,2.897817305224548E-70,4.291534423828125E-06,3.111507638930571E-61,7.555786372591432E+22,2.897817305224548E-70,0.0001220703125,2.350988701644575E-38;2.0303534698525194E-115,2.3283064365386963E-10,5.345529420184391E-51,8.156630584998156E-56,7.275957614183426E-12,5.345529420184391E-51,8.156630584998156E-56,7.275957614183426E-12,1.5340917079055395E-92,3.111507638930571E-61,7.965459555662261E-59,2.3283064365386963E-10,4.930380657631324E-32,1.8909140209225187E-124,3.111507638930571E-61,1.8273181228672674E-114,4.930380657631324E-32,1.8909140209225187E-124,7.965459555662261E-59,1.5340917079055395E-92;4.930380657631324E-32,5.65391060729083E+73,1.298074214633707E+33,1.9807040628566084E+28,1.7668470647783843E+72,1.298074214633707E+33,1.9807040628566084E+28,1.7668470647783843E+72,3.725290298461914E-09,7.555786372591432E+22,1.9342813113834067E+25,5.65391060729083E+73,1.1972621413014757E+52,4.591774807899561E-41,7.555786372591432E+22,4.930380657631324E-32,1.077535927171328E+53,4.591774807899561E-41,1.9342813113834067E+25,3.725290298461914E-09;1.8909140209225187E-124,2.168404344971009E-19,4.9784122222889134E-60,7.596454196607839E-65,6.776263578034403E-21,4.9784122222889134E-60,7.596454196607839E-65,6.776263578034403E-21,1.4287342391028437E-101,2.897817305224548E-70,7.418412301374843E-68,2.168404344971009E-19,4.591774807899561E-41,1.761050914342067E-133,2.897817305224548E-70,1.8909140209225187E-124,4.591774807899561E-41,1.5849458229078602E-132,7.418412301374843E-68,1.4287342391028437E-101;7.965459555662261E-59,9.134385233318143E+46,2097152,32,2.85449538541192E+45,2097152,32,2.85449538541192E+45,6.018531076210112E-36,0.0001220703125,0.03125,9.134385233318143E+46,1.9342813113834067E+25,7.418412301374843E-68,0.0001220703125,7.965459555662261E-59,1.9342813113834067E+25,7.418412301374843E-68,0.28125,6.018531076210112E-36;1.5340917079055395E-92,17592186044416,4.0389678347315804E-28,6.162975822039155E-33,549755813888,4.0389678347315804E-28,6.162975822039155E-33,549755813888,1.1591269220898192E-69,2.350988701644575E-38,6.018531076210112E-36,17592186044416,3.725290298461914E-09,1.4287342391028437E-101,2.350988701644575E-38,1.5340917079055395E-92,3.725290298461914E-09,1.4287342391028437E-101,6.018531076210112E-36,1.0432142298808373E-68"),
            "-1E+89",
            "1E+89",
            "2.7898078576873015E-23,65786056145275.47,1.3680945509740797E+46,1.442072031180521E-65,782698977.8941387,-5.027643857555136E+33,-5.073521044469966E-41,1.1730013789008158E-34,-7657735.799725139,3.3521461772944588E+62,-165004514.717017,-1.062148446019181E+58,5.892334686484354E+58,-3.0239576739635205E-67,9.991998628905614E-26,1.2858277832244846E-08,-1.7045892701684966E-29,2.5319993504335914E-48,-3.1279917678347036E-10,-4.0922531357646206E+63".Split(','),
            "1"),
        "k7e-935004-236" => new HeldOutProblem(
            name,
            "double",
            ParseMatrix("9.301181600167216E+68,3.417579257473456E+97,0.0001220703125,33554432,3.291009114642412E+63,2.7996809277222553E+101,1.1857109937901178E+80,9.975180266845737E+68,1.1857109937901178E+80,3.094850098213451E+26,0.0001220703125,3.094850098213451E+26,2.7996809277222553E+101,1.862645149230957E-09,3.417579257473456E+97,1.862645149230957E-09,0,3.291009114642412E+63,33554432;3.417579257473456E+97,4.819679741733546E+128,7.737125245533627E+25,2.1267647932558654E+37,2.0859248397665138E+93,1.7745086042373215E+131,7.515336264876266E+109,3.417579257473456E+97,7.515336264876266E+109,1.9615942923083377E+56,7.737125245533627E+25,1.9615942923083377E+56,1.7745086042373215E+131,1.1805916207174113E+21,4.00737416728399E+128,1.1805916207174113E+21,0,2.0859248397665138E+93,2.1267647932558654E+37;0.0001220703125,7.737125245533627E+25,6.14895201122737E-75,7.596454196607839E-65,7.450580596923828E-09,6.338253001141147E+29,268435456,0.0001220703125,268435456,7.006492321624085E-46,5.112611784615911E-75,7.006492321624085E-46,6.338253001141147E+29,4.2168791772922093E-81,7.737125245533627E+25,4.2168791772922093E-81,0,7.450580596923828E-09,7.596454196607839E-65;33554432,2.1267647932558654E+37,7.596454196607839E-65,4.6460167812150444E-52,2048,1.742245718635205E+41,7.378697629483821E+19,33554432,7.378697629483821E+19,1.925929944387236E-34,7.596454196607839E-65,1.925929944387236E-34,1.742245718635205E+41,1.1591269220898192E-69,2.1267647932558654E+37,1.1591269220898192E-69,0,2048,3.8629802450551265E-52;3.291009114642412E+63,2.0859248397665138E+93,7.450580596923828E-09,2048,4.469296435595408E+60,1.708789628736728E+97,7.237005577332262E+75,3.291009114642412E+63,7.237005577332262E+75,1.888946593147858E+22,7.450580596923828E-09,1.888946593147858E+22,1.708789628736728E+97,1.1368683772161603E-13,2.0859248397665138E+93,1.1368683772161603E-13,0,3.716044227348915E+60,2048;2.7996809277222553E+101,1.7745086042373215E+131,6.338253001141147E+29,1.742245718635205E+41,1.708789628736728E+97,2.50759359881991E+136,6.156563468186638E+113,2.7996809277222553E+101,6.156563468186638E+113,1.6069380442589903E+60,6.338253001141147E+29,1.6069380442589903E+60,2.6893032798937455E+136,9.671406556917033E+24,1.7745086042373215E+131,9.671406556917033E+24,0,1.708789628736728E+97,1.742245718635205E+41;1.1857109937901178E+80,7.515336264876266E+109,268435456,7.378697629483821E+19,7.237005577332262E+75,6.156563468186638E+113,4.497775435746664E+93,1.1857109937901178E+80,4.823701191960063E+93,6.80564733841877E+38,268435456,6.80564733841877E+38,6.156563468186638E+113,4096,7.515336264876266E+109,4096,0,7.237005577332262E+75,7.378697629483821E+19;9.975180266845737E+68,3.417579257473456E+97,0.0001220703125,33554432,3.291009114642412E+63,2.7996809277222553E+101,1.1857109937901178E+80,1.199717626688228E+69,1.1857109937901178E+80,3.094850098213451E+26,0.0001220703125,3.094850098213451E+26,2.7996809277222553E+101,1.862645149230957E-09,3.417579257473456E+97,1.862645149230957E-09,0,3.291009114642412E+63,33554432;1.1857109937901178E+80,7.515336264876266E+109,268435456,7.378697629483821E+19,7.237005577332262E+75,6.156563468186638E+113,4.823701191960063E+93,1.1857109937901178E+80,5.801478460600735E+93,6.80564733841877E+38,268435456,6.80564733841877E+38,6.156563468186638E+113,4096,7.515336264876266E+109,4096,0,7.237005577332262E+75,7.378697629483821E+19;3.094850098213451E+26,1.9615942923083377E+56,7.006492321624085E-46,1.925929944387236E-34,1.888946593147858E+22,1.6069380442589903E+60,6.80564733841877E+38,3.094850098213451E+26,6.80564733841877E+38,3.952393967665638E-14,7.006492321624085E-46,3.2862601528904634E-14,1.6069380442589903E+60,1.0691058840368783E-50,1.9615942923083377E+56,1.0691058840368783E-50,0,1.888946593147858E+22,1.925929944387236E-34;0.0001220703125,7.737125245533627E+25,5.112611784615911E-75,7.596454196607839E-65,7.450580596923828E-09,6.338253001141147E+29,268435456,0.0001220703125,268435456,7.006492321624085E-46,4.767165042412259E-75,7.006492321624085E-46,6.338253001141147E+29,4.2168791772922093E-81,7.737125245533627E+25,4.2168791772922093E-81,0,7.450580596923828E-09,7.596454196607839E-65;3.094850098213451E+26,1.9615942923083377E+56,7.006492321624085E-46,1.925929944387236E-34,1.888946593147858E+22,1.6069380442589903E+60,6.80564733841877E+38,3.094850098213451E+26,6.80564733841877E+38,3.2862601528904634E-14,7.006492321624085E-46,3.064215547965513E-14,1.6069380442589903E+60,1.0691058840368783E-50,1.9615942923083377E+56,1.0691058840368783E-50,0,1.888946593147858E+22,1.925929944387236E-34;2.7996809277222553E+101,1.7745086042373215E+131,6.338253001141147E+29,1.742245718635205E+41,1.708789628736728E+97,2.6893032798937455E+136,6.156563468186638E+113,2.7996809277222553E+101,6.156563468186638E+113,1.6069380442589903E+60,6.338253001141147E+29,1.6069380442589903E+60,3.234432323115517E+136,9.671406556917033E+24,1.7745086042373215E+131,9.671406556917033E+24,0,1.708789628736728E+97,1.742245718635205E+41;1.862645149230957E-09,1.1805916207174113E+21,4.2168791772922093E-81,1.1591269220898192E-69,1.1368683772161603E-13,9.671406556917033E+24,4096,1.862645149230957E-09,4096,1.0691058840368783E-50,4.2168791772922093E-81,1.0691058840368783E-50,9.671406556917033E+24,1.4316644545708248E-84,1.1805916207174113E+21,1.1903726925644817E-84,0,1.1368683772161603E-13,1.1591269220898192E-69;3.417579257473456E+97,4.00737416728399E+128,7.737125245533627E+25,2.1267647932558654E+37,2.0859248397665138E+93,1.7745086042373215E+131,7.515336264876266E+109,3.417579257473456E+97,7.515336264876266E+109,1.9615942923083377E+56,7.737125245533627E+25,1.9615942923083377E+56,1.7745086042373215E+131,1.1805916207174113E+21,3.736605642467603E+128,1.1805916207174113E+21,0,2.0859248397665138E+93,2.1267647932558654E+37;1.862645149230957E-09,1.1805916207174113E+21,4.2168791772922093E-81,1.1591269220898192E-69,1.1368683772161603E-13,9.671406556917033E+24,4096,1.862645149230957E-09,4096,1.0691058840368783E-50,4.2168791772922093E-81,1.0691058840368783E-50,9.671406556917033E+24,1.1903726925644817E-84,1.1805916207174113E+21,1.109942105229073E-84,0,1.1368683772161603E-13,1.1591269220898192E-69;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6.490371073168535E+32,0,0;3.291009114642412E+63,2.0859248397665138E+93,7.450580596923828E-09,2048,3.716044227348915E+60,1.708789628736728E+97,7.237005577332262E+75,3.291009114642412E+63,7.237005577332262E+75,1.888946593147858E+22,7.450580596923828E-09,1.888946593147858E+22,1.708789628736728E+97,1.1368683772161603E-13,2.0859248397665138E+93,1.1368683772161603E-13,0,3.464960157933539E+60,2048;33554432,2.1267647932558654E+37,7.596454196607839E-65,3.8629802450551265E-52,2048,1.742245718635205E+41,7.378697629483821E+19,33554432,7.378697629483821E+19,1.925929944387236E-34,7.596454196607839E-65,1.925929944387236E-34,1.742245718635205E+41,1.1591269220898192E-69,2.1267647932558654E+37,1.1591269220898192E-69,0,2048,3.6019680663352805E-52"),
            "-1E+89",
            "1E+89"),
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unknown witness"),
    };

    /// <summary>
    /// A seeded degenerate-structure problem (<paramref name="trial"/> mod 5 picks the covariance shape). The core block
    /// holds repeated 2×2 blocks (one block copied, correlation near 1 or moderate), groups of identical assets
    /// (<c>v·((1 − ρ)I + ρ11′)</c> per group, the groups sometimes identical to each other), one group of identical assets,
    /// repeated 3×3 blocks (one random correlation block copied) or a tied factor model (rows of loadings repeated); the
    /// core is scaled by a power drawn over the operating domain and up to two hedge legs of independent variance follow.
    /// </summary>
    private static HeldOutProblem? DegenerateStructureProblem(System.Random rng, string type, int trial, bool meanVariance)
    {
        var dec = type == "decimal";
        double Uniform(double lo, double hi) => lo + ((hi - lo) * rng.NextDouble());
        double Power(double lo, double hi) => Math.Pow(10, Uniform(lo, hi));
        var reach = dec ? 6.0 : 120.0;
        var n = rng.Next(2, 17);
        var tail = Math.Min(n - 2, rng.Next(0, 3));
        var core = n - tail;
        var cov = new double[n, n];
        var group = new int[n];
        var scale = Power(-reach, reach);
        var shape = trial % 5;
        switch (shape)
        {
            case 0:
                {
                    var a = Power(-1, 1);
                    var b = a * Power(-1, 1);
                    var rho = rng.Next(2) == 0 ? 1 - Power(dec ? -6 : -12, -1) : Uniform(-0.95, 0.95);
                    rho *= rng.Next(4) == 0 ? -1 : 1;
                    for (var i = 0; i < core; i++)
                    {
                        group[i] = i / 2;
                        cov[i, i] = i % 2 == 0 ? a : b;
                        if (i % 2 == 1)
                        {
                            cov[i, i - 1] = cov[i - 1, i] = rho * Math.Sqrt(a * b);
                        }
                    }

                    break;
                }

            case 1:
            case 2:
                {
                    var size = shape == 2 ? core : rng.Next(2, Math.Max(3, core));
                    var shared = rng.Next(2) == 0;
                    var v = Power(-1, 1);
                    var rho = rng.Next(2) == 0 ? Uniform(0, 0.99) : 1 - Power(dec ? -6 : -12, -1);
                    for (var i = 0; i < core; i++)
                    {
                        group[i] = i / size;
                    }

                    var groups = group.Take(core).Max() + 1;
                    var variance = new double[groups];
                    for (var g = 0; g < groups; g++)
                    {
                        variance[g] = shared ? v : Power(-1, 1);
                    }

                    for (var i = 0; i < core; i++)
                    {
                        for (var j = 0; j < core; j++)
                        {
                            if (group[i] == group[j])
                            {
                                cov[i, j] = variance[group[i]] * (i == j ? 1 : rho);
                            }
                        }
                    }

                    break;
                }

            case 3:
                {
                    var block = new double[3, 3];
                    var factor = new double[3, 3];
                    for (var i = 0; i < 3; i++)
                    {
                        for (var j = 0; j < 3; j++)
                        {
                            factor[i, j] = Uniform(-1, 1);
                        }
                    }

                    for (var i = 0; i < 3; i++)
                    {
                        for (var j = 0; j < 3; j++)
                        {
                            var acc = i == j ? 0.01 : 0.0;
                            for (var k = 0; k < 3; k++)
                            {
                                acc += factor[i, k] * factor[j, k];
                            }

                            block[i, j] = acc;
                        }
                    }

                    for (var i = 0; i < core; i++)
                    {
                        group[i] = i / 3;
                        for (var j = 0; j < core; j++)
                        {
                            if (i / 3 == j / 3)
                            {
                                cov[i, j] = block[i % 3, j % 3];
                            }
                        }
                    }

                    break;
                }

            default:
                {
                    double[][] rows = [[rng.Next(-2, 3), rng.Next(-2, 3)], [rng.Next(-2, 3), rng.Next(-2, 3)], [2, -1]];
                    var pick = new int[core];
                    for (var i = 0; i < core; i++)
                    {
                        pick[i] = rng.Next(rows.Length);
                        group[i] = pick[i];
                    }

                    var d = rng.Next(2) == 0 ? 1.0 : Power(dec ? -6 : -10, -1);
                    for (var i = 0; i < core; i++)
                    {
                        for (var j = 0; j < core; j++)
                        {
                            cov[i, j] = (rows[pick[i]][0] * rows[pick[j]][0]) + (rows[pick[i]][1] * rows[pick[j]][1]) + (i == j ? d : 0);
                        }
                    }

                    break;
                }
        }

        var lowest = dec ? -12.0 : -149.0;
        var highest = dec ? 12.0 : 149.0;
        for (var i = 0; i < core; i++)
        {
            for (var j = 0; j < core; j++)
            {
                cov[i, j] *= scale;
            }
        }

        for (var i = core; i < n; i++)
        {
            group[i] = 1000 + i;
            cov[i, i] = Power(lowest, highest);
        }

        var floor = dec ? 1e-14 : 1e-150;
        var cs = new string[n][];
        for (var i = 0; i < n; i++)
        {
            cs[i] = new string[n];
        }

        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                var value = cov[i, j];
                cs[i][j] = cs[j][i] = value == 0 || (i != j && Math.Abs(value) < floor) ? "0" : InType(type, value);
            }
        }

        var max = dec ? 1e14 : 1e150;
        var tight = new[] { 2, 4, 8, 16 }.Where(k => k <= n).ToArray();
        var (lb, ub) = rng.Next(5) switch
        {
            0 => ("0", "1"),
            1 => ("0", Reciprocal(type, tight[rng.Next(tight.Length)])),
            2 => (Reciprocal(type, 2 * n), "1"),
            _ => DegenerateBox(type, Power(0, dec ? 14 : 89)),
        };

        var name = "dstruct" + shape + "-" + trial + "-n" + n + "-t" + tail;
        if (!meanVariance)
        {
            return Admissible(new HeldOutProblem(name, type, cs, lb, ub));
        }

        var min = dec ? 1e-14 : 1e-150;
        double Clip(double x) => Math.Abs(x) > max ? Math.Sign(x) * max : Math.Abs(x) < min ? 0.0 : x;
        var variances = Enumerable.Range(0, n).Select(i => cov[i, i]).OrderBy(v => v).ToArray();
        var lambda = Clip(rng.Next(5) switch
        {
            0 => 1.0,
            1 => 1 / variances[n / 2],
            2 => 1 / variances[n - 1],
            3 => 1 / variances[0],
            _ => Power(dec ? -6 : -100, 0),
        });
        lambda = lambda == 0 ? min : lambda;
        var regime = rng.Next(6);
        var offset = Clip(Uniform(-1, 1) * Power(0, dec ? 13 : 140));
        var groupMean = new Dictionary<int, double>();
        var means = new string[n];
        for (var i = 0; i < n; i++)
        {
            var m = regime switch
            {
                0 => offset,
                1 => groupMean.TryGetValue(group[i], out var gm) ? gm : groupMean[group[i]] = Clip(Uniform(-1, 1) * Power(lowest / 2, highest / 2)),
                2 => offset == 0 ? 0 : dec ? offset * (1 + (rng.Next(-2, 3) * 1e-15)) : BitShift(offset, rng.Next(-2, 3)),
                3 => Clip(Uniform(-1, 1) * Power(lowest / 2, highest / 2)),
                4 => Clip(lambda * cov[i, i] * Uniform(-1, 1)),
                _ => 0.0,
            };
            means[i] = InType(type, Clip(m));
        }

        return Admissible(new HeldOutProblem(name + "-m" + regime, type, cs, lb, ub, means, InType(type, lambda)));
    }

    // A double moved by the given number of units in its last place.
    private static double BitShift(double x, int units)
    {
        for (var k = 0; k < Math.Abs(units); k++)
        {
            x = units > 0 ? Math.BitIncrement(x) : Math.BitDecrement(x);
        }

        return x;
    }

    private static (string Lb, string Ub) DegenerateBox(string type, double limit) =>
        (InType(type, -limit), InType(type, limit));

    // Block-diagonal covariance: the 2×2 block copied, then a diagonal tail.
    private static string[][] RepeatedBlocks(string a, string c, string b, int copies, string[] tail)
    {
        var n = (2 * copies) + tail.Length;
        var cov = new string[n][];
        for (var i = 0; i < n; i++)
        {
            cov[i] = Enumerable.Repeat("0", n).ToArray();
        }

        for (var k = 0; k < copies; k++)
        {
            cov[2 * k][2 * k] = a;
            cov[2 * k][(2 * k) + 1] = cov[(2 * k) + 1][2 * k] = c;
            cov[(2 * k) + 1][(2 * k) + 1] = b;
        }

        for (var t = 0; t < tail.Length; t++)
        {
            cov[(2 * copies) + t][(2 * copies) + t] = tail[t];
        }

        return cov;
    }

    private static HeldOutProblem DegenerateStructureWitness(string name) => name switch
    {
        "1_-/XYX-1Z.YY.b#3" => new HeldOutProblem(
            name,
            "double",
            RepeatedBlocks("8.405462403308149E+126", "1.092705323317454E+127", "1.4205231461590772E+127", 7, ["1.7090488521246833E-19", "5.060333107232601E-105"]),
            "-1.04E+89",
            "1.04E+89",
            ["-9.407660280636708E+131", "-9.40766028063671E+131", "-9.407660280636712E+131", "-9.40766028063671E+131", "-9.40766028063671E+131", "-9.40766028063671E+131", "-9.40766028063671E+131", "-9.407660280636708E+131", "-9.40766028063671E+131", "-9.407660280636708E+131", "-9.40766028063671E+131", "-9.40766028063671E+131", "-9.40766028063671E+131", "-9.407660280636708E+131", "-9.40766028063671E+131", "-9.407660280636708E+131"],
            "1"),
        "y1ub-35#0" => new HeldOutProblem(
            name,
            "double",
            RepeatedBlocks("1.3691304236447578E+34", "9.507168626193864E+33", "6.708739075859313E+33", 3, ["3.299886348196973E-38", "1.9546594293509715E-35"]),
            "-1.14E+49",
            "1.14E+49",
            ["1.0663526914195195E-62", "-1.0357403333386212E-62", "1.3800073291452487E-62", "-1.534276468125472E-62", "-1.2282938940546585E-62", "-1.3482839022675458E-62", "-1.1714902969497267E-62", "1.6565699156642848E-62"],
            "1.0795825705352401E-95"),
        "y1uh-38#0" => PermutedY1uh38(name, Enumerable.Range(0, 16).ToArray(), "1"),
        "y1uh-38#0/perm20" => PermutedY1uh38(name, [4, 6, 5, 2, 8, 9, 14, 0, 13, 3, 7, 12, 11, 10, 1, 15], "0.25"),
        "y1uh-38#0/perm29" => PermutedY1uh38(name, [15, 12, 10, 0, 2, 5, 14, 13, 4, 9, 3, 1, 8, 7, 6, 11], "0.25"),
        "y1uh-38#0/perm30" => PermutedY1uh38(name, [10, 14, 3, 6, 2, 0, 15, 13, 9, 1, 12, 5, 7, 8, 11, 4], "0.25"),
        "y1uh-38#0/perm46" => PermutedY1uh38(name, [5, 0, 10, 6, 4, 1, 12, 8, 14, 9, 3, 11, 15, 7, 2, 13], "0.25"),
        "y1uh-38#0/perm54" => PermutedY1uh38(name, [1, 10, 12, 11, 13, 14, 3, 5, 15, 0, 7, 6, 4, 8, 9, 2], "0.25"),
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unknown witness"),
    };

    // Σ'ᵢⱼ = Σ_{p(i), p(j)} of y1uh-38#0, with the box [0, ub].
    private static HeldOutProblem PermutedY1uh38(string name, int[] p, string ub)
    {
        var n = s_y1uh38.Length;
        var cov = new string[n][];
        for (var i = 0; i < n; i++)
        {
            cov[i] = new string[n];
            for (var j = 0; j < n; j++)
            {
                cov[i][j] = s_y1uh38[p[i]][p[j]];
            }
        }

        return new HeldOutProblem(name, "double", cov, "0", ub);
    }

    // The covariance of the held-out min-variance problem y1uh-38#0 (n = 16, scales from 1e-131 to 1e143).
    private static readonly string[][] s_y1uh38 =
    [
        ["4.770533902190355", "-5.041231355709151E+29", "-1.2738322299806545E-30", "9.839382993390982E+45", "4.840112092202438E-07", "-1.8074408788139134E+49", "-4.1366683530942995E+37", "3.9475099406556034E-57", "-6.790408626989282E-19", "2.2841685556155067E+21", "-9.096492911635907E+69", "-2.3016914024914185E+28", "-2.2502504324272428E-66", "1.3100957019106279E+54", "-2.170611086808623E+29", "93165073578.55447"],
        ["-5.041231355709151E+29", "6.937991884734619E+59", "-0.5265713444397822", "6.253865641549998E+75", "-5.125402342232521E+22", "-4.874906544187978E+79", "-1.6113593929988154E+67", "-1.4024495660336595E-27", "1056723539459.173", "2.3931153609504546E+55", "2.5063363585902382E+100", "3.260953650726983E+57", "-3.242152902622041E-37", "3.1288476985282436E+83", "7.338435596376707E+58", "-1.0409918030090854E+41"],
        ["-1.2738322299806545E-30", "-0.5265713444397822", "5.207875607392619E-59", "25415958799133964", "-3.888639909298052E-37", "1.2455308393597695E+20", "61214869.65141995", "-1.3994257632474027E-86", "-1.257187192854409E-47", "-0.0001667583037612548", "1.0567232053721805E+42", "-0.02691047300706029", "-3.5117911323072168E-96", "-4.2932877618325014E+24", "2.072304767475257", "-8.589813425907621E-19"],
        ["9.839382993390982E+45", "6.253865641549998E+75", "25415958799133964", "1.0804137058262993E+93", "9.561211025928321E+39", "-2.3125407170176183E+96", "-1.1805167667630812E+84", "3.358350462329479E-11", "3.5103392548867396E+28", "1.261606764861584E+71", "6.567778785265692E+117", "-3.6680158127601685E+74", "-2.888328010444464E-20", "2.1717161951866567E+100", "5.9510941052927E+75", "-6.957782502704977E+57"],
        ["4.840112092202438E-07", "-5.125402342232521E+22", "-3.888639909298052E-37", "9.561211025928321E+39", "4.684023243532486E-12", "-2.981042959856482E+43", "-6.754758979431537E+31", "3.4434140648839896E-63", "2.960953636418012E-24", "4.5349995163951735E+19", "1.3533768164256723E+65", "6.937904802083749E+20", "-1.747747045873343E-72", "-4.1532892659046455E+47", "6.806537608549651E+22", "-132948.03325284345"],
        ["-1.8074408788139134E+49", "-4.874906544187978E+79", "1.2455308393597695E+20", "-2.3125407170176183E+96", "-2.981042959856482E+43", "2.17779237274091E+100", "-7.533340324748959E+87", "1.3410735385938963E-07", "7.465479926455638E+31", "-6.066423636034023E+74", "-1.7136740500410145E+121", "4.644524395112374E+77", "2.392695651107916E-17", "-2.1781115640205553E+104", "2.377927350706682E+78", "2.6535191707501266E+61"],
        ["-4.1366683530942995E+37", "-1.6113593929988154E+67", "61214869.65141995", "-1.1805167667630812E+84", "-6.754758979431537E+31", "-7.533340324748959E+87", "7.554989965629451E+76", "-6.393567728178172E-19", "-1.4883592922440578E+20", "-7.045298976505024E+63", "-2.259531130065434E+109", "3.452758583883944E+66", "1.4521439326020335E-28", "2.911690606970725E+92", "6.018902608050595E+65", "-6.359852974588742E+48"],
        ["3.9475099406556034E-57", "-1.4024495660336595E-27", "-1.3994257632474027E-86", "3.358350462329479E-11", "3.4434140648839896E-63", "1.3410735385938963E-07", "-6.393567728178172E-19", "4.992299256300453E-113", "1.934224653230028E-74", "4.942147355859225E-32", "919118178397533.4", "-2.292385049237701E-29", "2.4374434697108722E-123", "-0.005734692561588731", "4.704715381230376E-28", "4.6116187371765735E-46"],
        ["-6.790408626989282E-19", "1056723539459.173", "-1.257187192854409E-47", "3.5103392548867396E+28", "2.960953636418012E-24", "7.465479926455638E+31", "-1.4883592922440578E+20", "1.934224653230028E-74", "6.234744163378674E-35", "292809374.1425066", "4.786397626648648E+53", "33830164095.545525", "4.214318473075904E-85", "-1.0926681476872275E+37", "157515800419.15192", "1.3221665831797953E-07"],
        ["2.2841685556155067E+21", "2.3931153609504546E+55", "-0.0001667583037612548", "1.261606764861584E+71", "4.5349995163951735E+19", "-6.066423636034023E+74", "-7.045298976505024E+63", "4.942147355859225E-32", "292809374.1425066", "1.0705268058752804E+52", "-4.648051482971058E+95", "1.8876723419380888E+54", "-1.3116258838930016E-40", "-6.960244905667863E+79", "-3.393293152549129E+55", "-1.3739078401328478E+37"],
        ["-9.096492911635907E+69", "2.5063363585902382E+100", "1.0567232053721805E+42", "6.567778785265692E+117", "1.3533768164256723E+65", "-1.7136740500410145E+121", "-2.259531130065434E+109", "919118178397533.4", "4.786397626648648E+53", "-4.648051482971058E+95", "3.2201464766191883E+143", "-4.7735640346524395E+99", "-54921.68290389942", "-1.0729789213990727E+125", "1.949203357848649E+101", "-6.908836513379307E+82"],
        ["-2.3016914024914185E+28", "3.260953650726983E+57", "-0.02691047300706029", "-3.6680158127601685E+74", "6.937904802083749E+20", "4.644524395112374E+77", "3.452758583883944E+66", "-2.292385049237701E-29", "33830164095.545525", "1.8876723419380888E+54", "-4.7735640346524395E+99", "2.129005751935083E+57", "4.5823752399086567E-39", "-1.263183236389777E+82", "-3.368161406735174E+57", "-8.440067603685988E+39"],
        ["-2.2502504324272428E-66", "-3.242152902622041E-37", "-3.5117911323072168E-96", "-2.888328010444464E-20", "-1.747747045873343E-72", "2.392695651107916E-17", "1.4521439326020335E-28", "2.4374434697108722E-123", "4.214318473075904E-85", "-1.3116258838930016E-40", "-54921.68290389942", "4.5823752399086567E-39", "1.2879237332078968E-131", "-2.2985798862935555E-12", "2.4380616584069948E-37", "2.0327519302151143E-55"],
        ["1.3100957019106279E+54", "3.1288476985282436E+83", "-4.2932877618325014E+24", "2.1717161951866567E+100", "-4.1532892659046455E+47", "-2.1781115640205553E+104", "2.911690606970725E+92", "-0.005734692561588731", "-1.0926681476872275E+37", "-6.960244905667863E+79", "-1.0729789213990727E+125", "-1.263183236389777E+82", "-2.2985798862935555E-12", "9.071417324911721E+108", "-1.2289522312784899E+83", "-1.4093202483321002E+65"],
        ["-2.170611086808623E+29", "7.338435596376707E+58", "2.072304767475257", "5.9510941052927E+75", "6.806537608549651E+22", "2.377927350706682E+78", "6.018902608050595E+65", "4.704715381230376E-28", "157515800419.15192", "-3.393293152549129E+55", "1.949203357848649E+101", "-3.368161406735174E+57", "2.4380616584069948E-37", "-1.2289522312784899E+83", "5.590182721972449E+59", "-7.444740343180673E+40"],
        ["93165073578.55447", "-1.0409918030090854E+41", "-8.589813425907621E-19", "-6.957782502704977E+57", "-132948.03325284345", "2.6535191707501266E+61", "-6.359852974588742E+48", "4.6116187371765735E-46", "1.3221665831797953E-07", "-1.3739078401328478E+37", "-6.908836513379307E+82", "-8.440067603685988E+39", "2.0327519302151143E-55", "-1.4093202483321002E+65", "-7.444740343180673E+40", "2.188455336553976E+23"],
    ];

    private static HeldOutProblem? ExtremeScaleProblem(System.Random rng, string type, int trial, bool meanVariance)
    {
        var dec = type == "decimal";
        double Uniform(double lo, double hi) => lo + ((hi - lo) * rng.NextDouble());
        var n = rng.Next(2, 11);
        double lo = dec ? -12 : -149, hi = dec ? 12 : 149;
        var mode = rng.Next(3);
        if (mode == 1)
        {
            var mid = Uniform(lo, hi);
            (lo, hi) = (Math.Max(lo, mid - 7), Math.Min(hi, mid + 7));
        }
        else if (mode == 2)
        {
            var a = Uniform(lo, hi);
            var b = Uniform(lo, hi);
            (lo, hi) = (Math.Min(a, b), Math.Max(a, b));
        }

        var exponent = new double[n];
        for (var i = 0; i < n; i++)
        {
            exponent[i] = Uniform(lo, hi);
        }

        var diagonal = rng.Next(3) == 0;
        var factor = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                factor[i, j] = Uniform(-1, 1);
            }
        }

        var correlation = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var acc = 0.0;
                for (var k = 0; k < n; k++)
                {
                    acc += factor[i, k] * factor[j, k];
                }

                correlation[i, j] = (acc / n) + (i == j ? 0.05 : 0);
            }
        }

        // Domain floor of a nonzero entry: decimal keeps every entry at 1e-14 or above (an off-diagonal entry below it is
        // dropped to zero); double entries are formed as ρ·10^((eᵢ + eⱼ)/2), at least about 1e-150.
        var cov = new string[n][];
        for (var i = 0; i < n; i++)
        {
            cov[i] = new string[n];
        }

        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                var rho = i == j ? 1.0 : diagonal ? 0.0 : correlation[i, j] / Math.Sqrt(correlation[i, i] * correlation[j, j]);
                var value = rho * Math.Pow(10, (exponent[i] + exponent[j]) / 2);
                if (i != j && (rho == 0 || Math.Abs(value) < (dec ? 1e-14 : 1e-150)))
                {
                    cov[i][j] = cov[j][i] = "0";
                    continue;
                }

                cov[i][j] = cov[j][i] = InType(type, value);
            }
        }

        var max = dec ? 1e14 : 1e150;
        var (lb, ub) = rng.Next(2) == 0 ? ("0", "1") : (InType(type, -max), InType(type, max));
        if (!meanVariance)
        {
            return Admissible(new HeldOutProblem("xs-mv0-" + trial + "-n" + n, type, cov, lb, ub));
        }

        var variance = exponent.Select(e => Math.Pow(10, e)).OrderBy(v => v).ToArray();
        var min = dec ? 1e-14 : 1e-150;
        double Clip(double x) => Math.Abs(x) > max ? Math.Sign(x) * max : Math.Abs(x) < min ? 0.0 : x;
        var lambda = Clip(rng.Next(5) switch
        {
            0 => 1 / variance[n / 2],
            1 => 1 / variance[n - 1],
            2 => 1 / variance[0],
            3 => max,
            _ => 1.0,
        });
        lambda = lambda == 0 ? min : lambda;
        var means = new string[n];
        var regime = rng.Next(4);
        for (var i = 0; i < n; i++)
        {
            var s = Math.Pow(10, exponent[i]);
            var m = regime switch
            {
                0 => lambda * s * Uniform(-1, 1) * Math.Pow(10, Uniform(-3, 1)),
                1 => Uniform(-1, 1) * Math.Pow(10, Uniform(dec ? -12 : -149, dec ? 12 : 149)),
                2 => 0.0,
                _ => Math.Sqrt(s) * Uniform(-1, 1),
            };
            means[i] = InType(type, Clip(m));
        }

        return Admissible(new HeldOutProblem("xs-mv-" + trial + "-n" + n, type, cov, lb, ub, means, InType(type, lambda)));
    }

    private static HeldOutProblem ExtremeScaleWitness(string name) => name switch
    {
        "cZZ!Y1. #3" => new HeldOutProblem(
            name,
            "double",
            [
                ["8.427017438445987E-97", "-9.363352709384088E-98", "-9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "-9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "-9.363352709384088E-98", "0"],
                ["-9.363352709384088E-98", "8.427017438445987E-97", "-9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "-9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "-9.363352709384088E-98", "0"],
                ["-9.363352709384088E-98", "-9.363352709384088E-98", "8.427017438445987E-97", "9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "-9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "-9.363352709384088E-98", "0"],
                ["9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "8.427017438445987E-97", "-9.363352709384088E-98", "-9.363352709384088E-98", "9.363352709384088E-98", "-9.363352709384088E-98", "-9.363352709384088E-98", "9.363352709384088E-98", "0"],
                ["9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "-9.363352709384088E-98", "8.427017438445987E-97", "-9.363352709384088E-98", "9.363352709384088E-98", "-9.363352709384088E-98", "-9.363352709384088E-98", "9.363352709384088E-98", "0"],
                ["9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "-9.363352709384088E-98", "-9.363352709384088E-98", "8.427017438445987E-97", "9.363352709384088E-98", "-9.363352709384088E-98", "-9.363352709384088E-98", "9.363352709384088E-98", "0"],
                ["-9.363352709384088E-98", "-9.363352709384088E-98", "-9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "8.427017438445987E-97", "9.363352709384088E-98", "9.363352709384088E-98", "-9.363352709384088E-98", "0"],
                ["9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "-9.363352709384088E-98", "-9.363352709384088E-98", "-9.363352709384088E-98", "9.363352709384088E-98", "8.427017438445987E-97", "-9.363352709384088E-98", "9.363352709384088E-98", "0"],
                ["9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "-9.363352709384088E-98", "-9.363352709384088E-98", "-9.363352709384088E-98", "9.363352709384088E-98", "-9.363352709384088E-98", "8.427017438445987E-97", "9.363352709384088E-98", "0"],
                ["-9.363352709384088E-98", "-9.363352709384088E-98", "-9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "-9.363352709384088E-98", "9.363352709384088E-98", "9.363352709384088E-98", "8.427017438445987E-97", "0"],
                ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1.5242912843339806E+141"],
            ],
            "-3.977E+48",
            "3.977E+48",
            ["-2.8699205863697714E-28", "-2.871317733728218E-28", "-2.8706863521102752E-28", "2.870096206757598E-28", "2.867192913321508E-28", "2.8670833954845745E-28", "-2.867907912566052E-28", "2.8690817611342123E-28", "2.867967751260693E-28", "-2.8686191305677176E-28", "6.602076122226958E-29"],
            "1.08946E+33"),
        "x9b-167#3" => new HeldOutProblem(
            name,
            "double",
            [
                ["3.119013907084961E-139", "2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "0"],
                ["2.399241466985832E-140", "3.119013907084961E-139", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "0"],
                ["-2.399241466985832E-140", "2.399241466985832E-140", "3.119013907084961E-139", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "0"],
                ["-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "3.119013907084961E-139", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "0"],
                ["2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "3.119013907084961E-139", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "0"],
                ["-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "3.119013907084961E-139", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "0"],
                ["2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "3.119013907084961E-139", "2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "0"],
                ["-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "3.119013907084961E-139", "2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "0"],
                ["2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "3.119013907084961E-139", "-2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "0"],
                ["2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "3.119013907084961E-139", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "0"],
                ["2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "3.119013907084961E-139", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "0"],
                ["-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "3.119013907084961E-139", "-2.399241466985832E-140", "2.399241466985832E-140", "0"],
                ["-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "3.119013907084961E-139", "2.399241466985832E-140", "0"],
                ["2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "-2.399241466985832E-140", "2.399241466985832E-140", "2.399241466985832E-140", "3.119013907084961E-139", "0"],
                ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "4.3556142965880123E+40"],
            ],
            "-5.225E+43",
            "5.225E+43",
            ["-8.018161913559204E-24", "8.010612460830284E-24", "-8.020358792551635E-24", "-8.021448403147072E-24", "8.021851625482571E-24", "-8.012780409629851E-24", "8.016767577797281E-24", "-8.018940271394958E-24", "8.010627421059252E-24", "8.010935014044681E-24", "8.008390013014176E-24", "-8.013822862976607E-24", "-8.019825041161152E-24", "8.021194392206369E-24", "1.05187734810572E-28"],
            "3.18365E+83"),
        "x9c-98#2" => new HeldOutProblem(
            name,
            "double",
            [
                ["5461.333333333421", "2730.6666666665783", "-2730.6666666665783", "0", "0", "0", "0"],
                ["2730.6666666665783", "5461.333333333421", "2730.6666666665783", "0", "0", "0", "0"],
                ["-2730.6666666665783", "2730.6666666665783", "5461.333333333421", "0", "0", "0", "0"],
                ["0", "0", "0", "6.681911775230489E-52", "0", "0", "0"],
                ["0", "0", "0", "0", "1.3538426240824291E+126", "0", "0"],
                ["0", "0", "0", "0", "0", "2.210859150104178E-75", "0"],
                ["0", "0", "0", "0", "0", "0", "6.717876107567089E-139"],
            ],
            "-3.999E+51",
            "3.999E+51",
            ["-5.091764888685851E-52", "5.083947353504893E-52", "-5.087837010348514E-52", "-5.6757859321440736E-58", "4.564613099583501E-57", "1.5372718766870204E-52", "2.851722612203483E-57"],
            "2.5997789389151146E-94"),
        "x9d-47#0" => new HeldOutProblem(
            name,
            "double",
            [
                ["3.379879973204907E-112", "-4.1182068023845603E-41", "-24006711.5422728", "-1.4232376220583187E-84", "-4.738705704834081E-130", "1.0062406978496091E-70", "-7.171298929716162E-70", "-1.469369611894865E-72", "1.2849274195051125E-15"],
                ["-4.1182068023845603E-41", "4.601353927439351E+32", "-2.0667064076722487E+79", "1.6708085276253878E-12", "-5.155666834333769E-58", "-156.56963007888672", "-139.3529569317964", "-0.26533993287768254", "-6.843049024945188E+57"],
                ["-24006711.5422728", "-2.0667064076722487E+79", "1.5512004406388467E+128", "3.56623654867565E+36", "3.283926783445762E-10", "2.6076848141329605E+49", "1.2314277031162925E+50", "1.1031578354223622E+47", "-1.55988204727678E+106"],
                ["-1.4232376220583187E-84", "1.6708085276253878E-12", "3.56623654867565E+36", "1.3525104515862278E-55", "9.798175202897242E-102", "-2.019908040149217E-43", "3.287127096254474E-42", "3.1060226734635995E-45", "-485260464832357.75"],
                ["-4.738705704834081E-130", "-5.155666834333769E-58", "3.283926783445762E-10", "9.798175202897242E-102", "3.716035151504718E-146", "4.423762301437155E-88", "-3.348631469632286E-87", "-2.175782931358552E-90", "8.901570420252138E-32"],
                ["1.0062406978496091E-70", "-156.56963007888672", "2.6076848141329605E+49", "-2.019908040149217E-43", "4.423762301437155E-88", "3.069547801741615E-28", "-5.955943412251145E-28", "-3.7287926359549546E-31", "-8.238919595981418E+27"],
                ["-7.171298929716162E-70", "-139.3529569317964", "1.2314277031162925E+50", "3.287127096254474E-42", "-3.348631469632286E-87", "-5.955943412251145E-28", "5.561404385013871E-27", "2.704341062263219E-30", "-5.800710385183116E+27"],
                ["-1.469369611894865E-72", "-0.26533993287768254", "1.1031578354223622E+47", "3.1060226734635995E-45", "-2.175782931358552E-90", "-3.7287926359549546E-31", "2.704341062263219E-30", "1.8114735087825274E-32", "1.8016052931613283E+24"],
                ["1.2849274195051125E-15", "-6.843049024945188E+57", "-1.55988204727678E+106", "-485260464832357.75", "8.901570420252138E-32", "-8.238919595981418E+27", "-5.800710385183116E+27", "1.8016052931613283E+24", "4.727565270341332E+84"],
            ],
            "-1E+150",
            "1E+150",
            ["-2667412.3887141496", "-2.296340448775168E+78", "1.7235560420064004E+127", "3.96248504684704E+35", "3.648807530497837E-11", "2.8974275659670323E+48", "1.3682530009635931E+49", "1.225730926008415E+46", "-1.7332022715864865E+105"],
            "1"),
        "x9d-241#1" => new HeldOutProblem(
            name,
            "double",
            [
                ["0.0003910083171195083", "3.812279460713506E+33", "-4.996065917204392E-56", "3.0026378649828806E+60", "6.537680206119318E+49", "4.042095714395396E-21"],
                ["3.812279460713506E+33", "4.964650453887578E+72", "4.93375473354423E-17", "-1.3230690686087607E+99", "1.5942063347349796E+88", "1.838349339266803E+18"],
                ["-4.996065917204392E-56", "4.93375473354423E-17", "6.419903634962838E-105", "18043693747.713837", "-0.8093869451600543", "5.708507636329627E-71"],
                ["3.0026378649828806E+60", "-1.3230690686087607E+99", "18043693747.713837", "4.765787553472205E+126", "-8.415002158808506E+114", "-2.33038114009078E+44"],
                ["6.537680206119318E+49", "1.5942063347349796E+88", "-0.8093869451600543", "-8.415002158808506E+114", "4.006095773463632E+104", "2.631535364276484E+33"],
                ["4.042095714395396E-21", "1.838349339266803E+18", "5.708507636329627E-71", "-2.33038114009078E+44", "2.631535364276484E+33", "1.4165972992157294E-36"],
            ],
            "0",
            "1",
            ["0", "0", "0", "0", "0", "0"],
            "1.5576557793702592E+104"),
        "x9e-98#1" => new HeldOutProblem(
            name,
            "double",
            [
                ["82.99762546029987", "-1.068657843333785E-30", "-1.652976978440034E-14", "-2.0293542689490905E-11", "4357566.927593941", "-1.1652913908701725E-05", "44300.62780087569", "1.4605679359638708E+20", "4.7184265881674514E+20", "-0.5153270918357425", "4141.170839912216", "1.614378700441206E+23", "-2.4797149968954813E-19", "2.8683941661950766E+23", "-1.4013761298192477E-27"],
                ["-1.068657843333785E-30", "1.1829759271955977E-61", "5.958676039048226E-46", "2.149017120756606E-43", "-2.0865112158502645E-25", "-6.62284780946665E-37", "-4.7680272622851694E-27", "-1.3788227583964009E-13", "-3.628714972878892E-11", "-6.515361560535679E-33", "-2.274561105089171E-27", "-3.4261156917884557E-09", "5.257018107678991E-51", "1.2800554906928846E-08", "3.9861599107500835E-59"],
                ["-1.652976978440034E-14", "5.958676039048226E-46", "6.960760213399388E-28", "-1.692854713046707E-26", "8.395538058156853E-09", "-1.2096394989982774E-19", "2.0072483194290952E-10", "-245660.7441878664", "-1429247.3037110693", "1.0552415302959636E-14", "-1.3059705828142762E-10", "15020617.750406917", "1.8458882473275835E-33", "-4950745047.000793", "-5.940161153959731E-42"],
                ["-2.0293542689490905E-11", "2.149017120756606E-43", "-1.692854713046707E-26", "2.757720490051422E-23", "-3.874260119674307E-06", "6.012245474204757E-18", "-7.591519756710074E-08", "10260126.987351105", "-195337835.22126332", "-3.682316842279336E-14", "-9.729746624855168E-09", "90561349112.95285", "-6.884660042879548E-31", "891439220701.567", "2.211920299319552E-39"],
                ["4357566.927593941", "-2.0865112158502645E-25", "8.395538058156853E-09", "-3.874260119674307E-06", "17294216313067.357", "8.026104094803797", "28319691730.47148", "5.703767326528559E+24", "-4.387865093111253E+26", "358843.9268709661", "-23293124335.546993", "2.413712664286069E+28", "8.934417393577936E-14", "-6.550868288212389E+29", "6.68670812598021E-22"],
                ["-1.1652913908701725E-05", "-6.62284780946665E-37", "-1.2096394989982774E-19", "6.012245474204757E-18", "8.026104094803797", "8.884969307044192E-11", "0.015150569205727119", "84480768738603.17", "430002656889932.2", "-1.9359294698021597E-06", "0.03752322706056598", "-67149142242022680", "4.420676417235908E-25", "1.9844493558161702E+18", "2.4429544702382514E-33"],
                ["44300.62780087569", "-4.7680272622851694E-27", "2.0072483194290952E-10", "-7.591519756710074E-08", "28319691730.47148", "0.015150569205727119", "1035175474.0473787", "-3.439521843223466E+23", "-2.6252071408111347E+24", "11490.518218499645", "16897603.285410598", "1.1749751632067476E+26", "3.2424176641951374E-15", "-4.115485252186841E+27", "-4.590905947529941E-24"],
                ["1.4605679359638708E+20", "-1.3788227583964009E-13", "-245660.7441878664", "10260126.987351105", "5.703767326528559E+24", "84480768738603.17", "-3.439521843223466E+23", "2.3916355567523627E+39", "4.5363070493130183E+39", "-1.1546721437154351E+19", "-6.911787696773188E+22", "2.593290854177928E+41", "4.305087183769636", "1.3124299135913997E+43", "9.894997525056228E-09"],
                ["4.7184265881674514E+20", "-3.628714972878892E-11", "-1429247.3037110693", "-195337835.22126332", "-4.387865093111253E+26", "430002656889932.2", "-2.6252071408111347E+24", "4.5363070493130183E+39", "3.3933570279589644E+41", "-3.126643455649302E+20", "3.379741272759348E+24", "-1.287704882001595E+43", "-12.980827226286733", "-1.1420716345102396E+44", "5.388628000612949E-08"],
                ["-0.5153270918357425", "-6.515361560535679E-33", "1.0552415302959636E-14", "-3.682316842279336E-14", "358843.9268709661", "-1.9359294698021597E-06", "11490.518218499645", "-1.1546721437154351E+19", "-3.126643455649302E+20", "1.0038100047792136", "-7701.375187388538", "7.404946528560413E+21", "3.0647962130089696E-20", "-2.7009707680378456E+22", "-3.8911965935559974E-29"],
                ["4141.170839912216", "-2.274561105089171E-27", "-1.3059705828142762E-10", "-9.729746624855168E-09", "-23293124335.546993", "0.03752322706056598", "16897603.285410598", "-6.911787696773188E+22", "3.379741272759348E+24", "-7701.375187388538", "550498066.8222128", "-1.592709867920148E+26", "-2.197549145023388E-16", "-2.3470731588736438E+27", "7.217424613559614E-25"],
                ["1.614378700441206E+23", "-3.4261156917884557E-09", "15020617.750406917", "90561349112.95285", "2.413712664286069E+28", "-67149142242022680", "1.1749751632067476E+26", "2.593290854177928E+41", "-1.287704882001595E+43", "7.404946528560413E+21", "-1.592709867920148E+26", "4.178835970733331E+45", "-3136.958211398482", "1.6861511448207936E+46", "1.3665934508978114E-05"],
                ["-2.4797149968954813E-19", "5.257018107678991E-51", "1.8458882473275835E-33", "-6.884660042879548E-31", "8.934417393577936E-14", "4.420676417235908E-25", "3.2424176641951374E-15", "4.305087183769636", "-12.980827226286733", "3.0647962130089696E-20", "-2.197549145023388E-16", "-3136.958211398482", "1.0014605355645429E-37", "49162.72299871928", "-4.468883211178546E-47"],
                ["2.8683941661950766E+23", "1.2800554906928846E-08", "-4950745047.000793", "891439220701.567", "-6.550868288212389E+29", "1.9844493558161702E+18", "-4.115485252186841E+27", "1.3124299135913997E+43", "-1.1420716345102396E+44", "-2.7009707680378456E+22", "-2.3470731588736438E+27", "1.6861511448207936E+46", "49162.72299871928", "6.862083844534853E+47", "0.00011681983801640193"],
                ["-1.4013761298192477E-27", "3.9861599107500835E-59", "-5.940161153959731E-42", "2.211920299319552E-39", "6.68670812598021E-22", "2.4429544702382514E-33", "-4.590905947529941E-24", "9.894997525056228E-09", "5.388628000612949E-08", "-3.8911965935559974E-29", "7.217424613559614E-25", "1.3665934508978114E-05", "-4.468883211178546E-47", "0.00011681983801640193", "8.62769508275154E-55"],
            ],
            "0",
            "1",
            ["-7.337467338581321E+149", "-7.337467338581321E+149", "-7.337467338581321E+149", "-7.337467338581321E+149", "-7.337467338581321E+149", "-7.337467338581321E+149", "-7.337467338581321E+149", "-7.337467338581321E+149", "-7.336579713470317E+149", "-7.337467338581321E+149", "-7.337467338581321E+149", "-7.337467338581321E+149", "-7.337467338581321E+149", "-7.337467338581321E+149", "-7.337467338581321E+149"],
            "1E+150"),
        "x9e-165#1" => new HeldOutProblem(
            name,
            "double",
            [
                ["13377036531522942", "-2.1700968331940002E-38", "-8.000548534058082E-35", "-3.1353348922088224E-35", "-2.2754803599096858E-59", "7.018145940481625E+75", "3.6142230660170196E-59", "-5.299320445609314E+47", "4.7503339314279E-40", "22434079300.031303", "4.146518021677308E+20", "2.4648191088124005E+73", "6.575067549281725E-12", "2.0797826417681978E+77", "1.0854905508930806E-59"],
                ["-2.1700968331940002E-38", "8.92204369754438E-91", "7.90302270661653E-88", "-3.8056819862921794E-88", "-4.65367466275051E-113", "-1.8194650968076112E+23", "-2.3093229977610484E-112", "9.605801456313152E-06", "-4.968737731350637E-93", "-2.648975619018938E-43", "-3.345250286331724E-33", "1.900036187909334E+22", "1.1356959282280605E-64", "3.4331276083481433E+24", "1.8840443879810268E-112"],
                ["-8.000548534058082E-35", "7.90302270661653E-88", "4.20798437016866E-84", "-1.3865394359185177E-85", "-4.2301889893112585E-111", "-2.2938966633117834E+26", "-5.337446680930918E-109", "-0.016766165982061085", "-4.084454026976518E-89", "4.118533068647736E-40", "-6.534992490128808E-31", "-1.3906472497906528E+25", "-3.143100321482718E-61", "2.2327797516420555E+26", "1.9137346482125374E-109"],
                ["-3.1353348922088224E-35", "-3.8056819862921794E-88", "-1.3865394359185177E-85", "2.1692738676420274E-84", "6.784906012901332E-110", "8.6833715592649E+25", "1.9925140755422624E-109", "0.009571955162707615", "-1.549139199034335E-90", "-2.4305550310804045E-41", "6.1539702812666085E-30", "-1.497669012639126E+25", "-1.3648855591171077E-61", "-1.7896745328674417E+27", "-4.342973017004415E-109"],
                ["-2.2754803599096858E-59", "-4.65367466275051E-113", "-4.2301889893112585E-111", "6.784906012901332E-110", "5.4519699476612384E-133", "-12.026177992617335", "1.988793076818706E-133", "-3.556228597881757E-27", "5.714808685321551E-114", "3.193442851838348E-64", "4.0840182734563927E-54", "15.33482437929442", "-2.149318508762892E-85", "833.6713050237906", "-8.164166573130973E-134"],
                ["7.018145940481625E+75", "-1.8194650968076112E+23", "-2.2938966633117834E+26", "8.6833715592649E+25", "-12.026177992617335", "2.946521353870133E+137", "215.33597353678792", "1.1770625703614122E+108", "-7.680381008539268E+21", "-1.6011685454695237E+70", "1.235983252148001E+80", "-4.4053529007630316E+135", "1.3480605306167427E+49", "-2.2441259050463255E+138", "-23.45299036738476"],
                ["3.6142230660170196E-59", "-2.3093229977610484E-112", "-5.337446680930918E-109", "1.9925140755422624E-109", "1.988793076818706E-133", "215.33597353678792", "9.605325943779039E-133", "-1.2425111309639151E-26", "6.781316146950984E-114", "4.686505645566425E-64", "4.7193797284515E-54", "-18.0686521329426", "-1.2575973408981029E-85", "-1691.0796542985663", "-4.169026516160958E-135"],
                ["-5.299320445609314E+47", "9.605801456313152E-06", "-0.016766165982061085", "0.009571955162707615", "-3.556228597881757E-27", "1.1770625703614122E+108", "-1.2425111309639151E-26", "2.589353534338548E+81", "-3.0532213502995797E-07", "-1.7912750505739658E+43", "-2.9962834857184673E+53", "1.385850315155643E+108", "9.013823253475358E+21", "1.8181263072886436E+110", "-6.4561316727203384E-27"],
                ["4.7503339314279E-40", "-4.968737731350637E-93", "-4.084454026976518E-89", "-1.549139199034335E-90", "5.714808685321551E-114", "-7.680381008539268E+21", "6.781316146950984E-114", "-3.0532213502995797E-07", "2.1490335237526574E-93", "-8.162955818054633E-45", "2.491943131171736E-34", "-9.806230736721923E+19", "2.9389002887480324E-67", "-4.487112341602996E+22", "4.7597810603105614E-114"],
                ["22434079300.031303", "-2.648975619018938E-43", "4.118533068647736E-40", "-2.4305550310804045E-41", "3.193442851838348E-64", "-1.6011685454695237E+70", "4.686505645566425E-64", "-1.7912750505739658E+43", "-8.162955818054633E-45", "1282703.5184319774", "5393706610896496", "-1.0790948871225273E+70", "-3.384193724004156E-16", "1.5320842865135592E+72", "-1.7495678550993966E-64"],
                ["4.146518021677308E+20", "-3.345250286331724E-33", "-6.534992490128808E-31", "6.1539702812666085E-30", "4.0840182734563927E-54", "1.235983252148001E+80", "4.7193797284515E-54", "-2.9962834857184673E+53", "2.491943131171736E-34", "5393706610896496", "7.154600178727943E+26", "6.90685616867224E+79", "-3.6757827637949157E-06", "-2.586137740127943E+82", "1.1203353572320444E-54"],
                ["2.4648191088124005E+73", "1.900036187909334E+22", "-1.3906472497906528E+25", "-1.497669012639126E+25", "15.33482437929442", "-4.4053529007630316E+135", "-18.0686521329426", "1.385850315155643E+108", "-9.806230736721923E+19", "-1.0790948871225273E+70", "6.90685616867224E+79", "6.070103571726084E+135", "7.756359947058279E+48", "1.1244784727112044E+137", "-2.262943241049976"],
                ["6.575067549281725E-12", "1.1356959282280605E-64", "-3.143100321482718E-61", "-1.3648855591171077E-61", "-2.149318508762892E-85", "1.3480605306167427E+49", "-1.2575973408981029E-85", "9.013823253475358E+21", "2.9389002887480324E-67", "-3.384193724004156E-16", "-3.6757827637949157E-06", "7.756359947058279E+48", "2.8344733055771796E-37", "-4.888614546645816E+50", "8.292695693106313E-86"],
                ["2.0797826417681978E+77", "3.4331276083481433E+24", "2.2327797516420555E+26", "-1.7896745328674417E+27", "833.6713050237906", "-2.2441259050463255E+138", "-1691.0796542985663", "1.8181263072886436E+110", "-4.487112341602996E+22", "1.5320842865135592E+72", "-2.586137740127943E+82", "1.1244784727112044E+137", "-4.888614546645816E+50", "1.3160830300339024E+140", "-1515.7229475197823"],
                ["1.0854905508930806E-59", "1.8840443879810268E-112", "1.9137346482125374E-109", "-4.342973017004415E-109", "-8.164166573130973E-134", "-23.45299036738476", "-4.169026516160958E-135", "-6.4561316727203384E-27", "4.7597810603105614E-114", "-1.7495678550993966E-64", "1.1203353572320444E-54", "-2.262943241049976", "8.292695693106313E-86", "-1515.7229475197823", "3.2865894211338623E-133"],
            ],
            "-1E+150",
            "1E+150",
            ["-8.030203542332342E+39", "-1.2324484702635413E+40", "8.201279111645062E+39", "2.915047970393197E+39", "-1.3332633598254958E+40", "-8.158219552757653E+39", "-7.131898552404997E+39", "1.163454634107576E+40", "-1.4533699823893433E+39", "-1.0715363308622886E+40", "7.070162818623645E+39", "1.1464372919907961E+40", "-1.4542062540385786E+39", "-2.218451990658032E+39", "1.2092484481972444E+40"],
            "1.1989213397325335E-97"),
        "_/c!a/1X Xa9c9Xb#3" => new HeldOutProblem(
            name,
            "double",
            [
                ["1.0059166094699117E+18", "-9.922571646389228E+59", "-2.2957492713720692E-39", "-0.02346640421415119", "2.304161419040578E-31", "1.389519811161843E+56", "-1588985.7048384303", "3.8829577149899095E+39", "-5.933385774281838E+74"],
                ["-9.922571646389228E+59", "1.0885561492547326E+103", "-5336.36757355926", "-6.879129414569602E+40", "-320894984491.05005", "-8.806780061144466E+97", "-1.8544362005175147E+49", "-1.9340446864494142E+82", "1.0704610384926446E+118"],
                ["-2.2957492713720692E-39", "-5336.36757355926", "1.1948204795311242E-94", "3.866251259501624E-58", "-9.940090517956713E-88", "-0.02575199919291757", "-1.1030885134908497E-49", "-3.580164510132989E-17", "-2.210549737434941E+19"],
                ["-0.02346640421415119", "-6.879129414569602E+40", "3.866251259501624E-58", "1.9281046740239846E-20", "3.3693844981188186E-50", "2.7439435817946744E+37", "2.2134301381699451E-13", "-3.22979318108016E+21", "2.010273753312629E+56"],
                ["2.304161419040578E-31", "-320894984491.05005", "-9.940090517956713E-88", "3.3693844981188186E-50", "3.800353441369743E-79", "123617334.22161932", "-2.6619109286954917E-42", "-1.0826757952667548E-08", "-3.752413908383635E+26"],
                ["1.389519811161843E+56", "-8.806780061144466E+97", "-0.02575199919291757", "2.7439435817946744E+37", "123617334.22161932", "4.121881773792402E+95", "5.251736252763031E+45", "-8.151072708930602E+77", "-2.5058947789361997E+114"],
                ["-1588985.7048384303", "-1.8544362005175147E+49", "-1.1030885134908497E-49", "2.2134301381699451E-13", "-2.6619109286954917E-42", "5.251736252763031E+45", "0.0009235347505773426", "-8.215614091333828E+29", "9.107556610915673E+64"],
                ["3.8829577149899095E+39", "-1.9340446864494142E+82", "-3.580164510132989E-17", "-3.22979318108016E+21", "-1.0826757952667548E-08", "-8.151072708930602E+77", "-8.215614091333828E+29", "7.328594729583695E+63", "-3.3790322163249126E+98"],
                ["-5.933385774281838E+74", "1.0704610384926446E+118", "-2.210549737434941E+19", "2.010273753312629E+56", "-3.752413908383635E+26", "-2.5058947789361997E+114", "9.107556610915673E+64", "-3.3790322163249126E+98", "2.1024327256191594E+134"],
            ],
            "0",
            "1",
            ["-8.003539479339281E+149", "-8.003539479339281E+149", "-8.003539479339281E+149", "-8.003539479339281E+149", "-8.003539479339281E+149", "-8.003539479339281E+149", "-8.003539479339281E+149", "-8.003539479339281E+149", "-7.9883786099780345E+149"],
            "8.36945814983372E+93"),
        "9!-a -b-  a.c9b0_199_9XZ#1" => new HeldOutProblem(
            name,
            "double",
            [
                ["7.9886657531473335E-146", "3.6322947229321377E-90", "2.5338109561496776E-81", "-6.3404967369419776E-145", "-1.2726118304534704E-125", "-1.4713025373524325E-57", "4.8302427086842695E-80", "1.2222557121700536E-45", "-1.149870288073264E-13", "-1.4486203620911975E-15"],
                ["3.6322947229321377E-90", "1.4516413476046072E-33", "3.8360105526657557E-25", "4.8617343140420874E-88", "8.28785123111654E-69", "-0.19878825520694945", "6.183018885917943E-25", "108547007310.95387", "-1.701364502624926E+43", "4.095420377059647E+40"],
                ["2.5338109561496776E-81", "3.8360105526657557E-25", "1.694693765722465E-15", "7.02463680999874E-79", "-6.913132655211281E-59", "-94689364.50575869", "-2.6934941415188017E-15", "1.572577757821428E+20", "-5.71644099958811E+52", "-5.41389089487528E+50"],
                ["-6.3404967369419776E-145", "4.8617343140420874E-88", "7.02463680999874E-79", "5.624514036723019E-141", "1.5285177155185771E-121", "7.093081034814534E-55", "9.747964203702597E-79", "2.6126557004119336E-43", "-2.090146979781839E-11", "-5.850924412042926E-13"],
                ["-1.2726118304534704E-125", "8.28785123111654E-69", "-6.913132655211281E-59", "1.5285177155185771E-121", "2.963714011317465E-101", "4.312612071981287E-35", "2.6016720054784305E-58", "1.7314772697300603E-24", "1051439854.6080928", "20939710.79709554"],
                ["-1.4713025373524325E-57", "-0.19878825520694945", "-94689364.50575869", "7.093081034814534E-55", "4.312612071981287E-35", "2.5765037193543176E+32", "95161021.216634", "-1.4110301798599517E+41", "5.738955475232254E+75", "2.2158072118719965E+73"],
                ["4.8302427086842695E-80", "6.183018885917943E-25", "-2.6934941415188017E-15", "9.747964203702597E-79", "2.6016720054784305E-58", "95161021.216634", "4.916852804731617E-13", "4.721350433861285E+20", "-3.98281242268393E+53", "6.108687600855325E+51"],
                ["1.2222557121700536E-45", "108547007310.95387", "1.572577757821428E+20", "2.6126557004119336E-43", "1.7314772697300603E-24", "-1.4110301798599517E+41", "4.721350433861285E+20", "1.3416286825039427E+56", "-1.525554058213638E+88", "-1.1568879052006744E+85"],
                ["-1.149870288073264E-13", "-1.701364502624926E+43", "-5.71644099958811E+52", "-2.090146979781839E-11", "1051439854.6080928", "5.738955475232254E+75", "-3.98281242268393E+53", "-1.525554058213638E+88", "5.772120050500216E+120", "3.4514553685085106E+117"],
                ["-1.4486203620911975E-15", "4.095420377059647E+40", "-5.41389089487528E+50", "-5.850924412042926E-13", "20939710.79709554", "2.2158072118719965E+73", "6.108687600855325E+51", "-1.1568879052006744E+85", "3.4514553685085106E+117", "1.1560821723679555E+117"],
            ],
            "-1E+150",
            "1E+150",
            ["-0.002947191349614015", "2.2034371042719556E+110", "8.947228638128523E+129", "-3608.563013452396", "9.505174221221046E+41", "1E+150", "-2.2360503217917383E+131", "-1E+150", "1E+150", "1E+150"],
            "1.2517734887155907E+145"),
        "X91/Y/ c /.#3" => new HeldOutProblem(
            name,
            "double",
            [
                ["5.334882991034893E-84", "-3.205591778776667E+30", "9.131177709207214E-45", "1.6336410449021989E-53", "-2.7764130149793286E-54", "-6.772340687552838E+17", "-8.287982940809701E-106", "5.7614995179273595E-09"],
                ["-3.205591778776667E+30", "6.604780908135928E+146", "4.206519783622639E+71", "-2.6356065161789297E+62", "8.224245723351996E+60", "1.6714927156561833E+133", "9067956470.778048", "-1.6636645713576776E+107"],
                ["9.131177709207214E-45", "4.206519783622639E+71", "0.0071232561390709", "2.4032420749405732E-12", "-6.475566473555153E-14", "7.903474611334156E+58", "-4.5851017207248384E-66", "-8.410992317749704E+32"],
                ["1.6336410449021989E-53", "-2.6356065161789297E+62", "2.4032420749405732E-12", "7.597956239608688E-21", "-7.753024030589605E-23", "5.020657652059957E+49", "-7.010306528650519E-75", "-4.078256488276393E+23"],
                ["-2.7764130149793286E-54", "8.224245723351996E+60", "-6.475566473555153E-14", "-7.753024030589605E-23", "1.071797357406009E-23", "1.0717104928291716E+47", "1.26504308471701E-75", "1.7891293216856686E+22"],
                ["-6.772340687552838E+17", "1.6714927156561833E+133", "7.903474611334156E+58", "5.020657652059957E+49", "1.0717104928291716E+47", "5.349941539948071E+120", "0.0007329971081901346", "-9.31091478236215E+93"],
                ["-8.287982940809701E-106", "9067956470.778048", "-4.5851017207248384E-66", "-7.010306528650519E-75", "1.26504308471701E-75", "0.0007329971081901346", "5.1325973337755565E-127", "-1.0564460117026527E-30"],
                ["5.7614995179273595E-09", "-1.6636645713576776E+107", "-8.410992317749704E+32", "-4.078256488276393E+23", "1.7891293216856686E+22", "-9.31091478236215E+93", "-1.0564460117026527E-30", "2.2641704446539575E+68"],
            ],
            "-1E+150",
            "1E+150",
            ["-1.942650901847955E+64", "-1E+150", "1.8651673346034673E+147", "2.1290504827664945E+128", "-5.073856076528406E+126", "1E+150", "8.998028689107748E+20", "-1E+150"],
            "1E+150"),
        "_9 .!b_.b_..-cbc 1X1./#0" => new HeldOutProblem(
            name,
            "double",
            [
                ["1.6982327693301437E-47", "-3.9042300904843788E-28", "-7.594425055052142E+33", "8.096321517535827E-55", "-7.588545324681956E-57", "-8.10510100014048E-93", "3.9226227423077E+25", "-6.205139607319206E-64", "-5.277471744843141E-58", "1.4977877231042767E+25", "8.723722649560424E-54", "-4.606698878673136E-59", "-2.0152717102987564E-22"],
                ["-3.9042300904843788E-28", "1.0663147552906006E-06", "-8.314281048090029E+54", "-7.87574138064308E-35", "-2.4292170818534637E-37", "2.2371617025289577E-72", "7.197317078671035E+45", "5.496718939308127E-44", "-6.91798646352565E-37", "-3.9337973965166124E+44", "6.7431910034443505E-34", "-2.8846179512551296E-39", "-0.2360221476113754"],
                ["-7.594425055052142E+33", "-8.314281048090029E+54", "3.77678084844741E+117", "-6.006957180962151E+27", "1.2716551463834274E+25", "8.11832176774492E-11", "-2.8526348224270793E+108", "2.3301873344883635E+18", "5.1125120705301124E+25", "1.0797922630356007E+107", "-8.512078240414509E+27", "-6.110017517675634E+23", "3.9924787365268744E+60"],
                ["8.096321517535827E-55", "-7.87574138064308E-35", "-6.006957180962151E+27", "1.4487864693845707E-61", "4.526537593788741E-64", "-7.519729319971739E-100", "-1.771203870552091E+18", "-3.9911059462590815E-71", "6.374246455048239E-65", "-1.017354068662048E+18", "4.517184399689191E-61", "2.8877512291858723E-66", "-5.660573390186057E-30"],
                ["-7.588545324681956E-57", "-2.4292170818534637E-37", "1.2716551463834274E+25", "4.526537593788741E-64", "7.517066996741078E-65", "3.424594127794408E-102", "-1.8089172867670125E+17", "-1.254620152488175E-72", "6.706163676929529E-66", "-1718846006665955", "-1.8958464959343205E-62", "1.2942351043520723E-67", "7.1236010630776105E-31"],
                ["-8.10510100014048E-93", "2.2371617025289577E-72", "8.11832176774492E-11", "-7.519729319971739E-100", "3.424594127794408E-102", "5.066956655070539E-137", "-2.357035503421139E-19", "2.725127414401211E-109", "-4.507540300270012E-102", "9.925135974050344E-21", "-2.8040482301777665E-99", "1.8778452536411716E-104", "-5.284034024916258E-67"],
                ["3.9226227423077E+25", "7.197317078671035E+45", "-2.8526348224270793E+108", "-1.771203870552091E+18", "-1.8089172867670125E+17", "-2.357035503421139E-19", "1.0745936342393641E+100", "-7626280362.601576", "-1.035285219214808E+17", "2.2065420320372273E+98", "1.4645466865439494E+19", "-865362778609572.9", "-9.494179823505735E+51"],
                ["-6.205139607319206E-64", "5.496718939308127E-44", "2.3301873344883635E+18", "-3.9911059462590815E-71", "-1.254620152488175E-72", "2.725127414401211E-109", "-7626280362.601576", "2.149630673944235E-79", "-2.685686038763325E-74", "-2309922036.663863", "-6.750114913854237E-71", "2.294348380406778E-75", "-8.055884073422385E-40"],
                ["-5.277471744843141E-58", "-6.91798646352565E-37", "5.1125120705301124E+25", "6.374246455048239E-65", "6.706163676929529E-66", "-4.507540300270012E-102", "-1.035285219214808E+17", "-2.685686038763325E-74", "4.736745516100322E-66", "-3686040689463959", "-2.3232379402199098E-63", "1.5347442376050836E-68", "4.2545353449048475E-31"],
                ["1.4977877231042767E+25", "-3.9337973965166124E+44", "1.0797922630356007E+107", "-1.017354068662048E+18", "-1718846006665955", "9.925135974050344E-21", "2.2065420320372273E+98", "-2309922036.663863", "-3686040689463959", "1.2052054144092936E+98", "4.250833738632682E+18", "-194610029951441.22", "-1.8705929317333604E+50"],
                ["8.723722649560424E-54", "6.7431910034443505E-34", "-8.512078240414509E+27", "4.517184399689191E-61", "-1.8958464959343205E-62", "-2.8040482301777665E-99", "1.4645466865439494E+19", "-6.750114913854237E-71", "-2.3232379402199098E-63", "4.250833738632682E+18", "2.325284458542641E-59", "-2.135315995614102E-65", "-8.588897454466882E-28"],
                ["-4.606698878673136E-59", "-2.8846179512551296E-39", "-6.110017517675634E+23", "2.8877512291858723E-66", "1.2942351043520723E-67", "1.8778452536411716E-104", "-865362778609572.9", "2.294348380406778E-75", "1.5347442376050836E-68", "-194610029951441.22", "-2.135315995614102E-65", "1.7127801837039457E-69", "5.890281605617577E-33"],
                ["-2.0152717102987564E-22", "-0.2360221476113754", "3.9924787365268744E+60", "-5.660573390186057E-30", "7.1236010630776105E-31", "-5.284034024916258E-67", "-9.494179823505735E+51", "-8.055884073422385E-40", "4.2545353449048475E-31", "-1.8705929317333604E+50", "-8.588897454466882E-28", "5.890281605617577E-33", "210549.05804641324"],
            ],
            "-1E+150",
            "1E+150",
            ["1.1534178205686901E+142", "2.7111362218482283E-88", "-1.0121324926989953E+33", "-3.4338437836156447E+71", "1.7173487820603498E+102", "6.339323072341313E-17", "-6.610728156092115E-39", "2.2646755028402544E+146", "6.858443550171197E+29", "-4.880187181537298E-44", "-1.8414919428780256E-05", "2.08002888688832E+118", "3.30537201221986E+82"],
            "1E+150"),
        "._c!bcZ cc 0a_X a0/YZ#1" => new HeldOutProblem(
            name,
            "double",
            [
                ["2.759043726014193E+117", "-1.95769401291862E+67", "1.2447082686115647E+59", "-7.846199294978797E+47", "-1.9262781975534803E+52", "-1.0376923113897553E+107", "-1.9497756686838857E+109", "-4.102586082058224E+100", "9.54966074767839E+33", "-1.779853867812874E+25", "-0.00012640547307194835", "158512.4098036678", "2.985628016554577E+30"],
                ["-1.95769401291862E+67", "8.209346672915905E+19", "27990646250.059395", "0.113669715421199", "1867.745215462823", "-4.388950968656526E+57", "3.3020438306429135E+60", "-4.877574684194799E+52", "1.5335688437587339E-15", "1.8444111768054E-24", "-2.7122235650977952E-52", "-2.875007245535081E-45", "-8.106088168121911E-19"],
                ["1.2447082686115647E+59", "27990646250.059395", "104.00075977459271", "-9.646250447929355E-11", "-1.4662175423261327E-05", "-2.0326486181957564E+49", "-5.596084119809158E+50", "-1.4576709751296164E+43", "4.837132571027929E-24", "-1.657471300029557E-33", "-2.269829350062304E-61", "-5.235465764443349E-53", "-7.811585811755856E-28"],
                ["-7.846199294978797E+47", "0.113669715421199", "-9.646250447929355E-11", "1.2703643720228062E-21", "-2.428442301947479E-17", "4.258032806626927E+37", "-1.981514019066655E+39", "-1.3326324922573264E+32", "-1.6557553235600553E-35", "3.102619752344064E-44", "2.805344767808492E-73", "-4.588237371842944E-65", "-1.6243087635139797E-39"],
                ["-1.9262781975534803E+52", "1867.745215462823", "-1.4662175423261327E-05", "-2.428442301947479E-17", "1.3637374254609618E-11", "2.415708510019394E+42", "2.3117870741710974E+45", "2.187851059231516E+37", "-2.896881401107906E-31", "-2.560619764346444E-40", "-3.745844435548471E-68", "6.286121249552889E-60", "-7.554458354373685E-35"],
                ["-1.0376923113897553E+107", "-4.388950968656526E+57", "-2.0326486181957564E+49", "4.258032806626927E+37", "2.415708510019394E+42", "2.803712439941275E+97", "-2.1065130483808587E+99", "8.789680926974442E+90", "-2.672312623764574E+23", "3348597394755161.5", "3.4337327050978574E-14", "1.7520010650199088E-05", "8.521777962664443E+20"],
                ["-1.9497756686838857E+109", "3.3020438306429135E+60", "-5.596084119809158E+50", "-1.981514019066655E+39", "2.3117870741710974E+45", "-2.1065130483808587E+99", "1.9571595756573598E+102", "-4.2710989143814143E+92", "-2.336833882710825E+26", "6.364107862933619E+17", "-3.4385382832956266E-11", "-0.008177377492948848", "-1.5593930950951364E+23"],
                ["-4.102586082058224E+100", "-4.877574684194799E+52", "-1.4576709751296164E+43", "-1.3326324922573264E+32", "2.187851059231516E+37", "8.789680926974442E+90", "-4.2710989143814143E+92", "3.315455793591147E+86", "-5.961766624391392E+18", "8302581842.2925205", "2.914597534000688E-19", "-1.3774362080050383E-10", "-376126585923489.1"],
                ["9.54966074767839E+33", "1.5335688437587339E-15", "4.837132571027929E-24", "-1.6557553235600553E-35", "-2.896881401107906E-31", "-2.672312623764574E+23", "-2.336833882710825E+26", "-5.961766624391392E+18", "1.5588090888196936E-48", "-7.298755630484173E-58", "-5.7618560540252626E-87", "7.882221463307092E-78", "1.488254156884906E-52"],
                ["-1.779853867812874E+25", "1.8444111768054E-24", "-1.657471300029557E-33", "3.102619752344064E-44", "-2.560619764346444E-40", "3348597394755161.5", "6.364107862933619E+17", "8302581842.2925205", "-7.298755630484173E-58", "1.1833015298415217E-65", "2.358384213861508E-95", "-2.810890305278025E-86", "-6.359613407100658E-62"],
                ["-0.00012640547307194835", "-2.7122235650977952E-52", "-2.269829350062304E-61", "2.805344767808492E-73", "-3.745844435548471E-68", "3.4337327050978574E-14", "-3.4385382832956266E-11", "2.914597534000688E-19", "-5.7618560540252626E-87", "2.358384213861508E-95", "3.86202023890498E-123", "8.594067362438942E-116", "4.5890946241108675E-90"],
                ["158512.4098036678", "-2.875007245535081E-45", "-5.235465764443349E-53", "-4.588237371842944E-65", "6.286121249552889E-60", "1.7520010650199088E-05", "-0.008177377492948848", "-1.3774362080050383E-10", "7.882221463307092E-78", "-2.810890305278025E-86", "8.594067362438942E-116", "3.263747738072801E-106", "2.841281621984001E-81"],
                ["2.985628016554577E+30", "-8.106088168121911E-19", "-7.811585811755856E-28", "-1.6243087635139797E-39", "-7.554458354373685E-35", "8.521777962664443E+20", "-1.5593930950951364E+23", "-376126585923489.1", "1.488254156884906E-52", "-6.359613407100658E-62", "4.5890946241108675E-90", "2.841281621984001E-81", "1.7938615767576092E-55"],
            ],
            "0",
            "1",
            ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
            "3.624444189018466E-118"),
        "9YX XYc1#0" => new HeldOutProblem(
            name,
            "double",
            [
                ["1.4760268307036447E+80", "4.663538869130737E-17", "-1.173520977973903E-23", "19.679151871095346", "-5.2251241869042084E+66", "2.2229042143172857E+68", "5.284137701237363E+67", "-6250.599663317242", "4.115821683555062E+106", "-2.2009990983573574E+72", "1.1649655661971665E+33", "56564649.3331734"],
                ["4.663538869130737E-17", "1.9325996612616535E-112", "-1.8275814206814547E-119", "2.4101883007123603E-95", "-6.069646132991315E-30", "-1.9388237826669796E-28", "1.5410950518637126E-28", "-1.1041552790680137E-92", "-30071785749.089657", "-3.272468882857679E-25", "-3.1781805830531115E-63", "-1.428438105382164E-88"],
                ["-1.173520977973903E-23", "-1.8275814206814547E-119", "1.842597895078497E-125", "5.111051509055544E-103", "-1.5025894289382992E-36", "-1.4195232240186744E-34", "-1.1553089880876668E-35", "4.761941622061635E-99", "-10805.632851826955", "6.3405994791233265E-31", "-3.3937512904609276E-70", "6.665955140321629E-95"],
                ["19.679151871095346", "2.4101883007123603E-95", "5.111051509055544E-103", "1.8746704417561604E-77", "-2.1098609553847578E-12", "-2.758329476734795E-10", "1.0085457847584622E-11", "-2.561638386651851E-75", "1.3462877803805518E+27", "-1.9304987729950226E-08", "-1.115127227057165E-45", "3.5528559751698915E-71"],
                ["-5.2251241869042084E+66", "-6.069646132991315E-30", "-1.5025894289382992E-36", "-2.1098609553847578E-12", "5.941182577124462E+54", "5.266738556735051E+55", "-1.1096895283399823E+55", "1.1503684930340777E-09", "1.2008443541647733E+94", "1.231008060282033E+59", "-9.311619090187728E+20", "-5.623822182148195E-06"],
                ["2.2229042143172857E+68", "-1.9388237826669796E-28", "-1.4195232240186744E-34", "-2.758329476734795E-10", "5.266738556735051E+55", "3.1977153449973505E+58", "1.1719037667956136E+57", "-1.1419060656021595E-08", "4.205910581700884E+94", "2.685264900265166E+61", "4.930161901373011E+22", "-0.0019912160049582483"],
                ["5.284137701237363E+67", "1.5410950518637126E-28", "-1.1553089880876668E-35", "1.0085457847584622E-11", "-1.1096895283399823E+55", "1.1719037667956136E+57", "4.508809043880035E+56", "-1.1212442573083293E-08", "-1.3032250944002469E+95", "3.829231122314003E+60", "-1.20817631081516E+21", "-0.00042324305537975396"],
                ["-6250.599663317242", "-1.1041552790680137E-92", "4.761941622061635E-99", "-2.561638386651851E-75", "1.1503684930340777E-09", "-1.1419060656021595E-08", "-1.1212442573083293E-08", "3.373454425423382E-72", "6.267661844711761E+30", "7.760365172862423E-05", "-5.096376466146162E-43", "1.7845615578709777E-68"],
                ["4.115821683555062E+106", "-30071785749.089657", "-10805.632851826955", "1.3462877803805518E+27", "1.2008443541647733E+94", "4.205910581700884E+94", "-1.3032250944002469E+95", "6.267661844711761E+30", "1.9109485026481365E+134", "-2.0868634726029252E+99", "-4.172569624503838E+60", "1.1441281469036003E+35"],
                ["-2.2009990983573574E+72", "-3.272468882857679E-25", "6.3405994791233265E-31", "-1.9304987729950226E-08", "1.231008060282033E+59", "2.685264900265166E+61", "3.829231122314003E+60", "7.760365172862423E-05", "-2.0868634726029252E+99", "4.0648624407223966E+65", "-6.571043712505495E+25", "-9.774759554120617"],
                ["1.1649655661971665E+33", "-3.1781805830531115E-63", "-3.3937512904609276E-70", "-1.115127227057165E-45", "-9.311619090187728E+20", "4.930161901373011E+22", "-1.20817631081516E+21", "-5.096376466146162E-43", "-4.172569624503838E+60", "-6.571043712505495E+25", "8.950845741190186E-13", "4.9544249033982485E-39"],
                ["56564649.3331734", "-1.428438105382164E-88", "6.665955140321629E-95", "3.5528559751698915E-71", "-5.623822182148195E-06", "-0.0019912160049582483", "-0.00042324305537975396", "1.7845615578709777E-68", "1.1441281469036003E+35", "-9.774759554120617", "4.9544249033982485E-39", "1.5618405528940264E-63"],
            ],
            "-1E+150",
            "1E+150",
            ["-1.002522792327139E+120", "-6.323408254495493E-48", "2.764598359343102E+110", "4.5508652018791986E-85", "-5.626977307089361E-61", "3.607496180328845E+29", "5.4024195925576635E+63", "-3.8277167725680325E-107", "1.2034432579772638E+118", "4.1205805350983584E-129", "8.921222226295606E-65", "5.159442101829834E-19"],
            "1"),
        "aY!a-_cX!-!Z/.#1" => new HeldOutProblem(
            name,
            "double",
            [
                ["2.5818796201534083E-96", "1.5255783836370005E-08", "-1.3580343708665016E-55", "-8.831742449507604E-55", "2.008039560357764E-64", "1.0712859245190776E-116", "-1.5820895219334739E-56", "-5.48316289631531E-60"],
                ["1.5255783836370005E-08", "5.517172213788547E+80", "-2.1531756636261144E+34", "-5.424110236174622E+33", "2.739582980462551E+24", "-6.019891366767759E-29", "1.4707845518212597E+32", "-9.206407508963934E+27"],
                ["-1.3580343708665016E-55", "-2.1531756636261144E+34", "7.75428656552051E-12", "-2.1342623543981475E-13", "1.3998468347932623E-22", "-4.767263339887825E-75", "-2.443633077904853E-14", "9.377876773271751E-19"],
                ["-8.831742449507604E-55", "-5.424110236174622E+33", "-2.1342623543981475E-13", "7.576165474992983E-13", "-9.94201502857472E-24", "-6.236876459453625E-75", "2.0330193074409718E-14", "4.351890520300134E-18"],
                ["2.008039560357764E-64", "2.739582980462551E+24", "1.3998468347932623E-22", "-9.94201502857472E-24", "6.498990274524353E-31", "3.1046780120171317E-84", "4.8336429420238714E-24", "1.0117552399890162E-27"],
                ["1.0712859245190776E-116", "-6.019891366767759E-29", "-4.767263339887825E-75", "-6.236876459453625E-75", "3.1046780120171317E-84", "2.8487594588948894E-136", "-1.377882321628953E-76", "-4.0164011457470284E-80"],
                ["-1.5820895219334739E-56", "1.4707845518212597E+32", "-2.443633077904853E-14", "2.0330193074409718E-14", "4.8336429420238714E-24", "-1.377882321628953E-76", "1.2576255306822044E-15", "1.6956319221830232E-19"],
                ["-5.48316289631531E-60", "-9.206407508963934E+27", "9.377876773271751E-19", "4.351890520300134E-18", "1.0117552399890162E-27", "-4.0164011457470284E-80", "1.6956319221830232E-19", "5.788254066287055E-23"],
            ],
            "0",
            "1",
            ["5.301349547366309E+149", "5.301349547366309E+149", "5.301349547366309E+149", "5.301349547366309E+149", "5.301349547366309E+149", "5.301349547366309E+149", "5.301347918848895E+149", "5.301349547366309E+149"],
            "5.691427585404736E-144"),
        "0a-!0aZbYb1aY0.0a/ab/! Y#0" => new HeldOutProblem(
            name,
            "double",
            [
                ["1.9848261981928883E-128", "4.1918951335491664E-95", "6.998190449476969E-108", "5.51734926936245", "-97837.56257016354", "-3.5472540359276537E-113", "-1.6877961010246407E-123", "-2.4689998583654184E-74", "-3.1820969588054476E-77", "-346677.0350096048", "-1.685802174114604E-130", "-6.081930889052191E-101", "2.922080656284904E-35", "-5.863152179270405E-08"],
                ["4.1918951335491664E-95", "1.0277996700231407E-60", "2.2767633113209555E-74", "4.185581505132187E+34", "1.5095541965143134E+38", "-4.7195574800443304E-79", "-7.003136209100091E-90", "-5.0807365527234297E-39", "-2.233718940397449E-43", "-5.294176947045173E+39", "-2.1805295820280333E-96", "8.026821648522045E-67", "-66.01084821368686", "5.624645444358199E+26"],
                ["6.998190449476969E-108", "2.2767633113209555E-74", "1.4856513268421888E-86", "3.3606006231107413E+21", "-6.204811194496082E+25", "-3.2295311057272934E-92", "6.623400795876322E-104", "-5.640062422612506E-52", "3.16911127782013E-56", "2.282046895216459E+26", "-1.8656931289944578E-109", "5.813292211154275E-80", "-4.912455967380556E-12", "46468696292532.17"],
                ["5.51734926936245", "4.185581505132187E+34", "3.3606006231107413E+21", "2.2636990102083696E+130", "1.1720715990883338E+133", "-64936939916553190", "794665.8973797238", "7.428871774211352E+56", "-4.042968986366307E+52", "-1.4472321574614982E+134", "-0.1922075858732442", "-1.8843811715542775E+29", "-1.7391497014393525E+96", "1.6975926674041773E+121"],
                ["-97837.56257016354", "1.5095541965143134E+38", "-6.204811194496082E+25", "1.1720715990883338E+133", "2.6013714951067136E+138", "-2.0853922451650073E+19", "9262659996.905731", "-3.2863413569672765E+60", "2.406707727608529E+56", "2.2379743826327773E+138", "740.4816876603933", "1.5916647476456368E+33", "-3.360756721471432E+100", "-3.8217638668196525E+125"],
                ["-3.5472540359276537E-113", "-4.7195574800443304E-79", "-3.2295311057272934E-92", "-64936939916553190", "-2.0853922451650073E+19", "7.671150807824896E-97", "-6.764451410608788E-108", "5.626413964049306E-57", "2.317749328982017E-61", "4.4207712481947445E+21", "2.4857409996429692E-114", "5.510279912742958E-85", "2.5338028997778092E-17", "-598635469.1300132"],
                ["-1.6877961010246407E-123", "-7.003136209100091E-90", "6.623400795876322E-104", "794665.8973797238", "9262659996.905731", "-6.764451410608788E-108", "1.0748137453851762E-117", "-8.07992737650072E-68", "1.020741728254844E-71", "-36282659795.72585", "3.978893803151988E-125", "-1.072593533062782E-95", "1.79536060271566E-27", "0.022415055036328676"],
                ["-2.4689998583654184E-74", "-5.0807365527234297E-39", "-5.640062422612506E-52", "7.428871774211352E+56", "-3.2863413569672765E+60", "5.626413964049306E-57", "-8.07992737650072E-68", "3.5579749458926605E-16", "-1.0865952674683592E-21", "3.0683191949407123E+61", "2.152805144633777E-74", "4.133990742459114E-45", "1.121186448675907E+24", "5.349661441535419E+48"],
                ["-3.1820969588054476E-77", "-2.233718940397449E-43", "3.16911127782013E-56", "-4.042968986366307E+52", "2.406707727608529E+56", "2.317749328982017E-61", "1.020741728254844E-71", "-1.0865952674683592E-21", "1.3312274150059471E-24", "3.742523614486717E+57", "4.66948202187199E-79", "2.6213843185011064E-48", "2.609036172326938E+19", "9.265167433847254E+44"],
                ["-346677.0350096048", "-5.294176947045173E+39", "2.282046895216459E+26", "-1.4472321574614982E+134", "2.2379743826327773E+138", "4.4207712481947445E+21", "-36282659795.72585", "3.0683191949407123E+61", "3.742523614486717E+57", "1.4384464483831513E+140", "18240.155314102067", "-1.5093288549344378E+33", "-2.2120303262025146E+101", "-4.531121423090927E+126"],
                ["-1.685802174114604E-130", "-2.1805295820280333E-96", "-1.8656931289944578E-109", "-0.1922075858732442", "740.4816876603933", "2.4857409996429692E-114", "3.978893803151988E-125", "2.152805144633777E-74", "4.66948202187199E-79", "18240.155314102067", "5.347087101733787E-131", "-3.469088360350995E-102", "3.858310528997116E-34", "2.1143447517255594E-09"],
                ["-6.081930889052191E-101", "8.026821648522045E-67", "5.813292211154275E-80", "-1.8843811715542775E+29", "1.5916647476456368E+33", "5.510279912742958E-85", "-1.072593533062782E-95", "4.133990742459114E-45", "2.6213843185011064E-48", "-1.5093288549344378E+33", "-3.469088360350995E-102", "2.4620618421270044E-71", "-0.00017398835876796727", "2.8122079413683424E+21"],
                ["2.922080656284904E-35", "-66.01084821368686", "-4.912455967380556E-12", "-1.7391497014393525E+96", "-3.360756721471432E+100", "2.5338028997778092E-17", "1.79536060271566E-27", "1.121186448675907E+24", "2.609036172326938E+19", "-2.2120303262025146E+101", "3.858310528997116E-34", "-0.00017398835876796727", "2.9005685135790134E+64", "2.8374130883724228E+88"],
                ["-5.863152179270405E-08", "5.624645444358199E+26", "46468696292532.17", "1.6975926674041773E+121", "-3.8217638668196525E+125", "-598635469.1300132", "0.022415055036328676", "5.349661441535419E+48", "9.265167433847254E+44", "-4.531121423090927E+126", "2.1143447517255594E-09", "2.8122079413683424E+21", "2.8374130883724228E+88", "1.1240457602031626E+115"],
            ],
            "-1E+150",
            "1E+150",
            ["-5.04575806485627E+19", "-2.7657087432031024E+88", "-1.1094064859599424E+61", "-1E+150", "-1E+150", "7.164851146646064E+53", "-3.826685975191167E+31", "4.272182500912103E+134", "-2.2113043908657106E+123", "-1E+150", "-78285848376973810", "1.6245357614862648E+79", "-1E+150", "-1E+150"],
            "1E+150"),
        "x9y-24#1" => new HeldOutProblem(
            name,
            "double",
            [
                ["1.3125398124022886E-136", "6.601432672802956E-71", "-6.162911392833563E-82", "2.74153364405507E-134", "9.851557994171074E-70", "3.502393192648665E-61", "-7.922448864082823E-129", "-3.592387880374298E-119", "3.5199967084920547E-06", "3.7067294636423183E-67", "-2.9860738376643872E-143", "-6.824023485339248E-131", "-1.469850919175241E-75", "8.810135106224818E-126", "-4.5667661402302965E-118"],
                ["6.601432672802956E-71", "0.0002941184036321358", "1.3478886727452792E-16", "2.690145061934141E-68", "0.003223444498600661", "-372366.9376968276", "1.4140214542745446E-63", "-3.9771545518667682E-53", "1.3700600204979963E+61", "-0.3027150954089173", "-6.942219515914102E-77", "3.661409017916928E-66", "-1.5015636850367923E-09", "-6.393058681591086E-60", "-1.2027589220527746E-51"],
                ["-6.162911392833563E-82", "1.3478886727452792E-16", "4.526007282221433E-26", "-8.011863505652868E-79", "2.509277625828655E-14", "-7.224296945727393E-07", "1.9762475812790405E-73", "9.648408534326294E-65", "5.638757554905787E+50", "1.4393049708118803E-11", "-5.054458416944428E-88", "2.5647004957999516E-75", "1.2196153392203827E-20", "-1.6106130142044236E-70", "-1.8523731308399865E-62"],
                ["2.74153364405507E-134", "2.690145061934141E-68", "-8.011863505652868E-79", "1.8553801861769153E-129", "4.82653513286039E-66", "1.1442109743559417E-57", "-5.091561287922655E-125", "1.1176491035964002E-115", "-0.1379272482342312", "-1.3634759788155738E-63", "2.9248405838561944E-140", "4.839339204197579E-128", "-4.5263909072263803E-72", "-9.16002721360979E-123", "-1.9174404218793916E-114"],
                ["9.851557994171074E-70", "0.003223444498600661", "2.509277625828655E-14", "4.82653513286039E-66", "0.1772936828003967", "10726437.198156605", "5.517339772048226E-62", "-1.5390201827364944E-51", "1.0548133544727956E+63", "8.38088269660869", "-5.038588895118494E-76", "1.9630963711216697E-63", "5.510536948841309E-08", "-1.735051583863603E-58", "-4.104980790180084E-50"],
                ["3.502393192648665E-61", "-372366.9376968276", "-7.224296945727393E-07", "1.1442109743559417E-57", "10726437.198156605", "64452223826246840", "-6.294871477530069E-53", "1.5433371020853663E-42", "4.217560601542123E+70", "8545223139.531143", "1.549548926620756E-66", "9.314483760743491E-55", "-20.353516410340333", "3.4915631002308007E-50", "-5.4829120722258734E-42"],
                ["-7.922448864082823E-129", "1.4140214542745446E-63", "1.9762475812790405E-73", "-5.091561287922655E-125", "5.517339772048226E-62", "-6.294871477530069E-53", "9.47464511049094E-120", "1.187834468612058E-110", "9876.118206279538", "-7.693944131824641E-59", "-5.958319300874641E-135", "1.393295299154097E-122", "1.9906541556244046E-66", "3.8363042397438487E-118", "-2.6804205905413237E-110"],
                ["-3.592387880374298E-119", "-3.9771545518667682E-53", "9.648408534326294E-65", "1.1176491035964002E-115", "-1.5390201827364944E-51", "1.5433371020853663E-42", "1.187834468612058E-110", "3.780935609429905E-100", "-44268292129514.14", "2.6660376272698054E-50", "9.878163588989626E-126", "4.284643412090626E-113", "3.8743777532132034E-57", "6.6175515362784875E-108", "2.0657235369002694E-99"],
                ["3.5199967084920547E-06", "1.3700600204979963E+61", "5.638757554905787E+50", "-0.1379272482342312", "1.0548133544727956E+63", "4.217560601542123E+70", "9876.118206279538", "-44268292129514.14", "7.888702716334028E+127", "1.3450341042327448E+65", "-4.0616249319208854E-11", "46.641816010269075", "3.743341236563247E+57", "-875490.1170214813", "-148691116457100"],
                ["3.7067294636423183E-67", "-0.3027150954089173", "1.4393049708118803E-11", "-1.3634759788155738E-63", "8.38088269660869", "8545223139.531143", "-7.693944131824641E-59", "2.6660376272698054E-50", "1.3450341042327448E+65", "27955.546822258013", "-2.6575625648001176E-73", "9.331154955106836E-62", "-3.0039932547610813E-05", "-4.499714667078776E-56", "2.6889842794218327E-48"],
                ["-2.9860738376643872E-143", "-6.942219515914102E-77", "-5.054458416944428E-88", "2.9248405838561944E-140", "-5.038588895118494E-76", "1.549548926620756E-66", "-5.958319300874641E-135", "9.878163588989626E-126", "-4.0616249319208854E-11", "-2.6575625648001176E-73", "2.0089161458102165E-148", "2.432255784219592E-137", "-1.6252249568168753E-81", "-6.957989762538975E-133", "-5.383092822190089E-124"],
                ["-6.824023485339248E-131", "3.661409017916928E-66", "2.5647004957999516E-75", "4.839339204197579E-128", "1.9630963711216697E-63", "9.314483760743491E-55", "1.393295299154097E-122", "4.284643412090626E-113", "46.641816010269075", "9.331154955106836E-62", "2.432255784219592E-137", "3.349234239883833E-124", "-2.931626048781547E-69", "-1.4246433742083885E-119", "-1.745764506325238E-111"],
                ["-1.469850919175241E-75", "-1.5015636850367923E-09", "1.2196153392203827E-20", "-4.5263909072263803E-72", "5.510536948841309E-08", "-20.353516410340333", "1.9906541556244046E-66", "3.8743777532132034E-57", "3.743341236563247E+57", "-3.0039932547610813E-05", "-1.6252249568168753E-81", "-2.931626048781547E-69", "1.9528074263696788E-12", "5.66309983891727E-64", "4.7497869166160135E-57"],
                ["8.810135106224818E-126", "-6.393058681591086E-60", "-1.6106130142044236E-70", "-9.16002721360979E-123", "-1.735051583863603E-58", "3.4915631002308007E-50", "3.8363042397438487E-118", "6.6175515362784875E-108", "-875490.1170214813", "-4.499714667078776E-56", "-6.957989762538975E-133", "-1.4246433742083885E-119", "5.66309983891727E-64", "4.20994234258E-114", "1.027352170368311E-106"],
                ["-4.5667661402302965E-118", "-1.2027589220527746E-51", "-1.8523731308399865E-62", "-1.9174404218793916E-114", "-4.104980790180084E-50", "-5.4829120722258734E-42", "-2.6804205905413237E-110", "2.0657235369002694E-99", "-148691116457100", "2.6889842794218327E-48", "-5.383092822190089E-124", "-1.745764506325238E-111", "4.7497869166160135E-57", "1.027352170368311E-106", "7.502256705677487E-98"],
            ],
            "-1E+150",
            "1E+150"),
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unknown witness"),
    };

    private static readonly Rational s_decimalMax = Rational.FromDecimal(decimal.MaxValue);

    // max |λ·Σᵢⱼ·w*ⱼ| over the exact optimum w* of a mean-variance problem.
    private static Rational MaxQuadraticTerm(HeldOutProblem p)
    {
        var q = QpInstance.From(HeldOutCase(p));
        var (w0, _) = ExactOracle.ActiveSet(q);
        var best = Rational.Zero;
        for (var i = 0; i < q.N; i++)
        {
            for (var j = 0; j < q.N; j++)
            {
                var term = (q.Lambda * q.Cov[i, j] * w0[j]).Abs();
                best = term > best ? term : best;
            }
        }

        return best;
    }

    private static Rational PowerOfTen(int e) => e >= 0
        ? Rational.Of(BigInteger.Pow(10, e), BigInteger.One)
        : Rational.Of(BigInteger.One, BigInteger.Pow(10, -e));

    private static HeldOutProblem? MeanVarianceDomainEdgeRangeProblem(System.Random rng, string type, int trial)
    {
        var dec = type == "decimal";
        double LogUniform(double lo, double hi) => Math.Pow(10, lo + ((hi - lo) * rng.NextDouble()));
        var oshape = rng.Next(5) != 0;
        var k = oshape ? 2 * rng.Next(1, 5) : rng.Next(2, 9);
        var s = new int[k];
        for (var i = 0; i < k; i++)
        {
            s[i] = oshape ? (i < k / 2 ? 1 : -1) : (rng.Next(2) == 0 ? 1 : -1);
        }

        s = [.. s.OrderBy(_ => rng.Next())];
        var le = dec ? -18 + (3.5 * rng.NextDouble()) : -13 + (7 * rng.NextDouble());
        var eps = Rational.Of(new BigInteger(Math.Round(Math.Pow(10, le - Math.Floor(le)) * 1e6)), BigInteger.Pow(10, 6 - (int)Math.Floor(le)));
        var scale = dec
            ? Rational.FromInt(rng.Next(1, 10)) * PowerOfTen(rng.Next(-2, 14))
            : Rational.FromDouble(LogUniform(-100, 140));
        var cov = new string[k][];
        for (var i = 0; i < k; i++)
        {
            cov[i] = new string[k];
        }

        for (var i = 0; i < k; i++)
        {
            for (var j = i; j < k; j++)
            {
                var e = (i == j ? Rational.One : Rational.Zero) - ((Rational.One - eps) / Rational.FromInt(k) * Rational.FromInt(s[i] * s[j]));
                cov[i][j] = cov[j][i] = InType(type, scale * e);
            }
        }

        // The optimum moves by about m/(λ·c·ε) along s; λ is chosen from that target and must stay in the domain.
        var m = (dec ? LogUniform(12.5, 14) : LogUniform(60, 150)) * (0.1 + (0.89 * rng.NextDouble()));
        var target = dec ? LogUniform(6, 13.5) : LogUniform(3, 140);
        var lambdaValue = m / (double.Parse(Scientific(scale * eps, 17), CultureInfo.InvariantCulture) * target);
        if (!double.IsFinite(lambdaValue) || lambdaValue > (dec ? 1e14 : 1e150) || lambdaValue < (dec ? 1e-12 : 1e-150))
        {
            return null;
        }

        var mu = new string[k];
        for (var i = 0; i < k; i++)
        {
            mu[i] = InType(type, m * s[i] * (1 + (2e-3 * (rng.NextDouble() - 0.5))));
        }

        var box = rng.Next(3) == 0 ? target * (0.3 + (0.65 * rng.NextDouble())) : target * (1.5 + (48.5 * rng.NextDouble()));
        box = Math.Max(1, Math.Min(dec ? 1e14 : 1e150, box));
        return Admissible(new HeldOutProblem(
            "mv-range-" + trial + "-n" + k, type, cov, InType(type, -box), InType(type, box), mu, InType(type, lambdaValue)));
    }

    private static HeldOutProblem DomainEdgeRangeWitness(string name) => name switch
    {
        "range-witness-1" => new HeldOutProblem(
            name,
            "decimal",
            [
                ["100000000", "99999999.9999999"],
                ["99999999.9999999", "100000000"],
            ],
            "-100000000000000",
            "100000000000000",
            ["100000000000000", "-100000000000000"],
            "100000000"),
        "_c0b9c _9-.YXYc #2" => new HeldOutProblem(
            name,
            "decimal",
            [
                ["78750.000000000001818361125000", "-11249.999999999998181638875000", "-11249.999999999998181638875000", "11249.999999999998181638875000", "-11249.999999999998181638875000", "11249.999999999998181638875000", "11249.999999999998181638875000", "11249.999999999998181638875000"],
                ["-11249.999999999998181638875000", "78750.000000000001818361125000", "-11249.999999999998181638875000", "11249.999999999998181638875000", "-11249.999999999998181638875000", "11249.999999999998181638875000", "11249.999999999998181638875000", "11249.999999999998181638875000"],
                ["-11249.999999999998181638875000", "-11249.999999999998181638875000", "78750.000000000001818361125000", "11249.999999999998181638875000", "-11249.999999999998181638875000", "11249.999999999998181638875000", "11249.999999999998181638875000", "11249.999999999998181638875000"],
                ["11249.999999999998181638875000", "11249.999999999998181638875000", "11249.999999999998181638875000", "78750.000000000001818361125000", "11249.999999999998181638875000", "-11249.999999999998181638875000", "-11249.999999999998181638875000", "-11249.999999999998181638875000"],
                ["-11249.999999999998181638875000", "-11249.999999999998181638875000", "-11249.999999999998181638875000", "11249.999999999998181638875000", "78750.000000000001818361125000", "11249.999999999998181638875000", "11249.999999999998181638875000", "11249.999999999998181638875000"],
                ["11249.999999999998181638875000", "11249.999999999998181638875000", "11249.999999999998181638875000", "-11249.999999999998181638875000", "11249.999999999998181638875000", "78750.000000000001818361125000", "-11249.999999999998181638875000", "-11249.999999999998181638875000"],
                ["11249.999999999998181638875000", "11249.999999999998181638875000", "11249.999999999998181638875000", "-11249.999999999998181638875000", "11249.999999999998181638875000", "-11249.999999999998181638875000", "78750.000000000001818361125000", "-11249.999999999998181638875000"],
                ["11249.999999999998181638875000", "11249.999999999998181638875000", "11249.999999999998181638875000", "-11249.999999999998181638875000", "11249.999999999998181638875000", "-11249.999999999998181638875000", "-11249.999999999998181638875000", "78750.000000000001818361125000"],
            ],
            "-62129451956669.97",
            "62129451956669.97",
            ["-33932315702498.05", "-33889073830476.25", "-33940252303569.61", "33947132651630.87", "-33890086609738.35", "33898380344481.3", "33918650053272.2", "33894945356321.39"],
            "1341826673279.158"),
        "/a-_Y!X/cbX1!1#3" => new HeldOutProblem(
            name,
            "decimal",
            [
                ["30000.000000000000010206640000", "9999.999999999999989793360000", "-9999.999999999999989793360000", "9999.999999999999989793360000"],
                ["9999.999999999999989793360000", "30000.000000000000010206640000", "9999.999999999999989793360000", "-9999.999999999999989793360000"],
                ["-9999.999999999999989793360000", "9999.999999999999989793360000", "30000.000000000000010206640000", "9999.999999999999989793360000"],
                ["9999.999999999999989793360000", "-9999.999999999999989793360000", "9999.999999999999989793360000", "30000.000000000000010206640000"],
            ],
            "-100000000000000",
            "100000000000000",
            ["7554412896878.071", "-7567537722613.821", "7553430820719.624", "-7552971317697.357"],
            "35163013708199.15"),
        "/1XX11Y 1-#1" => new HeldOutProblem(
            name,
            "decimal",
            [
                ["1666.6666666666666685298760000", "-333.33333333333333147012400000", "333.33333333333333147012400000", "333.33333333333333147012400000", "-333.33333333333333147012400000", "333.33333333333333147012400000"],
                ["-333.33333333333333147012400000", "1666.6666666666666685298760000", "333.33333333333333147012400000", "333.33333333333333147012400000", "-333.33333333333333147012400000", "333.33333333333333147012400000"],
                ["333.33333333333333147012400000", "333.33333333333333147012400000", "1666.6666666666666685298760000", "-333.33333333333333147012400000", "333.33333333333333147012400000", "-333.33333333333333147012400000"],
                ["333.33333333333333147012400000", "333.33333333333333147012400000", "-333.33333333333333147012400000", "1666.6666666666666685298760000", "333.33333333333333147012400000", "-333.33333333333333147012400000"],
                ["-333.33333333333333147012400000", "-333.33333333333333147012400000", "333.33333333333333147012400000", "333.33333333333333147012400000", "1666.6666666666666685298760000", "333.33333333333333147012400000"],
                ["333.33333333333333147012400000", "333.33333333333333147012400000", "-333.33333333333333147012400000", "-333.33333333333333147012400000", "333.33333333333333147012400000", "1666.6666666666666685298760000"],
            ],
            "-100000000000000",
            "100000000000000",
            ["-2185631235420.109", "-2185970558106.44", "2187964656329.805", "2185871602609.115", "-2185629008312.525", "2187058332606.595"],
            "37907444074213.02"),
        "_c/X _1-b_c Y_ccbZZb-#2" => new HeldOutProblem(
            name,
            "decimal",
            [
                ["16666.666666666666733600033333", "3333.3333333333332663999666667", "-3333.3333333333332663999666667", "3333.3333333333332663999666667", "-3333.3333333333332663999666667", "3333.3333333333332663999666667"],
                ["3333.3333333333332663999666667", "16666.666666666666733600033333", "3333.3333333333332663999666667", "-3333.3333333333332663999666667", "3333.3333333333332663999666667", "-3333.3333333333332663999666667"],
                ["-3333.3333333333332663999666667", "3333.3333333333332663999666667", "16666.666666666666733600033333", "3333.3333333333332663999666667", "-3333.3333333333332663999666667", "3333.3333333333332663999666667"],
                ["3333.3333333333332663999666667", "-3333.3333333333332663999666667", "3333.3333333333332663999666667", "16666.666666666666733600033333", "3333.3333333333332663999666667", "-3333.3333333333332663999666667"],
                ["-3333.3333333333332663999666667", "3333.3333333333332663999666667", "-3333.3333333333332663999666667", "3333.3333333333332663999666667", "16666.666666666666733600033333", "3333.3333333333332663999666667"],
                ["3333.3333333333332663999666667", "-3333.3333333333332663999666667", "3333.3333333333332663999666667", "-3333.3333333333332663999666667", "3333.3333333333332663999666667", "16666.666666666666733600033333"],
            ],
            "-100000000000000",
            "100000000000000",
            ["-3104128175596.314", "3103998483686.86", "-3099371908124.109", "3101076081437.751", "-3102048260926.07", "3104227700063.026"],
            "1232914760388.892"),
        "1bcb-bZ! ///__-9YX#2" => new HeldOutProblem(
            name,
            "decimal",
            [
                ["35000.000000000000097017700000", "4999.9999999999999029823000000", "4999.9999999999999029823000000", "-4999.9999999999999029823000000", "-4999.9999999999999029823000000", "4999.9999999999999029823000000", "-4999.9999999999999029823000000", "4999.9999999999999029823000000"],
                ["4999.9999999999999029823000000", "35000.000000000000097017700000", "-4999.9999999999999029823000000", "4999.9999999999999029823000000", "4999.9999999999999029823000000", "-4999.9999999999999029823000000", "4999.9999999999999029823000000", "-4999.9999999999999029823000000"],
                ["4999.9999999999999029823000000", "-4999.9999999999999029823000000", "35000.000000000000097017700000", "4999.9999999999999029823000000", "4999.9999999999999029823000000", "-4999.9999999999999029823000000", "4999.9999999999999029823000000", "-4999.9999999999999029823000000"],
                ["-4999.9999999999999029823000000", "4999.9999999999999029823000000", "4999.9999999999999029823000000", "35000.000000000000097017700000", "-4999.9999999999999029823000000", "4999.9999999999999029823000000", "-4999.9999999999999029823000000", "4999.9999999999999029823000000"],
                ["-4999.9999999999999029823000000", "4999.9999999999999029823000000", "4999.9999999999999029823000000", "-4999.9999999999999029823000000", "35000.000000000000097017700000", "4999.9999999999999029823000000", "-4999.9999999999999029823000000", "4999.9999999999999029823000000"],
                ["4999.9999999999999029823000000", "-4999.9999999999999029823000000", "-4999.9999999999999029823000000", "4999.9999999999999029823000000", "4999.9999999999999029823000000", "35000.000000000000097017700000", "4999.9999999999999029823000000", "-4999.9999999999999029823000000"],
                ["-4999.9999999999999029823000000", "4999.9999999999999029823000000", "4999.9999999999999029823000000", "-4999.9999999999999029823000000", "-4999.9999999999999029823000000", "4999.9999999999999029823000000", "35000.000000000000097017700000", "4999.9999999999999029823000000"],
                ["4999.9999999999999029823000000", "-4999.9999999999999029823000000", "-4999.9999999999999029823000000", "4999.9999999999999029823000000", "4999.9999999999999029823000000", "-4999.9999999999999029823000000", "4999.9999999999999029823000000", "35000.000000000000097017700000"],
            ],
            "-29595398709158.94",
            "29595398709158.94",
            ["-4943670641418.56", "4939115659557.131", "4940530372943.004", "-4944081941861.864", "-4943065573201.987", "4940106418368.592", "-4938866382005.989", "4942636674874.398"],
            "8051288535088.155"),
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unknown witness"),
    };

    // A double formatted for the working type (decimal: through 16 significant digits).
    private static string InType(string type, double x) => type == "decimal"
        ? decimal.Parse(x.ToString("G16", CultureInfo.InvariantCulture), NumberStyles.Float, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture)
        : x.ToString("R", CultureInfo.InvariantCulture);

    // A rational rounded into the working type (to nearest through 40 or 28 significant digits).
    private static string InType(string type, Rational r) => r.Sign == 0
        ? "0"
        : type == "double"
            ? double.Parse(Scientific(r, 40), NumberStyles.Float, CultureInfo.InvariantCulture).ToString("R", CultureInfo.InvariantCulture)
            : decimal.Parse(Scientific(r, 28), NumberStyles.Float, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture);

    private static double Gaussian(System.Random rng) =>
        Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());

    // The problem if it is feasible, in the operating domain and positive definite; otherwise null.
    private static HeldOutProblem? Admissible(HeldOutProblem p)
    {
        if (Feasible(p) is null)
        {
            return null;
        }

        var q = QpInstance.From(HeldOutCase(p));
        return q.InDomain(p.Type) && ExactOracle.IsPositiveDefinite(q.Cov) ? p : null;
    }

    private static HeldOutProblem? MeanVarianceNearMarginProblem(System.Random rng, string type, int trial)
    {
        var dec = type == "decimal";
        var oshape = trial % 4 < 2;
        var k = oshape ? 2 * rng.Next(1, 6) : rng.Next(2, 11);
        var s = new int[k];
        for (var i = 0; i < k; i++)
        {
            s[i] = oshape ? (i < k / 2 ? 1 : -1) : (rng.Next(2) == 0 ? 1 : -1);
        }

        s = [.. s.OrderBy(_ => rng.Next())];
        var le = dec ? -27.5 + (4 * rng.NextDouble()) : -15.5 + (4 * rng.NextDouble());
        var eps = Rational.Of(new BigInteger(Math.Round(Math.Pow(10, le - Math.Floor(le)) * 1e6)), BigInteger.Pow(10, 6 - (int)Math.Floor(le)));
        var vv = dec ? Rational.Of(rng.Next(1, 1000), 100) : Rational.FromDouble(Math.ScaleB(1 + rng.NextDouble(), rng.Next(-20, 21)));
        var cov = new string[k][];
        for (var i = 0; i < k; i++)
        {
            cov[i] = new string[k];
        }

        for (var i = 0; i < k; i++)
        {
            for (var j = i; j < k; j++)
            {
                var e = (i == j ? Rational.One : Rational.Zero) - ((Rational.One - eps) / Rational.FromInt(k) * Rational.FromInt(s[i] * s[j]));
                cov[i][j] = cov[j][i] = InType(type, vv * e);
            }
        }

        var regime = rng.Next(3);
        var lambda = dec ? new[] { "0.000001", "2", "1000000" }[regime] : new[] { "1E-40", "2", "1E+40" }[regime];
        var scale = double.Parse(lambda, CultureInfo.InvariantCulture) * double.Parse(Scientific(vv, 17), CultureInfo.InvariantCulture) / k * Math.Pow(10, -2 + (4 * rng.NextDouble()));
        var mu = new string[k];
        for (var i = 0; i < k; i++)
        {
            mu[i] = InType(type, Gaussian(rng) * scale);
        }

        var (lb, ub) = rng.Next(3) switch
        {
            0 => ("-1000", "1000"),
            1 => ("0", "1"),
            _ => ("-1", "2"),
        };

        return Admissible(new HeldOutProblem("mv-nearmargin-" + trial + "-n" + k, type, cov, lb, ub, mu, lambda));
    }

    private static HeldOutProblem? MeanVarianceDegenerateAtBoundProblem(System.Random rng, string type, int trial)
    {
        var dec = type == "decimal";
        var n = rng.Next(3, 11);
        var a = new int[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                a[i, j] = rng.Next(-4, 5);
            }
        }

        var unit = new double[n];
        for (var i = 0; i < n; i++)
        {
            unit[i] = new[] { 1, 0.5, 2, 4 }[rng.Next(4)];
        }

        var shift = rng.Next(1, 4);
        var cov = new string[n][];
        var diagonal = 0.0;
        for (var i = 0; i < n; i++)
        {
            cov[i] = new string[n];
            for (var j = 0; j < n; j++)
            {
                long acc = 0;
                for (var q = 0; q < n; q++)
                {
                    acc += a[i, q] * a[j, q];
                }

                var v = (acc + (i == j ? shift : 0)) * unit[i] * unit[j];
                cov[i][j] = InType(type, v);
                if (i == j)
                {
                    diagonal += v / n;
                }
            }
        }

        var lambda = dec ? new[] { "1", "0.5", "10", "0.000001" }[rng.Next(4)] : new[] { "1", "0.5", "10", "1E-30" }[rng.Next(4)];
        var scale = double.Parse(lambda, CultureInfo.InvariantCulture) * diagonal / n * Math.Pow(10, -1 + (2 * rng.NextDouble()));
        var mu = new string[n];
        for (var i = 0; i < n; i++)
        {
            mu[i] = InType(type, Gaussian(rng) * scale);
        }

        var (lb, ub) = rng.Next(3) switch
        {
            0 => ("0", "1"),
            1 => ("-0.5", "1"),
            _ => ("0", InType(type, Math.Round(2.0 / n, 4))),
        };

        var start = Admissible(new HeldOutProblem("mv-degenb-" + trial + "-n" + n, type, cov, lb, ub, mu, lambda));
        if (start is null)
        {
            return null;
        }

        var instance = QpInstance.From(HeldOutCase(start));
        var (w, pattern) = ExactOracle.ActiveSet(instance);
        var free = Enumerable.Range(0, n).Where(i => pattern[i] == 0).ToArray();
        var atBound = Enumerable.Range(0, n).Where(i => pattern[i] != 0).ToArray();
        if (free.Length == 0 || atBound.Length == 0)
        {
            return null;
        }

        Rational Gradient(int i)
        {
            var g = -instance.Mu[i];
            for (var j = 0; j < n; j++)
            {
                g += instance.Lambda * instance.Cov[i, j] * w[j];
            }

            return g;
        }

        var nu = Gradient(free[0]);
        var b = atBound[rng.Next(atBound.Length)];
        var multiplier = pattern[b] < 0 ? Gradient(b) - nu : nu - Gradient(b);
        var eta = rng.Next(3) == 0
            ? Rational.Zero
            : multiplier * Rational.Of(rng.Next(2) == 0 ? 1 : -1, BigInteger.Pow(10, dec ? rng.Next(20, 27) : rng.Next(10, 15)));
        var target = pattern[b] < 0 ? instance.Mu[b] + (multiplier - eta) : instance.Mu[b] - (multiplier - eta);
        if (target.Abs() > Rational.FromInt(dec ? 100_000_000_000_000 : 1_000_000_000_000_000_000))
        {
            return null;
        }

        var moved = (string[])mu.Clone();
        moved[b] = InType(type, target);
        return Admissible(start with { Name = start.Name + "-b" + b, Mu = moved });
    }

    private static HeldOutProblem? MeanVarianceCancellationOrMaskingProblem(System.Random rng, string type, int trial)
    {
        var dec = type == "decimal";
        double LogUniform(double lo, double hi) => Math.Pow(10, lo + ((hi - lo) * rng.NextDouble()));
        double[,] m;
        double lb, ub;
        if (trial % 2 == 0)
        {
            // D1: V0 asset plus a block with correlation ρ = −(1 − ε)/(k − 1).
            var k = rng.Next(2, 7);
            var n = k + 1;
            var v = dec ? LogUniform(-12, -6) : LogUniform(-150, -100);
            var eps = LogUniform(-9, -1);
            var rho = -(1 - eps) / (k - 1);
            m = new double[n, n];
            m[0, 0] = dec ? LogUniform(-8, 2) : LogUniform(-100, 10);
            for (var i = 1; i < n; i++)
            {
                for (var j = 1; j < n; j++)
                {
                    m[i, j] = i == j ? v : v * rho;
                }
            }

            lb = rng.Next(2) == 0 ? 0 : -(dec ? LogUniform(2, 12) : LogUniform(2, 140));
            ub = dec ? LogUniform(2, 12) : LogUniform(2, 140);
        }
        else
        {
            // D4: one or two high-variance assets beside a low-variance block.
            var n = rng.Next(3, 9);
            var h = rng.Next(1, Math.Min(3, n - 1) + 1);
            var k = n - h;
            var v = dec ? LogUniform(-10, -4) : LogUniform(-60, -4);
            var high = dec ? LogUniform(4, 10) : LogUniform(4, 60);
            var q = new double[k, k];
            for (var i = 0; i < k; i++)
            {
                for (var j = 0; j < k; j++)
                {
                    q[i, j] = (2 * rng.NextDouble()) - 1;
                }
            }

            m = new double[n, n];
            for (var i = 0; i < h; i++)
            {
                m[i, i] = high * (0.5 + (1.5 * rng.NextDouble()));
            }

            for (var i = 0; i < k; i++)
            {
                for (var j = 0; j < k; j++)
                {
                    double acc = 0;
                    for (var p = 0; p < k; p++)
                    {
                        acc += q[i, p] * q[j, p];
                    }

                    m[h + i, h + j] = v * ((acc / k) + (i == j ? 0.1 : 0));
                }
            }

            for (var i = 0; i < h; i++)
            {
                for (var j = 0; j < k; j++)
                {
                    m[i, h + j] = m[h + j, i] = (rng.NextDouble() - 0.5) * Math.Sqrt(m[i, i] * m[h + j, h + j]) / k;
                }
            }

            lb = new[] { 0.0, -Math.Round(LogUniform(-2, 1), 3), Math.Round(rng.NextDouble() / n, 3) }[rng.Next(3)];
            ub = Math.Round(1.0 / n * (1 + LogUniform(-2, 0.7)), 4);
        }

        var size = m.GetLength(0);
        var cov = new string[size][];
        for (var i = 0; i < size; i++)
        {
            cov[i] = new string[size];
            for (var j = 0; j < size; j++)
            {
                cov[i][j] = InType(type, i <= j ? m[i, j] : m[j, i]);
            }
        }

        var lambdaValue = rng.Next(2) == 0 ? 1.0 : LogUniform(-2, 2);
        var lambda = InType(type, lambdaValue);
        var exactLambda = HeldOutParse(type, lambda);
        var mu = new string[size];
        if (trial % 2 == 0)
        {
            // μ = λΣ(1/n·1 + δz), z ⟂ 1, rounded once.
            var z = new Rational[size];
            var total = Rational.Zero;
            for (var i = 0; i < size; i++)
            {
                z[i] = Rational.FromInt(rng.Next(-1000, 1001));
                total += z[i];
            }

            var delta = Rational.Of(BigInteger.One, BigInteger.Pow(10, dec ? rng.Next(14, 29) : rng.Next(8, 18)));
            for (var i = 0; i < size; i++)
            {
                var acc = Rational.Zero;
                for (var j = 0; j < size; j++)
                {
                    acc += HeldOutParse(type, cov[i][j]) * (Rational.Of(BigInteger.One, size) + (delta * (z[j] - (total / Rational.FromInt(size)))));
                }

                mu[i] = InType(type, exactLambda * acc);
            }
        }
        else
        {
            var small = lambdaValue * m[size - 1, size - 1] / size * LogUniform(-3, 0);
            for (var i = 0; i < size; i++)
            {
                mu[i] = InType(type, Gaussian(rng) * small);
            }

            mu[rng.Next(size)] = InType(type, (rng.Next(2) == 0 ? 1 : -1) * lambdaValue * m[0, 0] / size * LogUniform(0, 3));
        }

        return Admissible(new HeldOutProblem(
            (trial % 2 == 0 ? "mv-cancel-" : "mv-mask-") + trial + "-n" + size, type, cov, InType(type, lb), InType(type, ub), mu, lambda));
    }

    private static HeldOutProblem? ZeroCorrelationProblem(System.Random rng, string type, int trial)
    {
        var dec = type == "decimal";
        double LogUniform(double a, double b) => Math.Pow(10, a + ((b - a) * rng.NextDouble()));
        double Uniform(double a, double b) => a + ((b - a) * rng.NextDouble());

        // v·(δᵢⱼ − (1−ε)/k·sᵢsⱼ): eigenvalue v·ε along s, v elsewhere.
        double[,] Block(double v, double eps, int[] s)
        {
            var k = s.Length;
            var block = new double[k, k];
            for (var i = 0; i < k; i++)
            {
                for (var j = 0; j < k; j++)
                {
                    block[i, j] = v * ((i == j ? 1.0 : 0.0) - ((1 - eps) / k * s[i] * s[j]));
                }
            }

            return block;
        }

        double[,] m;
        double lb, ub;
        if (trial % 3 != 2)
        {
            var k = rng.Next(1, 12);
            var n = k + 1;
            var big = dec ? LogUniform(10, 13) : LogUniform(100, 149);
            var eps = LogUniform(-16, -10);
            var v = dec ? LogUniform(-10, -4) : LogUniform(-149, -135);
            var s = new int[k];
            for (var i = 0; i < k; i++)
            {
                s[i] = rng.Next(2) == 0 ? 1 : -1;
            }

            var block = Block(v, eps, s);
            var coupling = Uniform(0.3, 0.95) * Math.Sqrt(big * v * eps);
            m = new double[n, n];
            m[0, 0] = big;
            for (var i = 0; i < k; i++)
            {
                m[0, i + 1] = m[i + 1, 0] = coupling * s[i] / Math.Sqrt(k);
                for (var j = 0; j < k; j++)
                {
                    m[i + 1, j + 1] = block[i, j];
                }
            }

            lb = rng.Next(2) == 0 ? Math.Round(Uniform(0.05, 1.0 / n), 3) : -(dec ? LogUniform(8, 13) : LogUniform(100, 149));
            ub = dec ? LogUniform(8, 13) : LogUniform(100, 149);
        }
        else
        {
            var second = rng.Next(1, 3);
            var n = 2 + second + 1;
            var v1 = dec ? LogUniform(-12, -4) : LogUniform(-149, -100);
            var v2 = dec ? LogUniform(-12, -4) : LogUniform(-149, -100);
            var e1 = LogUniform(-16, -10);
            var e2 = LogUniform(-16, -10);
            var big = dec ? LogUniform(8, 13) : LogUniform(100, 149);
            var box = dec ? LogUniform(8, 13) : LogUniform(100, 149);
            m = new double[n, n];
            var b1 = Block(v1, e1, [1, -1]);
            for (var i = 0; i < 2; i++)
            {
                for (var j = 0; j < 2; j++)
                {
                    m[i, j] = b1[i, j];
                }
            }

            if (second == 2)
            {
                var b2 = Block(v2, e2, [1, -1]);
                for (var i = 0; i < 2; i++)
                {
                    for (var j = 0; j < 2; j++)
                    {
                        m[2 + i, 2 + j] = b2[i, j];
                    }
                }
            }
            else
            {
                m[2, 2] = v2;
            }

            m[n - 1, n - 1] = big;
            var low1 = v1 * e1;
            var low2 = second == 2 ? v2 * e2 : v2;
            var pair = Uniform(0.2, 0.9) * Math.Sqrt(low1 * low2) / 2;
            int[] sign = [1, -1];
            for (var i = 0; i < 2; i++)
            {
                for (var j = 0; j < second; j++)
                {
                    m[i, 2 + j] = m[2 + j, i] = pair * sign[i] * sign[j];
                }
            }

            var dominant = Uniform(0.2, 0.9) * Math.Sqrt(big * low1) / 2;
            for (var i = 0; i < 2; i++)
            {
                m[i, n - 1] = m[n - 1, i] = dominant * sign[i];
            }

            lb = -box;
            ub = box * Uniform(0.5, 2);
        }

        string Format(double x) => dec
            ? decimal.Parse(x.ToString("G16", CultureInfo.InvariantCulture), NumberStyles.Float, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture)
            : x.ToString("R", CultureInfo.InvariantCulture);

        var size = m.GetLength(0);
        var cov = new string[size][];
        for (var i = 0; i < size; i++)
        {
            cov[i] = new string[size];
            for (var j = 0; j < size; j++)
            {
                cov[i][j] = Format(i <= j ? m[i, j] : m[j, i]);
            }
        }

        var problem = Feasible(new HeldOutProblem("zc-" + trial + "-n" + size, type, cov, Format(lb), Format(ub)));
        if (problem is null)
        {
            return null;
        }

        var q = QpInstance.From(HeldOutCase(problem));
        return q.InDomain(type) && ExactOracle.IsPositiveDefinite(q.Cov) ? problem : null;
    }

    internal static void AssertCensus(List<HeldOutResult> results, int minimumQualified)
    {
        var failures = results.Where(r =>
            r.Outcome is HeldOutOutcome.Exhausted or HeldOutOutcome.Wrong or HeldOutOutcome.Error
            || (r.Outcome == HeldOutOutcome.Refused && r.Qualified)).ToList();
        var summary = results.Count + " problems, " + results.Count(r => r.Qualified) + " qualified; "
            + string.Join(", ", results.GroupBy(r => r.Outcome + (r.Qualified ? "|Q" : "|U")).OrderBy(g => g.Key, StringComparer.Ordinal).Select(g => g.Key + " " + g.Count()));
        Assert.True(results.Count(r => r.Qualified) >= minimumQualified, "the census must cover qualified cases: " + summary);
        Assert.True(failures.Count == 0, summary + " | " + string.Join(" | ", failures.Take(6).Select(f => f.Outcome + (f.Qualified ? " [Q] " : " [U] ") + f.Detail)));
    }

    /// <summary>
    /// The refusal direction of the conditioning guard on a census (design step 4, and step 5 for mean variance): every
    /// member whose deciding block <c>B = F*</c> crosses the margin <c>48·n·u·κ_eq ≥ 1</c> is refused naming
    /// stationarity, and their number is the one the committed seed produces. <see cref="AssertCensus"/> accepts a
    /// refusal-permitted member that returns a vector within the bound, so without this a guard that never fires on one
    /// entry point or one type passes every census.
    /// </summary>
    internal static void AssertMarginCrossersAreRefusedNamingStationarity(List<HeldOutResult> results, int expectedCrossers)
    {
        var crossers = results.Where(r => r.CrossesTheMargin).ToList();
        var notRefused = crossers.Where(r => r.Outcome != HeldOutOutcome.Refused || !r.Detail.Contains("stationarity", StringComparison.Ordinal)).ToList();
        Assert.True(
            notRefused.Count == 0,
            notRefused.Count + " of " + crossers.Count + " members whose deciding block crosses the margin were not refused naming stationarity: "
            + string.Join(" | ", notRefused.Take(6).Select(r => r.Outcome + " " + r.Detail)));
        Assert.True(
            crossers.Count == expectedCrossers,
            "the committed seed must produce " + expectedCrossers + " members whose deciding block crosses the margin, found " + crossers.Count);
    }

    internal enum HeldOutOutcome
    {
        Ok,
        Refused,
        Exhausted,
        Wrong,
        Error,
    }

    // Mu / Lambda null: a min-variance problem; otherwise the mean-variance problem with those means and risk aversion.
    internal sealed record HeldOutProblem(string Name, string Type, string[][] Cov, string Lb, string Ub, string[]? Mu = null, string? Lambda = null);

    // CrossesTheMargin: in the operating domain with a representable exact optimum, and the deciding block B = F* of
    // that optimum has 48·n·u·κ_eq ≥ 1, so the conditioning guard of design step 4 refuses it naming stationarity.
    internal sealed record HeldOutResult(HeldOutOutcome Outcome, bool Qualified, string Detail, bool CrossesTheMargin = false);

    private static HeldOutProblem HeldOutWitness(string name) => name switch
    {
        "degen-120#1" => RepeatedStructure(name, "double", 5, 3, "3", "1E-8", (1.0 / 6).ToString("R", CultureInfo.InvariantCulture), "1"),
        "v3probe-239#1" => RepeatedStructure(name, "decimal", 9, 2, "0.95", "1E-8", "0.1000000000000000000000000000", "1"),
        "v3probe-188#2" => Diagonal(name, "double", ["441029.3620155419", "7.41199517350585E+88"], "0", "1"),
        "diag-59#2" => Diagonal(name, "double", ["3.941309654548258E-121", "5.043442595471698E+38", "8.38828807510335E+55"], "0", "1"),
        "a1Y9. /1 0#1" => new HeldOutProblem(
            name,
            "double",
            [
                ["2.7385846168598383E+123", "-1.0480340939535158E-20", "1.0480340939535158E-20", "1.0480340939535158E-20", "-1.0480340939535158E-20"],
                ["-1.0480340939535158E-20", "2.1181073767866432E-148", "7.060357922622136E-149", "7.060357922622136E-149", "-7.060357922622136E-149"],
                ["1.0480340939535158E-20", "7.060357922622136E-149", "2.1181073767866432E-148", "-7.060357922622136E-149", "7.060357922622136E-149"],
                ["1.0480340939535158E-20", "7.060357922622136E-149", "-7.060357922622136E-149", "2.1181073767866432E-148", "7.060357922622136E-149"],
                ["-1.0480340939535158E-20", "-7.060357922622136E-149", "7.060357922622136E-149", "7.060357922622136E-149", "2.1181073767866432E-148"],
            ],
            "-1.812747161683739E+112",
            "5.180313214203733E+137"),
        "b2all-58#2" => new HeldOutProblem(
            name,
            "double",
            [
                ["1.444765204704803E-103", "1.4447652047048028E-103", "5.225555924775371E-116", "27000499.302796084"],
                ["1.4447652047048028E-103", "1.444765204704803E-103", "-5.225555924775371E-116", "-27000499.302796084"],
                ["5.225555924775371E-116", "-5.225555924775371E-116", "2.121483687017583E-111", "0"],
                ["27000499.302796084", "-27000499.302796084", "0", "1.6946148566801546E+134"],
            ],
            "-1.0347090339649893E+142",
            "1.167820423506846E+142"),
        "b2all-256#2" => new HeldOutProblem(
            name,
            "double",
            [
                ["6.361264850021478E-104", "6.361264850021477E-104", "1.4072849659489092E-121", "1.979066558802193E-08"],
                ["6.361264850021477E-104", "6.361264850021478E-104", "-1.4072849659489092E-121", "-1.979066558802193E-08"],
                ["1.4072849659489092E-121", "-1.4072849659489092E-121", "1.5040580591642194E-122", "0"],
                ["1.979066558802193E-08", "-1.979066558802193E-08", "0", "2.627914784223261E+104"],
            ],
            "-7.833870272127124E+146",
            "6.257504218561206E+146"),
        "b2zc-57#3" => new HeldOutProblem(
            name,
            "double",
            [
                ["5.270435198544766E+107", "-3.530391742090507E-26", "3.530391742090507E-26", "3.530391742090507E-26", "-3.530391742090507E-26"],
                ["-3.530391742090507E-26", "2.5859211576794033E-142", "8.619737192264676E-143", "8.619737192264676E-143", "-8.619737192264676E-143"],
                ["3.530391742090507E-26", "8.619737192264676E-143", "2.5859211576794033E-142", "-8.619737192264676E-143", "8.619737192264676E-143"],
                ["3.530391742090507E-26", "8.619737192264676E-143", "-8.619737192264676E-143", "2.5859211576794033E-142", "8.619737192264676E-143"],
                ["-3.530391742090507E-26", "-8.619737192264676E-143", "8.619737192264676E-143", "8.619737192264676E-143", "2.5859211576794033E-142"],
            ],
            "-1.9355800066314786E+124",
            "5.844122145476373E+127"),
        "b2zc-115#3" => new HeldOutProblem(
            name,
            "double",
            [
                ["2.817206895202302E+139", "1.1755222743587216E-09", "-1.1755222743587216E-09", "1.1755222743587216E-09", "-1.1755222743587216E-09"],
                ["1.1755222743587216E-09", "1.4245108048260116E-141", "4.748369349420038E-142", "-4.748369349420038E-142", "4.748369349420038E-142"],
                ["-1.1755222743587216E-09", "4.748369349420038E-142", "1.4245108048260116E-141", "4.748369349420038E-142", "-4.748369349420038E-142"],
                ["1.1755222743587216E-09", "-4.748369349420038E-142", "4.748369349420038E-142", "1.4245108048260116E-141", "4.748369349420038E-142"],
                ["-1.1755222743587216E-09", "4.748369349420038E-142", "-4.748369349420038E-142", "4.748369349420038E-142", "1.4245108048260116E-141"],
            ],
            "-8.066652328217586E+131",
            "2.951882292780023E+140"),
        "b2zc-302#0" => new HeldOutProblem(
            name,
            "double",
            [
                ["5.37290657430612E+125", "4.0985647495832944E-20", "-4.0985647495832944E-20", "4.0985647495832944E-20", "-4.0985647495832944E-20"],
                ["4.0985647495832944E-20", "8.471259025279296E-149", "2.823753008426431E-149", "-2.823753008426431E-149", "2.823753008426431E-149"],
                ["-4.0985647495832944E-20", "2.823753008426431E-149", "8.471259025279296E-149", "2.823753008426431E-149", "-2.823753008426431E-149"],
                ["4.0985647495832944E-20", "-2.823753008426431E-149", "2.823753008426431E-149", "8.471259025279296E-149", "2.823753008426431E-149"],
                ["-4.0985647495832944E-20", "2.823753008426431E-149", "-2.823753008426431E-149", "2.823753008426431E-149", "8.471259025279296E-149"],
            ],
            "-1.0628821245589749E+111",
            "5.13201021500903E+111"),
        "b2zc2-133#3" => new HeldOutProblem(
            name,
            "double",
            [
                ["1.1360252394740702E-105", "1.13602523947407E-105", "9.829125367811154E-132", "0.42962716413286434"],
                ["1.13602523947407E-105", "1.1360252394740702E-105", "-9.829125367811154E-132", "-0.42962716413286434"],
                ["9.829125367811154E-132", "-9.829125367811154E-132", "2.469744103202485E-141", "0"],
                ["0.42962716413286434", "-0.42962716413286434", "0", "9.172157423925018E+120"],
            ],
            "-5.048352946084747E+146",
            "3.7874421113889157E+146"),
        "b2zc2-138#2" => new HeldOutProblem(
            name,
            "double",
            [
                ["1.209451576749324E-124", "1.2094515767493236E-124", "1.0920567071787496E-137", "4.310981734512907E-11"],
                ["1.2094515767493236E-124", "1.209451576749324E-124", "-1.0920567071787496E-137", "-4.310981734512907E-11"],
                ["1.0920567071787496E-137", "-1.0920567071787496E-137", "2.0400861913448223E-134", "0"],
                ["4.310981734512907E-11", "-4.310981734512907E-11", "0", "1.93063360050203E+119"],
            ],
            "-4.665838736090831E+131",
            "7.362912272022498E+131"),
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unknown witness"),
    };

    private static HeldOutProblem Diagonal(string name, string type, string[] variances, string lb, string ub)
    {
        var n = variances.Length;
        var cov = new string[n][];
        for (var i = 0; i < n; i++)
        {
            cov[i] = Enumerable.Repeat("0", n).ToArray();
            cov[i][i] = variances[i];
        }

        return new HeldOutProblem(name, type, cov, lb, ub);
    }

    // Σᵢⱼ = (i ≡ j mod groups ? c : c/2) + (i = j ? d : 0), formed in the working type.
    private static HeldOutProblem RepeatedStructure(string name, string type, int n, int groups, string c, string d, string lb, string ub)
    {
        var cov = new string[n][];
        for (var i = 0; i < n; i++)
        {
            cov[i] = new string[n];
            for (var j = 0; j < n; j++)
            {
                var same = i % groups == j % groups;
                cov[i][j] = type == "double"
                    ? ((same ? ParseDouble(c) : ParseDouble(c) / 2) + (i == j ? ParseDouble(d) : 0.0)).ToString("R", CultureInfo.InvariantCulture)
                    : ((same ? ParseDecimal(c) : ParseDecimal(c) / 2) + (i == j ? ParseDecimal(d) : 0m)).ToString(CultureInfo.InvariantCulture);
            }
        }

        return new HeldOutProblem(name, type, cov, lb, ub);
    }

    /// <summary>
    /// The corpus-membership margin: a case is qualified only when <c>48·n·u·κ_eq &lt; 1</c>, where <c>κ_eq</c> is the
    /// exact 1-norm condition number of its free block after the covariance-only equilibration the solver applies and
    /// its conditioning guard measures (<c>QpScaling&lt;T&gt;.EquilibrateCovariance</c>, see <see cref="EquilibratedCovariance"/>).
    /// </summary>
    private static bool QualifiesAtTheMargin(int n, Rational unitRoundoff, Rational condEq) =>
        Rational.FromInt(48L * n) * unitRoundoff * condEq < Rational.One;

    /// <summary>
    /// The free block that decides the exact optimum <paramref name="w"/>, on which corpus membership is judged
    /// (section 4 with the deciding block <c>B = F*</c> of design step 4): the variables strictly inside the box. A
    /// variable the oracle leaves free but whose exact value lands on a bound with a zero multiplier (degenerate at the
    /// bound) is not in it, and at the exact optimum no variable on a bound has a negative multiplier, so none joins.
    /// </summary>
    private static int[] DecidingBlock(Rational[] w, Rational lb, Rational ub) =>
        Enumerable.Range(0, w.Length).Where(i => w[i] > lb && w[i] < ub).ToArray();

    /// <summary>
    /// <c>D⁻¹ΣD⁻¹</c> exactly, with <c>D = diag(radix^eᵢ)</c> the exponents of the covariance-only scaling
    /// <c>QpScaling&lt;T&gt;.EquilibrateCovariance</c> applied to the case's covariance: the scaling of design step 1
    /// (amendment #11) that the qualification definition is stated after and that the solver's conditioning guard
    /// measures. The bounds and the linear term take no part.
    /// </summary>
    private static Rational[,] EquilibratedCovariance(CorpusCase c, QpInstance q)
    {
        var exponents = c.Type == "double" ? ScalingExponents<double>(c, ParseDouble) : ScalingExponents<decimal>(c, ParseDecimal);
        var radix = c.Type == "double" ? 2 : 10;
        var n = q.N;
        var scaled = new Rational[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var k = exponents[i] + exponents[j];
                var power = BigInteger.Pow(radix, Math.Abs(k));
                scaled[i, j] = k >= 0 ? q.Cov[i, j] / Rational.Of(power, BigInteger.One) : q.Cov[i, j] * Rational.Of(power, BigInteger.One);
            }
        }

        return scaled;
    }

    private static int[] ScalingExponents<T>(CorpusCase c, Func<string, T> parse)
        where T : IFloatingPoint<T>
    {
        var n = c.N;
        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                cov[i, j] = parse(c.Cov[i][j]);
            }
        }

        return Boutquin.Numerics.Internal.QpScaling<T>.EquilibrateCovariance(cov).Exponents;
    }

    private static double ParseDouble(string s) => double.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture);

    private static decimal ParseDecimal(string s) => decimal.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture);

    // 1/k rounded in the working type, as the string the problem carries.
    private static string Reciprocal(string type, int k) => type == "double"
        ? (1.0 / k).ToString("R", CultureInfo.InvariantCulture)
        : (1m / k).ToString(CultureInfo.InvariantCulture);

    private static HeldOutProblem? Feasible(HeldOutProblem p)
    {
        var n = Rational.FromInt(p.Cov.Length);
        var lb = HeldOutParse(p.Type, p.Lb);
        var ub = HeldOutParse(p.Type, p.Ub);
        return lb * n > Rational.One || ub * n < Rational.One || lb > ub ? null : p;
    }

    private static Rational HeldOutParse(string type, string s) => type == "double"
        ? Rational.FromDouble(ParseDouble(s))
        : Rational.FromDecimal(ParseDecimal(s));

    private static HeldOutProblem? RepeatedStructureProblem(System.Random rng, string type, int trial)
    {
        var n = rng.Next(2, 13);
        var groups = rng.Next(1, 4);
        var c = new[] { "0.5", "0.95", "1.5", "2", "3" }[rng.Next(5)];
        var d = new[] { "1E-8", "1E-6", "1E-4", "0.05" }[rng.Next(4)];
        var (lb, ub) = rng.Next(5) switch
        {
            0 => (Reciprocal(type, n + 1), "1"),
            1 => ("0", n > 2 ? Reciprocal(type, n - 1) : "1"),
            2 => (Reciprocal(type, n + 1), n > 2 ? Reciprocal(type, n - 1) : "1"),
            3 => ("-0.5", n > 2 ? "0.5" : "1"),
            _ => (Reciprocal(type, 6), "1"),
        };

        return Feasible(RepeatedStructure("repeated-" + trial + "-n" + n + "-g" + groups + "-c" + c + "-d" + d, type, n, groups, c, d, lb, ub));
    }

    private static HeldOutProblem? DiagonalProblem(System.Random rng, string type, int trial)
    {
        var n = rng.Next(2, 13);
        var span = type == "double" ? new[] { 2, 20, 80, 149 }[rng.Next(4)] : new[] { 2, 6, 13 }[rng.Next(3)];
        var variances = new double[n];
        var cov = new string[n][];
        for (var i = 0; i < n; i++)
        {
            cov[i] = new string[n];
            var exponent = rng.Next(-span, span + 1);
            variances[i] = type == "double"
                ? Math.Pow(10, exponent + rng.NextDouble())
                : (double)ParseDecimal(rng.Next(1, 1_000_000).ToString(CultureInfo.InvariantCulture) + "E" + (exponent - 5).ToString(CultureInfo.InvariantCulture));
        }

        // Near-diagonal problems (every third trial) couple each pair at ρ·√(Σᵢᵢ·Σⱼⱼ) with ρ < 1/(2n).
        var rho = trial % 3 == 2 ? rng.NextDouble() / (2 * n) : 0.0;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var value = i == j ? variances[i] : rho * Math.Sqrt(variances[i]) * Math.Sqrt(variances[j]);
                cov[i][j] = type == "double"
                    ? value.ToString("R", CultureInfo.InvariantCulture)
                    : ((decimal)double.Parse(value.ToString("G12", CultureInfo.InvariantCulture), CultureInfo.InvariantCulture)).ToString(CultureInfo.InvariantCulture);
            }
        }

        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < i; j++)
            {
                cov[i][j] = cov[j][i];
            }
        }

        var (lb, ub) = rng.Next(4) switch
        {
            0 or 1 => ("0", "1"),
            2 => ("0", n > 2 ? Reciprocal(type, n - 1) : "1"),
            _ => ("-0.5", "1"),
        };

        var problem = Feasible(new HeldOutProblem("diag-" + trial + "-n" + n + "-span" + span + (rho > 0 ? "-near" : ""), type, cov, lb, ub));
        if (problem is null)
        {
            return null;
        }

        var q = QpInstance.From(HeldOutCase(problem));
        return q.InDomain(type) && ExactOracle.IsPositiveDefinite(q.Cov) ? problem : null;
    }

    private static CorpusCase HeldOutCase(HeldOutProblem p) => new()
    {
        Id = p.Name,
        Type = p.Type,
        Mode = p.Mu is null ? "minvar" : "meanvar",
        Cov = p.Cov,
        Mu = p.Mu ?? [],
        Lambda = p.Lambda ?? "",
        Lb = p.Lb,
        Ub = p.Ub,
        Representable = true,
    };

    internal static HeldOutResult SolveHeldOut(HeldOutProblem p) =>
        ClassifyHeldOut(p, () => CorpusRun.Solve(HeldOutCase(p)));

    // The output of solve, judged against the exact optimum with the AC-3.1 bound (a refusal is judged by the caller).
    private static HeldOutResult ClassifyHeldOut(HeldOutProblem p, Func<Rational[]> solve)
    {
        var c = HeldOutCase(p);
        var q = QpInstance.From(c);
        var (w0, pattern) = ExactOracle.ActiveSet(q);
        var free = DecidingBlock(w0, q.Lb, q.Ub);
        var u = p.Type == "double"
            ? Rational.Of(BigInteger.One, BigInteger.One << 53)
            : Rational.Of(BigInteger.One, BigInteger.Pow(10, 28));
        var inDomain = q.InDomain(p.Type) && w0.All(x => HeldOutRepresentable(p.Type, x));
        var qualified = inDomain && QualifiesAtTheMargin(q.N, u, ExactOracle.Cond1(EquilibratedCovariance(c, q), free));
        var crosses = inDomain && !qualified;
        var label = p.Name + " (" + p.Type + ")";
        Rational[] output;
        try
        {
            output = solve();
        }
        catch (InvalidOperationException ex)
        {
            var outcome = ex.Message.Contains("iteration budget", StringComparison.Ordinal) ? HeldOutOutcome.Exhausted : HeldOutOutcome.Refused;

            return new HeldOutResult(outcome, qualified, label + ": " + ex.Message, crosses);
        }
        catch (Exception ex)
        {
            return new HeldOutResult(HeldOutOutcome.Error, qualified, label + ": " + ex.GetType().Name + ": " + ex.Message, crosses);
        }

        var m = w0.Select(x => x.Abs()).Max();
        var floor = Rational.FromInt(4) * HeldOutUlp(p.Type, m);
        var error = w0.Select((x, i) => (output[i] - x).Abs()).Max();
        if (error <= floor)
        {
            return new HeldOutResult(HeldOutOutcome.Ok, qualified, label, crosses);
        }

        var sensitivity = ExactOracle.Sensitivity(q, c, w0, pattern, exactFallback: true);
        var scaled = sensitivity is null ? floor : Rational.FromInt(4) * sensitivity.Value * m;
        var bound = scaled > floor ? scaled : floor;
        return error <= bound
            ? new HeldOutResult(HeldOutOutcome.Ok, qualified, label, crosses)
            : new HeldOutResult(HeldOutOutcome.Wrong, qualified, label + ": error " + Scientific(error, 6) + " > bound " + Scientific(bound, 6), crosses);
    }

    private static bool HeldOutRepresentable(string type, Rational x)
    {
        if (x.Sign == 0)
        {
            return true;
        }

        var a = x.Abs();
        return type == "double"
            ? a < Rational.Of(BigInteger.One << 1023, BigInteger.One) && a > Rational.Of(BigInteger.One, BigInteger.One << 1022)
            : a < Rational.Of(BigInteger.Parse("79228162514264337593543950335", CultureInfo.InvariantCulture), BigInteger.One);
    }

    // One unit in the last place of the working type at the correctly rounded |m|.
    private static Rational HeldOutUlp(string type, Rational m)
    {
        if (type == "double")
        {
            var d = Math.Abs(double.Parse(Scientific(m, 40), NumberStyles.Float, CultureInfo.InvariantCulture));
            return Rational.FromDouble(Math.BitIncrement(d)) - Rational.FromDouble(d);
        }

        return ExactOracle.DecimalUnit(Math.Abs(decimal.Parse(Scientific(m, 28), NumberStyles.Float, CultureInfo.InvariantCulture)));
    }

    // A rational as a decimal scientific string with the given number of significant digits (truncated).
    private static string Scientific(Rational r, int digits)
    {
        var num = BigInteger.Abs(r.Num);
        if (num.IsZero)
        {
            return "0";
        }

        var magnitude = num.ToString(CultureInfo.InvariantCulture).Length - r.Den.ToString(CultureInfo.InvariantCulture).Length;
        var shift = digits - magnitude;
        var q = shift >= 0 ? num * BigInteger.Pow(10, shift) / r.Den : num / (r.Den * BigInteger.Pow(10, -shift));
        return (r.Sign < 0 ? "-" : "") + q.ToString(CultureInfo.InvariantCulture) + "E" + (-shift).ToString(CultureInfo.InvariantCulture);
    }

    private static readonly Lazy<NewCoreRun> s_newCore = new(() => NewCoreRun.Execute(s_run.Value));

    /// <summary>
    /// The production <see cref="ActiveSetQpSolver{T}.SolveMinVariance"/> on every min-variance case of the
    /// corpus, classified against the exact optimum with the AC-3.1 bound; also writes
    /// <c>observed-new-core.json</c> (per family and type: ok / refused / wrong, the float best-effort pass and
    /// the solve-time ratios) for the build report.
    /// </summary>
    private sealed class NewCoreRun
    {
        public bool Loaded { get; private set; }

        /// <summary>Per "family|type": cases, qualified, ok, refused, wrong, qualified_refused, qualified_wrong.</summary>
        public SortedDictionary<string, int[]> Table { get; } = new(StringComparer.Ordinal);

        public Dictionary<(string Mode, string Type, bool Qualified), CaseStats> ByType { get; } = [];

        public int FloatQualifiedCases { get; private set; }

        public List<string> FloatQualifiedRefusals { get; } = [];

        public Dictionary<string, double[]> SolveTimes { get; } = [];

        public double ElapsedSeconds { get; private set; }

        public double CpuSeconds { get; private set; }

        public CaseStats Stats(string type, bool qualified, string mode = "minvar") =>
            ByType.TryGetValue((mode, type, qualified), out var s) ? s : new CaseStats();

        public static NewCoreRun Execute(CorpusRun baseline)
        {
            var run = new NewCoreRun { Loaded = baseline.Loaded };
            if (!baseline.Loaded)
            {
                return run;
            }

            var clock = Stopwatch.StartNew();
            var cpu0 = Environment.CpuUsage.TotalTime;
            var times = new Dictionary<string, List<double>>(StringComparer.Ordinal);
            foreach (var c in baseline.Files.SelectMany(f => f.Cases))
            {
                var key = c.Mode + "|" + c.Family + "|" + c.Type;
                if (!run.Table.TryGetValue(key, out var row))
                {
                    row = new int[s_outcomeColumns.Length];
                    run.Table[key] = row;
                }

                if (!run.ByType.TryGetValue((c.Mode, c.Type, c.Qualified), out var stats))
                {
                    stats = new CaseStats();
                    run.ByType[(c.Mode, c.Type, c.Qualified)] = stats;
                }

                row[0]++;
                stats.Cases++;
                if (c.Qualified)
                {
                    row[1]++;
                }

                Rational[]? output = null;
                Exception? refusal = null;
                var t0 = Stopwatch.GetTimestamp();
                try
                {
                    output = CorpusRun.Solve(c);
                }
                catch (Exception ex)
                {
                    refusal = ex;
                }

                var ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                if (!times.TryGetValue(c.Type, out var list))
                {
                    list = [];
                    times[c.Type] = list;
                }

                list.Add(ms);
                if (refusal is not null)
                {
                    row[3]++;
                    if (c.Qualified)
                    {
                        row[5]++;
                    }

                    stats.Refused.Add(refusal.GetType().Name + ": " + c.Id + ": " + refusal.Message);
                }
                else if (baseline.ExactOptima.TryGetValue(c.Id, out var w0) && CorpusRun.Classify(c, w0, output!) == Kind.Ok)
                {
                    row[2]++;
                    stats.Ok++;
                    var needed = CorpusRun.NeededToleranceScale(c);
                    if (needed > stats.MaxNeededKappa)
                    {
                        stats.MaxNeededKappa = needed;
                        stats.MaxNeededKappaCase = c.Id;
                    }
                }
                else
                {
                    row[4]++;
                    if (c.Qualified)
                    {
                        row[6]++;
                    }

                    stats.Wrong.Add(c.Id);
                }

                if (c.Type == "double" && c.Qualified && c.Mode == "minvar")
                {
                    try
                    {
                        if (CorpusRun.SolveAsFloat(c) is not null)
                        {
                            run.FloatQualifiedCases++;
                        }
                    }
                    catch (Exception ex)
                    {
                        // Inputs passed the float-domain filter, so the exception is the solver's.
                        run.FloatQualifiedCases++;
                        run.FloatQualifiedRefusals.Add(c.Id + ": " + ex.GetType().Name + ": " + ex.Message);
                    }
                }
            }

            foreach (var (type, list) in times)
            {
                list.Sort();
                var p99Index = Math.Max(0, (int)Math.Ceiling(0.99 * list.Count) - 1);
                var median = list.Count % 2 == 1 ? list[list.Count / 2] : (list[(list.Count / 2) - 1] + list[list.Count / 2]) / 2;
                run.SolveTimes[type] = [median, list[p99Index], list.Count];
            }

            run.MeasureM1Ratios(baseline);
            run.ElapsedSeconds = clock.Elapsed.TotalSeconds;
            run.CpuSeconds = (Environment.CpuUsage.TotalTime - cpu0).TotalSeconds;
            run.WriteObserved(CorpusRun.CalibrationWorkloadMs());
            return run;
        }

        /// <summary>Per type: median and p99 calibrated ratio on the M-1 case set, and the case count.</summary>
        public Dictionary<string, double[]> M1Ratios { get; } = [];

        // The solve-time ratios on the M-1 case set with M-1's estimator: every M-1 case (min-variance through the
        // production entry point, mean-variance as it stands) timed three times, keeping its minimum; median and p99 of
        // those minima, each divided by the calibration workload timed in the same run.
        private void MeasureM1Ratios(CorpusRun baseline)
        {
            var calibration = CorpusRun.CalibrationWorkloadMs();
            foreach (var file in baseline.Files)
            {
                var minima = new List<double>();
                foreach (var c in file.Cases.Where(c => s_m1Families.Contains(c.Family)))
                {
                    var best = double.PositiveInfinity;
                    for (var rep = 0; rep < 3; rep++)
                    {
                        var t0 = Stopwatch.GetTimestamp();
                        try
                        {
                            _ = CorpusRun.Solve(c);
                        }
                        catch (InvalidOperationException)
                        {
                            // A refusal is timed like a result.
                        }

                        best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
                    }

                    minima.Add(best);
                }

                minima.Sort();
                var p99Index = Math.Max(0, (int)Math.Ceiling(0.99 * minima.Count) - 1);
                var median = minima.Count % 2 == 1 ? minima[minima.Count / 2] : (minima[(minima.Count / 2) - 1] + minima[minima.Count / 2]) / 2;
                M1Ratios[file.Type] = [median / calibration, minima[p99Index] / calibration, minima.Count];
            }
        }

        private void WriteObserved(double calibrationMs)
        {
            using var stream = new MemoryStream();
            using (var w = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
            {
                w.WriteStartObject();
                w.WriteString("solver", "ActiveSetQpSolver<T>.SolveMinVariance and SolveMeanVariance (production entry points); families keyed mode|family|type");
                w.WriteStartArray("families");
                foreach (var (key, row) in Table)
                {
                    var parts = key.Split('|');
                    w.WriteStartObject();
                    w.WriteString("mode", parts[0]);
                    w.WriteString("family", parts[1]);
                    w.WriteString("type", parts[2]);
                    for (var i = 0; i < s_outcomeColumns.Length; i++)
                    {
                        w.WriteNumber(s_outcomeColumns[i], row[i]);
                    }

                    w.WriteEndObject();
                }

                w.WriteEndArray();
                w.WriteStartObject("needed_tolerance_scale");
                foreach (var ((mode, type, qualified), stats) in ByType.OrderBy(kv => kv.Key.Mode, StringComparer.Ordinal).ThenBy(kv => kv.Key.Type, StringComparer.Ordinal).ThenBy(kv => kv.Key.Qualified))
                {
                    w.WriteStartObject(mode + "|" + type + (qualified ? "|qualified" : "|refusal-permitted"));
                    w.WriteNumber("max", stats.MaxNeededKappa);
                    w.WriteString("case", stats.MaxNeededKappaCase);
                    w.WriteEndObject();
                }

                w.WriteEndObject();
                w.WriteNumber("float_qualified_cases", FloatQualifiedCases);
                w.WriteNumber("float_qualified_refusals", FloatQualifiedRefusals.Count);
                w.WriteStartArray("float_refusal_sample");
                foreach (var r in FloatQualifiedRefusals.Take(10))
                {
                    w.WriteStringValue(r);
                }

                w.WriteEndArray();
                w.WriteStartArray("refusal_sample");
                foreach (var r in ByType.Values.SelectMany(s => s.Refused).Take(25))
                {
                    w.WriteStringValue(r);
                }

                w.WriteEndArray();
                w.WriteStartObject("timing_single_pass");
                w.WriteNumber("calibration_ms", Math.Round(calibrationMs, 4));
                foreach (var (type, t) in SolveTimes)
                {
                    w.WriteStartObject(type);
                    w.WriteNumber("cases", (int)t[2]);
                    w.WriteNumber("median_ms", Math.Round(t[0], 6));
                    w.WriteNumber("p99_ms", Math.Round(t[1], 6));
                    w.WriteNumber("median_ratio", Math.Round(t[0] / calibrationMs, 6));
                    w.WriteNumber("p99_ratio", Math.Round(t[1] / calibrationMs, 6));
                    w.WriteEndObject();
                }

                w.WriteEndObject();
                w.WriteStartObject("m1_case_set_min_of_three");
                foreach (var (type, r) in M1Ratios)
                {
                    w.WriteStartObject(type);
                    w.WriteNumber("cases", (int)r[2]);
                    w.WriteNumber("median_ratio", Math.Round(r[0], 6));
                    w.WriteNumber("p99_ratio", Math.Round(r[1], 6));
                    w.WriteEndObject();
                }

                w.WriteEndObject();
                w.WriteEndObject();
            }

            Directory.CreateDirectory(CorpusRun.FixtureDirectory);
            File.WriteAllText(Path.Combine(CorpusRun.FixtureDirectory, "observed-new-core.json"), Encoding.UTF8.GetString(stream.ToArray()));
        }
    }

    private sealed class CaseStats
    {
        public int Cases { get; set; }

        public int Ok { get; set; }

        public List<string> Wrong { get; } = [];

        public List<string> Refused { get; } = [];

        /// <summary>Largest smallest-accepting tolerance scale (a power of two) over the correct outputs.</summary>
        public double MaxNeededKappa { get; set; }

        public string MaxNeededKappaCase { get; set; } = "";

        public string Describe() =>
            Cases + " cases, " + Ok + " ok, " + Wrong.Count + " wrong [" + string.Join(", ", Wrong.Take(15)) + "], "
            + Refused.Count + " refused [" + string.Join(" | ", Refused.Take(8)) + "]";
    }

    private static readonly string[] s_outcomeColumns = ["cases", "qualified", "ok", "refused", "wrong", "qualified_refused", "qualified_wrong"];

    private sealed class PerformanceBaseline
    {
        public Dictionary<string, double[]> Ratios { get; } = [];

        public List<string> Errors { get; } = [];

        public int CorpusCases { get; private set; }

        public int M1CorpusCases { get; private set; }

        public double? MeasuredRuntimeSeconds { get; private set; }

        public string CalibrationWorkload { get; private set; } = "";

        public double CalibrationMs { get; private set; }

        public static PerformanceBaseline Load(string path)
        {
            var file = new PerformanceBaseline();
            using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
            var root = doc.RootElement;
            file.CorpusCases = root.TryGetProperty("corpus_cases", out var cc) ? cc.GetInt32() : -1;
            file.M1CorpusCases = root.TryGetProperty("m1_corpus_cases", out var m1) ? m1.GetInt32() : -1;
            file.MeasuredRuntimeSeconds = root.TryGetProperty("measured_runtime_seconds", out var rt) ? rt.GetDouble() : null;
            if (root.TryGetProperty("timing", out var timing))
            {
                var calibration = timing.GetProperty("calibration");
                file.CalibrationWorkload = calibration.GetProperty("workload").GetString()!;
                file.CalibrationMs = calibration.GetProperty("ms").GetDouble();
                foreach (var type in new[] { "double", "decimal" })
                {
                    var t = timing.GetProperty(type);
                    file.Ratios[type] =
                    [
                        t.GetProperty("median_ratio").GetDouble(),
                        t.GetProperty("p99_ratio").GetDouble(),
                        t.GetProperty("cases").GetInt32(),
                    ];
                }
            }
            else
            {
                file.Errors.Add("no timing object");
            }

            return file;
        }
    }

    // ------------------------------------------------------------------------------------------
    // Pipeline: load -> exact oracle verification -> production solver pass -> timing
    // ------------------------------------------------------------------------------------------

    private enum Kind
    {
        Ok,
        Refused,
        Wrong,
    }

    private sealed record FamilySpec(string Family, string Generator, List<int> Seeds, int Count);

    private sealed class CorpusCase
    {
        public string Id { get; set; } = "";

        public string Family { get; set; } = "";

        public string Type { get; set; } = "";

        public string Mode { get; set; } = "";

        public string[][] Cov { get; set; } = [];

        public string[] Mu { get; set; } = [];

        public string Lambda { get; set; } = "";

        public string Lb { get; set; } = "";

        public string Ub { get; set; } = "";

        public string Pattern { get; set; } = "";

        public string[] Optimum { get; set; } = [];

        public bool InDomain { get; set; }

        public bool Representable { get; set; }

        public string Cond1 { get; set; } = "";

        public string CondEq { get; set; } = "";

        public bool Qualified { get; set; }

        public string Sensitivity { get; set; } = "";

        public int N => Cov.Length;
    }

    private sealed class CorpusFile
    {
        public string Type { get; set; } = "";

        public List<FamilySpec> Families { get; } = [];

        public List<CorpusCase> Cases { get; } = [];
    }

    private sealed class CorpusRun
    {
        public bool Loaded { get; private set; }

        public string LoadReport { get; private set; } = "";

        public List<string> LoadErrors { get; } = [];

        public List<CorpusFile> Files { get; } = [];

        public Dictionary<string, Rational[]> ExactOptima { get; } = [];

        public List<string> OptimumFailures { get; } = [];

        public List<string> FlagFailures { get; } = [];

        public List<string> SensitivityFailures { get; } = [];

        public HashSet<string> SensitivityChecked { get; } = [];

        public double ElapsedSeconds { get; private set; }

        public double CpuSeconds { get; private set; }

        public static string FixtureDirectory => Path.Combine(AppContext.BaseDirectory, "Verification", "qp-corpus");

        public static CorpusRun Execute()
        {
            var run = new CorpusRun();
            var clock = Stopwatch.StartNew();
            var cpu0 = Environment.CpuUsage.TotalTime;
            run.Load();
            if (run.Loaded)
            {
                run.VerifyExactOptima();
                run.VerifyFlags();
                run.VerifySensitivitySample();
            }

            run.ElapsedSeconds = clock.Elapsed.TotalSeconds;
            run.CpuSeconds = (Environment.CpuUsage.TotalTime - cpu0).TotalSeconds;
            return run;
        }

        // ---- production solver pass, classification, timing ----------------------------------

        private static Rational[] RunSolver<T>(CorpusCase c, Func<string, T> parse, Func<T, Rational?> toRational, int forcedReference = -1)
            where T : IFloatingPoint<T>
        {
            var n = c.N;
            var cov = new T[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    cov[i, j] = parse(c.Cov[i][j]);
                }
            }

            var lb = parse(c.Lb);
            var ub = parse(c.Ub);
            var weights = c.Mode == "minvar"
                ? forcedReference >= 0
                    ? ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(cov, lb, ub, forcedReference: forcedReference)
                    : ActiveSetQpSolver<T>.SolveMinVariance(cov, lb, ub)
                : ActiveSetQpSolver<T>.SolveMeanVariance(cov, c.Mu.Select(parse).ToArray(), parse(c.Lambda), lb, ub);
            var result = new Rational[weights.Length];
            for (var i = 0; i < weights.Length; i++)
            {
                var r = toRational(weights[i]);
                if (r is null)
                {
                    throw new ArithmeticException("non-finite weight returned");
                }

                result[i] = r.Value;
            }

            return result;
        }

        public static Rational[] Solve(CorpusCase c, int forcedReference = -1) => c.Type == "double"
            ? RunSolver<double>(
                c,
                s => double.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture),
                x => double.IsFinite(x) ? Rational.FromDouble(x) : null,
                forcedReference)
            : RunSolver<decimal>(
                c,
                s => decimal.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture),
                x => Rational.FromDecimal(x),
                forcedReference);

        /// <summary>
        /// The smallest power-of-two tolerance scale at which the independent certificate accepts the production
        /// solver's output for this case (bounds clamped to the equal-weight frontier as the solver documents).
        /// </summary>
        public static double NeededToleranceScale(CorpusCase c) => c.Type == "double"
            ? NeededToleranceScale<double>(c, s => double.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture))
            : NeededToleranceScale<decimal>(c, s => decimal.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture));

        private static double NeededToleranceScale<T>(CorpusCase c, Func<string, T> parse)
            where T : IFloatingPoint<T>
        {
            var n = c.N;
            var cov = new T[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    cov[i, j] = parse(c.Cov[i][j]);
                }
            }

            var inverse = T.One / T.CreateChecked(n);
            var lower = T.Min(parse(c.Lb), inverse);
            var upper = T.Max(parse(c.Ub), inverse);
            var minVariance = c.Mode == "minvar";
            var means = minVariance ? null : c.Mu.Select(parse).ToArray();
            var riskAversion = minVariance ? T.One : parse(c.Lambda);
            var weights = minVariance
                ? ActiveSetQpSolver<T>.SolveMinVariance(cov, parse(c.Lb), parse(c.Ub))
                : ActiveSetQpSolver<T>.SolveMeanVariance(cov, means, riskAversion, parse(c.Lb), parse(c.Ub));
            var kappa = T.One;
            for (var k = 0; k < 64; k++)
            {
                if (Boutquin.Numerics.Internal.QpKktCertificate<T>.Check(cov, means, riskAversion, lower, upper, weights, kappa).Accepted)
                {
                    return double.CreateChecked(kappa);
                }

                kappa += kappa;
            }

            return double.PositiveInfinity;
        }

        /// <summary>
        /// The float best-effort pass: the double case's inputs rounded to float, through the production entry
        /// point; <see langword="null"/> when a rounded input leaves float's operating domain (zero or
        /// <c>1e-15 ≤ |x| ≤ 1e15</c>), where float is not exercised.
        /// </summary>
        public static float[]? SolveAsFloat(CorpusCase c)
        {
            static bool InDomain(float x) => x == 0f || (Math.Abs(x) >= 1e-15f && Math.Abs(x) <= 1e15f);
            var n = c.N;
            var cov = new float[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    cov[i, j] = float.Parse(c.Cov[i][j], NumberStyles.Float, CultureInfo.InvariantCulture);
                    if (!InDomain(cov[i, j]))
                    {
                        return null;
                    }
                }
            }

            if (!InDomain(float.Parse(c.Lb, NumberStyles.Float, CultureInfo.InvariantCulture))
                || !InDomain(float.Parse(c.Ub, NumberStyles.Float, CultureInfo.InvariantCulture)))
            {
                return null;
            }

            return ActiveSetQpSolver<float>.SolveMinVariance(
                cov,
                float.Parse(c.Lb, NumberStyles.Float, CultureInfo.InvariantCulture),
                float.Parse(c.Ub, NumberStyles.Float, CultureInfo.InvariantCulture));
        }

        /// <summary>Accuracy bound of AC-3.1: max(4 x the case's sensitivity bound, 4 ulps of the working type), absolute.</summary>
        private static Rational Bound(CorpusCase c, Rational[] w0)
        {
            var maxIndex = 0;
            for (var i = 1; i < w0.Length; i++)
            {
                if (w0[i].Abs() > w0[maxIndex].Abs())
                {
                    maxIndex = i;
                }
            }

            var m = w0[maxIndex].Abs();
            Rational ulp;
            if (c.Type == "double")
            {
                var md = Math.Abs(double.Parse(c.Optimum[maxIndex], NumberStyles.Float, CultureInfo.InvariantCulture));
                ulp = Rational.FromDouble(Math.BitIncrement(md) - md);
            }
            else
            {
                ulp = ExactOracle.DecimalUnit(Math.Abs(decimal.Parse(c.Optimum[maxIndex], NumberStyles.Float, CultureInfo.InvariantCulture)));
            }

            var floor = Rational.FromInt(4) * ulp;
            var sensitivity = double.Parse(c.Sensitivity, NumberStyles.Float, CultureInfo.InvariantCulture);
            if (!double.IsFinite(sensitivity))
            {
                return Rational.Of(BigInteger.Pow(10, 400), BigInteger.One);
            }

            var scaled = Rational.FromDouble(sensitivity) * Rational.FromInt(4) * m;
            return scaled > floor ? scaled : floor;
        }

        public static Kind Classify(CorpusCase c, Rational[] w0, Rational[] output)
        {
            if (!c.Representable || output.Length != w0.Length)
            {
                return Kind.Wrong;
            }

            var bound = Bound(c, w0);
            for (var i = 0; i < w0.Length; i++)
            {
                if ((output[i] - w0[i]).Abs() > bound)
                {
                    return Kind.Wrong;
                }
            }

            return Kind.Ok;
        }

        public static double CalibrationWorkloadMs() => MeasureCalibrationMs();

        /// <summary>The AC-5.1 calibration workload: a 200 x 200 CholeskyDecomposition&lt;double&gt; factorisation.</summary>
        private static double MeasureCalibrationMs()
        {
            const int n = 200;
            var rng = new System.Random(20261008);
            var b = new double[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    b[i, j] = rng.NextDouble() - 0.5;
                }
            }

            var a = new double[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    var sum = i == j ? n : 0.0;
                    for (var k = 0; k < n; k++)
                    {
                        sum += b[i, k] * b[j, k];
                    }

                    a[i, j] = sum;
                }
            }

            var best = double.PositiveInfinity;
            for (var rep = 0; rep < 7; rep++)
            {
                var t0 = Stopwatch.GetTimestamp();
                _ = CholeskyDecomposition<double>.Decompose(a);
                best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
            }

            return best;
        }

        private static string Req(JsonElement e, string name, List<string> errors, string ctx)
        {
            if (e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String)
            {
                return v.GetString()!;
            }

            errors.Add(ctx + ": missing string field '" + name + "'");
            return "";
        }

        private static bool ReqBool(JsonElement e, string name, List<string> errors, string ctx)
        {
            if (e.TryGetProperty(name, out var v) && (v.ValueKind == JsonValueKind.True || v.ValueKind == JsonValueKind.False))
            {
                return v.GetBoolean();
            }

            errors.Add(ctx + ": missing boolean field '" + name + "'");
            return false;
        }

        private void Load()
        {
            var report = new StringBuilder();
            var any = false;
            foreach (var type in new[] { "double", "decimal" })
            {
                var path = Path.Combine(FixtureDirectory, "corpus-" + type + ".json");
                if (!File.Exists(path))
                {
                    report.Append("missing ").Append(path).Append("; ");
                    continue;
                }

                any = true;
                using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
                var root = doc.RootElement;
                var file = new CorpusFile { Type = type };
                if (root.TryGetProperty("families", out var fams))
                {
                    foreach (var f in fams.EnumerateArray())
                    {
                        var seeds = new List<int>();
                        if (f.TryGetProperty("seeds", out var sd))
                        {
                            seeds.AddRange(sd.EnumerateArray().Select(x => x.GetInt32()));
                        }

                        file.Families.Add(new FamilySpec(
                            Req(f, "family", LoadErrors, type + " family"),
                            Req(f, "generator", LoadErrors, type + " family"),
                            seeds,
                            f.TryGetProperty("count", out var cnt) ? cnt.GetInt32() : -1));
                    }
                }
                else
                {
                    LoadErrors.Add(type + ": no families array");
                }

                if (root.TryGetProperty("cases", out var cs))
                {
                    foreach (var c in cs.EnumerateArray())
                    {
                        file.Cases.Add(ParseCase(c, type));
                    }
                }
                else
                {
                    LoadErrors.Add(type + ": no cases array");
                }

                Files.Add(file);
            }

            Loaded = any && Files.Count == 2;
            LoadReport = report.ToString();
        }

        private CorpusCase ParseCase(JsonElement c, string type)
        {
            var id = Req(c, "id", LoadErrors, type + " case");
            var cc = new CorpusCase
            {
                Id = id,
                Type = type,
                Family = Req(c, "family", LoadErrors, id),
                Mode = Req(c, "mode", LoadErrors, id),
                Lb = Req(c, "lb", LoadErrors, id),
                Ub = Req(c, "ub", LoadErrors, id),
                Pattern = Req(c, "pattern", LoadErrors, id),
                Cond1 = Req(c, "cond1", LoadErrors, id),
                CondEq = Req(c, "cond_eq", LoadErrors, id),
                Sensitivity = Req(c, "sensitivity", LoadErrors, id),
                InDomain = ReqBool(c, "in_domain", LoadErrors, id),
                Representable = ReqBool(c, "representable", LoadErrors, id),
                Qualified = ReqBool(c, "qualified", LoadErrors, id),
            };
            if (c.TryGetProperty("cov", out var cov))
            {
                cc.Cov = cov.EnumerateArray().Select(r => r.EnumerateArray().Select(x => x.GetString()!).ToArray()).ToArray();
            }
            else
            {
                LoadErrors.Add(id + ": missing cov");
            }

            if (c.TryGetProperty("optimum", out var opt))
            {
                cc.Optimum = opt.EnumerateArray().Select(x => x.GetString()!).ToArray();
            }
            else
            {
                LoadErrors.Add(id + ": missing optimum");
            }

            if (cc.Mode == "meanvar")
            {
                cc.Lambda = Req(c, "lambda", LoadErrors, id);
                if (c.TryGetProperty("mu", out var mu))
                {
                    cc.Mu = mu.EnumerateArray().Select(x => x.GetString()!).ToArray();
                }
                else
                {
                    LoadErrors.Add(id + ": missing mu");
                }
            }

            if (cc.Cov.Length == 0 || cc.Cov.Any(r => r.Length != cc.Cov.Length) || cc.Optimum.Length != cc.Cov.Length
                || cc.Pattern.Length != cc.Cov.Length)
            {
                LoadErrors.Add(id + ": inconsistent dimensions");
            }

            return cc;
        }

        // ---- exact oracle verification --------------------------------------------------------

        private static void Fail(List<string> sink, string message)
        {
            if (sink.Count < 25)
            {
                sink.Add(message);
            }
        }

        private static int[] ToPattern(string p) => p.Select(ch => ch == '-' ? -1 : ch == '+' ? 1 : 0).ToArray();

        private void VerifyExactOptima()
        {
            foreach (var file in Files)
            {
                foreach (var c in file.Cases)
                {
                    var q = QpInstance.From(c);
                    var opt = ExactOracle.Optimum(q, ToPattern(c.Pattern));
                    if (opt is null)
                    {
                        Fail(OptimumFailures, c.Id + ": the committed active set is not a KKT point of the exact problem");
                        continue;
                    }

                    var (w, _) = opt.Value;
                    ExactOptima[c.Id] = w;
                    for (var i = 0; i < w.Length; i++)
                    {
                        var committed = Rational.ParseScientific(c.Optimum[i]);
                        var tol = w[i].Abs() * Rational.ParseScientific("1e-58");
                        if ((committed - w[i]).Abs() > tol)
                        {
                            Fail(OptimumFailures, c.Id + ": committed optimum[" + i + "] differs from the exact optimum");
                            break;
                        }
                    }
                }
            }
        }

        private void VerifyFlags()
        {
            foreach (var file in Files)
            {
                var unit = file.Type == "double" ? Rational.Of(1, BigInteger.One << 53) : Rational.Of(1, BigInteger.Pow(10, 28));
                var maxMagnitude = file.Type == "double" ? Rational.FromDouble(double.MaxValue) : Rational.FromDecimal(decimal.MaxValue);
                foreach (var c in file.Cases)
                {
                    if (!ExactOptima.TryGetValue(c.Id, out var w))
                    {
                        continue;
                    }

                    var q = QpInstance.From(c);
                    var inDomain = q.InDomain(c.Type);
                    var representable = w.All(x => x.Abs() <= maxMagnitude);
                    var free = DecidingBlock(w, q.Lb, q.Ub);
                    var cond = ExactOracle.Cond1(q.Cov, free);
                    var condEq = ExactOracle.Cond1(EquilibratedCovariance(c, q), free);
                    var qualified = inDomain && representable && QualifiesAtTheMargin(q.N, unit, condEq);
                    // A missing cond_eq is reported by the loader (LoadErrors); only a present value is compared.
                    if (c.CondEq.Length > 0 && c.CondEq != "inf")
                    {
                        var committedEq = Rational.FromDouble(double.Parse(c.CondEq, CultureInfo.InvariantCulture));
                        if ((committedEq - condEq).Abs() > condEq * Rational.ParseScientific("1e-9"))
                        {
                            Fail(FlagFailures, c.Id + ": committed cond_eq " + c.CondEq + " differs from the exact condition number of the equilibrated free block");
                        }
                    }

                    if (inDomain != c.InDomain)
                    {
                        Fail(FlagFailures, c.Id + ": in_domain flag disagrees with the exact definition");
                    }

                    if (representable != c.Representable)
                    {
                        Fail(FlagFailures, c.Id + ": representable flag disagrees with the exact definition");
                    }

                    if (qualified != c.Qualified)
                    {
                        Fail(FlagFailures, c.Id + ": qualified flag disagrees with the exact definition 48·n·u·cond_eq < 1 (cond1=" + c.Cond1 + ", cond_eq=" + c.CondEq + ")");
                    }

                    if (c.Cond1 != "inf")
                    {
                        var committed = Rational.FromDouble(double.Parse(c.Cond1, CultureInfo.InvariantCulture));
                        if ((committed - cond).Abs() > cond * Rational.ParseScientific("1e-9"))
                        {
                            Fail(FlagFailures, c.Id + ": committed cond1 " + c.Cond1 + " differs from the exact condition number");
                        }
                    }
                }
            }
        }

        private void VerifySensitivitySample()
        {
            var witnesses = new HashSet<string>(s_requiredWitnesses, StringComparer.Ordinal);
            var index = 0;
            foreach (var file in Files)
            {
                foreach (var c in file.Cases)
                {
                    index++;
                    var take = witnesses.Contains(c.Id) || (index % 9 == 0 && c.N <= 5);
                    if (!take || !ExactOptima.TryGetValue(c.Id, out var w))
                    {
                        continue;
                    }

                    var q = QpInstance.From(c);
                    var result = ExactOracle.Sensitivity(q, c, w, ToPattern(c.Pattern));
                    if (result is null)
                    {
                        Fail(SensitivityFailures, c.Id + ": sensitivity could not be recomputed");
                        continue;
                    }

                    SensitivityChecked.Add(c.Id);
                    var committed = Rational.FromDouble(double.Parse(c.Sensitivity, CultureInfo.InvariantCulture));
                    var diff = (committed - result.Value).Abs();
                    var scale = committed.Abs() > result.Value.Abs() ? committed.Abs() : result.Value.Abs();
                    if (diff > scale * Rational.ParseScientific("1e-9"))
                    {
                        Fail(SensitivityFailures, c.Id + ": committed sensitivity " + c.Sensitivity + " differs from the recomputed value");
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // Exact rational arithmetic and the exact KKT oracle (shares no code with the solver)
    // ------------------------------------------------------------------------------------------

    private readonly struct Rational : IEquatable<Rational>, IComparable<Rational>
    {
        private readonly BigInteger _den;

        private Rational(BigInteger num, BigInteger den)
        {
            Num = num;
            _den = den;
        }

        public static Rational Zero => default;

        public static Rational One => new(BigInteger.One, BigInteger.One);

        public BigInteger Num { get; }

        public BigInteger Den => _den.IsZero ? BigInteger.One : _den;

        public int Sign => Num.Sign;

        public static Rational Of(BigInteger num, BigInteger den)
        {
            if (den.IsZero)
            {
                throw new DivideByZeroException();
            }

            if (den.Sign < 0)
            {
                num = -num;
                den = -den;
            }

            if (num.IsZero)
            {
                return default;
            }

            var g = BigInteger.GreatestCommonDivisor(num, den);
            if (!g.IsOne)
            {
                num /= g;
                den /= g;
            }

            return new Rational(num, den);
        }

        public static Rational FromInt(long value) => Of(value, BigInteger.One);

        public static Rational FromDouble(double x)
        {
            if (!double.IsFinite(x))
            {
                throw new ArgumentOutOfRangeException(nameof(x));
            }

            var bits = BitConverter.DoubleToInt64Bits(x);
            var negative = bits < 0;
            var exponent = (int)((bits >> 52) & 0x7FF);
            var mantissa = bits & 0xFFFFFFFFFFFFFL;
            if (exponent == 0)
            {
                exponent = 1;
            }
            else
            {
                mantissa |= 1L << 52;
            }

            exponent -= 1075;
            BigInteger num = mantissa;
            if (negative)
            {
                num = -num;
            }

            return exponent >= 0 ? Of(num << exponent, BigInteger.One) : Of(num, BigInteger.One << -exponent);
        }

        public static Rational FromDecimal(decimal x)
        {
            var bits = decimal.GetBits(x);
            var magnitude = (new BigInteger((uint)bits[2]) << 64) | (new BigInteger((uint)bits[1]) << 32) | new BigInteger((uint)bits[0]);
            var scale = (bits[3] >> 16) & 0xFF;
            if (bits[3] < 0)
            {
                magnitude = -magnitude;
            }

            return Of(magnitude, BigInteger.Pow(10, scale));
        }

        public static Rational ParseScientific(string s)
        {
            s = s.Trim();
            var exponent = 0;
            var e = s.IndexOfAny(['E', 'e']);
            if (e >= 0)
            {
                exponent = int.Parse(s[(e + 1)..], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
                s = s[..e];
            }

            var negative = s.StartsWith('-');
            if (negative || s.StartsWith('+'))
            {
                s = s[1..];
            }

            var dot = s.IndexOf('.', StringComparison.Ordinal);
            var fractionDigits = 0;
            if (dot >= 0)
            {
                fractionDigits = s.Length - dot - 1;
                s = s.Remove(dot, 1);
            }

            var digits = BigInteger.Parse(s, NumberStyles.None, CultureInfo.InvariantCulture);
            if (negative)
            {
                digits = -digits;
            }

            exponent -= fractionDigits;
            return exponent >= 0
                ? Of(digits * BigInteger.Pow(10, exponent), BigInteger.One)
                : Of(digits, BigInteger.Pow(10, -exponent));
        }

        public static Rational operator +(Rational a, Rational b) => Of((a.Num * b.Den) + (b.Num * a.Den), a.Den * b.Den);

        public static Rational operator -(Rational a, Rational b) => Of((a.Num * b.Den) - (b.Num * a.Den), a.Den * b.Den);

        public static Rational operator *(Rational a, Rational b) => Of(a.Num * b.Num, a.Den * b.Den);

        public static Rational operator /(Rational a, Rational b) => Of(a.Num * b.Den, a.Den * b.Num);

        public static Rational operator -(Rational a) => new(-a.Num, a._den);

        public static bool operator <(Rational a, Rational b) => a.CompareTo(b) < 0;

        public static bool operator >(Rational a, Rational b) => a.CompareTo(b) > 0;

        public static bool operator <=(Rational a, Rational b) => a.CompareTo(b) <= 0;

        public static bool operator >=(Rational a, Rational b) => a.CompareTo(b) >= 0;

        public static bool operator ==(Rational a, Rational b) => a.CompareTo(b) == 0;

        public static bool operator !=(Rational a, Rational b) => a.CompareTo(b) != 0;

        public Rational Abs() => Sign < 0 ? -this : this;

        public int CompareTo(Rational other) => (Num * other.Den).CompareTo(other.Num * Den);

        public bool Equals(Rational other) => CompareTo(other) == 0;

        public override bool Equals(object? obj) => obj is Rational r && Equals(r);

        public override int GetHashCode() => HashCode.Combine(Num, Den);

        public override string ToString() => Num.ToString(CultureInfo.InvariantCulture) + "/" + Den.ToString(CultureInfo.InvariantCulture);
    }

    private sealed class QpInstance
    {
        public Rational[,] Cov { get; private init; } = new Rational[0, 0];

        public Rational[] Mu { get; private init; } = [];

        public Rational Lambda { get; private init; }

        public Rational Lb { get; private init; }

        public Rational Ub { get; private init; }

        public int N => Mu.Length;

        public static QpInstance From(CorpusCase c)
        {
            var n = c.N;
            Rational Parse(string s) => c.Type == "double"
                ? Rational.FromDouble(double.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture))
                : Rational.FromDecimal(decimal.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture));

            var cov = new Rational[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    cov[i, j] = Parse(c.Cov[i][j]);
                }
            }

            var minVariance = c.Mode == "minvar";
            return new QpInstance
            {
                Cov = cov,
                Mu = minVariance ? new Rational[n] : c.Mu.Select(Parse).ToArray(),
                Lambda = minVariance ? Rational.FromInt(2) : Parse(c.Lambda),
                Lb = Parse(c.Lb),
                Ub = Parse(c.Ub),
            };
        }

        public QpInstance With(Rational[,]? cov = null, Rational[]? mu = null, Rational? lambda = null, Rational? lb = null, Rational? ub = null)
            => new()
            {
                Cov = cov ?? Cov,
                Mu = mu ?? Mu,
                Lambda = lambda ?? Lambda,
                Lb = lb ?? Lb,
                Ub = ub ?? Ub,
            };

        public bool InDomain(string type)
        {
            var values = new List<Rational>();
            for (var i = 0; i < N; i++)
            {
                for (var j = 0; j < N; j++)
                {
                    values.Add(Cov[i, j]);
                }
            }

            values.Add(Lb);
            values.Add(Ub);
            values.AddRange(Mu);
            values.Add(Lambda);
            var high = type == "double" ? Rational.Of(BigInteger.Pow(10, 150), BigInteger.One) : Rational.Of(BigInteger.Pow(10, 14), BigInteger.One);
            var low = type == "double" ? Rational.Of(BigInteger.One, BigInteger.Pow(10, 150)) : Rational.Zero;
            foreach (var v in values)
            {
                var a = v.Abs();
                if (a.Sign == 0)
                {
                    continue;
                }

                if (a > high || a < low)
                {
                    return false;
                }
            }

            return true;
        }
    }

    private static class ExactOracle
    {
        public static Rational[]? SolveLinear(Rational[][] a, Rational[] b)
        {
            var n = a.Length;
            var m = new Rational[n][];
            for (var i = 0; i < n; i++)
            {
                m[i] = new Rational[n + 1];
                Array.Copy(a[i], m[i], n);
                m[i][n] = b[i];
            }

            for (var c = 0; c < n; c++)
            {
                var p = -1;
                for (var r = c; r < n; r++)
                {
                    if (m[r][c].Sign != 0)
                    {
                        p = r;
                        break;
                    }
                }

                if (p < 0)
                {
                    return null;
                }

                (m[c], m[p]) = (m[p], m[c]);
                var pivot = m[c][c];
                for (var r = 0; r < n; r++)
                {
                    if (r == c || m[r][c].Sign == 0)
                    {
                        continue;
                    }

                    var f = m[r][c] / pivot;
                    for (var k = c; k <= n; k++)
                    {
                        m[r][k] -= f * m[c][k];
                    }
                }
            }

            var x = new Rational[n];
            for (var i = 0; i < n; i++)
            {
                x[i] = m[i][n] / m[i][i];
            }

            return x;
        }

        /// <summary>The weights if the pattern (-1 at lb, +1 at ub, 0 free) satisfies the full KKT conditions exactly.</summary>
        public static Rational[]? Kkt(QpInstance q, int[] status)
        {
            var n = q.N;
            var fixedSum = Rational.Zero;
            var free = new List<int>();
            var w = new Rational[n];
            for (var i = 0; i < n; i++)
            {
                if (status[i] == 0)
                {
                    free.Add(i);
                }
                else
                {
                    w[i] = status[i] < 0 ? q.Lb : q.Ub;
                    fixedSum += w[i];
                }
            }

            var nu = Rational.Zero;
            var nuFree = free.Count == 0;
            if (free.Count > 0)
            {
                var k = free.Count;
                var a = new Rational[k + 1][];
                var rhs = new Rational[k + 1];
                for (var r = 0; r < k; r++)
                {
                    a[r] = new Rational[k + 1];
                    for (var c = 0; c < k; c++)
                    {
                        a[r][c] = q.Lambda * q.Cov[free[r], free[c]];
                    }

                    a[r][k] = Rational.One;
                    var sum = q.Mu[free[r]];
                    for (var j = 0; j < n; j++)
                    {
                        if (status[j] != 0)
                        {
                            sum -= q.Lambda * q.Cov[free[r], j] * w[j];
                        }
                    }

                    rhs[r] = sum;
                }

                a[k] = new Rational[k + 1];
                for (var c = 0; c < k; c++)
                {
                    a[k][c] = Rational.One;
                }

                rhs[k] = Rational.One - fixedSum;
                var x = SolveLinear(a, rhs);
                if (x is null)
                {
                    return null;
                }

                for (var r = 0; r < k; r++)
                {
                    w[free[r]] = x[r];
                    if (x[r] < q.Lb || x[r] > q.Ub)
                    {
                        return null;
                    }
                }

                nu = x[k];
            }
            else if (fixedSum != Rational.One)
            {
                return null;
            }

            Rational? lo = null;
            Rational? hi = null;
            for (var i = 0; i < n; i++)
            {
                if (status[i] == 0)
                {
                    continue;
                }

                var g = q.Mu[i];
                for (var j = 0; j < n; j++)
                {
                    g -= q.Lambda * q.Cov[i, j] * w[j];
                }

                if (nuFree)
                {
                    if (status[i] < 0 && (lo is null || g > lo.Value))
                    {
                        lo = g;
                    }

                    if (status[i] > 0 && (hi is null || g < hi.Value))
                    {
                        hi = g;
                    }
                }
                else
                {
                    g -= nu;
                    if ((status[i] < 0 && g.Sign > 0) || (status[i] > 0 && g.Sign < 0))
                    {
                        return null;
                    }
                }
            }

            if (nuFree && lo is not null && hi is not null && lo.Value > hi.Value)
            {
                return null;
            }

            return w;
        }

        /// <summary>The certified exact optimum: the hint pattern first, then (n &lt;= 6) full enumeration.</summary>
        public static (Rational[] W, int[] Pattern)? Optimum(QpInstance q, int[]? hint)
        {
            if (hint is not null)
            {
                var w = Kkt(q, hint);
                if (w is not null)
                {
                    return (w, hint);
                }
            }

            var n = q.N;
            if (n > 6)
            {
                return null;
            }

            var total = (int)Math.Pow(3, n);
            var st = new int[n];
            for (var code = 0; code < total; code++)
            {
                var rest = code;
                for (var i = 0; i < n; i++)
                {
                    st[i] = (rest % 3) - 1;
                    rest /= 3;
                }

                var w = Kkt(q, st);
                if (w is not null)
                {
                    return (w, (int[])st.Clone());
                }
            }

            return null;
        }

        public static bool IsPositiveDefinite(Rational[,] s)
        {
            var n = s.GetLength(0);
            var a = (Rational[,])s.Clone();
            for (var k = 0; k < n; k++)
            {
                if (a[k, k].Sign <= 0)
                {
                    return false;
                }

                for (var i = k + 1; i < n; i++)
                {
                    var f = a[i, k] / a[k, k];
                    for (var j = k + 1; j < n; j++)
                    {
                        a[i, j] -= f * a[k, j];
                    }
                }
            }

            return true;
        }

        private static Rational OneNorm(Rational[][] m)
        {
            var best = Rational.Zero;
            for (var c = 0; c < m.Length; c++)
            {
                var sum = Rational.Zero;
                for (var r = 0; r < m.Length; r++)
                {
                    sum += m[r][c].Abs();
                }

                if (sum > best)
                {
                    best = sum;
                }
            }

            return best;
        }

        /// <summary>Exact 1-norm condition number of the principal sub-block of <paramref name="s"/> on <paramref name="idx"/>.</summary>
        public static Rational Cond1(Rational[,] s, int[] idx)
        {
            var k = idx.Length;
            if (k == 0)
            {
                return Rational.One;
            }

            var block = new Rational[k][];
            for (var r = 0; r < k; r++)
            {
                block[r] = new Rational[k];
                for (var c = 0; c < k; c++)
                {
                    block[r][c] = s[idx[r], idx[c]];
                }
            }

            var inverseColumns = new Rational[k][];
            for (var c = 0; c < k; c++)
            {
                var e = new Rational[k];
                e[c] = Rational.One;
                inverseColumns[c] = SolveLinear(block, e)!;
            }

            var inverse = new Rational[k][];
            for (var r = 0; r < k; r++)
            {
                inverse[r] = new Rational[k];
                for (var c = 0; c < k; c++)
                {
                    inverse[r][c] = inverseColumns[c][r];
                }
            }

            return OneNorm(block) * OneNorm(inverse);
        }

        /// <summary>One decimal unit in the 28th significant digit (at least 1e-28), as an exact rational.</summary>
        public static Rational DecimalUnit(decimal d)
        {
            if (d == 0m)
            {
                return Rational.Of(BigInteger.One, BigInteger.Pow(10, 28));
            }

            var bits = decimal.GetBits(d);
            var magnitude = (new BigInteger((uint)bits[2]) << 64) | (new BigInteger((uint)bits[1]) << 32) | new BigInteger((uint)bits[0]);
            var scale = (bits[3] >> 16) & 0xFF;
            var adjusted = magnitude.ToString(CultureInfo.InvariantCulture).Length - 1 - scale;
            var exponent = Math.Max(adjusted - 27, -28);
            return exponent >= 0
                ? Rational.Of(BigInteger.Pow(10, exponent), BigInteger.One)
                : Rational.Of(BigInteger.One, BigInteger.Pow(10, -exponent));
        }

        private static Rational[] Neighbours(string type, string s)
        {
            if (type == "double")
            {
                var x = double.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture);
                return [Rational.FromDouble(Math.BitIncrement(x)), Rational.FromDouble(Math.BitDecrement(x))];
            }

            var d = decimal.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture);
            var unit = DecimalUnit(d);
            var v = Rational.FromDecimal(d);
            return [v + unit, v - unit];
        }

        /// <summary>
        /// The exact optimum by a primal active-set method in rational arithmetic (Nocedal and Wright 2006,
        /// Algorithm 16.3): the greedy feasible start, the exact subspace minimiser of each working set, the step
        /// length to the first blocking bound, and the release of the smallest-index variable with a negative
        /// multiplier. The result is certified by <see cref="Kkt"/>, which checks the full KKT conditions.
        /// </summary>
        public static (Rational[] W, int[] Pattern) ActiveSet(QpInstance q)
        {
            var n = q.N;
            var w = new Rational[n];
            var status = new int[n];
            var remaining = Rational.One;
            for (var i = 0; i < n; i++)
            {
                w[i] = q.Lb;
                remaining -= q.Lb;
            }

            var lastFilled = n - 1;
            for (var i = 0; i < n && remaining.Sign > 0; i++)
            {
                var room = q.Ub - q.Lb;
                var add = room < remaining ? room : remaining;
                w[i] += add;
                remaining -= add;
                lastFilled = i;
            }

            for (var i = 0; i < n; i++)
            {
                status[i] = w[i] == q.Lb ? -1 : w[i] == q.Ub ? 1 : 0;
            }

            if (!status.Contains(0))
            {
                status[lastFilled] = 0;
            }

            for (var guard = 0; guard < 100_000; guard++)
            {
                var free = Enumerable.Range(0, n).Where(i => status[i] == 0).ToArray();
                var x = SubspaceMinimiser(q, w, status, free);
                var moving = free.Any(j => x[j] != w[j]);
                if (moving)
                {
                    var alpha = Rational.One;
                    var blocking = -1;
                    foreach (var j in free)
                    {
                        var p = x[j] - w[j];
                        if (p.Sign == 0)
                        {
                            continue;
                        }

                        var t = p.Sign < 0 ? (w[j] - q.Lb) / -p : (q.Ub - w[j]) / p;
                        if (t < alpha)
                        {
                            alpha = t;
                            blocking = j;
                        }
                    }

                    foreach (var j in free)
                    {
                        w[j] += alpha * (x[j] - w[j]);
                    }

                    if (blocking >= 0)
                    {
                        var atLower = x[blocking] < w[blocking];
                        w[blocking] = atLower ? q.Lb : q.Ub;
                        status[blocking] = atLower ? -1 : 1;
                    }

                    continue;
                }

                // Descent gradient λΣw − μ; ν is its common value on the free variables.
                var gradient = new Rational[n];
                for (var i = 0; i < n; i++)
                {
                    var g = -q.Mu[i];
                    for (var j = 0; j < n; j++)
                    {
                        g += q.Lambda * q.Cov[i, j] * w[j];
                    }

                    gradient[i] = g;
                }

                var nu = gradient[free[0]];
                var release = -1;
                for (var i = 0; i < n && release < 0; i++)
                {
                    if (status[i] != 0 && (status[i] < 0 ? gradient[i] - nu : nu - gradient[i]).Sign < 0)
                    {
                        release = i;
                    }
                }

                if (release < 0)
                {
                    var certified = Kkt(q, status) ?? throw new InvalidOperationException("the exact active set did not certify");
                    return (certified, (int[])status.Clone());
                }

                status[release] = 0;
            }

            throw new InvalidOperationException("the exact active set did not terminate");
        }

        // The exact minimiser of the QP restricted to the free variables, the fixed ones held at their values.
        private static Rational[] SubspaceMinimiser(QpInstance q, Rational[] w, int[] status, int[] free)
        {
            var n = q.N;
            var x = (Rational[])w.Clone();
            var k = free.Length;
            var fixedSum = Rational.Zero;
            for (var i = 0; i < n; i++)
            {
                if (status[i] != 0)
                {
                    fixedSum += w[i];
                }
            }

            if (k == 1)
            {
                x[free[0]] = Rational.One - fixedSum;
                return x;
            }

            var a = new Rational[k + 1][];
            var rhs = new Rational[k + 1];
            for (var r = 0; r < k; r++)
            {
                a[r] = new Rational[k + 1];
                for (var c = 0; c < k; c++)
                {
                    a[r][c] = q.Lambda * q.Cov[free[r], free[c]];
                }

                a[r][k] = Rational.One;
                var sum = q.Mu[free[r]];
                for (var j = 0; j < n; j++)
                {
                    if (status[j] != 0)
                    {
                        sum -= q.Lambda * q.Cov[free[r], j] * w[j];
                    }
                }

                rhs[r] = sum;
            }

            a[k] = new Rational[k + 1];
            for (var c = 0; c < k; c++)
            {
                a[k][c] = Rational.One;
            }

            rhs[k] = Rational.One - fixedSum;
            var solution = SolveLinear(a, rhs) ?? throw new InvalidOperationException("singular subspace system");
            for (var r = 0; r < k; r++)
            {
                x[free[r]] = solution[r];
            }

            return x;
        }

        /// <summary>
        /// Largest relative change of the exact optimum under a single-entry one-ulp input change. With
        /// <paramref name="exactFallback"/>, a perturbed problem whose optimum leaves <paramref name="pattern"/> is
        /// solved by <see cref="ActiveSet"/> instead of counting as unresolved.
        /// </summary>
        public static Rational? Sensitivity(QpInstance q, CorpusCase c, Rational[] w0, int[] pattern, bool exactFallback = false)
        {
            var scale = w0.Select(x => x.Abs()).Max();
            var worst = Rational.Zero;
            var n = q.N;
            var ok = true;

            void Probe(QpInstance perturbed)
            {
                if (!IsPositiveDefinite(perturbed.Cov))
                {
                    return;
                }

                var r = Optimum(perturbed, pattern);
                if (r is null && exactFallback)
                {
                    r = ActiveSet(perturbed);
                }

                if (r is null)
                {
                    ok = false;
                    return;
                }

                var d = Rational.Zero;
                for (var i = 0; i < n; i++)
                {
                    var delta = (r.Value.W[i] - w0[i]).Abs();
                    if (delta > d)
                    {
                        d = delta;
                    }
                }

                d /= scale;
                if (d > worst)
                {
                    worst = d;
                }
            }

            for (var i = 0; i < n; i++)
            {
                for (var j = i; j < n; j++)
                {
                    foreach (var nx in Neighbours(c.Type, c.Cov[i][j]))
                    {
                        var cov = (Rational[,])q.Cov.Clone();
                        cov[i, j] = nx;
                        cov[j, i] = nx;
                        Probe(q.With(cov: cov));
                    }
                }
            }

            foreach (var nx in Neighbours(c.Type, c.Lb))
            {
                Probe(q.With(lb: nx));
            }

            foreach (var nx in Neighbours(c.Type, c.Ub))
            {
                Probe(q.With(ub: nx));
            }

            if (c.Mode == "meanvar")
            {
                for (var i = 0; i < n; i++)
                {
                    foreach (var nx in Neighbours(c.Type, c.Mu[i]))
                    {
                        var mu = (Rational[])q.Mu.Clone();
                        mu[i] = nx;
                        Probe(q.With(mu: mu));
                    }
                }

                foreach (var nx in Neighbours(c.Type, c.Lambda))
                {
                    if (nx.Sign > 0)
                    {
                        Probe(q.With(lambda: nx));
                    }
                }
            }

            return ok ? worst : null;
        }
    }
}
