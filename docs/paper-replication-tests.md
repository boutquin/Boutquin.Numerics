# Paper replication tests

This catalogue records tests that assert printed results, independently evaluated paper formulas, or explicitly identified variants of those formulas. The initial audit used source commit `7eb262e`; the 13 listed gaps were implemented and verified with the accompanying numerical changes on 2026-09-30.

The detailed catalogue contains **204 implemented entries across 166 distinct test methods**, with **no remaining unimplemented listed cases**. A Theory may occupy one entry covering all listed rows, or separate entries for named NIST datasets; entries are not a count of xUnit executions. The comparison index lists another **69 verification methods** whose existence alone does not establish paper replication.

An **Implemented** status means the named test exists and makes the described assertions. It does not mean that every result in that paper is covered. **Not yet implemented** means the specific described case has no matching assertion; the production algorithm can already exist. Each entry states whether its expected value is printed, independently derived, statistical, a companion-code oracle, or a library convention.

The 13 added cases include four declared seeded reconstructions (Figure 2.2 denoising, corrected bootstrap Table 2, and Jarre Tables 3–4), because their original realizations are unpublished. The three uniform Jarre tables agree with 49 of 60 printed cells; 11 Q/RNAK cells are checked against the literal preprint equations with independent controls and explicitly recorded as source discrepancies. Two circular block-length cells and the P² marker trace also require equation-based reconciliation. **Implemented** therefore describes the precise assertions below; it does not imply every original printed value was reproduced. The complete Release suite passes 1,995 tests with 13 existing stress-test skips.

For an array or matrix too large to inline, the Inputs/Output fields identify the complete committed JSON value with a JSON pointer, or a C# data factory and its exact fields. All Theory rows and tolerances are listed. Generic signatures retain `T` as declared; the Class field states the instantiated type. Unless an entry says otherwise, unmentioned optional arguments use their declared defaults.

Paper line numbers use the local source PDF identified by the SHA-256 below and this extraction convention:

```bash
pdftotext -layout paper-file-name.pdf paper-file-name.txt
nl -ba paper-file-name.txt
```

Form-feed page breaks do not add a line unless followed by a newline. PDF page numbers count from 1, including covers. Image-only papers have no usable text line range: their visually checked page/equation is given instead. NIST supplies data files rather than PDF papers. A missing primary source or unpublished random realization is stated explicitly; no line or input is invented.

The document inventories numeric assertions, rather than transcribing every plot, proof, timing experiment, or table in the reference books. API guards, generic-type parity, convergence regressions without a paper oracle, and architecture tests are outside the detailed replication catalogue. Relevant external comparisons and misleadingly named tests are indexed after it.

## Coverage index

| Paper / source | Implemented entries | Not yet implemented |
| --- | ---: | ---: |
| [bailey-lopezdeprado-2014-deflated-sharpe.pdf](#paper-dsr) | 9 | 0 |
| [bailey-lopezdeprado-2012-sharpe-frontier.pdf](#paper-mintrl) | 7 | 0 |
| [harvey-liu-2015-backtesting.pdf](#paper-haircut) | 7 | 0 |
| [ledoit-wolf-2004-well-conditioned-estimator.pdf](#paper-lw) | 2 | 0 |
| [ledoit-wolf-2004-honey-shrunk.pdf](#paper-lwcc) | 2 | 0 |
| [ledoit-wolf-2003-improved-estimation.pdf](#paper-lwsf) | 2 | 0 |
| [chen-wiesel-eldar-hero-2010-oas.pdf](#paper-oas) | 1 | 0 |
| [ledoit-wolf-2022-quadratic-shrinkage.pdf](#paper-qis) | 2 | 0 |
| [lam-2016-nercome.pdf](#paper-nercome) | 1 | 0 |
| [1201.0175.pdf](#paper-poet) | 1 | 0 |
| [lopez-de-prado-2020-machine-learning-for-asset-managers.pdf](#paper-mlfam) | 5 | 0 |
| [johnstone-2001-largest-eigenvalue-pca.pdf](#paper-johnstone) | 5 | 0 |
| [newey-west-1987-hac.pdf](#paper-nw) | 2 | 0 |
| [szekely-rizzo-bakirov-2007-distance-correlation.pdf](#paper-dcorr) | 9 | 0 |
| [press-et-al-2007-numerical-recipes-3e.pdf](#paper-fisher) | 4 | 0 |
| [welford-1962-corrected-sums-of-squares-and-products.pdf](#paper-welford) | 1 | 0 |
| [bailey-borwein-lopezdeprado-zhu-2015-pbo.pdf](#paper-pbo) | 7 | 0 |
| [sheppert-2026-gt-score.pdf](#paper-gt) | 11 | 0 |
| [Politis_White_2004.pdf](#paper-pw) | 5 | 0 |
| [Patton_Politis_White_2009.pdf](#paper-ppw) | 7 | 0 |
| [rbnew3dp.pdf](#paper-fdb) | 4 | 0 |
| [Mammen (1993), Bootstrap and Wild Bootstrap for High Dimensional Linear Models (PDF filename unavailable)](#paper-mammen) | 1 | 0 |
| [joe-kuo-notes.pdf](#paper-sobol) | 3 | 0 |
| [halton-1960-quasi-random-sequences.pdf](#paper-halton) | 2 | 0 |
| [harrell-davis-1982-distribution-free-quantile-estimator.pdf](#paper-hd) | 2 | 0 |
| [hyndman-1996-sample-quantiles.pdf](#paper-type7) | 3 | 0 |
| [oneill-2014-pcg.pdf](#paper-pcg) | 2 | 0 |
| [blackman-2018-xoshiro.pdf](#paper-xoshiro) | 2 | 0 |
| [fritsch-carlson-1980-monotone-piecewise-cubic-interpolation.pdf](#paper-fc) | 3 | 0 |
| [hagan-west-2006-interpolation-curve-construction.pdf](#paper-hw) | 6 | 0 |
| [jarre-2025-cubic-spline-functions-revisited.pdf](#paper-jarre) | 10 | 0 |
| [higham-2002-nearest-correlation-matrix.pdf](#paper-ncm) | 1 | 0 |
| [higham-1990-cholesky-semidefinite.pdf](#paper-chol) | 2 | 0 |
| [nelder-1965-simplex-method.pdf](#paper-nm) | 1 | 0 |
| [madsen-nielsen-tingleff-2004-nonlinear-least-squares.pdf](#paper-lm) | 1 | 0 |
| [jpmorgan-1996-riskmetrics-technical-document.pdf](#paper-ewma) | 1 | 0 |
| [jain-chlamtac-1985-p2-quantile.pdf](#paper-p2) | 1 | 0 |
| [NIST StRD nonlinear least squares (dataset collection; no PDF)](#paper-nist-nls) | 27 | 0 |
| [NIST StRD linear least squares (dataset collection; no PDF)](#paper-nist-lls) | 24 | 0 |
| [NIST StRD univariate statistics (dataset collection; no PDF)](#paper-nist-univ) | 18 | 0 |

<a id="paper-dsr"></a>

## Paper: bailey-lopezdeprado-2014-deflated-sharpe.pdf

Source PDF SHA-256: `ca4a1e834a2954b84eb560776e903329cd9cef318d458aadf6d67bc8166d10b5`.

### Class: `DeflatedSharpeRatio` (filename: [DeflatedSharpeRatio.cs](../src/Boutquin.Numerics/Statistics/DeflatedSharpeRatio.cs))

Function: `DsrResult ComputeFromReturns(ReadOnlySpan<decimal> returns, int numTrials, int tradingDaysPerYear = 252)`

#### Test #1: DeflatedSharpe_Scalar_MatchesReferenceImplementation

Paper line: 298–381; PDF page(s) 7–9; Equations (1)–(2).

Basis: Paper-derived formula.

**Inputs:**

- returns: [dsr.json](../tests/Verification/vectors/dsr.json), JSON pointer `/returns` (252 observations)
- numTrials: 100
- tradingDaysPerYear: 252

**Output / assertions:**

- ExpectedMaxSharpe: 2.5356389238408914 (annualized), absolute tolerance 1e-6
- DeflatedSharpe: 0.398550272717761, absolute tolerance 1e-4
- PValue = 1 − DSR: 0.3464936408246937, absolute tolerance 1e-6

Status: **Implemented** — [DsrVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/DsrVerificationTests.cs) :: `DeflatedSharpe_Scalar_MatchesReferenceImplementation`.

### Class: `DeflatedSharpeRatio<double>` (filename: [DeflatedSharpeRatio.cs](../src/Boutquin.Numerics/Statistics/DeflatedSharpeRatio.cs))

Function: `DsrResult<T> ComputeWithTrialVariance(T observedSharpe, int numTrials, T trialSharpeVariance, T backTestYears, T skewness, T kurtosis, int tradingDaysPerYear)`

#### Test #2: DeflatedSharpe_DisclosedTrialVariance_MatchesReferenceImplementation

Paper line: 298–381; PDF page(s) 7–9; Equations (1)–(2).

Basis: Paper-derived formula.

**Inputs:**

- observedSharpe: 2.9341891965586524
- numTrials: 100
- trialSharpeVariance: 0.5 (annualized)
- backTestYears: 1
- skewness: 0.022738483815877994
- kurtosis: 3.250881824521277
- tradingDaysPerYear: 252

**Output / assertions:**

- ExpectedMaxSharpe: 1.7894064662732079
- PValue: 0.12839243440931525
- Absolute tolerance: 1e-6

Status: **Implemented** — [DsrVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/DsrVerificationTests.cs) :: `DeflatedSharpe_DisclosedTrialVariance_MatchesReferenceImplementation`.

#### Test #3: DeflatedSharpe_PaperWorkedExample_Is0Point9004

Paper line: 407–449; worked investment example; the formulas on page 10 were checked visually; PDF page(s) 9–10.

Basis: Printed paper example.

**Inputs:**

- observedSharpe: 2.5
- numTrials: 100
- trialSharpeVariance: 0.5 (annualized)
- backTestYears: 5
- skewness: −3
- kurtosis: 10
- tradingDaysPerYear: 250

**Output / assertions:**

- Paper DSR: 0.9004; paper threshold per observation: 0.1132; absolute tolerance 5e-5
- Independent formula targets: DSR = 0.9003968344493904 and threshold = 0.11317200186513214; absolute tolerance 1e-6

Status: **Implemented** — [DsrVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/DsrVerificationTests.cs) :: `DeflatedSharpe_PaperWorkedExample_Is0Point9004`.

Paper DSR is the probability 1−PValue. The result property named DeflatedSharpe instead returns observedSharpe−ExpectedMaxSharpe; it is not the probability.

### Class: `DeflatedSharpeRatio<double>` (filename: [DeflatedSharpeRatio.cs](../src/Boutquin.Numerics/Statistics/DeflatedSharpeRatio.cs))

Function: `DsrResult<T> Compute(T observedSharpe, int numTrials, T backTestYears, T skewness, T kurtosis, int tradingDaysPerYear = 252)`

#### Test #4: Dsr_Compute_PValueDependsOnlyOnPerPeriodData_NotOnAnnualizationLabel

Paper line: 298–381; PDF page(s) 7–9; Equations (1)–(2).

Basis: Paper-derived formula.

**Inputs:**

- Theory rows, periodsPerYear: 52, 252, 365
- observedSharpe: 0.1 × sqrt(periodsPerYear)
- numTrials: 10
- backTestYears: 1000 / periodsPerYear
- skewness: 0
- kurtosis: 3
- tradingDaysPerYear: periodsPerYear

**Output / assertions:**

- PValue: 0.05680796385938458 for every row; absolute tolerance 1e-7

Status: **Implemented** — [ArenaR1B09DsrMinTrlTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/ArenaR1B09DsrMinTrlTests.cs) :: `Dsr_Compute_PValueDependsOnlyOnPerPeriodData_NotOnAnnualizationLabel`.

#### Test #5: Dsr_Compute_ThresholdIsPaperEquation1AtTheNullTrialVariance

Paper line: 298–381; PDF page(s) 7–9; Equations (1)–(2).

Basis: Paper-derived formula.

**Inputs:**

- observedSharpe: 2.5 (annualized)
- numTrials: 100
- backTestYears: 5
- skewness: 0
- kurtosis: 3 (Pearson)
- tradingDaysPerYear: 250

**Output / assertions:**

- ExpectedMaxSharpe: 1.1321729784527885
- DeflatedSharpe: 2.5 − 1.1321729784527885
- Absolute tolerance: 1e-7

Status: **Implemented** — [ArenaR1B09DsrMinTrlTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/ArenaR1B09DsrMinTrlTests.cs) :: `Dsr_Compute_ThresholdIsPaperEquation1AtTheNullTrialVariance`.

The overload without disclosed trial variance uses the null per-observation variance 1/(1250−1).

### Class: `DeflatedSharpeRatio<double>` (filename: [DeflatedSharpeRatio.cs](../src/Boutquin.Numerics/Statistics/DeflatedSharpeRatio.cs))

Function: `DsrResult<T> ComputeWithTrialVariance(T observedSharpe, int numTrials, T trialSharpeVariance, T backTestYears, T skewness, T kurtosis, int tradingDaysPerYear)`

#### Test #6: Dsr_ComputeWithTrialVariance_ZeroVarianceIsThePsrAgainstZero

Paper line: 298–381; PDF page(s) 7–9; Equations (1)–(2).

Basis: Paper-derived formula.

**Inputs:**

- observedSharpe: 2.5 (annualized)
- numTrials: 100
- backTestYears: 5
- skewness: 0
- kurtosis: 3 (Pearson)
- tradingDaysPerYear: 250
- trialSharpeVariance: 0

**Output / assertions:**

- ExpectedMaxSharpe: exactly 0
- PValue: 1.4013680051404235e-8; absolute tolerance 1e-11

Status: **Implemented** — [ArenaR1B09DsrMinTrlTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/ArenaR1B09DsrMinTrlTests.cs) :: `Dsr_ComputeWithTrialVariance_ZeroVarianceIsThePsrAgainstZero`.

### Class: `DeflatedSharpeRatio<double>` (filename: [DeflatedSharpeRatio.cs](../src/Boutquin.Numerics/Statistics/DeflatedSharpeRatio.cs))

Function: `DsrResult<T> Compute(T observedSharpe, int numTrials, T backTestYears, T skewness, T kurtosis, int tradingDaysPerYear = 252)`

#### Test #7: DsrDirectCompute_UsesAuthoritativeObservationCountDenominator

Paper line: 298–381; PDF page(s) 7–9; Equations (1)–(2).

Basis: Paper-derived formula.

**Inputs:**

- observedSharpe: 2
- numTrials: 10
- backTestYears: 1
- skewness: 0
- kurtosis: 3
- tradingDaysPerYear: 252

**Output / assertions:**

- PValue: 0.3373278190124802; absolute tolerance 1e-10

Status: **Implemented** — [C3ArenaFixTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/C3ArenaFixTests.cs) :: `DsrDirectCompute_UsesAuthoritativeObservationCountDenominator`.

### Class: `DeflatedSharpeRatio<double>` (filename: [DeflatedSharpeRatio.cs](../src/Boutquin.Numerics/Statistics/DeflatedSharpeRatio.cs))

Function: `DsrResult<T> ComputeWithTrialVariance(T observedSharpe, int numTrials, T trialSharpeVariance, T backTestYears, T skewness, T kurtosis, int tradingDaysPerYear)`

#### Test #8: Worked example with 46 trials

Paper line: 438–449; alternative trial counts in the worked example; PDF page(s) 10.

Basis: Printed paper result.

**Inputs:**

- observedSharpe: 2.5
- numTrials: 46
- trialSharpeVariance: 0.5
- backTestYears: 5
- skewness: -3
- kurtosis: 10
- tradingDaysPerYear: 250

**Output / assertions:**

- Paper DSR = 1−PValue: approximately 0.9505; printed rounding tolerance 5e-5
- Independently evaluated probability: 0.9505017068755787; absolute tolerance 1e-10
- Annualized expected maximum Sharpe: 1.586868412628726; absolute tolerance 1e-10
- Probability must exceed 0.95

Status: **Implemented** — [DsrVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/DsrVerificationTests.cs) :: `DeflatedSharpe_PaperAlternativeTrialCounts_Match0Point9505`.

#### Test #9: Worked example with 88 trials

Paper line: 438–449; alternative trial counts in the worked example; PDF page(s) 10.

Basis: Printed paper result.

**Inputs:**

- observedSharpe: 2.5
- numTrials: 88
- trialSharpeVariance: 0.5
- backTestYears: 5
- skewness: 0
- kurtosis: 3
- tradingDaysPerYear: 250

**Output / assertions:**

- Paper DSR = 1−PValue: approximately 0.9505; printed rounding tolerance 5e-5
- Independently evaluated probability: 0.9504908166760138; absolute tolerance 1e-10
- Annualized expected maximum Sharpe: 1.757368797553125; absolute tolerance 1e-10
- Probability must exceed 0.95

Status: **Implemented** — [DsrVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/DsrVerificationTests.cs) :: `DeflatedSharpe_PaperAlternativeTrialCounts_Match0Point9505`.

<a id="paper-mintrl"></a>

## Paper: bailey-lopezdeprado-2012-sharpe-frontier.pdf

Source PDF SHA-256: `7a756315dc645e343f9b619a727adbd10fde7347af14ae7000bdf3c99397b0e5`.

### Class: `MinimumTrackRecordLength` (filename: [MinimumTrackRecordLength.cs](../src/Boutquin.Numerics/Statistics/MinimumTrackRecordLength.cs))

Function: `double Compute(double observedSharpe, double benchmarkSharpe = 0.0, double skewness = 0.0, double kurtosis = 3.0, double significanceLevel = 0.05, int tradingDaysPerYear = 252)`

#### Test #10: MinimumTrackRecordLength_PaperWorkedExample_Is688Observations

Paper line: 461–514; PDF page(s) 10–11; Equation (13), Section 5 and Figures 8–10.

Basis: Printed paper example.

**Inputs:**

- observedSharpe: 2
- benchmarkSharpe: 1
- skewness: 0
- kurtosis: 3
- significanceLevel: 0.05
- tradingDaysPerYear: 252

**Output / assertions:**

- Printed result: 2.73 years (absolute tolerance 0.005)
- Equation (13) result: 688.2080373402349 observations (absolute tolerance 1e-6); additional rounded target 688.2 ± 0.05

Status: **Implemented** — [DsrVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/DsrVerificationTests.cs) :: `MinimumTrackRecordLength_PaperWorkedExample_Is688Observations`.

The prose prints 2.73 years. The observation count is the equation evaluated at 252 observations/year.

### Class: `MinimumTrackRecordLength<double>` (filename: [MinimumTrackRecordLength.cs](../src/Boutquin.Numerics/Statistics/MinimumTrackRecordLength.cs))

Function: `T Compute(T observedSharpe, T benchmarkSharpe, T skewness, T kurtosis, double significanceLevel, int tradingDaysPerYear)`

#### Test #11: MinTrl_PaperWorkedExample_AnnualizedTwoVersusOne_Is688Observations

Paper line: 461–514; PDF page(s) 10–11; Equation (13), Section 5 and Figures 8–10.

Basis: Printed paper example.

**Inputs:**

- observedSharpe: 2
- benchmarkSharpe: 1
- skewness: 0
- kurtosis: 3
- significanceLevel: 0.05
- tradingDaysPerYear: 252

**Output / assertions:**

- 688.2080373402348 observations; absolute tolerance 1e-5
- 2.73 years; absolute tolerance 0.005

Status: **Implemented** — [ArenaR1B09DsrMinTrlTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/ArenaR1B09DsrMinTrlTests.cs) :: `MinTrl_PaperWorkedExample_AnnualizedTwoVersusOne_Is688Observations`.

### Class: `MinimumTrackRecordLength` (filename: [MinimumTrackRecordLength.cs](../src/Boutquin.Numerics/Statistics/MinimumTrackRecordLength.cs))

Function: `double Compute(double observedSharpe, double benchmarkSharpe = 0.0, double skewness = 0.0, double kurtosis = 3.0, double significanceLevel = 0.05, int tradingDaysPerYear = 252)`

#### Test #12: MinimumTrackRecordLength_MatchesReferenceImplementation

Paper line: 461–514; PDF page(s) 10–11; Equation (13), Section 5 and Figures 8–10.

Basis: Paper-derived formula.

**Inputs:**

- observedSharpe: 2.9341891965586524
- benchmarkSharpe: 0
- skewness: 0.022738483815877994
- kurtosis: 3.250881824521277
- significanceLevel: 0.05
- tradingDaysPerYear: 252

**Output / assertions:**

- 81.38117726582657 observations; absolute tolerance 1e-10

Status: **Implemented** — [DsrVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/DsrVerificationTests.cs) :: `MinimumTrackRecordLength_MatchesReferenceImplementation`.

### Class: `MinimumTrackRecordLength<double>` (filename: [MinimumTrackRecordLength.cs](../src/Boutquin.Numerics/Statistics/MinimumTrackRecordLength.cs))

Function: `T Compute(T observedSharpe, T benchmarkSharpe, T skewness, T kurtosis, double significanceLevel, int tradingDaysPerYear)`

#### Test #13: MinTrl_ReturnsPerPeriodObservationCount_IndependentOfAnnualizationLabel

Paper line: 461–514; PDF page(s) 10–11; Equation (13), Section 5 and Figures 8–10.

Basis: Paper-derived formula.

**Inputs:**

- Theory rows, periodsPerYear: 12, 252
- observedSharpe: 0.1 × sqrt(periodsPerYear)
- benchmarkSharpe: 0
- skewness: 0
- kurtosis: 3
- significanceLevel: 0.05
- tradingDaysPerYear: periodsPerYear

**Output / assertions:**

- 272.90711713658885 observations for both rows; absolute tolerance 1e-6

Status: **Implemented** — [ArenaR1B09DsrMinTrlTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/ArenaR1B09DsrMinTrlTests.cs) :: `MinTrl_ReturnsPerPeriodObservationCount_IndependentOfAnnualizationLabel`.

#### Test #14: MinTRL_ReturnsObservationCount_NotAnnualizedByTradingDays

Paper line: 461–514; PDF page(s) 10–11; Equation (13), Section 5 and Figures 8–10.

Basis: Paper-derived formula.

**Inputs:**

- observedSharpe: 1
- benchmarkSharpe: 0
- skewness: 0
- kurtosis: 3
- significanceLevel: 0.05
- tradingDaysPerYear: 252

**Output / assertions:**

- 684.1497221590919 observations; absolute tolerance 1e-9

Status: **Implemented** — [StatisticsReviewFixTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/StatisticsReviewFixTests.cs) :: `MinTRL_ReturnsObservationCount_NotAnnualizedByTradingDays`.

#### Test #15: Figure 9 frequency example

Paper line: 505–514; weekly and monthly examples; PDF page(s) 11.

Basis: Printed paper result.

**Inputs:**

- observedSharpe: 2
- benchmarkSharpe: 1
- skewness: 0
- kurtosis: 3
- significanceLevel: 0.05
- tradingDaysPerYear: 52

**Output / assertions:**

- Printed minimum track record: approximately 2.83 years; API output is an observation count, so compare Compute(...) / 52

- Independent equation output: 147.0993465211522 observations = 2.828833586945235 years; observation tolerance 1e-9; printed-year tolerance 0.005

Status: **Implemented** — [DsrVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/DsrVerificationTests.cs) :: `MinimumTrackRecordLength_PaperWeeklyAndMonthlyFrequencies_MatchPrintedYears`.

#### Test #16: Figure 10 frequency example

Paper line: 505–514; weekly and monthly examples; PDF page(s) 11.

Basis: Printed paper result.

**Inputs:**

- observedSharpe: 2
- benchmarkSharpe: 1
- skewness: 0
- kurtosis: 3
- significanceLevel: 0.05
- tradingDaysPerYear: 12

**Output / assertions:**

- Printed minimum track record: approximately 3.24 years; API output is an observation count, so compare Compute(...) / 12

- Independent equation output: 38.87760835733575 observations = 3.239800696444646 years; observation tolerance 1e-9; printed-year tolerance 0.005

Status: **Implemented** — [DsrVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/DsrVerificationTests.cs) :: `MinimumTrackRecordLength_PaperWeeklyAndMonthlyFrequencies_MatchPrintedYears`.

<a id="paper-haircut"></a>

## Paper: harvey-liu-2015-backtesting.pdf

Source PDF SHA-256: `b32ec2e9e08d24501245079ac16e002656e546ddbc4f654670836fa1ee47c02f`.

### Class: `HaircutSharpe<double>` (filename: [HaircutSharpe.cs](../src/Boutquin.Numerics/Statistics/HaircutSharpe.cs))

Function: `HaircutSharpeResult<T> Compute(T observedSharpe, int numTrials, T backTestYears, int tradingDaysPerYear = 252)`

#### Test #17: HaircutSharpe_SingleTrial_MatchesReference

Paper line: 114–151; PDF page(s) 3; Equations (3)–(5), worked example on printed page 14.

Basis: Declared scalar variant.

**Inputs:**

- observedSharpe: 1.2
- numTrials: 1
- backTestYears: 5.0
- tradingDaysPerYear: 252

**Output / assertions:**

- BonferroniPValue: 0.007385885748577245; HolmPValue: 0.007385885748577245; BhyPValue: 0.007385885748577245
- HaircutSharpe: 1.2; HaircutAmount: 0.0
- P-values: relative tolerance 1e-9; haircut quantities: absolute tolerance 1e-9

Status: **Implemented** — [HaircutFisherVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/HaircutFisherVerificationTests.cs) :: `HaircutSharpe_SingleTrial_MatchesReference`.

The independent oracle uses SciPy Student-t tails. The scalar compatibility overload tested here accepts one observed Sharpe and assumes every other trial has p = 1. The full-panel APIs `ComputePanel` and `AdjustPValues` instead use all supplied trial p-values for Holm and BHY. The worked-example text instead illustrates independent trials with p_M = 1 − (1 − p_S)^N; its printed figures also follow a normal approximation. These are distinct adjustments, so the printed comparison has explicit bands.

#### Test #18: HaircutSharpe_ModerateTrials_MatchesReference

Paper line: 114–151; PDF page(s) 3; Equations (3)–(5), worked example on printed page 14.

Basis: Declared scalar variant.

**Inputs:**

- observedSharpe: 1.0
- numTrials: 10
- backTestYears: 5.0
- tradingDaysPerYear: 252

**Output / assertions:**

- BonferroniPValue: 0.25521967364992404; HolmPValue: 0.25521967364992404; BhyPValue: 0.7475303219087656
- HaircutSharpe: 0.5090547892862277; HaircutAmount: 0.4909452107137723
- P-values: relative tolerance 1e-9; haircut quantities: absolute tolerance 1e-9

Status: **Implemented** — [HaircutFisherVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/HaircutFisherVerificationTests.cs) :: `HaircutSharpe_ModerateTrials_MatchesReference`.

The independent oracle uses SciPy Student-t tails. The scalar compatibility overload tested here accepts one observed Sharpe and assumes every other trial has p = 1. The full-panel APIs `ComputePanel` and `AdjustPValues` instead use all supplied trial p-values for Holm and BHY. The worked-example text instead illustrates independent trials with p_M = 1 − (1 − p_S)^N; its printed figures also follow a normal approximation. These are distinct adjustments, so the printed comparison has explicit bands.

#### Test #19: HaircutSharpe_ManyTrials_MatchesReference

Paper line: 114–151; PDF page(s) 3; Equations (3)–(5), worked example on printed page 14.

Basis: Declared scalar variant.

**Inputs:**

- observedSharpe: 1.5
- numTrials: 100
- backTestYears: 3.0
- tradingDaysPerYear: 252

**Output / assertions:**

- BonferroniPValue: 0.9557531469396854; HolmPValue: 0.9557531469396854; BhyPValue: 1.0
- HaircutSharpe: 0.03204414787449062; HaircutAmount: 1.4679558521255094
- P-values: relative tolerance 1e-9; haircut quantities: absolute tolerance 1e-9

Status: **Implemented** — [HaircutFisherVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/HaircutFisherVerificationTests.cs) :: `HaircutSharpe_ManyTrials_MatchesReference`.

The independent oracle uses SciPy Student-t tails. The scalar compatibility overload tested here accepts one observed Sharpe and assumes every other trial has p = 1. The full-panel APIs `ComputePanel` and `AdjustPValues` instead use all supplied trial p-values for Holm and BHY. The worked-example text instead illustrates independent trials with p_M = 1 − (1 − p_S)^N; its printed figures also follow a normal approximation. These are distinct adjustments, so the printed comparison has explicit bands.

#### Test #20: HaircutSharpe_LowSharpeThousandTrials_MatchesReference

Paper line: 114–151; PDF page(s) 3; Equations (3)–(5), worked example on printed page 14.

Basis: Declared scalar variant.

**Inputs:**

- observedSharpe: 0.5
- numTrials: 1000
- backTestYears: 2.0
- tradingDaysPerYear: 252

**Output / assertions:**

- BonferroniPValue: 1.0; HolmPValue: 1.0; BhyPValue: 1.0
- HaircutSharpe: 0.0; HaircutAmount: 0.5
- P-values: relative tolerance 1e-9; haircut quantities: absolute tolerance 1e-9

Status: **Implemented** — [HaircutFisherVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/HaircutFisherVerificationTests.cs) :: `HaircutSharpe_LowSharpeThousandTrials_MatchesReference`.

The independent oracle uses SciPy Student-t tails. The scalar compatibility overload tested here accepts one observed Sharpe and assumes every other trial has p = 1. The full-panel APIs `ComputePanel` and `AdjustPValues` instead use all supplied trial p-values for Holm and BHY. The worked-example text instead illustrates independent trials with p_M = 1 − (1 − p_S)^N; its printed figures also follow a normal approximation. These are distinct adjustments, so the printed comparison has explicit bands.

#### Test #21: HaircutSharpe_NegativeSharpe_ShrinksTowardZero

Paper line: 114–151; PDF page(s) 3; Equations (3)–(5), worked example on printed page 14.

Basis: Declared scalar variant.

**Inputs:**

- observedSharpe: -0.3
- numTrials: 50
- backTestYears: 3.0
- tradingDaysPerYear: 252

**Output / assertions:**

- BonferroniPValue: 1.0; HolmPValue: 1.0; BhyPValue: 1.0
- HaircutSharpe: -0.0; HaircutAmount: -0.3
- P-values: relative tolerance 1e-9; haircut quantities: absolute tolerance 1e-9

Status: **Implemented** — [HaircutFisherVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/HaircutFisherVerificationTests.cs) :: `HaircutSharpe_NegativeSharpe_ShrinksTowardZero`.

The independent oracle uses SciPy Student-t tails. The scalar compatibility overload tested here accepts one observed Sharpe and assumes every other trial has p = 1. The full-panel APIs `ComputePanel` and `AdjustPValues` instead use all supplied trial p-values for Holm and BHY. The worked-example text instead illustrates independent trials with p_M = 1 − (1 − p_S)^N; its printed figures also follow a normal approximation. These are distinct adjustments, so the printed comparison has explicit bands.

#### Test #22: HaircutSharpe_HighSharpe_MatchesReference

Paper line: 114–151; PDF page(s) 3; Equations (3)–(5), worked example on printed page 14.

Basis: Declared scalar variant.

**Inputs:**

- observedSharpe: 3.0
- numTrials: 500
- backTestYears: 10.0
- tradingDaysPerYear: 252

**Output / assertions:**

- BonferroniPValue: 2.6573469090148465e-18; HolmPValue: 2.6573469090148465e-18; BhyPValue: 1.8050888345168948e-17
- HaircutSharpe: 2.780405187569587; HaircutAmount: 0.219594812430413
- P-values: relative tolerance 1e-9; haircut quantities: absolute tolerance 1e-9

Status: **Implemented** — [HaircutFisherVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/HaircutFisherVerificationTests.cs) :: `HaircutSharpe_HighSharpe_MatchesReference`.

The independent oracle uses SciPy Student-t tails. The scalar compatibility overload tested here accepts one observed Sharpe and assumes every other trial has p = 1. The full-panel APIs `ComputePanel` and `AdjustPValues` instead use all supplied trial p-values for Holm and BHY. The worked-example text instead illustrates independent trials with p_M = 1 − (1 − p_S)^N; its printed figures also follow a normal approximation. These are distinct adjustments, so the printed comparison has explicit bands.

#### Test #23: HaircutSharpe_HarveyLiuWorkedExample_MatchesPaper

Paper line: 114–151; PDF page(s) 3; Equations (3)–(5), worked example on printed page 14.

Basis: Printed paper example, approximate comparison.

**Inputs:**

- observedSharpe: 0.75
- numTrials: 200
- backTestYears: 20.0
- tradingDaysPerYear: 12

**Output / assertions:**

- BonferroniPValue: 0.18516106058571263; HolmPValue: 0.18516106058571263; BhyPValue: 1.0
- HaircutSharpe: 0.29714238270683013; HaircutAmount: 0.45285761729316987
- P-values: relative tolerance 1e-9; haircut quantities: absolute tolerance 1e-9
- Printed single-test p-value: 0.0008 ± 0.0002; adjusted p-value: 0.15 ± 0.04; haircut Sharpe: 0.32 ± 0.03

Status: **Implemented** — [HaircutFisherVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/HaircutFisherVerificationTests.cs) :: `HaircutSharpe_HarveyLiuWorkedExample_MatchesPaper`.

The independent oracle uses SciPy Student-t tails. The scalar compatibility overload tested here accepts one observed Sharpe and assumes every other trial has p = 1. The full-panel APIs `ComputePanel` and `AdjustPValues` instead use all supplied trial p-values for Holm and BHY. The worked-example text instead illustrates independent trials with p_M = 1 − (1 − p_S)^N; its printed figures also follow a normal approximation. These are distinct adjustments, so the printed comparison has explicit bands.


<a id="paper-lw"></a>

## Paper: ledoit-wolf-2004-well-conditioned-estimator.pdf

Source PDF SHA-256: `08e79244bf0ea47acb4a35189cfd7614bbc21cc058684da3d0cfe50d83717f1c`.

### Class: `LedoitWolfShrinkageEstimator<decimal>` (filename: [LedoitWolfShrinkageEstimator.cs](../src/Boutquin.Numerics/Statistics/LedoitWolfShrinkageEstimator.cs))

Function: `T[,] Estimate(T[,] returns)`

#### Test #24: LedoitWolfScaledIdentity_MatchesJmvaEquation14

Paper line: 599–641; PDF page(s) 12; Section 3.3, Equation (14).

Basis: Paper-derived formula.

**Inputs:**

- Theory rows: 120×8, 120×8 at small scale, 60×30, 40×50 (observations × variables)
- small_well_conditioned: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/ledoit_wolf_scaled_identity`
- small_scale_oas: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_scale_oas/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_scale_oas/ledoit_wolf_scaled_identity`
- moderate: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/ledoit_wolf_scaled_identity`
- overconcentrated: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/ledoit_wolf_scaled_identity`

**Output / assertions:**

- Every entry of the N×N matrix at the indicated JSON pointers
- Tolerance: max absolute error ≤ (maximum absolute reference entry) × 1e-9

Status: **Implemented** — [ShrinkageEstimatorPaperFidelityTests.cs](../tests/Boutquin.Numerics.Tests.Verification/ShrinkageEstimatorPaperFidelityTests.cs) :: `LedoitWolfScaledIdentity_MatchesJmvaEquation14`.

Oracle provenance: scikit-learn ledoit_wolf, rescaled from 1/T to 1/(T−1).

### Class: `LedoitWolfShrinkageEstimator` (filename: [LedoitWolfShrinkageEstimator.cs](../src/Boutquin.Numerics/Statistics/LedoitWolfShrinkageEstimator.cs))

Function: `decimal[,] Estimate(decimal[,] returns)`

#### Test #25: LedoitWolf_MatchesOwnFormulaReference

Paper line: 599–641; PDF page(s) 12; Section 3.3, Equation (14).

Basis: Paper-derived formula.

**Inputs:**

- returns: [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/returns` (120×5)

**Output / assertions:**

- Every matrix entry: [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/ledoit_wolf_scaled_identity`
- Absolute tolerance: 1e-6

Status: **Implemented** — [CovarianceVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CovarianceVerificationTests.cs) :: `LedoitWolf_MatchesOwnFormulaReference`.

Despite the older test name, the current generator calls scikit-learn rather than reproducing the C# estimator.


<a id="paper-lwcc"></a>

## Paper: ledoit-wolf-2004-honey-shrunk.pdf

Source PDF SHA-256: `45aa32c6ced2108e60e4c2a6539944ac6eb4f7ebeef704357d8b82481f49172f`.

### Class: `LedoitWolfConstantCorrelationEstimator<decimal>` (filename: [LedoitWolfConstantCorrelationEstimator.cs](../src/Boutquin.Numerics/Statistics/LedoitWolfConstantCorrelationEstimator.cs))

Function: `T[,] Estimate(T[,] returns)`

#### Test #26: LedoitWolfConstantCorrelation_MatchesHoneyAppendixB

Paper line: 519–670; PDF page(s) 12–15; Appendix B, shrinkage intensity.

Basis: Paper-derived formula.

**Inputs:**

- Theory rows: 120×8, 120×8 at small scale, 60×30, 40×50 (observations × variables)
- small_well_conditioned: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/lw_constant_correlation`
- small_scale_oas: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_scale_oas/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_scale_oas/lw_constant_correlation`
- moderate: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/lw_constant_correlation`
- overconcentrated: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/lw_constant_correlation`

**Output / assertions:**

- Every entry of the N×N matrix at the indicated JSON pointers
- Tolerance: max absolute error ≤ (maximum absolute reference entry) × 1e-9

Status: **Implemented** — [ShrinkageEstimatorPaperFidelityTests.cs](../tests/Boutquin.Numerics.Tests.Verification/ShrinkageEstimatorPaperFidelityTests.cs) :: `LedoitWolfConstantCorrelation_MatchesHoneyAppendixB`.

Oracle provenance: authors’ covCor.m moment equations, independently evaluated in NumPy.

### Class: `LedoitWolfConstantCorrelationEstimator` (filename: [LedoitWolfConstantCorrelationEstimator.cs](../src/Boutquin.Numerics/Statistics/LedoitWolfConstantCorrelationEstimator.cs))

Function: `decimal[,] Estimate(decimal[,] returns)`

#### Test #27: LwConstantCorrelation_MatchesPaperReference

Paper line: 519–670; PDF page(s) 12–15; Appendix B, shrinkage intensity.

Basis: Paper-derived formula.

**Inputs:**

- small_well_conditioned: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/lw_constant_correlation`
- moderate: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/lw_constant_correlation`
- overconcentrated: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/lw_constant_correlation`

**Output / assertions:**

- Every N×N matrix entry at the indicated pointers
- Per-entry abs(actual−reference) / max(abs(reference),1e-8) ≤ 1e-10

Status: **Implemented** — [CovarianceVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CovarianceVerificationTests.cs) :: `LwConstantCorrelation_MatchesPaperReference`.


<a id="paper-lwsf"></a>

## Paper: ledoit-wolf-2003-improved-estimation.pdf

Source PDF SHA-256: `c8bb93a7f817dbcf96f34073438a71e1d275493d5767949dd17ff864500867d3`.

### Class: `LedoitWolfSingleFactorEstimator<decimal>` (filename: [LedoitWolfSingleFactorEstimator.cs](../src/Boutquin.Numerics/Statistics/LedoitWolfSingleFactorEstimator.cs))

Function: `T[,] Estimate(T[,] returns)`

#### Test #28: LedoitWolfSingleFactor_MatchesLw2003Theorem2

Paper line: 478–555; PDF page(s) 10–11; Theorem 2 and its moment estimators.

Basis: Paper-derived formula.

**Inputs:**

- Theory rows: 120×8, 120×8 at small scale, 60×30, 40×50 (observations × variables)
- small_well_conditioned: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/lw_single_factor`
- small_scale_oas: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_scale_oas/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_scale_oas/lw_single_factor`
- moderate: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/lw_single_factor`
- overconcentrated: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/lw_single_factor`

**Output / assertions:**

- Every entry of the N×N matrix at the indicated JSON pointers
- Tolerance: max absolute error ≤ (maximum absolute reference entry) × 1e-9

Status: **Implemented** — [ShrinkageEstimatorPaperFidelityTests.cs](../tests/Boutquin.Numerics.Tests.Verification/ShrinkageEstimatorPaperFidelityTests.cs) :: `LedoitWolfSingleFactor_MatchesLw2003Theorem2`.

Oracle provenance: authors’ covMarket.m moment equations, independently evaluated in NumPy.

### Class: `LedoitWolfSingleFactorEstimator` (filename: [LedoitWolfSingleFactorEstimator.cs](../src/Boutquin.Numerics/Statistics/LedoitWolfSingleFactorEstimator.cs))

Function: `decimal[,] Estimate(decimal[,] returns)`

#### Test #29: LwSingleFactor_MatchesPaperReference

Paper line: 478–555; PDF page(s) 10–11; Theorem 2 and its moment estimators.

Basis: Paper-derived formula.

**Inputs:**

- small_well_conditioned: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/lw_single_factor`
- moderate: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/lw_single_factor`
- overconcentrated: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/lw_single_factor`

**Output / assertions:**

- Every N×N matrix entry at the indicated pointers
- Per-entry abs(actual−reference) / max(abs(reference),1e-8) ≤ 1e-10

Status: **Implemented** — [CovarianceVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CovarianceVerificationTests.cs) :: `LwSingleFactor_MatchesPaperReference`.


<a id="paper-oas"></a>

## Paper: chen-wiesel-eldar-hero-2010-oas.pdf

Source PDF SHA-256: `0a74b89eb3e63ee49b15272668185186e3d3a9a3da64a79d9d14abe912379333`.

### Class: `OracleApproximatingShrinkageEstimator` (filename: [OracleApproximatingShrinkageEstimator.cs](../src/Boutquin.Numerics/Statistics/OracleApproximatingShrinkageEstimator.cs))

Function: `decimal[,] Estimate(decimal[,] returns)`

#### Test #30: Oas_MatchesPaperReference

Paper line: 401–415; PDF page(s) 8; Theorem 3, Equations (22)–(23).

Basis: Paper-derived formula.

**Inputs:**

- small_well_conditioned: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/oas`
- small_scale_oas: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_scale_oas/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_scale_oas/oas`
- moderate: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/oas`
- overconcentrated: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/oas`

**Output / assertions:**

- Every N×N matrix entry at the indicated pointers
- Per-entry abs(actual−reference) / max(abs(reference),1e-8) ≤ 1e-10

Status: **Implemented** — [CovarianceVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CovarianceVerificationTests.cs) :: `Oas_MatchesPaperReference`.

OAS uses the known-zero-mean raw second moment divided by the observation count in Equation (2), and the exact 2/N corrections in Equation (23). The reference evaluates those equations directly, rather than scikit-learn’s simplified expression.


<a id="paper-qis"></a>

## Paper: ledoit-wolf-2022-quadratic-shrinkage.pdf

Source PDF SHA-256: `eb059693ccc80bcba7f1c4ea83cfde93e4092e3f15f444bb3a031b713d79437e`.

### Class: `QuadraticInverseShrinkageEstimator<decimal>` (filename: [QuadraticInverseShrinkageEstimator.cs](../src/Boutquin.Numerics/Statistics/QuadraticInverseShrinkageEstimator.cs))

Function: `T[,] Estimate(T[,] returns)`

#### Test #31: QuadraticInverseShrinkage_MatchesLw2022AppendixA

Paper line: 1651–1778; PDF page(s) 27–30; Appendix A, QIS reference code.

Basis: Paper-derived formula.

**Inputs:**

- Theory rows: 120×8, 120×8 at small scale, 60×30, 40×50 (observations × variables)
- small_well_conditioned: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/qis`
- small_scale_oas: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_scale_oas/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_scale_oas/qis`
- moderate: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/qis`
- overconcentrated: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/qis`

**Output / assertions:**

- Every entry of the N×N matrix at the indicated JSON pointers
- Tolerance: max absolute error ≤ (maximum absolute reference entry) × 1e-6

Status: **Implemented** — [ShrinkageEstimatorPaperFidelityTests.cs](../tests/Boutquin.Numerics.Tests.Verification/ShrinkageEstimatorPaperFidelityTests.cs) :: `QuadraticInverseShrinkage_MatchesLw2022AppendixA`.

Oracle provenance: authors’ Appendix A QIS(Y), including the p>n branch and trace normalization, independently evaluated in NumPy.

### Class: `QuadraticInverseShrinkageEstimator` (filename: [QuadraticInverseShrinkageEstimator.cs](../src/Boutquin.Numerics/Statistics/QuadraticInverseShrinkageEstimator.cs))

Function: `decimal[,] Estimate(decimal[,] returns)`

#### Test #32: QuadraticInverseShrinkage_MatchesPortedPipeline

Paper line: 1651–1778; PDF page(s) 27–30; Appendix A, QIS reference code.

Basis: Paper-derived formula.

**Inputs:**

- small_well_conditioned: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/qis`
- moderate: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/qis`
- overconcentrated: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/qis`

**Output / assertions:**

- Every matrix entry at the indicated JSON pointers
- Per-entry abs(actual−reference) / max(abs(reference),1e-8) ≤ 1e-6

Status: **Implemented** — [CovarianceVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CovarianceVerificationTests.cs) :: `QuadraticInverseShrinkage_MatchesPortedPipeline`.

The current generate_covariance_vectors.py oracle implements the paper’s Appendix A directly. The test’s older XML comment still calls it a translation-parity check; the provenance here follows the current generator.


<a id="paper-nercome"></a>

## Paper: lam-2016-nercome.pdf

Source PDF SHA-256: `a423ef16d0c97143310a8e0863e97c16c28226fa3f4c5a921645d1e6aba0ec44`.

### Class: `NercomeCovarianceEstimator` (filename: [NercomeCovarianceEstimator.cs](../src/Boutquin.Numerics/Statistics/NercomeCovarianceEstimator.cs))

Function: `decimal[,] Estimate(decimal[,] returns)`

#### Test #33: Nercome_MatchesPaperReference

Paper locus: Section 4.1, Equations (4.5)–(4.9), full reconstructed-matrix averaging and split selection.

Basis: Cross-language benchmark of the complete estimator and declared deterministic sampling policy. Independent exact rational spectral-projector unit tests separately verify the equations on exhaustive small panels; this benchmark does not reproduce a printed paper result.

**Inputs:**

- small_well_conditioned: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/nercome`
- Default seven-candidate grid; 50 distinct first subsets per candidate, capped by the number available; seed 0. Candidate sizes are floored, clamped and deduplicated, with first-candidate tie selection.

**Output / assertions:**

- Every 8×8 matrix entry at the indicated pointer
- Per-entry abs(actual−reference) / max(abs(reference),1e-8) ≤ 0.1

Status: **Implemented** — [CovarianceVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CovarianceVerificationTests.cs) :: `Nercome_MatchesPaperReference`.

This test is a NumPy implementation of the library’s fixed split. It does not reproduce Lam’s random-split averaging or optimized splitting. The loose tolerance and exclusion of rank-deficient regimes are material limits.


<a id="paper-poet"></a>

## Paper: 1201.0175.pdf

Source: [primary source](https://arxiv.org/pdf/1201.0175).

Source PDF SHA-256: `05933dd0123bf7ff84bf61e596271c7178451dc5207a643f2498ff39e5fe8ab0`.

### Class: `PoetCovarianceEstimator` (filename: [PoetCovarianceEstimator.cs](../src/Boutquin.Numerics/Statistics/PoetCovarianceEstimator.cs))

Function: `decimal[,] Estimate(decimal[,] returns)`

#### Test #34: Poet_MatchesPaperReference

Paper line: 720–747; PDF page(s) 15; Section 3.2, adaptive threshold Equation (3.2).

Basis: Paper-derived formula.

**Inputs:**

- small_well_conditioned: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/small_well_conditioned/poet`
- moderate: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/moderate/poet`
- overconcentrated: returns = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/returns`; expected matrix = [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/regimes/overconcentrated/poet`
- numFactors: 1
- thresholdMultiplier: 0.5

**Output / assertions:**

- Every N×N matrix entry at the indicated pointers
- Per-entry abs(actual−reference) / max(abs(reference),1e-8) ≤ 1e-4

Status: **Implemented** — [CovarianceVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CovarianceVerificationTests.cs) :: `Poet_MatchesPaperReference`.

POET oracle uses PCA on the T×T matrix; C# uses the N×N covariance matrix.


<a id="paper-mlfam"></a>

## Paper: lopez-de-prado-2020-machine-learning-for-asset-managers.pdf

Source PDF SHA-256: `ac101c48f631d2beee4ec4fa4a27c526efa5c0e20e9b02f00b5214107eba6c5a`.

### Class: `DenoisedCovarianceEstimator<double>` (filename: [DenoisedCovarianceEstimator.cs](../src/Boutquin.Numerics/Statistics/DenoisedCovarianceEstimator.cs))

Function: `T[,] Estimate(T[,] returns)`

#### Test #35: Denoised_RescalesReconstructionToUnitDiagonal_MatchingCov2Corr

Paper line: 1284–1314; PDF page(s) 35; Section 2.5.1 and Code Snippet 2.5.

Basis: Paper-derived formula.

**Inputs:**

- returns: BuildPanel(betas, 120, seed: 17), shape 120×6
- betas: [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]; standard deviations: [0.010, 0.015, 0.020, 0.012, 0.018, 0.025]
- applyLedoitWolfShrinkage: false

**Output / assertions:**

- Correlation upper triangle: [0.9999999999999998, 0.3998140235478781, 0.3778581598490972, 0.34727737950468307, 0.3055275951367473, 0.25001076112616455, 0.9999999999999998, 0.36381687654738815, 0.334372483996147, 0.29417412980065877, 0.24072031222636991, 1.0, 0.3160103549790205, 0.27801950110530693, 0.22750111016369617, 0.9999999999999999, 0.25551885351267395, 0.20908901214044553, 1.0, 0.18395227221509058, 1.0]
- Maximum absolute entry error: < 1e-10

Status: **Implemented** — [RmtCleaningFidelityTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/RmtCleaningFidelityTests.cs) :: `Denoised_RescalesReconstructionToUnitDiagonal_MatchingCov2Corr`.

BuildPanel creates an exactly centered and orthogonalized panel with population correlation C_ii=1 and C_ij=beta_i beta_j; the complete deterministic construction is [RmtCleaningFidelityTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/RmtCleaningFidelityTests.cs) :: `BuildPanel`. Output vectors list the upper triangle row by row, including the diagonal, after converting the returned covariance to correlation.

### Class: `DetonedCovarianceEstimator<double>` (filename: [DetonedCovarianceEstimator.cs](../src/Boutquin.Numerics/Statistics/DetonedCovarianceEstimator.cs))

Function: `T[,] Estimate(T[,] returns)`

#### Test #36: Detoned_FullDetoning_RemovesTheMarketComponentAndRescales

Paper line: 1413–1428; printed page 32, Section 2.6 detoning equations; PDF page(s) 38.

Basis: Paper-derived formula.

**Inputs:**

- returns: BuildPanel(betas, 120, seed: 37), shape 120×6
- betas: [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]; standard deviations: [0.010, 0.015, 0.020, 0.012, 0.018, 0.025]
- detoningAlpha: 1

**Output / assertions:**

- Correlation upper triangle: [1.0, -0.21315168875412208, -0.21627923929524404, -0.21584600870570964, -0.2074548964968825, -0.18500881944207304, 1.0, -0.2182567500916197, -0.21649466762675795, -0.20679861418210524, -0.18338054048356384, 1.0, -0.21540580361065495, -0.2042062191863684, -0.17980872362268266, 1.0, -0.19767795134781704, -0.17263786452855923, 1.0, -0.15934798639584255, 1.0]
- Maximum absolute entry error: < 1e-7

Status: **Implemented** — [RmtCleaningFidelityTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/RmtCleaningFidelityTests.cs) :: `Detoned_FullDetoning_RemovesTheMarketComponentAndRescales`.

BuildPanel creates an exactly centered and orthogonalized panel with population correlation C_ii=1 and C_ij=beta_i beta_j; the complete deterministic construction is [RmtCleaningFidelityTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/RmtCleaningFidelityTests.cs) :: `BuildPanel`. Output vectors list the upper triangle row by row, including the diagonal, after converting the returned covariance to correlation. Section 2.6 supplies the detoning formula; Code Snippet 2.6 is targeted shrinkage, a different operation.

#### Test #37: Detoned_AllNoiseSpectrum_AveragesWithoutMarketRemoval

Paper line: 1284–1314; PDF page(s) 35; Section 2.5.1 and Code Snippet 2.5.

Basis: Declared all-noise variant.

**Inputs:**

- returns (observations × variables): [[1,−1,0],[0,1,−1],[−1,0,1]]
- detoningAlpha: 1

**Output / assertions:**

- SignalCount: 0
- Output correlation equals [[1,0,0],[0,1,0],[0,0,1]] within 1e-9

Status: **Implemented** — [RmtCleaningFidelityTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/RmtCleaningFidelityTests.cs) :: `Detoned_AllNoiseSpectrum_AveragesWithoutMarketRemoval`.

At q=1 the fit uses fallback variance 1 and MP edge 4. The sample eigenvalues (1.5, 1.5, 0) all lie below that edge, so no market component qualifies, and snippet 2.5 still replaces every residual eigenvalue by their mean, 1, with the trace preserved. The output is therefore the identity correlation, not the sample correlation [[1,−0.5,−0.5],[−0.5,1,−0.5],[−0.5,−0.5,1]]. This all-noise averaging is a library policy; it is not an empirical result printed in the book.

### Class: `DenoisedCovarianceEstimator<double>` (filename: [DenoisedCovarianceEstimator.cs](../src/Boutquin.Numerics/Statistics/DenoisedCovarianceEstimator.cs))

Function: `internal static (T[,] Correlation, T[] StdDevs, int SignalCount) DenoiseCorrelation(T[,] sampleCov, int t)`

#### Test #38: Denoising_FitsNoiseVarianceBeforeSelectingSignalCount

Paper line: 1167–1266; printed pages 26–28, Section 2.4 and Code Snippet 2.4; PDF page(s) 32–34.

Basis: Paper-derived formula.

**Inputs:**

- sampleCov: SampleCovarianceEstimator<double>.Estimate(BuildPanel(17)); panel shape 500×50
- t: 500
- BuildPanel uses uint32 LCG state ← 1664525×state+1013904223; Normal() sums 12 uniforms and subtracts 6; return[i,j]=1.5×factor[i]×(0.8+0.4×j/49)+Normal()
- Complete construction: [RmtFittedVarianceTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/RmtFittedVarianceTests.cs) :: `BuildPanel`

**Output / assertions:**

- SignalCount: exactly 2

Status: **Implemented** — [RmtFittedVarianceTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/RmtFittedVarianceTests.cs) :: `Denoising_FitsNoiseVarianceBeforeSelectingSignalCount`.

Independent NumPy/SciPy fit gives sigma²≈0.30076066 and edge≈0.521. These support the expected signal count; the test does not assert those two numeric intermediates.

### Class: `CovarianceHelpers<double>` (filename: [CovarianceHelpers.cs](../src/Boutquin.Numerics/Statistics/CovarianceHelpers.cs))

Function: `internal static double FitNoiseVariance(T[] eigenvalues, double q)`

#### Test #39: Fitted noise variance in Figure 2.2

Paper line: 1190–1266; Code Snippets 2.3–2.4 and Figure 2.2; PDF page(s) 33–34.

Basis: Seeded reconstruction of the published experiment; independent NumPy/SciPy oracle.

**Inputs:**

- observations: 10000; variables: 1000; factors: 100; q=T/N=10; alpha: 0.995; KDE bandwidth: 0.01
- NumPy RandomState seed: 2026; normal return matrix; normal factor loadings; uniform diagonal noise, in Code Snippet 2.3 draw order
- Construction: [generate_rmt_paper_vectors.py](../tests/Verification/generate_rmt_paper_vectors.py)
- Complete descending correlation spectrum: [paper_rmt.json](../tests/Verification/vectors/paper_rmt.json), JSON pointer `/eigenvalues` (1000 values)

**Output / assertions:**

- Fitted noise variance: 0.676858649698627; absolute tolerance 1e-8 against the independent SciPy optimum
- MP upper edge: 1.17262751207523; absolute tolerance 2e-8
- Recovered signal-factor count: exactly 100; spectrum trace: 1000 within 1e-9

Status: **Implemented** — [RmtPaperReplicationTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/RmtPaperReplicationTests.cs) :: `Figure22_SeededBookExperiment_Recovers100Factors`.

The book prints sigma²≈0.6768 and 100 factors but supplies no realization/seed. This declared seed reproduces the experiment and factor count; the printed variance is context, not an exact golden for an unpublished realization. NumPy supplies the spectrum; production performs the fit and the MP edge computation. The oracle's dense scan and SciPy refinement verify the library's documented global minimum, which can differ from the book's local search.

<a id="paper-johnstone"></a>

## Paper: johnstone-2001-largest-eigenvalue-pca.pdf

Source PDF SHA-256: `57f0813b8523c54dc0fc24524fa3e5c9a135a4360866ead084e4608b0bb9e01f`.

### Class: `TracyWidomDenoisedCovarianceEstimator<double>` (filename: [TracyWidomDenoisedCovarianceEstimator.cs](../src/Boutquin.Numerics/Statistics/TracyWidomDenoisedCovarianceEstimator.cs))

Function: `T[,] Estimate(T[,] returns)`

#### Test #40: TracyWidom_RescalesReconstructionToUnitDiagonal_MatchingCov2Corr

Paper line: 218–252; PDF page(s) 5–6; Theorem 1.1, Equations (1.3)–(1.4).

Basis: Paper-derived formula.

**Inputs:**

- returns: BuildPanel(betas, 120, seed: 19), shape 120×6
- betas: [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]; standard deviations: [0.010, 0.015, 0.020, 0.012, 0.018, 0.025]

**Output / assertions:**

- Correlation upper triangle: [0.9999999999999998, 0.3998140235478781, 0.3778581598490972, 0.34727737950468307, 0.3055275951367473, 0.25001076112616455, 0.9999999999999998, 0.36381687654738815, 0.334372483996147, 0.29417412980065877, 0.24072031222636991, 1.0, 0.3160103549790205, 0.27801950110530693, 0.22750111016369617, 0.9999999999999999, 0.25551885351267395, 0.20908901214044553, 1.0, 0.18395227221509058, 1.0]
- Maximum absolute entry error: < 1e-10

Status: **Implemented** — [RmtCleaningFidelityTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/RmtCleaningFidelityTests.cs) :: `TracyWidom_RescalesReconstructionToUnitDiagonal_MatchingCov2Corr`.

BuildPanel creates an exactly centered and orthogonalized panel with population correlation C_ii=1 and C_ij=beta_i beta_j; the complete deterministic construction is [RmtCleaningFidelityTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/RmtCleaningFidelityTests.cs) :: `BuildPanel`. Output vectors list the upper triangle row by row, including the diagonal, after converting the returned covariance to correlation. Johnstone selects one signal; reconstruction/rescaling also uses the book’s Code Snippet 2.5.

#### Test #41: TracyWidom_ThresholdIsMuNpOverTPlusSigmaNpOverTTimesTheTw1NinetyNinthPercentile

Paper line: 218–252; PDF page(s) 5–6; Theorem 1.1, Equations (1.3)–(1.4).

Basis: Paper-derived formula.

**Inputs:**

- returns: BuildPanel(betas, 120, seed: 23), shape 120×6
- betas: [0.523595, 0.458146, 0.392696, 0.327247, 0.261798, 0.196348]

**Output / assertions:**

- Correlation upper triangle: [0.9999999999999998, 0.18463144904482312, 0.17140074210378167, 0.1540548276398625, 0.13204104139215828, 0.10506521877440422, 1.0, 0.16280416174665518, 0.14632822920770927, 0.12541854134440963, 0.09979568735436921, 1.0, 0.13584233458972794, 0.11643103691823924, 0.09264431904633447, 0.9999999999999999, 0.10464810772819928, 0.08326862782107883, 0.9999999999999999, 0.07136989149404202, 1.0]
- Maximum absolute entry error: < 1e-10

Status: **Implemented** — [RmtCleaningFidelityTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/RmtCleaningFidelityTests.cs) :: `TracyWidom_ThresholdIsMuNpOverTPlusSigmaNpOverTTimesTheTw1NinetyNinthPercentile`.

BuildPanel creates an exactly centered and orthogonalized panel with population correlation C_ii=1 and C_ij=beta_i beta_j; the complete deterministic construction is [RmtCleaningFidelityTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/RmtCleaningFidelityTests.cs) :: `BuildPanel`. Output vectors list the upper triangle row by row, including the diagonal, after converting the returned covariance to correlation. Largest eigenvalue ≈1.67088 exceeds the Johnstone threshold ≈1.66578. The test asserts the resulting matrix, not a separately returned threshold.

#### Test #42: TracyWidom_EigenvalueBelowTheNinetyNinthPercentileThreshold_IsNoise

Paper line: 218–252; PDF page(s) 5–6; Theorem 1.1, Equations (1.3)–(1.4).

Basis: Paper-derived decision with declared fallback.

**Inputs:**

- returns: BuildPanel(betas, 120, seed: 29), shape 120×6
- betas: [0.516879, 0.452269, 0.387659, 0.323049, 0.25844, 0.19383]

**Output / assertions:**

- Output correlation equals sample correlation within 1e-9

Status: **Implemented** — [RmtCleaningFidelityTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/RmtCleaningFidelityTests.cs) :: `TracyWidom_EigenvalueBelowTheNinetyNinthPercentileThreshold_IsNoise`.

BuildPanel creates an exactly centered and orthogonalized panel with population correlation C_ii=1 and C_ij=beta_i beta_j; the complete deterministic construction is [RmtCleaningFidelityTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/RmtCleaningFidelityTests.cs) :: `BuildPanel`. Output vectors list the upper triangle row by row, including the diagonal, after converting the returned covariance to correlation. Largest eigenvalue ≈1.65378 is below ≈1.66578. All-noise passthrough is a documented library fallback, not a printed Johnstone matrix.

### Class: `TracyWidomDenoisedCovarianceEstimator<double>` (filename: [TracyWidomDenoisedCovarianceEstimator.cs](../src/Boutquin.Numerics/Statistics/TracyWidomDenoisedCovarianceEstimator.cs))

Function: `internal static double TracyWidomShift(int t, int n)`

#### Test #43: TracyWidomShift_MatchesJohnstoneSigmaNp

Paper line: 218–252; PDF page(s) 5–6; Theorem 1.1, Equations (1.3)–(1.4).

Basis: Paper-derived formula.

**Inputs:**

- Theory rows (t observations, n variables): (120,8), (60,30), (40,50)
- 99% TW1 quantile: 2.0234

**Output / assertions:**

- Shift values respectively: 0.17687077, 0.30121121, 0.45170049; absolute tolerance 1e-7

Status: **Implemented** — [NewEstimatorsTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NewEstimatorsTests.cs) :: `TracyWidomShift_MatchesJohnstoneSigmaNp`.

Computes sigma_np/t × F1^-1(0.99). Table 1 prints the quantile rounded to 2.02; the test uses 2.0234.

#### Test #44: TracyWidomThreshold_ShiftDependsOnObservationCount

Paper line: 218–252; PDF page(s) 5–6; Theorem 1.1, Equations (1.3)–(1.4).

Basis: Paper-derived property.

**Inputs:**

- n: 20; t: 40 versus 200

**Output / assertions:**

- Shift at t=200 is strictly less than shift at t=40

Status: **Implemented** — [NewEstimatorsTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NewEstimatorsTests.cs) :: `TracyWidomThreshold_ShiftDependsOnObservationCount`.


<a id="paper-nw"></a>

## Paper: newey-west-1987-hac.pdf

Source PDF SHA-256: `c93e71c778d5a84065aedd40c94ead23e4a2e68ee0b05938bb361fc7e97a8017`.

### Class: `NeweyWestVariance` (filename: [NeweyWestVariance.cs](../src/Boutquin.Numerics/Statistics/NeweyWestVariance.cs))

Function: `decimal MeanVariance(ReadOnlySpan<decimal> series, int lags)`

#### Test #45: NeweyWestVariance_MatchesReference

Paper line: unavailable in the text extraction; image-only scan: PDF page 5, printed page 3, Equation (5).

Basis: Paper-derived formula.

**Inputs:**

- series: [scalar_stats.json](../tests/Verification/vectors/scalar_stats.json), JSON pointer `/newey_west/series` (500 observations)
- lags: 8

**Output / assertions:**

- Variance of the sample mean: 0.003602326939948279; absolute tolerance 1e-10

Status: **Implemented** — [ScalarStatisticsVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/ScalarStatisticsVerificationTests.cs) :: `NeweyWestVariance_MatchesReference`.

The current Python generator obtains the intercept-only HAC sandwich variance from statsmodels with the small-sample correction disabled. The older C# comment still calls it a hand port.

### Class: `NeweyWestVariance<double>` (filename: [NeweyWestVariance.cs](../src/Boutquin.Numerics/Statistics/NeweyWestVariance.cs))

Function: `T MeanVariance(ReadOnlySpan<T> series, int lags)`

#### Test #46: NeweyWest_ZeroLags_KnownAnswer

Paper line: unavailable in the text extraction; image-only scan: PDF page 5, printed page 3, Equation (5).

Basis: Paper-derived formula.

**Inputs:**

- series: [0.01,0.02,−0.005,0.015,−0.01,0.02,0.005]
- lags: 0

**Output / assertions:**

- Variance of the sample mean: 1.7201166180758018e-5; absolute tolerance 1e-15

Status: **Implemented** — [StatisticsReviewFixTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/StatisticsReviewFixTests.cs) :: `NeweyWest_ZeroLags_KnownAnswer`.


<a id="paper-dcorr"></a>

## Paper: szekely-rizzo-bakirov-2007-distance-correlation.pdf

Source PDF SHA-256: `a687015855e729b84ba343524176ba123f054f2de213810b065c4dd10ad8ce36`.

### Class: `DistanceCorrelation` (filename: [DistanceCorrelation.cs](../src/Boutquin.Numerics/Statistics/DistanceCorrelation.cs))

Function: `double Compute(ReadOnlySpan<double> x, ReadOnlySpan<double> y)`

#### Test #47: DistanceCorrelation_MatchesSzekely2007Reference

Paper line: 246–292; PDF page(s) 5–6; Section 2.2, Equations (2.8)–(2.10).

Basis: Paper-derived formula.

**Inputs:**

- x: [correlation.json](../tests/Verification/vectors/correlation.json), JSON pointer `/x`
- y: [correlation.json](../tests/Verification/vectors/correlation.json), JSON pointer `/y`

**Output / assertions:**

- Distance correlation: [correlation.json](../tests/Verification/vectors/correlation.json), JSON pointer `/distance_correlation`
- Absolute tolerance: 1e-6

Status: **Implemented** — [CorrelationVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CorrelationVerificationTests.cs) :: `DistanceCorrelation_MatchesSzekely2007Reference`.

Independent NumPy pairwise distances, double centering and the biased n²-normalized estimator from Equations (2.8)–(2.10).

### Class: `DistanceCorrelation<double>` (filename: [DistanceCorrelation.cs](../src/Boutquin.Numerics/Statistics/DistanceCorrelation.cs))

Function: `T Compute(ReadOnlySpan<T> x, ReadOnlySpan<T> y)`

#### Test #48: DistanceCorrelation_KnownAnswer_MatchesReference

Paper line: 246–292; PDF page(s) 5–6; Section 2.2, Equations (2.8)–(2.10).

Basis: Paper-derived formula.

**Inputs:**

- x: [1,2,3,4,5]
- y: [1,4,9,16,25]

**Output / assertions:**

- Distance correlation: 0.9869160440537484; absolute tolerance 1e-12

Status: **Implemented** — [StatisticsReviewFixTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/StatisticsReviewFixTests.cs) :: `DistanceCorrelation_KnownAnswer_MatchesReference`.

#### Test #49: DistanceCorrelation_PerfectLinear_EqualsOne

Paper line: 246–292; PDF page(s) 5–6; Section 2.2, Equations (2.8)–(2.10).

Basis: Paper-derived property.

**Inputs:**

- x: [1.0, 2.0, 3.0, 4.0]
- y: [1.0, 2.0, 3.0, 4.0]

**Output / assertions:**

- Distance correlation: 1; absolute tolerance 1e-12

Status: **Implemented** — [StatisticsReviewFixTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/StatisticsReviewFixTests.cs) :: `DistanceCorrelation_PerfectLinear_EqualsOne`.

#### Test #50: DistanceCorrelation_ConstantSeries_ReturnsZero

Paper line: 246–292; PDF page(s) 5–6; Section 2.2, Equations (2.8)–(2.10).

Basis: Paper-derived property.

**Inputs:**

- x: [5.0, 5.0, 5.0, 5.0]
- y: [1.0, 2.0, 3.0, 4.0]

**Output / assertions:**

- Distance correlation: 0, exactly

Status: **Implemented** — [StatisticsReviewFixTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/StatisticsReviewFixTests.cs) :: `DistanceCorrelation_ConstantSeries_ReturnsZero`.

### Class: `DistanceCorrelation` (filename: [DistanceCorrelation.cs](../src/Boutquin.Numerics/Statistics/DistanceCorrelation.cs))

Function: `double Compute(ReadOnlySpan<double> x, ReadOnlySpan<double> y)`

#### Test #51: DistanceCorrelation_QuadraticDependence_MatchesReferenceValue

Paper line: 246–292; PDF page(s) 5–6; Section 2.2, Equations (2.8)–(2.10).

Basis: Paper-derived formula.

**Inputs:**

- x: [1,2,3,4,5]
- y: [1,4,9,16,25]

**Output / assertions:**

- Distance correlation: 0.9869160440537484; absolute tolerance 1e-6

Status: **Implemented** — [CorrelationAndMomentTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/CorrelationAndMomentTests.cs) :: `DistanceCorrelation_QuadraticDependence_MatchesReferenceValue`.

Independent evaluation of the biased double-centered distance formula, not a printed sample.

### Class: `DistanceCorrelation<double>` (filename: [DistanceCorrelation.cs](../src/Boutquin.Numerics/Statistics/DistanceCorrelation.cs))

Function: `T Compute(ReadOnlySpan<T> x, ReadOnlySpan<T> y)`

#### Test #52: Compute_OneSeriesTranslated_MatchesReferenceFormula

Paper line: 246–292; PDF page(s) 5–6; Section 2.2, Equations (2.8)–(2.10).

Basis: Paper-derived formula.

**Inputs:**

- x: [1e16,1e16+2,1e16+4,1e16+8]
- y: [0,1,4,9]

**Output / assertions:**

- Distance correlation: 0.9917026572959774; absolute tolerance 1e-14

Status: **Implemented** — [DistanceCorrelationTranslationTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/DistanceCorrelationTranslationTests.cs) :: `Compute_OneSeriesTranslated_MatchesReferenceFormula`.

Independent evaluation of the biased double-centered distance formula, not a printed sample.

### Class: `DistanceCorrelation<float>` (filename: [DistanceCorrelation.cs](../src/Boutquin.Numerics/Statistics/DistanceCorrelation.cs))

Function: `T Compute(ReadOnlySpan<T> x, ReadOnlySpan<T> y)`

#### Test #53: Compute_FloatSeriesTranslated_MatchesReferenceFormula

Paper line: 246–292; PDF page(s) 5–6; Section 2.2, Equations (2.8)–(2.10).

Basis: Paper-derived formula.

**Inputs:**

- x: [1e7,1e7+1,1e7+2,1e7+4]
- y: [0,1,4,9]

**Output / assertions:**

- Distance correlation: 0.9917026572959774 (rounded to float); absolute tolerance 1e-6

Status: **Implemented** — [DistanceCorrelationTranslationTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/DistanceCorrelationTranslationTests.cs) :: `Compute_FloatSeriesTranslated_MatchesReferenceFormula`.

Independent evaluation of the biased double-centered distance formula, not a printed sample.

### Class: `DistanceCorrelation<double>` (filename: [DistanceCorrelation.cs](../src/Boutquin.Numerics/Statistics/DistanceCorrelation.cs))

Function: `T Compute(ReadOnlySpan<T> x, ReadOnlySpan<T> y)`

#### Test #54: Compute_BothSeriesTranslated_MatchesIndependentGolden

Paper line: 246–292; PDF page(s) 5–6; Section 2.2, Equations (2.8)–(2.10).

Basis: Paper-derived formula.

**Inputs:**

- x: [0,2,4,8,12,18,24,30]+1e16
- y: [8,0,12,2,24,4,30,18]+1e16

**Output / assertions:**

- Distance correlation: 0.6518494559566712980269140176320893593376513379720926947322965195912174; absolute tolerance 1e-14

Status: **Implemented** — [DistanceCorrelationTranslationTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/DistanceCorrelationTranslationTests.cs) :: `Compute_BothSeriesTranslated_MatchesIndependentGolden`.

Independent evaluation of the biased double-centered distance formula, not a printed sample.

#### Test #55: Compute_ExactlyAffineAtLargeOffset_ReturnsOne

Paper line: 246–292; PDF page(s) 5–6; Section 2.2, Equations (2.8)–(2.10).

Basis: Paper-derived formula.

**Inputs:**

- x: [1e16,1e16+2,1e16+4,1e16+6]
- y: [0,1,2,3]

**Output / assertions:**

- Distance correlation: 1; absolute tolerance 1e-14

Status: **Implemented** — [DistanceCorrelationTranslationTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/DistanceCorrelationTranslationTests.cs) :: `Compute_ExactlyAffineAtLargeOffset_ReturnsOne`.

Independent evaluation of the biased double-centered distance formula, not a printed sample.


<a id="paper-fisher"></a>

## Paper: press-et-al-2007-numerical-recipes-3e.pdf

Source PDF SHA-256: `76817c1b3b28aa71c5e275044eeaf92e48118b8e5e40ea2b3e1ce7639f9b5926`.

### Class: `FisherZTransform<double>` (filename: [FisherZTransform.cs](../src/Boutquin.Numerics/Statistics/FisherZTransform.cs))

Function: `T Forward(T r)`

#### Test #56: FisherZTransform_Forward_MatchesReference

Paper line: 51917–51950; PDF page(s) 770–771; Section 14.5, Equations (14.5.6)–(14.5.8); Fisher transform and its sampling scale.

Basis: Paper-derived formula.

**Inputs:**

- r values: [-0.9, -0.5, 0.0, 0.3, 0.7, 0.95]

**Output / assertions:**

- z values in the same order: [-1.4722194895832204, -0.5493061443340549, 0.0, 0.3095196042031118, 0.8673005276940532, 1.8317808230648227]
- Absolute tolerance: 1e-10

Status: **Implemented** — [HaircutFisherVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/HaircutFisherVerificationTests.cs) :: `FisherZTransform_Forward_MatchesReference`.

These are analytical transforms, not numerical examples printed in Fisher’s 1915 paper. Numerical Recipes explicitly gives the transform used here.

### Class: `FisherZTransform<double>` (filename: [FisherZTransform.cs](../src/Boutquin.Numerics/Statistics/FisherZTransform.cs))

Function: `T Inverse(T z)`

#### Test #57: FisherZTransform_Inverse_MatchesReference

Paper line: 51917–51950; PDF page(s) 770–771; Section 14.5, Equations (14.5.6)–(14.5.8); Fisher transform and its sampling scale.

Basis: Paper-derived formula.

**Inputs:**

- z values: [-2.0, -1.0, 0.0, 1.0, 2.0]

**Output / assertions:**

- r values in the same order: [-0.9640275800758169, -0.7615941559557649, 0.0, 0.7615941559557649, 0.9640275800758169]
- Absolute tolerance: 1e-10

Status: **Implemented** — [HaircutFisherVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/HaircutFisherVerificationTests.cs) :: `FisherZTransform_Inverse_MatchesReference`.

### Class: `FisherZTransform<double>` (filename: [FisherZTransform.cs](../src/Boutquin.Numerics/Statistics/FisherZTransform.cs))

Function: `(T Lower, T Upper) ConfidenceInterval(T r, int n, double confidenceLevel = 0.95)`

#### Test #58: FisherZTransform_ConfidenceInterval_MatchesReference

Paper line: 51917–51950; PDF page(s) 770–771; Section 14.5, Equations (14.5.6)–(14.5.8); Fisher transform and its sampling scale.

Basis: Paper-derived formula.

**Inputs:**

- (r,n,confidenceLevel) rows: [(0.5, 50, 0.95), (0.5, 50, 0.99), (0.8, 30, 0.95), (-0.4, 100, 0.95), (0.0, 20, 0.95), (0.9, 200, 0.99)]

**Output / assertions:**

- (Lower,Upper) rows in the same order: [(0.257487860768284, 0.6832563020988135), (0.17186017608342016, 0.7282680432333951), (0.6177862851491421, 0.9006795310331298), (-0.5529725840467599, -0.22094057506769604), (-0.44252078215149937, 0.44252078215149937), (0.8587854888996732, 0.9296410618000578)]
- Absolute tolerance: 1e-8

Status: **Implemented** — [HaircutFisherVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/HaircutFisherVerificationTests.cs) :: `FisherZTransform_ConfidenceInterval_MatchesReference`.

Normal intervals in z space with standard error 1/sqrt(n−3), inverted with tanh. This does not assert Fisher’s original finite-sample distribution.

#### Test #59: FisherZTransform_CiAtZeroCorrelation_IsSymmetric

Paper line: 51917–51950; PDF page(s) 770–771; Section 14.5, Equations (14.5.6)–(14.5.8); Fisher transform and its sampling scale.

Basis: Paper-derived property.

**Inputs:**

- r: 0
- n: 20
- confidenceLevel: 0.95

**Output / assertions:**

- Lower + Upper: 0 within 1e-15

Status: **Implemented** — [HaircutFisherVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/HaircutFisherVerificationTests.cs) :: `FisherZTransform_CiAtZeroCorrelation_IsSymmetric`.


<a id="paper-welford"></a>

## Paper: welford-1962-corrected-sums-of-squares-and-products.pdf

Source PDF SHA-256: `a9b8fe36e7811203e9c7f64ebfddc35a71fd405a494c30c076667d2aa6847a9b`.

### Class: `WelfordMoments<double>` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Add(T value)`

#### Test #60: WelfordMoments_SmallIntegerSeries_MatchesReference

Paper line: 69–134; PDF page(s) 2–3; Printed pages 419–420, mean and corrected-sum recurrence.

Basis: Paper-derived formula.

**Inputs:**

- values added in order: [1,2,3,4,5]

**Output / assertions:**

- Count: 5; Mean: 3; sample Variance: 2.5; PopulationVariance: 2; StdDev: sqrt(2.5)
- Absolute tolerance: 1e-10

Status: **Implemented** — [WelfordPcaVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/WelfordPcaVerificationTests.cs) :: `WelfordMoments_SmallIntegerSeries_MatchesReference`.

The fixture generator is a C# port. This small exact case is independently checkable by summing deviations; the arbitrary float-series port is listed separately as parity evidence.


<a id="paper-pbo"></a>

## Paper: bailey-borwein-lopezdeprado-zhu-2015-pbo.pdf

Source PDF SHA-256: `d8bfbadaaedb430d9dec929646d92f3580fdf18238f34121ae7738fa1c69a63d`.

### Class: `ProbabilityOfBacktestOverfitting` (filename: [ProbabilityOfBacktestOverfitting.cs](../src/Boutquin.Numerics/Statistics/ProbabilityOfBacktestOverfitting.cs))

Function: `PboResult Compute(decimal[,] returns, int splitCount = 16)`

#### Test #61: Pbo_MatchesReference

Paper line: 399–501; PDF page(s) 10–12; Section 2.2, Algorithm 2.3.

Basis: Paper-derived formula.

**Inputs:**

- returns: [scalar_stats.json](../tests/Verification/vectors/scalar_stats.json), JSON pointer `/pbo/returns` (10×128)
- splitCount: 16

**Output / assertions:**

- Pbo: 0.22525252525252526; LogitMedian: 0.9808292530117263
- Absolute tolerance: 1e-8

Status: **Implemented** — [ScalarStatisticsVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/ScalarStatisticsVerificationTests.cs) :: `Pbo_MatchesReference`.

Independent NumPy enumeration of C(16,8)=12,870 joins. The prose in the paper has a 12,780 typo; the combinatorial value is 12,870.

### Class: `ProbabilityOfBacktestOverfitting<double>` (filename: [ProbabilityOfBacktestOverfitting.cs](../src/Boutquin.Numerics/Statistics/ProbabilityOfBacktestOverfitting.cs))

Function: `PboResult<T> Compute(T[,] returns, int splitCount = 16)`

#### Test #62: Pbo_SharpeIsComputedOnJoinedBlocks

Paper line: 399–501; PDF page(s) 10–12; Section 2.2, Algorithm 2.3.

Basis: Paper-derived formula.

**Inputs:**

- returns: LcgPanel(6,24,seed:20)
- splitCount: 4

**Output / assertions:**

- Pbo: exactly 0.5; LogitMedian: −0.31430432971118716
- LogitValues: [−1.791759469228055,−0.9162907318741551,−0.9162907318741551,0.28768207245178085,0.28768207245178085,0.28768207245178085]; absolute tolerance 1e-12

Status: **Implemented** — [ArenaR1B11PboGenScoreTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/ArenaR1B11PboGenScoreTests.cs) :: `Pbo_SharpeIsComputedOnJoinedBlocks`.

Exact input generator: [ArenaR1B11PboGenScoreTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/ArenaR1B11PboGenScoreTests.cs) :: `LcgPanel`; uint32 state ← 1664525×state+1013904223, return=((state & 0xFFFFFF)/16777216−0.5)×0.04, strategies as the outer loop.

#### Test #63: Pbo_WinnerWithExactOutOfSampleTwin_TakesAverageRank

Paper line: 399–501; PDF page(s) 10–12; Section 2.2, Algorithm 2.3.

Basis: Paper-derived procedure with explicit tie convention.

**Inputs:**

- noise: LcgPanel(2,32,seed:7)
- returns rows: noise[0]+0.01, noise[0]+0.01, noise[1]−0.01
- splitCount: 4

**Output / assertions:**

- Pbo: exactly 0; all six LogitValues: ln(5/3) within 1e-12

Status: **Implemented** — [ArenaR1B11PboGenScoreTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/ArenaR1B11PboGenScoreTests.cs) :: `Pbo_WinnerWithExactOutOfSampleTwin_TakesAverageRank`.

Exact input generator: [ArenaR1B11PboGenScoreTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/ArenaR1B11PboGenScoreTests.cs) :: `LcgPanel`; uint32 state ← 1664525×state+1013904223, return=((state & 0xFFFFFF)/16777216−0.5)×0.04, strategies as the outer loop. Tied ranks use their average; the paper does not specify this numerical tie fixture.

#### Test #64: Pbo_ConstantStrategy_HasZeroSharpeRegardlessOfRounding

Paper line: 399–501; PDF page(s) 10–12; Section 2.2, Algorithm 2.3.

Basis: Paper-derived procedure with declared zero-variance convention.

**Inputs:**

- noise: LcgPanel(2,24,seed:3)
- returns rows: constant 0.01, noise[0]+0.002, noise[1]+0.001
- splitCount: 4

**Output / assertions:**

- Pbo: 2/3 within 1e-15; LogitMedian: −ln(3) within 1e-12

Status: **Implemented** — [ArenaR1B11PboGenScoreTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/ArenaR1B11PboGenScoreTests.cs) :: `Pbo_ConstantStrategy_HasZeroSharpeRegardlessOfRounding`.

Exact input generator: [ArenaR1B11PboGenScoreTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/ArenaR1B11PboGenScoreTests.cs) :: `LcgPanel`; uint32 state ← 1664525×state+1013904223, return=((state & 0xFFFFFF)/16777216−0.5)×0.04, strategies as the outer loop. Zero standard deviation is assigned Sharpe 0 by the library; it is not a printed paper result.

#### Test #65: Pbo_CountsAtOrBelowMedian_ExactZeroLogitCounted

Paper line: 399–501; PDF page(s) 10–12; Section 2.2, Algorithm 2.3.

Basis: Paper-derived formula.

**Inputs:**

- returns: [[−0.0146,0.0139,0.0106,−0.0098,−0.0002,−0.002,0.0061,0.0115],[−0.0162,−0.0189,0.0134,−0.0027,0.0105,−0.0199,−0.0022,0.0089],[−0.0108,0.0178,0.0161,−0.0188,−0.019,0.0017,0.0176,−0.0048]]
- splitCount: 4

**Output / assertions:**

- Pbo: 1 within 1e-12; all six folds are at or below the OOS median

Status: **Implemented** — [StatisticsReviewFixTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/StatisticsReviewFixTests.cs) :: `Pbo_CountsAtOrBelowMedian_ExactZeroLogitCounted`.

Two folds have zero logit and are included in the probability count.

#### Test #66: S27_Pbo_ReferencePin_AtOrBelowMedian

Paper line: 399–501; PDF page(s) 10–12; Section 2.2, Algorithm 2.3.

Basis: Paper-derived formula.

**Inputs:**

- returns: [[−0.0146,0.0139,0.0106,−0.0098,−0.0002,−0.002,0.0061,0.0115],[−0.0162,−0.0189,0.0134,−0.0027,0.0105,−0.0199,−0.0022,0.0089],[−0.0108,0.0178,0.0161,−0.0188,−0.019,0.0017,0.0176,−0.0048]]
- splitCount: 4

**Output / assertions:**

- Pbo: 1 within 1e-12; all six folds are at or below the OOS median

Status: **Implemented** — [StatisticsReviewFixTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/StatisticsReviewFixTests.cs) :: `S27_Pbo_ReferencePin_AtOrBelowMedian`.

Two folds have zero logit and are included in the probability count.

#### Test #67: Pbo_CountsLogitsAtOrBelowZero

Paper line: 399–501; PDF page(s) 10–12; Section 2.2, Algorithm 2.3.

Basis: Paper-derived formula.

**Inputs:**

- returns: the complete 3×8 literal matrix in [BacktestStatsTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/BacktestStatsTests.cs) :: `Pbo_CountsLogitsAtOrBelowZero`
- splitCount: 4

**Output / assertions:**

- Pbo: 1 within 1e-12; two of six folds have zero logit

Status: **Implemented** — [BacktestStatsTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/BacktestStatsTests.cs) :: `Pbo_CountsLogitsAtOrBelowZero`.


<a id="paper-gt"></a>

## Paper: sheppert-2026-gt-score.pdf

Source PDF SHA-256: `a9bca7ee4e4e809a1625ec27969219eb398afc39d70561deb04fa236c35faf12`.

The following GT-inspired variant tests exercise sample SD, cumulative arithmetic-return R²,
zero-target downside RMS over all observations, additive epsilon and the inclusive minimum-count
safeguard. They do not reproduce authenticated author component values or establish empirical
generalization. Author code uses different components and its singleton-loss path can return −Inf.
The independent rational discriminator suite is [GtInspiredVariantContractTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/GtInspiredVariantContractTests.cs).

### Class: `GtInspiredScore<double>` (filename: [GtInspiredScore.cs](../src/Boutquin.Numerics/Statistics/GtInspiredScore.cs))

Function: `GtInspiredScoreResult<T> Compute(ReadOnlySpan<T> returns, T benchmarkMean, T epsilon)`

#### Test #68: Compute_OutperformingStrategy_MatchesVariantStandardBranch

Paper line: 192–260; PDF page(s) 4–5; Equations (1)–(2) and Algorithm 1; local preprint version.

Basis: Independently tested library variant; no author-code or literal paper-result parity.

**Inputs:**

- returns: [0.010,0.012,0.008,0.015,0.009,0.011,0.013,−0.004,0.014,0.010,0.012,0.009]
- benchmarkMean: 0
- epsilon: default 1e-6

**Output / assertions:**

- Mu: 0.009916666666666666 ± 1e-15; Z: 7.053452892053309 ± 1e-9
- RSquared: 0.9891432653097217 ± 1e-12; DownsideDeviation: 0.0011547005383792514 ± 1e-15
- Score: −16.580470598518065 ± 1e-9

Status: **Implemented** — [GtInspiredScoreTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/GtInspiredScoreTests.cs) :: `Compute_OutperformingStrategy_MatchesVariantStandardBranch`.

#### Test #69: S27_GtInspiredScore_ReferencePin_VariantFormula

Paper line: 192–260; PDF page(s) 4–5; Equations (1)–(2) and Algorithm 1; local preprint version.

Basis: Independently tested library variant; no author-code or literal paper-result parity.

**Inputs:**

- returns: [0.010,0.012,0.008,0.015,0.009,0.011,0.013,−0.004,0.014,0.010,0.012,0.009]
- benchmarkMean: 0
- epsilon: default 1e-6

**Output / assertions:**

- Mu: 0.009916666666666666 ± 1e-15; Z: 7.053452892053309 ± 1e-9
- RSquared: 0.9891432653097217 ± 1e-12; DownsideDeviation: 0.0011547005383792514 ± 1e-15
- Score: −16.580470598518065 ± 1e-9

Status: **Implemented** — [StatisticsReviewFixTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/StatisticsReviewFixTests.cs) :: `S27_GtInspiredScore_ReferencePin_VariantFormula`.

#### Test #70: Compute_Underperformer_UsesLargePenaltyBranch

Paper line: 192–260; PDF page(s) 4–5; Equations (1)–(2) and Algorithm 1; local preprint version.

Basis: Independently tested library variant; no author-code or literal paper-result parity.

**Inputs:**

- returns: [0.001,−0.002,0.0015,−0.001,0.0005,−0.0008]
- benchmarkMean: 0.0005
- epsilon: default 1e-6

**Output / assertions:**

- Z: −1.1541678045384953; Score: 188.40003151324478; absolute tolerance 1e-9

Status: **Implemented** — [GtInspiredScoreTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/GtInspiredScoreTests.cs) :: `Compute_Underperformer_UsesLargePenaltyBranch`.

#### Test #71: Compute_MarginalOutperformer_UsesSmoothTransitionBranch

Paper line: 192–260; PDF page(s) 4–5; Equations (1)–(2) and Algorithm 1; local preprint version.

Basis: Independently tested library variant; no author-code or literal paper-result parity.

**Inputs:**

- returns: [0.002,0.001,0.003,0.0015,0.0025,0.001,0.002,0.0018]
- benchmarkMean: 0.00175
- epsilon: default 1e-6

**Output / assertions:**

- Z: 0.40824829046386324; Score: 44.664288567984045; absolute tolerance 1e-9

Status: **Implemented** — [GtInspiredScoreTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/GtInspiredScoreTests.cs) :: `Compute_MarginalOutperformer_UsesSmoothTransitionBranch`.

#### Test #72: Compute_ZeroDownsideDeviation_EpsilonGuardsDenominator

Paper line: 192–260; PDF page(s) 4–5; Equations (1)–(2) and Algorithm 1; local preprint version.

Basis: Independently tested library variant; no author-code or literal paper-result parity.

**Inputs:**

- returns: [0.01,0.02,0.015,0.012,0.018,0.011,0.013,0.016]
- benchmarkMean: 0
- epsilon: default 1e-6

**Output / assertions:**

- DownsideDeviation: exactly 0; Score: −35101.78228497962 ± 1e-6

Status: **Implemented** — [GtInspiredScoreTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/GtInspiredScoreTests.cs) :: `Compute_ZeroDownsideDeviation_EpsilonGuardsDenominator`.

#### Test #73: Compute_JustAboveZEqualsOne_UsesStandardBranchFormula

Paper line: 192–260; PDF page(s) 4–5; Equations (1)–(2) and Algorithm 1; local preprint version.

Basis: Independently tested library variant; no author-code or literal paper-result parity.

**Inputs:**

- returns: [0.006,0.002,0.006,0.002,0.004]
- benchmarkMean: 0.0030966285370900852
- epsilon: default 1e-6

**Output / assertions:**

- Z: 1.01 ± 1e-9; Score: −38.84225537863123 ± 1e-6

Status: **Implemented** — [GtInspiredScoreTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/GtInspiredScoreTests.cs) :: `Compute_JustAboveZEqualsOne_UsesStandardBranchFormula`.

#### Test #74: Compute_JustBelowZEqualsOne_UsesMarginalBranchFormula

Paper line: 192–260; PDF page(s) 4–5; Equations (1)–(2) and Algorithm 1; local preprint version.

Basis: Independently tested library variant; no author-code or literal paper-result parity.

**Inputs:**

- returns: [0.006,0.002,0.006,0.002,0.004]
- benchmarkMean: 0.0031145170809100836
- epsilon: default 1e-6

**Output / assertions:**

- Z: 0.99 ± 1e-9; Score: 0.9950166250832115 ± 1e-6

Status: **Implemented** — [GtInspiredScoreTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/GtInspiredScoreTests.cs) :: `Compute_JustBelowZEqualsOne_UsesMarginalBranchFormula`.

#### Test #75: Compute_IsMultiplicativeRatio_NotAdditiveWeightedSum

Paper line: 192–260; PDF page(s) 4–5; Equations (1)–(2) and Algorithm 1; local preprint version.

Basis: Paper-derived property.

**Inputs:**

- returns: [0.010,0.012,0.008,0.015,0.009,0.011,0.013,−0.004,0.014,0.010,0.012,0.009]
- benchmarkMean: 0
- epsilon: default 1e-6

**Output / assertions:**

- Score < −1

Status: **Implemented** — [GtInspiredScoreTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/GtInspiredScoreTests.cs) :: `Compute_IsMultiplicativeRatio_NotAdditiveWeightedSum`.

#### Test #76: GtInspiredScore_SignificanceGate_AnchorsAtZEqualsOne

Paper line: 192–260; PDF page(s) 4–5; Equations (1)–(2) and Algorithm 1; local preprint version.

Basis: Paper-derived property.

**Inputs:**

- returns: [0.006,0.002,0.006,0.002,0.004]
- benchmarkMean: mean(returns) − sampleSd(returns)/sqrt(5)
- epsilon: default 1e-6

**Output / assertions:**

- Z: 1 within 1e-9; Score: 0 within 1e-9
- Score is also in [−38.84225537863123,0.9950166250832115+1e-6]

Status: **Implemented** — [StatisticsReviewFixTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/StatisticsReviewFixTests.cs) :: `GtInspiredScore_SignificanceGate_AnchorsAtZEqualsOne`.

This boundary value alone cannot distinguish the two adjacent branches because ln(1)=0; the z=0.99 and z=1.01 tests above do.

### Class: `GtInspiredScore` (filename: [GtInspiredScore.cs](../src/Boutquin.Numerics/Statistics/GtInspiredScore.cs))

Function: `GtInspiredScoreResult Compute(ReadOnlySpan<double> returns, double benchmarkMean = 0.0)`

#### Test #77: GtInspiredScore_MatchesReference

Paper line: 192–260; PDF page(s) 4–5; Equations (1)–(2) and Algorithm 1; local preprint version.

Basis: Independently tested library variant; no author-code or literal paper-result parity.

**Inputs:**

- returns: [scalar_stats.json](../tests/Verification/vectors/scalar_stats.json), JSON pointer `/generalization_score/returns` (504 observations)
- benchmarkMean: 0
- epsilon: default 1e-6

**Output / assertions:**

- Score: −0.1737672316087327; Mu: 0.0011970247611585343; Z: 2.6727444602446058
- RSquared: 0.9535117825376453; DownsideDeviation: 0.006456457975401322; absolute tolerance 1e-10

Status: **Implemented** — [ScalarStatisticsVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/ScalarStatisticsVerificationTests.cs) :: `GtInspiredScore_MatchesReference`.

### Class: `GtInspiredScore<double>` (filename: [GtInspiredScore.cs](../src/Boutquin.Numerics/Statistics/GtInspiredScore.cs))

Function: `GtInspiredScoreResult<T> Compute(ReadOnlySpan<T> returns, T benchmarkMean, T epsilon)`

#### Test #78: GtScore_ConstantSeries_TakesZeroZ

Paper line: 192–260; PDF page(s) 4–5; Equations (1)–(2) and Algorithm 1; local preprint version.

Basis: Declared zero-variance variant.

**Inputs:**

- returns: a constant value repeated count times; (value,count) rows: (0.01,3),(0.01,4),(0.01,10),(0.01,100),(0.003,7),(−0.02,10)
- benchmarkMean: 0
- epsilon: default 1e-6

**Output / assertions:**

- Z: exactly 0; Mu: exactly value; Score: 100+100×(1−exp(−1)) within 1e-12

Status: **Implemented** — [ArenaR1B11PboGenScoreTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/ArenaR1B11PboGenScoreTests.cs) :: `GtScore_ConstantSeries_TakesZeroZ`.

The library defines Z=0 when sample standard deviation is zero; substituting that convention into Algorithm 1 gives the asserted score.


<a id="paper-pw"></a>

## Paper: Politis_White_2004.pdf

Source: [primary source](https://public.econ.duke.edu/~ap172/Politis_White_2004.pdf).

Source PDF SHA-256: `5c0171ca2a8c355392973770c25c2264d407a36b478eddd709cafd4420ee2368`.

### Class: `PolitisWhiteBlockLength` (filename: [PolitisWhiteBlockLength.cs](../src/Boutquin.Numerics/MonteCarlo/PolitisWhiteBlockLength.cs))

Function: `double EstimateDouble(ReadOnlySpan<double> series)`

#### Test #79: PolitisWhiteBlockLength_MatchesPaperReference

Paper line: 360–415; PDF page(s) 6; Equation (8), flat-top kernel and stationary optimum.

Basis: Paper-derived formula.

**Inputs:**

- Theory row politis_white: series = [bootstrap.json](../tests/Verification/vectors/bootstrap.json), JSON pointer `/politis_white/series` (512 observations)
- Theory row politis_white_negative: series = [bootstrap.json](../tests/Verification/vectors/bootstrap.json), JSON pointer `/politis_white_negative/series` (512 observations)

**Output / assertions:**

- Block lengths respectively: 9.295216713475261, 13.573602735472155; absolute tolerance 1e-10

Status: **Implemented** — [BootstrapVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/BootstrapVerificationTests.cs) :: `PolitisWhiteBlockLength_MatchesPaperReference`.

Applies Patton–Politis–White 2009 corrected D_SB=2g(0)² (corrections 1 and 4, lines 63–90). The selector begins the stability window at m, following the corrected companion code. The oracle follows the complete corrected author listing: common truncated-row correlation and sample covariance, separate column means, strict significant-lag fallback and upper-only cap.

#### Test #80: EstimateDouble_WindowIsFiveLags_EvenWhenLog10NExceedsFive

Paper line: 360–415; PDF page(s) 6; Equation (8), flat-top kernel and stationary optimum.

Basis: Paper-derived formula.

**Inputs:**

- n: 200000; independent Gaussian e and f arrays of length n+7 generated alternately
- GaussianSampler/Xoshiro seed: 20260930
- series[t]: e[t+7]+0.5×e[t+6]+0.3×(f[t+7]+f[t]), t=0..n−1

**Output / assertions:**

- Block length: (G/g0)^(2/3)×cbrt(n), with G=2×R1+4×R2+3×R3, g0=R0+2×R1+2×R2+R3; R_k is the sample covariance of columns x[4..n−1] and x[4−k..n−1−k], using separate column means and denominator n−5
- Relative tolerance: 1e-9

Status: **Implemented** — [PolitisWhiteBlockLengthLagRuleTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/PolitisWhiteBlockLengthLagRuleTests.cs) :: `EstimateDouble_WindowIsFiveLags_EvenWhenLog10NExceedsFive`.

The five-lag window uses max(5,ceil(sqrt(log10(n)))) and selects m=2, M=4 with the corrected companion window. The search horizon and cap follow companion opt_block_length_REV_dec07.m; neither is an extra numeric table result in the 2004 paper.

#### Test #81: EstimateDouble_TranslatedRamp_PreservesBlockLength

Paper line: 360–415; PDF page(s) 6; Equation (8), flat-top kernel and stationary optimum.

Basis: Paper-derived formula.

**Inputs:**

- series[i]=1e16+2×(i mod 16), i=0..127

**Output / assertions:**

- Block length: 33.85832450374236; absolute tolerance 1e-12

Status: **Implemented** — [PolitisWhiteTranslationTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/PolitisWhiteTranslationTests.cs) :: `EstimateDouble_TranslatedRamp_PreservesBlockLength`.

Analytical/high-precision oracle for the corrected flat-top formula and author lag rules on translated data; not a printed paper series.

#### Test #82: EstimateDouble_TranslatedSteps_MatchesDecimalOracle

Paper line: 360–415; PDF page(s) 6; Equation (8), flat-top kernel and stationary optimum.

Basis: Paper-derived formula.

**Inputs:**

- count=64,128,256; series[i]=1e16+2×((i/8 integer division) mod 4)

**Output / assertions:**

- Block length: 7.211866879624393, 14.725319703185178, 48 respectively; absolute tolerance 1e-11

Status: **Implemented** — [PolitisWhiteTranslationTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/PolitisWhiteTranslationTests.cs) :: `EstimateDouble_TranslatedSteps_MatchesDecimalOracle`.

Analytical/high-precision oracle for the corrected flat-top formula and author lag rules on translated data; not a printed paper series.

#### Test #83: EstimateDouble_TranslatedOscillation_MatchesPaperFormula

Paper line: 360–415; PDF page(s) 6; Equation (8), flat-top kernel and stationary optimum.

Basis: Paper-derived formula.

**Inputs:**

- series[i]=1e16+2×Round(5×sin(0.3×i)), i=0..199; Round uses .NET ties-to-even

**Output / assertions:**

- Block length: 43; absolute tolerance 1e-12

Status: **Implemented** — [PolitisWhiteTranslationTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/PolitisWhiteTranslationTests.cs) :: `EstimateDouble_TranslatedOscillation_MatchesPaperFormula`.

Analytical/high-precision oracle for the corrected flat-top formula and author lag rules on translated data; not a printed paper series.


<a id="paper-ppw"></a>

## Paper: Patton_Politis_White_2009.pdf

Source: [primary source](https://public.econ.duke.edu/~ap172/Patton_Politis_White_2009.pdf).

Source PDF SHA-256: `5ec74ea3d71e28e11d0154cfb751dd442f18ba08d64951ad00b19b59acf6a150`.

### Class: `PolitisWhiteBlockLength` (filename: [PolitisWhiteBlockLength.cs](../src/Boutquin.Numerics/MonteCarlo/PolitisWhiteBlockLength.cs))

Function: `double EstimateDouble(ReadOnlySpan<double> series)`

#### Test #84: EstimateDouble_PositiveAr1_MeanTracksStationaryBootstrapOptimum

Paper line: 61–122; PDF page(s) 2–3; Corrections 1–5 and corrected Table 1.

Basis: Paper-derived statistical envelope.

**Inputs:**

- AR(1): x_t = 0.5 × x_(t−1) + epsilon_t, epsilon_t ~ N(0,1)
- Observations per replication: 5000; burn-in: 200; replications: 40
- GaussianSampler/Xoshiro seed: 20260927

**Output / assertions:**

- Theoretical optimum b*=(2|phi|/(1−phi²))^(2/3)×5000^(1/3) = 20.714883373025724
- Asserted mean estimate / b*: in [0.85,1.1]

Status: **Implemented** — [PolitisWhiteBlockLengthTheoryTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/PolitisWhiteBlockLengthTheoryTests.cs) :: `EstimateDouble_PositiveAr1_MeanTracksStationaryBootstrapOptimum`.

These are new AR(1) experiments against the corrected asymptotic formula, not the finite-sample experiment printed in corrected Tables 2–4.

#### Test #85: EstimateDouble_NegativeAr1_MeanTracksStationaryBootstrapOptimum

Paper line: 61–122; PDF page(s) 2–3; Corrections 1–5 and corrected Table 1.

Basis: Paper-derived statistical envelope.

**Inputs:**

- AR(1): x_t = -0.4 × x_(t−1) + epsilon_t, epsilon_t ~ N(0,1)
- Observations per replication: 5000; burn-in: 200; replications: 40
- GaussianSampler/Xoshiro seed: 20260928

**Output / assertions:**

- Theoretical optimum b*=(2|phi|/(1−phi²))^(2/3)×5000^(1/3) = 16.552507832792617
- Asserted mean estimate / b*: in [0.85,1.15]

Status: **Implemented** — [PolitisWhiteBlockLengthTheoryTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/PolitisWhiteBlockLengthTheoryTests.cs) :: `EstimateDouble_NegativeAr1_MeanTracksStationaryBootstrapOptimum`.

These are new AR(1) experiments against the corrected asymptotic formula, not the finite-sample experiment printed in corrected Tables 2–4.

#### Test #86: EstimateDouble_PersistentAr1_LagWindowIsNotTruncatedBelowPaperHorizon

Paper line: 61–122; PDF page(s) 2–3; Corrections 1–5 and corrected Table 1.

Basis: Paper-derived statistical envelope.

**Inputs:**

- AR(1): x_t = 0.95 × x_(t−1) + epsilon_t, epsilon_t ~ N(0,1)
- Observations per replication: 5000; burn-in: 200; replications: 40
- GaussianSampler/Xoshiro seed: 20260929

**Output / assertions:**

- Theoretical optimum b*=(2|phi|/(1−phi²))^(2/3)×5000^(1/3) = 123.8290834152461
- Asserted mean estimate / b*: in [0.75,1.1]

Status: **Implemented** — [PolitisWhiteBlockLengthTheoryTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/PolitisWhiteBlockLengthTheoryTests.cs) :: `EstimateDouble_PersistentAr1_LagWindowIsNotTruncatedBelowPaperHorizon`.

These are new AR(1) experiments against the corrected asymptotic formula, not the finite-sample experiment printed in corrected Tables 2–4.

#### Test #87: EstimateDouble_NearUnitRootSeries_IsCappedAtPattonBmax

Paper line: 61–122; PDF page(s) 2–3; Corrections 1–5 and corrected Table 1.

Basis: Companion author-code rule.

**Inputs:**

- series: Gaussian random walk, 100000 observations, initial level 0
- GaussianSampler/Xoshiro seed: 20261001

**Output / assertions:**

- Block length: exactly ceil(min(3×sqrt(100000),100000/3)) = 949

Status: **Implemented** — [PolitisWhiteBlockLengthLagRuleTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/PolitisWhiteBlockLengthLagRuleTests.cs) :: `EstimateDouble_NearUnitRootSeries_IsCappedAtPattonBmax`.

The binding cap comes from opt_block_length_REV_dec07.m, not a formula printed in the correction PDF.

### Class: `PolitisWhiteBlockLength` (filename: [PolitisWhiteBlockLength.cs](../src/Boutquin.Numerics/MonteCarlo/PolitisWhiteBlockLength.cs))

Function: `internal static double StationaryOptimum(double covarianceMoment, double spectralSum, int sampleLength)`

#### Test #88: Corrected Table 1 AR(1) optima

Paper line: 112–122; corrected Table 1; PDF page(s) 3.

Basis: Printed stationary optima; independent population autocovariances; reconciled circular formula.

**Inputs:**

- Stationary Gaussian AR(1), unit innovation variance
- (phi,N) rows: (0.7,200),(0.7,800),(0.1,200),(0.1,800),(−0.4,200),(−0.4,800)
- R(k)=phi^|k|/(1−phi²); independently sum 512 signed two-sided lags for G=Σ|k|R(k), g(0)=ΣR(k); omitted tail <1e-73
- Formula under test is the internal `StationaryOptimum` also called by the public finite-sample estimator

**Output / assertions:**

- Population stationary-bootstrap optima: [11.46520712622, 18.1998818532, 2.013445441238, 3.196145411501, 5.660878050251, 8.986083772031]; agreement with the independent AR(1) closed form within 1e-12
- Printed stationary values [11.47,18.20,2.01,3.20,5.66,8.99]; rounding tolerance 0.005
- Circular formula b_CB=(3/2)^(1/3)×b_SB: [13.12438589121208,20.83366397014665,2.304819673188722,3.658673173816779,6.480087729479396,10.28649807862182]
- Reconciled circular rounding [13.12,20.83,2.30,3.66,6.48,10.29]; tolerance 0.005

Status: **Implemented** — [PolitisWhiteCorrectedTableTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/PolitisWhiteCorrectedTableTests.cs) :: `StationaryOptimum_CorrectedTable1_MatchesPopulationAutocovariances`.

These are population optima, not values an estimated finite sample must equal. Circular selection is a theoretical ratio check; it is not exposed by the public method. The printed circular cells 2.31 and 10.23 disagree with the equations: 2.30481967 rounds to 2.30 and 10.28649808 rounds to 10.29.

### Class: `PolitisWhiteBlockLength` (filename: [PolitisWhiteBlockLength.cs](../src/Boutquin.Numerics/MonteCarlo/PolitisWhiteBlockLength.cs))

Function: `double EstimateDouble(ReadOnlySpan<double> series)`

#### Test #89: Corrected Table 2 finite-sample experiment

Paper line: 129–144; corrected Table 2; PDF page(s) 3.

Basis: Seeded statistical reconstruction with an independent exact oracle and declared finite-sample conventions.

**Inputs:**

- Same six (phi,N) rows as corrected Table 1; 1000 replications per row
- GaussianSampler / Xoshiro256StarStar seeds: 20090000,20090001,20090002,20090003,20090004,20090005 in row order
- Initial x is N(0,1/(1−phi²)); every observation applies x←phi×x+N(0,1); no burn-in
- Complete RNG, covariance and kernel construction: [generate_ppw2009_vectors.py](../tests/Verification/generate_ppw2009_vectors.py)
- Selected author convention: corrected window m..m+K_N−1, common truncated rows, separate column means, sample covariance and no lower floor

**Output / assertions:**

- Seeded mean estimated stationary block length / population optimum: [0.8439659223592, 0.9237496704359, 0.9530250375115, 0.8998700500305, 1.111442573101, 1.093909408596]
- Seeded standard deviations: [0.3560823001363, 0.2529527654155, 1.211536698457, 0.362297628604, 0.7088814004366, 0.3869628824337]
- Seeded RMSE: [0.3886057684223, 0.2640742872535, 1.211841572829, 0.3757050957883, 0.7172375995056, 0.3980069209598]
- All summaries and selected individual block estimates match independent NumPy within 1e-9; complete goldens/checkpoints: [ppw2009.json](../tests/Verification/vectors/ppw2009.json), `/rows/0` through `/rows/5`
- Printed mean [0.859,0.927,0.959,0.881,1.062,1.081], SD [0.342,0.244,0.943,0.323,0.644,0.368], RMSE [0.370,0.254,0.943,0.344,0.646,0.377] are checked with row-specific statistical envelopes
- Each envelope = six combined Monte Carlo standard errors + 0.0005 printed rounding; SD/RMSE errors use empirical fourth/influence moments, not a Gaussian approximation to the heavy-tailed estimates. Exact tolerance fields are `/rows/i/mean_paper_tolerance`, `sd_paper_tolerance`, and `rmse_paper_tolerance`

Status: **Implemented** — [PolitisWhite2009VerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/PolitisWhite2009VerificationTests.cs) :: `EstimateDouble_CorrectedTable2_SeededExperimentMatchesIndependentOracle`.

The paper's exact seeds/observations are unpublished. These are declared independent replications of its six configurations, with exact controls from an independent NumPy implementation of the selected author listing. They do not replay the unpublished 1000 original samples.

#### Test #204: Held-out negative-AR(1) selector regression

Paper line: 61–144; corrected stationary optimum and Table 2 negative-autocorrelation regime; PDF page(s) 2–3.

Basis: Independent-seed statistical control of the corrected companion selector.

**Inputs:**

- phi: −0.4; N: 200; replications: 1000; unit Gaussian innovations
- GaussianSampler / Xoshiro seed: 20090101; stationary Gaussian initial state; construction as Test #89

**Output / assertions:**

- Mean estimated block length / population optimum: within [0.8,1.4]
- Independent NumPy control: 1.06946595; the former window gives 2.43446469 and violates this bound

Status: **Implemented** — [PolitisWhiteCorrectedTableTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/PolitisWhiteCorrectedTableTests.cs) :: `EstimateDouble_NegativeAr1_ShortSampleUsesCorrectedCompanionWindow`.

This held-out stream is separate from the six Table 2 exact-oracle seeds.

<a id="paper-fdb"></a>

## Paper: rbnew3dp.pdf

Source: [primary source](https://www.russell-davidson.research.mcgill.ca/articles/rbnew3dp.pdf).

Source PDF SHA-256: `3c09fe9f734ae4058c505330e9cff97cbf68f72337d3614b89d7b5d271f2fbd6`.

### Class: `FastDoubleBootstrap` (filename: [FastDoubleBootstrap.cs](../src/Boutquin.Numerics/MonteCarlo/FastDoubleBootstrap.cs))

Function: `decimal PValue(ReadOnlySpan<decimal> observations, decimal observed, Func<ReadOnlySpan<decimal>, decimal> statistic, PValueTail tail = PValueTail.RightTail)`

#### Test #90: PValue_MatchesHandComputedFastDoubleBootstrap

Paper line: 437–535; PDF page(s) 9–11; Section 5, Equations (22)–(24); Davidson–MacKinnon 2006 author manuscript, published 2007.

Basis: Paper-derived formula.

**Inputs:**

- observations: [1,2,3]
- outerCount: 101; Pcg64 seed: 7
- statistic: scripted calls alternate outer b and inner b+shift, b=0..100; ignores the resampled data
- (tail,observed,shift) rows: (RightTail,89.5,−20), (RightTail,1000,−20), (LeftTail,10.5,+20), (TwoSided,89.5,−20)

**Output / assertions:**

- P-values respectively: 31/101, 20/101, 31/101, 62/101; absolute tolerance 1e-20

Status: **Implemented** — [FastDoubleBootstrapPaperOracleTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/FastDoubleBootstrapPaperOracleTests.cs) :: `PValue_MatchesHandComputedFastDoubleBootstrap`.

Hand derivation from Equations (22)–(24). The library uses strict empirical tail counts, excluding equality at both ranks, and type-7 interpolation for inner quantiles; the paper discusses nonunique empirical quantiles. The two-sided option folds one-sided tests and is not an absolute-statistic symmetric test.

#### Test #91: PValue_TwoSided_EqualsFoldedOneSidedFastDoubleBootstrapValues

Paper line: 437–535; PDF page(s) 9–11; Section 5, Equations (22)–(24); Davidson–MacKinnon 2006 author manuscript, published 2007.

Basis: Paper-derived property.

**Inputs:**

- observations: [1]; observed: 60.25; outerCount: 101; Pcg64 seed: 3
- Scripted outer statistics b=0..100; inner statistics b−7

**Output / assertions:**

- TwoSided p-value: exactly min(2×min(RightTail p-value,LeftTail p-value),1)

Status: **Implemented** — [FastDoubleBootstrapPaperOracleTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/FastDoubleBootstrapPaperOracleTests.cs) :: `PValue_TwoSided_EqualsFoldedOneSidedFastDoubleBootstrapValues`.

### Class: `FastDoubleBootstrap` (filename: [FastDoubleBootstrap.cs](../src/Boutquin.Numerics/MonteCarlo/FastDoubleBootstrap.cs))

Function: `decimal PValueImposingNull(ReadOnlySpan<decimal> observations, decimal observed, Func<ReadOnlySpan<decimal>, decimal> statistic, Action<Span<decimal>> imposeNull, PValueTail tail = PValueTail.RightTail)`

#### Test #92: FastDoubleBootstrap_PValue_MatchesReferenceWithin8Percent

Paper line: 437–535; PDF page(s) 9–11; Section 5, Equations (22)–(24); Davidson–MacKinnon 2006 author manuscript, published 2007.

Basis: Paper-derived statistical comparison.

**Inputs:**

- observations: [bootstrap.json](../tests/Verification/vectors/bootstrap.json), JSON pointer `/fast_double_bootstrap/series` (100 observations)
- outerCount: 2000; C# Pcg64 seed: 20260701; NumPy reference seed: 20260505
- statistic: arithmetic mean; imposeNull: subtract sample mean from every outer and inner empirical population
- Theory rows: RightTail, LeftTail, TwoSided, each at observed=0.11707565530303186 and observed_upper=0.32820039908383963

**Output / assertions:**

- At observed: expected p-values 0.155, 0.845, 0.31 respectively
- At observed_upper: expected p-values 0.0015, 0.9985, 0.003 respectively
- Absolute error: < 0.08 (8 percentage points)

Status: **Implemented** — [BootstrapVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/BootstrapVerificationTests.cs) :: `FastDoubleBootstrap_PValue_MatchesReferenceWithin8Percent`.

The actual assertion allows 0.08, despite the older method name saying 2 percent. RNG streams differ; this checks the sampling distribution/conventions, not identical resamples.

#### Test #93: PValue_WithNullImposition_HasNominalSizeUnderTheNull

Paper line: 437–535; PDF page(s) 9–11; Section 5, Equations (22)–(24); Davidson–MacKinnon 2006 author manuscript, published 2007.

Basis: Paper-derived statistical envelope.

**Inputs:**

- 400 N(0,1) samples, each of size 20; GaussianSampler/Xoshiro seed 20261002
- observed and statistic: absolute arithmetic mean
- outerCount: 199; Pcg64 seed per replication r: 1000+r; imposeNull: recenter outer and inner populations to mean 0
- Nominal rejection level: 0.05

**Output / assertions:**

- Empirical rejection frequency: in [0.02,0.10]

Status: **Implemented** — [FastDoubleBootstrapNullImpositionTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/FastDoubleBootstrapNullImpositionTests.cs) :: `PValue_WithNullImposition_HasNominalSizeUnderTheNull`.

Tests the null-imposed standard FDB. The source comment cites the 2023 improved paper, but the implemented statistic is the earlier unconditional FDB; it does not reproduce the 2023 conditional refinement.


<a id="paper-mammen"></a>

## Paper: Mammen (1993), Bootstrap and Wild Bootstrap for High Dimensional Linear Models (PDF filename unavailable)

Source: [primary source](https://doi.org/10.1214/aos/1176349025).

### Class: `WildBootstrapResampler` (filename: [WildBootstrapResampler.cs](../src/Boutquin.Numerics/MonteCarlo/WildBootstrapResampler.cs))

Function: `decimal[] Resample(ReadOnlySpan<decimal> residuals)`

#### Test #94: WildMammen_WeightDistribution_MatchesMammenAtoms

Paper line: unavailable in the text extraction; primary PDF could not be retrieved; exact page/line of the two-point law remains unverified.

Basis: Analytical weight-law test; paper locus unverified.

**Inputs:**

- observations: 20000 ones
- weights: Mammen; seed: 20260601

**Output / assertions:**

- Theoretical negative probability: (sqrt(5)+1)/(2×sqrt(5)) = 0.7236067977499789
- Theoretical mean: 0; variance: 1
- Probability tolerance: 4×sqrt(p×(1−p)/20000); mean tolerance: 4/sqrt(20000); variance tolerance: 0.05

Status: **Implemented** — [BootstrapVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/BootstrapVerificationTests.cs) :: `WildMammen_WeightDistribution_MatchesMammenAtoms`.

Mammen atoms are (1−sqrt(5))/2 and (1+sqrt(5))/2. The test checks probability, mean and variance; it does not assert the third moment or reproduce a regression-size table. Exact primary-page attribution remains unresolved.


<a id="paper-sobol"></a>

## Paper: joe-kuo-notes.pdf

Source: [primary source](https://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-notes.pdf).

Source PDF SHA-256: `f4ebdf9c4370fea05519d4f026e9f8458678c8acce986d8fb2862d367eb9806d`.

### Class: `SobolSequence` (filename: [SobolSequence.cs](../src/Boutquin.Numerics/MonteCarlo/SobolSequence.cs))

Function: `double[] Next()`

#### Test #95: SobolSequence_MatchesReferencePointsExactly

Paper line: 8–90; PDF page(s) 1–3; Direction-number recurrence and Gray-code Equations (4)–(5).

Basis: Companion direction-table comparison.

**Inputs:**

- dimension: 4; skip: 0 (the default stream, which emits the origin first); unscrambled
- Generate 16 points; emitted point j is author index j (indices 0..15, including the origin)

**Output / assertions:**

- Every coordinate: [qmc.json](../tests/Verification/vectors/qmc.json), JSON pointer `/sobol` at array indices 0..15
- Absolute tolerance: 1e-10

Status: **Implemented** — [QmcVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/QmcVerificationTests.cs) :: `SobolSequence_MatchesReferencePointsExactly`.

Uses the Joe–Kuo D(6) table via SciPy. Gray-code order follows Equations (4)–(5); the default stream emits the origin (author index 0) as its first point, and `skip: 1` recovers the former stream that omitted it.

#### Test #96: SobolSequence_Dimension32Point2MatchesJoeKuo

Paper line: 8–90; PDF page(s) 1–3; Direction-number recurrence and Gray-code Equations (4)–(5).

Basis: Companion direction-table comparison.

**Inputs:**

- dimension: 32; skip: 1; call Next twice

**Output / assertions:**

- Second emitted point (author index 2), coordinate 32 (zero-based index 31): exactly 0.25

Status: **Implemented** — [QmcVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/QmcVerificationTests.cs) :: `SobolSequence_Dimension32Point2MatchesJoeKuo`.

`skip: 1` starts after the origin, so the second emitted point is author index 2. With the default `skip: 0` the second emitted point is author index 1, whose coordinates are all 0.5.

#### Test #97: SobolSequence_AllEmbeddedRowsAndLaterBitsMatchSciPy

Paper line: 8–90; PDF page(s) 1–3; Direction-number recurrence and Gray-code Equations (4)–(5).

Basis: Companion direction-table comparison.

**Inputs:**

- dimension: 32; skip: 0; unscrambled
- Compare sequence indices: [1, 2, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023]

**Output / assertions:**

- Every coordinate in all 13×32 selected points: [qmc_sobol_extended.json](../tests/Verification/vectors/qmc_sobol_extended.json), JSON pointer `/sobol`
- Absolute tolerance: 1e-10

Status: **Implemented** — [QmcVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/QmcVerificationTests.cs) :: `SobolSequence_AllEmbeddedRowsAndLaterBitsMatchSciPy`.


<a id="paper-halton"></a>

## Paper: halton-1960-quasi-random-sequences.pdf

Source PDF SHA-256: `0a12f2692f67ef83d845e3b17b8a72ec65a0009987f8416eecf80ccb60fa78d5`.

### Class: `HaltonSequence` (filename: [HaltonSequence.cs](../src/Boutquin.Numerics/MonteCarlo/HaltonSequence.cs))

Function: `double[] Next()`

#### Test #98: HaltonSequence_MatchesReferencePointsExactly

Paper line: unavailable in the text extraction; image-only scan: PDF page 3, printed page 85, Equations (7)–(11).

Basis: Paper-derived formula.

**Inputs:**

- dimension: 4; radices: 2,3,5,7; skip: 0; unscrambled
- Generate 15 points after the origin (sequence indices 1..15)

**Output / assertions:**

- Every coordinate: [qmc.json](../tests/Verification/vectors/qmc.json), JSON pointer `/halton` at array indices 1..15
- Absolute tolerance: 1e-10

Status: **Implemented** — [QmcVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/QmcVerificationTests.cs) :: `HaltonSequence_MatchesReferencePointsExactly`.

#### Test #99: HaltonSequence_AllPrimeRadices_MatchSciPyPointAtIndexOne

Paper line: unavailable in the text extraction; image-only scan: PDF page 3, printed page 85, Equations (7)–(11).

Basis: Paper-derived formula.

**Inputs:**

- dimension: 50; skip: 0; first emitted point

**Output / assertions:**

- Coordinate j: 1 / prime[j], where primes are [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229]
- Absolute tolerance: 1e-10

Status: **Implemented** — [QmcVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/QmcVerificationTests.cs) :: `HaltonSequence_AllPrimeRadices_MatchSciPyPointAtIndexOne`.


<a id="paper-hd"></a>

## Paper: harrell-davis-1982-distribution-free-quantile-estimator.pdf

Source PDF SHA-256: `64ae5f465dfe665af62a992198ccd94e1cea121193a1f0ab978d8b3b7287752f`.

### Class: `HarrellDavisPercentile` (filename: [HarrellDavisPercentile.cs](../src/Boutquin.Numerics/MonteCarlo/HarrellDavisPercentile.cs))

Function: `T Compute<T>(ReadOnlySpan<T> sorted, double p)`

#### Test #100: HarrellDavisPercentile_MatchesReference

Paper line: 54–101; PDF page(s) 2; Printed page 636, beta-weighted quantile estimator.

Basis: Paper-derived formula.

**Inputs:**

- sorted: [scalar_stats.json](../tests/Verification/vectors/scalar_stats.json), JSON pointer `/harrell_davis/sorted_sample` (50 observations)
- Theory p: 0.1,0.25,0.5,0.75,0.9

**Output / assertions:**

- Percentiles respectively: −1.5800891311891596,−0.9175510530432394,−0.37946356297952655,0.23469849662755865,0.9480702975454546
- Absolute error ≤ 1e-8×max(abs(reference),1)

Status: **Implemented** — [ScalarStatisticsVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/ScalarStatisticsVerificationTests.cs) :: `HarrellDavisPercentile_MatchesReference`.

Independent scipy.special.betainc weights from Equation (3), not a port of C#.

#### Test #101: HarrellDavis_FivePointSample_MatchesIndependentBetaWeights

Paper line: 54–101; PDF page(s) 2; Printed page 636, beta-weighted quantile estimator.

Basis: Analytical median plus independent beta-density integration.

**Inputs:**

- sorted: [1,2,3,4,5]
- p: 0.5 and 0.25

**Output / assertions:**

- p=0.5: 3 ± 1e-9 (independent symmetry identity)
- p=0.25: 1.769924414314333208 ± 1e-12 (80-digit beta-density quadrature); also 1 < q25 < 2

Status: **Implemented** — [McRandomReviewFixTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/McRandomReviewFixTests.cs) :: `HarrellDavis_FivePointSample_MatchesIndependentBetaWeights`.

The q25 target comes from direct integration of the Beta(3/2,9/2) density over five equal intervals, then weighting those masses by observations 1 through 5. SciPy beta weights and hdquantiles independently agree. This is a formula-derived result, not a printed Harrell–Davis value.


<a id="paper-type7"></a>

## Paper: hyndman-1996-sample-quantiles.pdf

Source PDF SHA-256: `8bff77e70cd6af119e6948a94c755f05ad0e75250b66579f21b35cbbf0fbed6d`.

### Class: `Percentile` (filename: [Percentile.cs](../src/Boutquin.Numerics/MonteCarlo/Percentile.cs))

Function: `T Compute<T>(ReadOnlySpan<T> sorted, double p)`

#### Test #102: Compute_EndPoints

Paper line: unavailable in the text extraction; image-only scan: PDF page 4, printed page 363, Definition 7.

Basis: Paper-derived formula.

**Inputs:**

- sorted (decimal): [1,2,3,4,5]
- p: 0 and 1

**Output / assertions:**

- Percentile: 1 and 5, exactly

Status: **Implemented** — [PercentileTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/PercentileTests.cs) :: `Compute_EndPoints`.

Definition 7: one-based h=1+(n−1)p, with linear interpolation between adjacent order statistics. These are analytical examples, not the paper’s Table 4 comparison sample.

#### Test #103: Compute_Median_MatchesMiddle

Paper line: unavailable in the text extraction; image-only scan: PDF page 4, printed page 363, Definition 7.

Basis: Paper-derived formula.

**Inputs:**

- sorted (decimal): [1,2,3,4,5]
- p: 0.5

**Output / assertions:**

- Percentile: 3, exactly

Status: **Implemented** — [PercentileTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/PercentileTests.cs) :: `Compute_Median_MatchesMiddle`.

Definition 7: one-based h=1+(n−1)p, with linear interpolation between adjacent order statistics. These are analytical examples, not the paper’s Table 4 comparison sample.

#### Test #104: Compute_LinearInterpolation_MatchesNumPyConvention

Paper line: unavailable in the text extraction; image-only scan: PDF page 4, printed page 363, Definition 7.

Basis: Paper-derived formula.

**Inputs:**

- sorted (decimal): [1,2,3,4]
- p: 0.25

**Output / assertions:**

- Percentile: 1.75, exactly

Status: **Implemented** — [PercentileTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/PercentileTests.cs) :: `Compute_LinearInterpolation_MatchesNumPyConvention`.

Definition 7: one-based h=1+(n−1)p, with linear interpolation between adjacent order statistics. These are analytical examples, not the paper’s Table 4 comparison sample.


<a id="paper-pcg"></a>

## Paper: oneill-2014-pcg.pdf

Source PDF SHA-256: `7e516ebe5de1a2c6d3ed16cc76e7b5cd5d501fbc7faeeb7bd7ff0cbc699ed7dd`.

### Class: `Pcg64RandomSource` (filename: [Pcg64RandomSource.cs](../src/Boutquin.Numerics/Random/Pcg64RandomSource.cs))

Function: `ulong NextULong()`

#### Test #105: NextULong_Seed42Stream54_MatchesPcgCReferenceOutput

Paper line: 1960–2045; PDF page(s) 40–41; PCG output permutations; exact stream oracle comes from companion author code.

Basis: Companion author-code known-answer test.

**Inputs:**

- seed: 42; streamId: 54; six consecutive NextULong calls

**Output / assertions:**

- Exact 64-bit hexadecimal words: [86b1da1d72062b68,1304aa46c9853d39,a3670e9e0dd50358,f9090e529a7dae00,c85b9fd837996f2c,606121f8e3919196]

Status: **Implemented** — [Pcg64RandomSourceOracleTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Random/Pcg64RandomSourceOracleTests.cs) :: `NextULong_Seed42Stream54_MatchesPcgCReferenceOutput`.

Oracle is pcg-c test-high/expected/check-pcg64.out, Round 1, not a printed word sequence in the PDF. Author source: [author source](https://github.com/imneme/pcg-c/blob/master/test-high/expected/check-pcg64.out)

#### Test #106: Pcg64_MatchesIndependentReferencePort

Paper line: 1960–2045; PDF page(s) 40–41; PCG output permutations; exact stream oracle comes from companion author code.

Basis: Independent recurrence oracle.

**Inputs:**

- seed: 0xDEADBEEFCAFEF00D; 32 consecutive words

**Output / assertions:**

- Exact word sequence from [McRandomReviewFixTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/McRandomReviewFixTests.cs) :: `PcgReference`

Status: **Implemented** — [McRandomReviewFixTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/McRandomReviewFixTests.cs) :: `Pcg64_MatchesIndependentReferencePort`.

The complete expected recurrence and initialization are inline in the linked test helper; this is not a printed sequence or a statistical-quality replication.


<a id="paper-xoshiro"></a>

## Paper: blackman-2018-xoshiro.pdf

Source PDF SHA-256: `3aee2f264786742812e651595898fb3c06e184abf589c56c14cc7c7b205668f5`.

### Class: `Xoshiro256StarStarRandomSource` (filename: [Xoshiro256StarStarRandomSource.cs](../src/Boutquin.Numerics/Random/Xoshiro256StarStarRandomSource.cs))

Function: `ulong NextULong()`

#### Test #107: Xoshiro256_MatchesIndependentReferencePort

Paper line: 260–374; PDF page(s) 5–8; Xoshiro engine, star-star scrambler and jump discussion.

Basis: Independent recurrence oracle.

**Inputs:**

- seed: 0x0123456789ABCDEF; 32 consecutive words

**Output / assertions:**

- Exact word sequence from [McRandomReviewFixTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/McRandomReviewFixTests.cs) :: `XoshiroReference`

Status: **Implemented** — [McRandomReviewFixTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/McRandomReviewFixTests.cs) :: `Xoshiro256_MatchesIndependentReferencePort`.

The complete expected recurrence and initialization are inline in the linked test helper; this is not a printed sequence or a statistical-quality replication.

### Class: `Xoshiro256StarStarRandomSource` (filename: [Xoshiro256StarStarRandomSource.cs](../src/Boutquin.Numerics/Random/Xoshiro256StarStarRandomSource.cs))

Function: `void Jump()`

#### Test #108: Xoshiro256_JumpPartitionsTheStream

Paper line: 260–374; PDF page(s) 5–8; Xoshiro engine, star-star scrambler and jump discussion.

Basis: Independent recurrence oracle.

**Inputs:**

- seed: 7; initialize both legacy and generic<double> engines; call Jump once; then eight NextULong calls

**Output / assertions:**

- Exact 64-bit words: [156617fd83df2a74,1ccb4975f3ae6cbc,c6b79bd4fd3989f0,335a2f71da09ba0e,cae07fd267535616,3cd5ab0d0dab15f7,a8ecdfbee86de85c,83139c6298ae2616]

Status: **Implemented** — [RandomSourceTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Random/RandomSourceTests.cs) :: `Xoshiro256_JumpPartitionsTheStream`.

The 2^128-step jump oracle was derived by squaring the GF(2) transition matrix 128 times; it uses no implementation jump constants.


<a id="paper-fc"></a>

## Paper: fritsch-carlson-1980-monotone-piecewise-cubic-interpolation.pdf

Source PDF SHA-256: `7630c6e75eacb32c485b47b2bd9de75dfe7aa2bed105c1365f027719607c89a3`.

### Class: `MonotoneCubicInterpolator<double>` (filename: [MonotoneCubicInterpolator.cs](../src/Boutquin.Numerics/Interpolation/MonotoneCubicInterpolator.cs))

Function: `T Interpolate(T x, ReadOnlySpan<T> xs, ReadOnlySpan<T> ys)`

#### Test #109: Interpolate_MonotoneQuadraticOnNonUniformGrid_UsesFritschCarlsonStepOne

Paper line: 196–289; PDF page(s) 4–6; Algorithm Steps 1–2A, three-point differences and monotonicity region.

Basis: Paper-derived formula.

**Inputs:**

- xs: [0,1,3]
- ys: [0,1,9]
- Theory query x: 0.5, 2, 2.5

**Output / assertions:**

- Interpolated values respectively: 0.25, 4, 6.25; absolute tolerance 1e-14

Status: **Implemented** — [MonotoneCubicInterpolatorTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Interpolation/MonotoneCubicInterpolatorTests.cs) :: `Interpolate_MonotoneQuadraticOnNonUniformGrid_UsesFritschCarlsonStepOne`.

Nonuniform three-point initialization reproduces x² here. This is an analytical example of Step 1, not a data row printed in the paper.

### Class: `MonotoneCubicSpline<double>` (filename: [MonotoneCubicSpline.cs](../src/Boutquin.Numerics/Interpolation/MonotoneCubicSpline.cs))

Function: `T Interpolate(T x)`

#### Test #110: Interpolate_MonotoneQuadraticOnNonUniformGrid_UsesFritschCarlsonStepOne

Paper line: 196–289; PDF page(s) 4–6; Algorithm Steps 1–2A, three-point differences and monotonicity region.

Basis: Paper-derived formula.

**Inputs:**

- xs: [0,1,3]
- ys: [0,1,9]
- Theory query x: 0.5, 2, 2.5

**Output / assertions:**

- Interpolated values respectively: 0.25, 4, 6.25; absolute tolerance 1e-14

Status: **Implemented** — [MonotoneCubicSplineTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Interpolation/MonotoneCubicSplineTests.cs) :: `Interpolate_MonotoneQuadraticOnNonUniformGrid_UsesFritschCarlsonStepOne`.

Nonuniform three-point initialization reproduces x² here. This is an analytical example of Step 1, not a data row printed in the paper.

### Class: `MonotoneCubicInterpolator<double>` (filename: [MonotoneCubicInterpolator.cs](../src/Boutquin.Numerics/Interpolation/MonotoneCubicInterpolator.cs))

Function: `T Interpolate(T x, ReadOnlySpan<T> xs, ReadOnlySpan<T> ys)`

#### Test #111: Interpolate_DecreasingDataWithClippedTangent_StaysWithinSegment

Paper line: 196–289; PDF page(s) 4–6; Algorithm Steps 1–2A, three-point differences and monotonicity region.

Basis: Paper-derived property.

**Inputs:**

- xs: [0,1,2]
- ys: [0,−1,−10]
- x: k/20 for k=1..19; also constructs MonotoneCubicSpline<double>

**Output / assertions:**

- Every queried value, from both classes, is in [−1,0]

Status: **Implemented** — [MonotoneCubicInterpolatorTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Interpolation/MonotoneCubicInterpolatorTests.cs) :: `Interpolate_DecreasingDataWithClippedTangent_StaysWithinSegment`.

Pins sign-preserving radius-3 clipping in monotonicity region S2, not a printed numerical table.


<a id="paper-hw"></a>

## Paper: hagan-west-2006-interpolation-curve-construction.pdf

Source PDF SHA-256: `0c509508c555b28013ff6c372a4c4056f742c7851da53f4775b5a5a4d16a1ef8`.

### Class: `MonotoneConvexInterpolator<double>` (filename: [MonotoneConvexInterpolator.cs](../src/Boutquin.Numerics/Interpolation/MonotoneConvexInterpolator.cs))

Function: `T Interpolate(T x, ReadOnlySpan<T> xs, ReadOnlySpan<T> ys)`

#### Test #112: Interpolate_IncreasingForwards_UsesSectorTwoFlatSegment

Paper line: 983–1235; PDF page(s) 20–25; Section 6, endpoint forwards and monotone-convex sectors.

Basis: Paper-derived formula.

**Inputs:**

- xs: [0,1,2,3]
- ys: [0,0.01,0.021,0.051]
- x: 1.315

**Output / assertions:**

- Interpolated NCR: 0.0133075; absolute tolerance 1e-14

Status: **Implemented** — [MonotoneConvexInterpolatorTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Interpolation/MonotoneConvexInterpolatorTests.cs) :: `Interpolate_IncreasingForwards_UsesSectorTwoFlatSegment`.

Hand-derived sector (ii) integral: f1=0.0105, f2=0.0205, g_left=−0.0005, g_right=0.0095, eta=0.85; Equations (49)–(50).

#### Test #113: Interpolate_SectorThreeAndFour_MatchPaperFormulas

Paper line: 983–1235; PDF page(s) 20–25; Section 6, endpoint forwards and monotone-convex sectors.

Basis: Paper-derived formula.

**Inputs:**

- xs: [0,1,2,3]; x: 1.5
- Sector (iii) ys: [0,0.03,0.04,0.048]
- Sector (iv) ys: [0,0.01,0.04,0.06]

**Output / assertions:**

- Interpolated NCR respectively: 0.0355, 0.02515625; absolute tolerance 1e-14

Status: **Implemented** — [MonotoneConvexInterpolatorTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Interpolation/MonotoneConvexInterpolatorTests.cs) :: `Interpolate_SectorThreeAndFour_MatchPaperFormulas`.

Independent hand integrals of Equations (52) and (55)–(56).

#### Test #114: Interpolate_IncreasingForwards_ForwardCurveIsNonDecreasing

Paper line: 983–1235; PDF page(s) 20–25; Section 6, endpoint forwards and monotone-convex sectors.

Basis: Paper-derived property.

**Inputs:**

- xs: [0,1,2,3]; ys: [0,0.01,0.021,0.051]
- t: 0.01,0.02,...,2.99; finite-difference epsilon: 1e-6

**Output / assertions:**

- Central-difference forward at each t is at least the previous forward minus 1e-8

Status: **Implemented** — [MonotoneConvexInterpolatorTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Interpolation/MonotoneConvexInterpolatorTests.cs) :: `Interpolate_IncreasingForwards_ForwardCurveIsNonDecreasing`.

### Class: `MonotoneConvexInterpolator` (filename: [MonotoneConvexInterpolator.cs](../src/Boutquin.Numerics/Interpolation/MonotoneConvexInterpolator.cs))

Function: `double Interpolate(double x, ReadOnlySpan<double> xs, ReadOnlySpan<double> ys)`

#### Test #115: MonotoneConvexInterpolator_MatchesHaganWestReference

Paper line: 983–1235; PDF page(s) 20–25; Section 6, endpoint forwards and monotone-convex sectors.

Basis: Paper-derived formula.

**Inputs:**

- xs: [0.0, 0.25, 1.0, 5.0, 10.0, 30.0]
- ys (NCR): [0.0, 0.005012541823544286, 0.030459207484708574, 0.18632957819149348, 0.3856624808119846, 1.3862943611198906]
- x queries: [0.1, 0.5, 3.0, 7.5, 20.0]

**Output / assertions:**

- Interpolated values in query order: [0.0018592901667502748, 0.013096971200456112, 0.10759529252929927, 0.2847567170024309, 0.8554833804920464]
- Absolute tolerance: 1e-10

Status: **Implemented** — [InterpolationVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/InterpolationVerificationTests.cs) :: `MonotoneConvexInterpolator_MatchesHaganWestReference`.

Python evaluates the paper’s piecewise forward formula and integrates with scipy.integrate.quad; the library uses closed-form integrals. Inputs/results also appear at [interpolation.json](../tests/Verification/vectors/interpolation.json), JSON pointer `/monotone_convex`.

### Class: `LogLinearInterpolator` (filename: [LogLinearInterpolator.cs](../src/Boutquin.Numerics/Interpolation/LogLinearInterpolator.cs))

Function: `double Interpolate(double x, ReadOnlySpan<double> xs, ReadOnlySpan<double> ys)`

#### Test #116: LogLinearInterpolator_MatchesNumpyReference

Paper line: 321–372; Section 3, piecewise constant forwards/log-linear discount interpolation; PDF page(s) 7–8.

Basis: Paper-derived formula.

**Inputs:**

- xs: [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
- ys (discount factors): [1.0, 0.9875, 0.975, 0.95, 0.9, 0.78, 0.6]
- x queries: [0.1, 0.375, 0.75, 1.5, 3.0, 7.0]

**Output / assertions:**

- Interpolated values: [0.9949811239751459, 0.9812300953395182, 0.9624188277460078, 0.9246621004453465, 0.8580775517949689, 0.7022912569332717]
- Absolute tolerance: 1e-10

Status: **Implemented** — [InterpolationVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/InterpolationVerificationTests.cs) :: `LogLinearInterpolator_MatchesNumpyReference`.

Equivalent discount-factor formula D(x)=exp((1−w)log(D_left)+w log(D_right)); synthetic formula check, not a paper curve table.

### Class: `FlatForwardInterpolator` (filename: [FlatForwardInterpolator.cs](../src/Boutquin.Numerics/Interpolation/FlatForwardInterpolator.cs))

Function: `double Interpolate(double x, ReadOnlySpan<double> xs, ReadOnlySpan<double> ys)`

#### Test #117: FlatForwardInterpolator_MatchesNumpyReference

Paper line: 321–372; Section 3, piecewise constant forwards/log-linear discount interpolation; PDF page(s) 7–8.

Basis: Paper-derived formula.

**Inputs:**

- xs: [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
- ys (discount factors): [1.0, 0.9875, 0.975, 0.95, 0.9, 0.78, 0.6]
- x queries: [0.1, 0.375, 0.75, 1.5, 3.0, 7.0]

**Output / assertions:**

- Interpolated values: [0.9949811239751459, 0.9812300953395182, 0.9624188277460078, 0.9246621004453465, 0.8580775517949689, 0.7022912569332718]
- Absolute tolerance: 1e-10

Status: **Implemented** — [InterpolationVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/InterpolationVerificationTests.cs) :: `FlatForwardInterpolator_MatchesNumpyReference`.

Equivalent discount-factor formula D(x)=exp((1−w)log(D_left)+w log(D_right)); synthetic formula check, not a paper curve table.


<a id="paper-jarre"></a>

## Paper: jarre-2025-cubic-spline-functions-revisited.pdf

Source PDF SHA-256: `331bcd0aa5bd3828bb2d7762e8846d064b7a8421edb5a9a7c5adfe0f2470b0f5`.

### Class: `CubicSplineInterpolator` (filename: [CubicSplineInterpolator.cs](../src/Boutquin.Numerics/Interpolation/CubicSplineInterpolator.cs))

Function: `double Interpolate(double x)`

#### Test #118: QSpline_BetweenNodes_ReproducesQuadraticAndCubicExactly

Paper line: 461–527; PDF page(s) 10–11; Five-point endpoint curvature, Theorem 2 and polynomial exactness.

Basis: Paper-derived formula.

**Inputs:**

- boundary: SplineBoundary.QSpline
- Four Theory rows: uniform/irregular grid × quadratic/cubic function
- Uniform xs: [0,1,2,3,4,5]; irregular xs: [0,0.2,1.1,2.3,4.5,5.7]
- ys: f(xs); quadratic f(x)=2x²−3x+1, cubic f(x)=x³−2x²+3x−1
- Queries per interval: left+fraction×width, fractions=[0.25,0.5,0.75]

**Output / assertions:**

- Every query equals the corresponding polynomial f(x) within 1e-10

Status: **Implemented** — [CubicSplineInterpolatorTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Interpolation/CubicSplineInterpolatorTests.cs) :: `QSpline_BetweenNodes_ReproducesQuadraticAndCubicExactly`.

Theorem 2 has a zero error bound for degree≤3; testing between knots exercises the endpoint condition, unlike node-only interpolation. QSpline requires at least six knots, and both grids have six.

#### Test #119: QSpline_Quartic_UndampedOnZeroFifthDifference

Paper line: 461–527; PDF page(s) 10–11; Five-point endpoint curvature, Theorem 2 and polynomial exactness.

Basis: Paper-derived formula.

**Inputs:**

- boundary: SplineBoundary.QSpline; ys=x⁴
- Uniform xs: [0,1,2,3,4,5]; query x: 0.5 and 3.5
- Irregular xs: [0,0.2,1.1,2.3,4.5,5.7]; query x: 0.1 and 3.4

**Output / assertions:**

- Uniform values respectively: −7/76, 5701/38
- Irregular values respectively: 6844729/1819220000, 1194051751/9096100
- Absolute tolerance: 1e-10

Status: **Implemented** — [CubicSplineInterpolatorTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Interpolation/CubicSplineInterpolatorTests.cs) :: `QSpline_Quartic_UndampedOnZeroFifthDifference`.

Five-point degree-4 endpoint fits give exact f″=12x², and the fifth difference vanishes, so the safeguard leaves them undamped. The expected between-knot values solve the spline moment equations on the six knots with those endpoint curvatures.

#### Test #120: QSpline_ApproximatesSinWell

Paper line: 461–527; PDF page(s) 10–11; Five-point endpoint curvature, Theorem 2 and polynomial exactness.

Basis: Library accuracy envelope.

**Inputs:**

- boundary: SplineBoundary.QSpline
- xs[i]=i×pi/10, i=0..10; ys[i]=sin(xs[i])
- Queries: each interval midpoint

**Output / assertions:**

- Maximum midpoint absolute error: < 0.01

Status: **Implemented** — [CubicSplineInterpolatorTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Interpolation/CubicSplineInterpolatorTests.cs) :: `QSpline_ApproximatesSinWell`.

This is not Jarre Table 1: it uses 11 knots and checks midpoints with a loose bound, while the paper tabulates supremum errors at 6,12,24,48,96 knots.

#### Test #121: Table 1 maximum errors

Paper line: 618–627; Table 1; PDF page(s) 13.

Basis: Printed uniform results plus independent source-equation control for NAT and NAK; Q and RNAK follow the authenticated October 29, 2025 author listing.

**Inputs:**

- f(x): sin(x); interval: [0,pi]
- knot counts: [6,12,24,48,96]
- mesh: equidistant
- ys: f(xs); boundary modes: Natural, NotAKnot, QSpline, RevisedNotAKnot (RNAK)

**Output / assertions:**

- Printed supremum errors in knot-count order (retained for comparison):
- NAT: [0.00045, 1.8e-05, 9.1e-07, 5.2e-08, 3.1e-09]
- NAK: [0.0027, 5.5e-05, 1.4e-06, 5.2e-08, 3.1e-09]
- Q: [0.0022, 4e-05, 9.6e-07, 5.6e-08, 3.1e-09]
- RNAK: [0.0016, 1.8e-05, 9.1e-07, 5.2e-08, 3.1e-09]

Independent oracle maxima actually asserted against production (NAT and NAK only; same knot-count order; absolute tolerance 5e-13):

- NAT: [0.0004472573474413, 1.768167316751e-05, 9.107118865703e-07, 5.204294217798e-08, 3.115247615426e-09]
- NAK: [0.002715199618203, 5.451326246471e-05, 1.379862092338e-06, 5.204294217798e-08, 3.115247615426e-09]

July 2025 preprint oracle maxima for Q and RNAK (historical comparison, not asserted: production implements the October 2025 author-listing Q and RNAK algorithms):

- Q: [0.002192658654907, 7.163038654348e-05, 1.960928728804e-06, 5.612839323529e-08, 3.115247615426e-09]
- RNAK: [0.001622012771438, 1.772349121021e-05, 9.107125179542e-07, 5.204294217798e-08, 3.115247615426e-09]

Status: **Implemented** — [JarreSplineVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/JarreSplineVerificationTests.cs) :: `Table1_SineZeroEndpointCurvature_UniformMesh` (NAT and NAK maxima; count of July printed-cell disagreements), and, in the same file, `AuthenticatedOctoberAuthorCoefficients_MatchBetweenKnots` (Q and RNAK against the October author listing).

18 of 20 July-oracle cells agree with printed rounding. The July source-equation discrepancies are Q at 12 knots (printed 4e-05, July oracle 7.163038654348e-05) and Q at 24 knots (printed 9.6e-07, July oracle 1.960928728804e-06). The test asserts the count of these printed-cell disagreements; their July values are not asserted against production.

SciPy independently locates extrema through roots of s′−f′; the C# test maximizes |s−f| over subdivided intervals and also checks the oracle extremum location for NAT and NAK. RNAK means **revised** not-a-knot. The July 2025 preprint (arXiv:2507.05083v1, Sections 4–4.1, both safeguards) is archived in this row as a historical comparison; production implements the algorithms of Jarre JCAM 478 (2026), Appendix, and the authenticated October 29, 2025 author code, whose maxima and between-knot coefficients for this function and these knot counts are asserted by `AuthenticatedOctoberAuthorCoefficients_MatchBetweenKnots`. Full controls: [generate_jarre_spline_vectors.py](../tests/Verification/generate_jarre_spline_vectors.py) and [jarre_splines.json](../tests/Verification/vectors/jarre_splines.json), `/tables/0/cases/*/results`.

#### Test #122: Table 2 maximum errors

Paper line: 637–643; Table 2; PDF page(s) 13–14.

Basis: Printed uniform results plus independent source-equation control for NAT and NAK; Q and RNAK follow the authenticated October 29, 2025 author listing.

**Inputs:**

- f(x): sin(x); interval: [pi/4,5pi/4]
- knot counts: [6,12,24,48,96]
- mesh: equidistant
- ys: f(xs); boundary modes: Natural, NotAKnot, QSpline, RevisedNotAKnot (RNAK)

**Output / assertions:**

- Printed supremum errors in knot-count order (retained for comparison):
- NAT: [0.014, 0.0029, 0.00065, 0.00016, 3.8e-05]
- NAK: [0.0043, 0.00017, 7.9e-06, 4.3e-07, 2.5e-08]
- Q: [0.0016, 5.5e-05, 2.2e-06, 1.1e-07, 6e-09]
- RNAK: [0.0016, 4.6e-05, 9.1e-07, 5.2e-08, 3.1e-09]

Independent oracle maxima actually asserted against production (NAT and NAK only; same knot-count order; absolute tolerance 5e-13):

- NAT: [0.01445459759592, 0.002863674724086, 0.0006492806960999, 0.00015518879498, 3.796770707343e-05]
- NAK: [0.004320703775415, 0.0001655711916734, 7.859469968508e-06, 4.253363977691e-07, 2.469610316247e-08]

July 2025 preprint oracle maxima for Q and RNAK (historical comparison, not asserted: production implements the October 2025 author-listing Q and RNAK algorithms):

- Q: [0.00304314616394, 5.760391782006e-05, 2.692611706268e-06, 1.275674285095e-07, 6.642025374148e-09]
- RNAK: [0.002299760117412, 8.063730221264e-05, 9.09993739473e-07, 5.203567665646e-08, 3.11514125606e-09]

Status: **Implemented** — [JarreSplineVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/JarreSplineVerificationTests.cs) :: `Table2_ShiftedSine_UniformMesh` (NAT and NAK maxima; count of July printed-cell disagreements), and, in the same file, `AuthenticatedOctoberAuthorCoefficients_MatchBetweenKnots` (Q and RNAK against the October author listing).

13 of 20 July-oracle cells agree with printed rounding. The July source-equation discrepancies are Q at 6 knots (printed 0.0016, July oracle 0.00304314616394), RNAK at 6 knots (printed 0.0016, July oracle 0.002299760117412), Q at 12 knots (printed 5.5e-05, July oracle 5.760391782006e-05), RNAK at 12 knots (printed 4.6e-05, July oracle 8.063730221264e-05), Q at 24 knots (printed 2.2e-06, July oracle 2.692611706268e-06), Q at 48 knots (printed 1.1e-07, July oracle 1.275674285095e-07) and Q at 96 knots (printed 6e-09, July oracle 6.642025374148e-09). The test asserts the count of these printed-cell disagreements; their July values are not asserted against production.

SciPy independently locates extrema through roots of s′−f′; the C# test maximizes |s−f| over subdivided intervals and also checks the oracle extremum location for NAT and NAK. RNAK means **revised** not-a-knot. The July 2025 preprint (arXiv:2507.05083v1, Sections 4–4.1, both safeguards) is archived in this row as a historical comparison; production implements the algorithms of Jarre JCAM 478 (2026), Appendix, and the authenticated October 29, 2025 author code, whose maxima and between-knot coefficients for this function and these knot counts are asserted by `AuthenticatedOctoberAuthorCoefficients_MatchBetweenKnots`. Full controls: [generate_jarre_spline_vectors.py](../tests/Verification/generate_jarre_spline_vectors.py) and [jarre_splines.json](../tests/Verification/vectors/jarre_splines.json), `/tables/1/cases/*/results`.

#### Test #123: Table 3 maximum errors

Paper line: 645–652; Table 3; PDF page(s) 14.

Basis: Seeded reconstruction of the published random-mesh protocol; independent source-equation control for NAT and NAK.

**Inputs:**

- f(x): sin(x); interval: [pi/4,5pi/4]
- knot counts: [6,12,24,48,96]
- mesh: independent irregular reconstruction; xorshift32 seed 250705086; sorted uniform values affinely rescaled to include both endpoints
- Complete knots/ordinates: [jarre_splines.json](../tests/Verification/vectors/jarre_splines.json), `/tables/2/cases/0` through `/tables/2/cases/4`
- ys: f(xs); boundary modes: Natural, NotAKnot, QSpline, RevisedNotAKnot (RNAK)

**Output / assertions:**

- Printed supremum errors in knot-count order (retained for comparison):
- NAT: [0.042, 0.0012, 0.0052, 0.00017, 0.00034]
- NAK: [0.047, 0.0012, 0.00052, 1.4e-05, 8.5e-07]
- Q: [0.046, 0.0012, 0.00052, 1.4e-05, 8.5e-07]
- RNAK: [0.047, 0.0012, 0.00052, 1.4e-05, 8.5e-07]

Independent oracle maxima actually asserted against production (NAT and NAK only; same knot-count order; absolute tolerance 5e-13):

- NAT: [0.05298554802772, 0.002118253898924, 0.0001796901755506, 4.608935730244e-05, 6.76858747779e-05]
- NAK: [0.04865949732349, 0.00211436805878, 9.164139740458e-05, 3.177639573693e-06, 3.259908699649e-07]

July 2025 preprint oracle maxima for Q and RNAK (historical comparison, not asserted: production implements the October 2025 author-listing Q and RNAK algorithms, and the author listing covers only uniform meshes):

- Q: [0.04531348599238, 0.002114502859007, 8.604355105868e-05, 3.177639573693e-06, 3.259908699649e-07]
- RNAK: [0.04865445217132, 0.002114410081047, 8.997987556869e-05, 3.177639573693e-06, 3.259908699649e-07]

Status: **Implemented** — [JarreSplineVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/JarreSplineVerificationTests.cs) :: `Table3_ShiftedSine_ReconstructedRandomMesh` (NAT and NAK maxima).

The original random knots and seed are unpublished. All 20 cells use this declared mesh reconstruction; the printed rows are context, not exact goldens.

SciPy independently locates extrema through roots of s′−f′; the C# test maximizes |s−f| over subdivided intervals and also checks the oracle extremum location for NAT and NAK. RNAK means **revised** not-a-knot. The July 2025 preprint (arXiv:2507.05083v1, Sections 4–4.1, both safeguards) is archived in this row as a historical comparison; production implements the algorithms of Jarre JCAM 478 (2026), Appendix, and the authenticated October 29, 2025 author code. That listing (`AuthenticatedOctoberAuthorCoefficients_MatchBetweenKnots`) covers only the uniform-mesh Tables 1, 2 and 5, so this table has no asserted Q or RNAK values. Full controls: [generate_jarre_spline_vectors.py](../tests/Verification/generate_jarre_spline_vectors.py) and [jarre_splines.json](../tests/Verification/vectors/jarre_splines.json), `/tables/2/cases/*/results`.

#### Test #124: Table 4 maximum errors

Paper line: 660–667; Table 4; PDF page(s) 14.

Basis: Seeded reconstruction of the published random-mesh protocol; independent source-equation control for NAT and NAK.

**Inputs:**

- f(x): 1/(1+x²); interval: [−1,3]
- knot counts: [6,12,24,48,96]
- mesh: independent irregular reconstruction; xorshift32 seed 250705087; sorted uniform values affinely rescaled to include both endpoints
- Complete knots/ordinates: [jarre_splines.json](../tests/Verification/vectors/jarre_splines.json), `/tables/3/cases/0` through `/tables/3/cases/4`
- ys: f(xs); boundary modes: Natural, NotAKnot, QSpline, RevisedNotAKnot (RNAK)

**Output / assertions:**

- Printed supremum errors in knot-count order (retained for comparison):
- NAT: [0.11, 0.015, 0.0017, 0.00018, 5.7e-05]
- NAK: [0.14, 0.0062, 0.0013, 0.00018, 3.6e-05]
- Q: [0.27, 0.03, 0.0013, 0.00018, 3.6e-05]
- RNAK: [0.14, 0.021, 0.0013, 0.00018, 3.6e-05]

Independent oracle maxima actually asserted against production (NAT and NAK only; same knot-count order; absolute tolerance 5e-13):

- NAT: [0.08090688885297, 0.02440154036276, 0.002694359711084, 0.0009370243665545, 1.350681848955e-05]
- NAK: [0.3039509598655, 0.02439878317319, 0.0005311877439691, 9.421139475108e-05, 1.350681848955e-05]

July 2025 preprint oracle maxima for Q and RNAK (historical comparison, not asserted: production implements the October 2025 author-listing Q and RNAK algorithms, and the author listing covers only uniform meshes):

- Q: [0.8050066322339, 0.02439743323407, 0.0007970161223222, 9.876272691056e-05, 1.350681848955e-05]
- RNAK: [0.435875438129, 0.02439877779183, 0.0004851742303938, 9.421139475085e-05, 1.350681848955e-05]

Status: **Implemented** — [JarreSplineVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/JarreSplineVerificationTests.cs) :: `Table4_RungeFunction_ReconstructedRandomMesh` (NAT and NAK maxima).

The original random knots and seed are unpublished. All 20 cells use this declared mesh reconstruction; the printed rows are context, not exact goldens.

SciPy independently locates extrema through roots of s′−f′; the C# test maximizes |s−f| over subdivided intervals and also checks the oracle extremum location for NAT and NAK. RNAK means **revised** not-a-knot. The July 2025 preprint (arXiv:2507.05083v1, Sections 4–4.1, both safeguards) is archived in this row as a historical comparison; production implements the algorithms of Jarre JCAM 478 (2026), Appendix, and the authenticated October 29, 2025 author code. That listing (`AuthenticatedOctoberAuthorCoefficients_MatchBetweenKnots`) covers only the uniform-mesh Tables 1, 2 and 5, so this table has no asserted Q or RNAK values. Full controls: [generate_jarre_spline_vectors.py](../tests/Verification/generate_jarre_spline_vectors.py) and [jarre_splines.json](../tests/Verification/vectors/jarre_splines.json), `/tables/3/cases/*/results`.

#### Test #125: Table 5 maximum errors

Paper line: 681–687; Table 5; PDF page(s) 14–15.

Basis: Printed uniform results plus independent source-equation control for NAT and NAK; Q and RNAK follow the authenticated October 29, 2025 author listing.

**Inputs:**

- f(x): 1/(1+exp(−x)); interval: [−1,4]
- knot counts: [6,12,24,48,96]
- mesh: equidistant
- ys: f(xs); boundary modes: Natural, NotAKnot, QSpline, RevisedNotAKnot (RNAK)

**Output / assertions:**

- Printed supremum errors in knot-count order (retained for comparison):
- NAT: [0.0055, 0.00096, 0.00021, 5.1e-05, 1.2e-05]
- NAK: [0.00058, 0.00013, 8e-06, 4.6e-07, 2.7e-08]
- Q: [0.0023, 0.00011, 8.2e-07, 1e-07, 6.6e-09]
- RNAK: [0.00095, 0.00013, 1e-06, 4.4e-08, 2.7e-09]

Independent oracle maxima actually asserted against production (NAT and NAK only; same knot-count order; absolute tolerance 5e-13):

- NAT: [0.005493488271729, 0.0009581611135632, 0.0002126105366964, 5.058263376961e-05, 1.236185403614e-05]
- NAK: [0.0005770927489316, 0.0001294452633607, 7.990722889595e-06, 4.604732519731e-07, 2.728463788237e-08]

July 2025 preprint oracle maxima for Q and RNAK (historical comparison, not asserted: production implements the October 2025 author-listing Q and RNAK algorithms):

- Q: [0.002344481745218, 0.0001099670288977, 8.208328675097e-07, 1.020582600098e-07, 6.646755534856e-09]
- RNAK: [0.0006218770058481, 0.0001280851806115, 1.681603657488e-06, 4.373369610899e-08, 2.670369370072e-09]

Status: **Implemented** — [JarreSplineVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/JarreSplineVerificationTests.cs) :: `Table5_Logistic_UniformMesh` (NAT and NAK maxima; count of July printed-cell disagreements), and, in the same file, `AuthenticatedOctoberAuthorCoefficients_MatchBetweenKnots` (Q and RNAK against the October author listing).

18 of 20 July-oracle cells agree with printed rounding. The July source-equation discrepancies are RNAK at 6 knots (printed 0.00095, July oracle 0.0006218770058481) and RNAK at 24 knots (printed 1e-06, July oracle 1.681603657488e-06). The test asserts the count of these printed-cell disagreements; their July values are not asserted against production.

SciPy independently locates extrema through roots of s′−f′; the C# test maximizes |s−f| over subdivided intervals and also checks the oracle extremum location for NAT and NAK. RNAK means **revised** not-a-knot. The July 2025 preprint (arXiv:2507.05083v1, Sections 4–4.1, both safeguards) is archived in this row as a historical comparison; production implements the algorithms of Jarre JCAM 478 (2026), Appendix, and the authenticated October 29, 2025 author code, whose maxima and between-knot coefficients for this function and these knot counts are asserted by `AuthenticatedOctoberAuthorCoefficients_MatchBetweenKnots`. Full controls: [generate_jarre_spline_vectors.py](../tests/Verification/generate_jarre_spline_vectors.py) and [jarre_splines.json](../tests/Verification/vectors/jarre_splines.json), `/tables/4/cases/*/results`.

### Class: `CubicSplineInterpolator<decimal>` / `CubicSplineInterpolator<double>` (filename: [CubicSplineInterpolator.cs](../src/Boutquin.Numerics/Interpolation/CubicSplineInterpolator.cs))

Function: `T Interpolate(T x)`

#### Test #202: Revised not-a-knot reproduces a cubic on irregular knots

Paper line: 557–604; Sections 4–4.1, revised third-derivative jumps and safeguards; PDF page(s) 12–13.

Basis: Exact polynomial consequence of the published divided-difference equations.

**Inputs:**

- xs: [0,0.25,1,2,4,7]; ys: x³−2x+1
- boundary: RevisedNotAKnot; decimal and double instantiations
- Queries: x_i+(x_(i+1)−x_i)/3, i=0..4

**Output / assertions:**

- f(query)=query³−2×query+1; decimal absolute tolerance 1e-23; double absolute tolerance 2e-12

Status: **Implemented** — [RevisedNotAKnotTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Interpolation/RevisedNotAKnotTests.cs) :: `CubicOnIrregularKnots_IsReproducedInDecimalAndDouble`.

Fourth/fifth divided differences vanish; the published revised rule reduces to classical not-a-knot.

#### Test #203: Revised not-a-knot quartic has the prescribed endpoint jumps

Paper line: 557–604; Sections 4–4.1, revised third-derivative jumps and safeguards; PDF page(s) 12–13.

Basis: Exact polynomial consequence of the published divided-difference equations.

**Inputs:**

- Theory scale rows: 1 and 10000000000
- xs: [0,scale,2×scale,3×scale,4×scale,5×scale]; ys: [0,1,16,81,256,625]
- boundary: RevisedNotAKnot; decimal and double instantiations
- Query x=(i+u)×scale, i=0..4, u=0.25,0.5,0.75

**Output / assertions:**

- Expected s(x)=(i+u)^4−u²(1−u)²; decimal absolute tolerance 1e-22; double absolute tolerance 2e-12

Status: **Implemented** — [RevisedNotAKnotTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Interpolation/RevisedNotAKnotTests.cs) :: `QuarticOnUniformKnots_MatchesAnalyticSplineIncludingAbscissaScaling`.

The exact knot curvatures are 12×i²−2 in normalized coordinates; fifth divided differences vanish, exercising undamped fourth-difference jumps at both ends and abscissa scaling.

<a id="paper-ncm"></a>

## Paper: higham-2002-nearest-correlation-matrix.pdf

Source PDF SHA-256: `65f31d5557693fb761fb7286a7c5621c129b8eb9e90ba645f79b15fa393c77bc`.

### Class: `NearestPsdProjection<double>` (filename: [NearestPsdProjection.cs](../src/Boutquin.Numerics/LinearAlgebra/NearestPsdProjection.cs))

Function: `T[,] Higham(T[,] a, int maxIterations, T tolerance)`

#### Test #126: Higham_KnownInvalidCorrelation_MatchesReferenceNearest

Paper line: 356–389; PDF page(s) 9; Section 2, explicit 3×3 nearest-correlation example.

Basis: Printed paper example.

**Inputs:**

- a: [[1,1,0],[1,1,1],[0,1,1]]
- maxIterations: 200; tolerance: 1e-13

**Output / assertions:**

- Nearest correlation: [[1,0.7607,0.1573],[0.7607,1,0.7607],[0.1573,0.7607,1]]
- Three upper off-diagonal entries: absolute tolerance 1e-3; diagonal: 1 within 1e-12; IsPsd: true

Status: **Implemented** — [LinAlgInterpReviewFixTests.cs](../tests/Boutquin.Numerics.Tests.Unit/ReviewFixes/LinAlgInterpReviewFixTests.cs) :: `Higham_KnownInvalidCorrelation_MatchesReferenceNearest`.


<a id="paper-chol"></a>

## Paper: higham-1990-cholesky-semidefinite.pdf

Source PDF SHA-256: `0447f520ce25cf30b523edbaf35d8000d70581b6a9ae5c74a5872d3a6bfb10db`.

### Class: `CholeskyDecomposition<double>` (filename: [CholeskyDecomposition.cs](../src/Boutquin.Numerics/LinearAlgebra/CholeskyDecomposition.cs))

Function: `PivotedCholeskyResult<T> DecomposePivoted(T[,] a)`

#### Test #127: DecomposePivoted_DefaultOverload_UsesHighamRelativeTolerance_NotAbsoluteZero

Paper line: 604–619; PDF page(s) 14; Section 4, relative stopping criterion (4.3).

Basis: Paper-derived formula.

**Inputs:**

- b: [[1,2],[3,1],[2,−1],[0.5,4]]
- a: b×bᵀ; default relative tolerance

**Output / assertions:**

- Rank: exactly 2

Status: **Implemented** — [CholeskyDecompositionTests.cs](../tests/Boutquin.Numerics.Tests.Unit/LinearAlgebra/CholeskyDecompositionTests.cs) :: `DecomposePivoted_DefaultOverload_UsesHighamRelativeTolerance_NotAbsoluteZero`.

An independently rank-2 matrix exercises the paper’s relative stopping rule. The library selects epsilon=n×unitRoundoff; this is not one of the printed test matrices.

### Class: `CholeskyDecomposition<decimal>` (filename: [CholeskyDecomposition.cs](../src/Boutquin.Numerics/LinearAlgebra/CholeskyDecomposition.cs))

Function: `PivotedCholeskyResult<decimal> DecomposePivoted(decimal[,] a)`

#### Test #128: DecomposePivoted_DecimalDefaultTolerance_UsesHighamRelativeTolerance_NotAbsoluteZero

Paper line: 604–619; PDF page(s) 14; Section 4, relative stopping criterion (4.3).

Basis: Paper-derived formula.

**Inputs:**

- l: [[4,0,0],[2,3,0],[1,0.5,1e−14]]
- a: l×lᵀ in decimal arithmetic; omitted tolerance

**Output / assertions:**

- Numerical rank: exactly 2

Status: **Implemented** — [CholeskyDecompositionTests.cs](../tests/Boutquin.Numerics.Tests.Unit/LinearAlgebra/CholeskyDecompositionTests.cs) :: `DecomposePivoted_DecimalDefaultTolerance_UsesHighamRelativeTolerance_NotAbsoluteZero`.

The third pivot is 1e−28, below the relative cutoff. This checks a numerical-rank convention derived from (4.3), not exact algebraic rank.


<a id="paper-nm"></a>

## Paper: nelder-1965-simplex-method.pdf

Source PDF SHA-256: `de30981aad63ec5e372217d806f81f7a5d424425cc4035eb58d030aebf7f8200`.

### Class: `NelderMeadMinimizer<double>` (filename: [NelderMeadMinimizer.cs](../src/Boutquin.Numerics/Solvers/NelderMeadMinimizer.cs))

Function: `SimplexResult<T> Minimize(Func<T[], T> objective, T[] initialGuess, SimplexOptions<T>? options = null)`

#### Test #129: Minimize_Rosenbrock_RecoversMinimum

Paper line: 115–124; PDF page(s) 2; Rosenbrock benchmark and its zero minimum.

Basis: Paper benchmark, analytical optimum.

**Inputs:**

- objective: (1−x)²+100(y−x²)²
- initialGuess: [0,0]; default SimplexOptions

**Output / assertions:**

- Converged: true; Argmin: [1,1] within 1e-4 per coordinate; MinValue: < 1e-7

Status: **Implemented** — [NelderMeadMinimizerTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NelderMeadMinimizerTests.cs) :: `Minimize_Rosenbrock_RecoversMinimum`.

The paper uses start [−1.2,1]. This test starts at [0,0] and checks the same analytical optimum; it does not reproduce the paper’s evaluation counts.


<a id="paper-lm"></a>

## Paper: madsen-nielsen-tingleff-2004-nonlinear-least-squares.pdf

Source PDF SHA-256: `42f7f28b789c6d840f8f3a7f7a2d28be82c3f2e6d844c57aaa49a4bc9db41ef1`.

### Class: `LevenbergMarquardtSolver` (filename: [LevenbergMarquardtSolver.cs](../src/Boutquin.Numerics/Solvers/LevenbergMarquardtSolver.cs))

Function: `MultivariateSolverResult Solve(Func<double[], double[]> residuals, double[] initialGuess, Func<double[], double[,]>? jacobian = null, double[]? lowerBounds = null, double[]? upperBounds = null)`

#### Test #130: Solve_Rosenbrock_ConvergesFromStandardStartingPoint

Paper line: 975–985; PDF page(s) 21; Modified Rosenbrock benchmark; lambda=0 gives the two-residual problem.

Basis: Paper benchmark, analytical optimum.

**Inputs:**

- residuals: [10(y−x²),1−x]
- jacobian: [[−20x,10],[−1,0]]
- initialGuess: [−1.2,1]; default solver options

**Output / assertions:**

- Converged: true; Parameters: [1,1] within 1e-6 per coordinate; FinalCost: < 1e-10; Iterations: ≤ 30

Status: **Implemented** — [LevenbergMarquardtSolverTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/LevenbergMarquardtSolverTests.cs) :: `Solve_Rosenbrock_ConvergesFromStandardStartingPoint`.

The optimum follows by setting lambda=0 in the printed modified problem. The 30-iteration allowance is a library acceptance bound, not a printed iteration count.


<a id="paper-ewma"></a>

## Paper: jpmorgan-1996-riskmetrics-technical-document.pdf

Source PDF SHA-256: `9ae51e8f9bbfa99b95de9aa7d52969c742b7c2dece437fea856176f8bf5193c7`.

### Class: `ExponentiallyWeightedCovarianceEstimator` (filename: [ExponentiallyWeightedCovarianceEstimator.cs](../src/Boutquin.Numerics/Statistics/ExponentiallyWeightedCovarianceEstimator.cs))

Function: `decimal[,] Estimate(decimal[,] returns)`

#### Test #131: Ewma_MatchesRiskMetricsFormula

Paper line: 4773–4785; PDF page(s) 91; Normalized weights, Equation [5.2]; covariance Table 5.4 at lines 4980–5018.

Basis: Declared weighted-mean covariance variant.

**Inputs:**

- returns: [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/returns` (120×5)
- lambda: 0.94
- weights: proportional to (1−lambda)×lambda^(T−1−t), normalized to sum to 1; weighted centering

**Output / assertions:**

- Every covariance entry: [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `/ewma_lambda_0_94`
- Per-entry abs(actual−reference) ≤ max(1e-16,1e-10×max(abs(actual),abs(reference)))

Status: **Implemented** — [CovarianceVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CovarianceVerificationTests.cs) :: `Ewma_MatchesRiskMetricsFormula`.

Uses normalized finite-window weights and an exponentially weighted mean. It does not reproduce the zero-mean recursive forecasts or market-data tables in RiskMetrics.


<a id="paper-p2"></a>

## Paper: jain-chlamtac-1985-p2-quantile.pdf

Source PDF SHA-256: `b9545f47bf17556aa2facbffa34577455ecf080b8384c2fb1a8f42f1af0fd99e`.

### Class: `PSquareEstimator<double>` and `PSquareEstimator` (filename: [PSquareEstimator.cs](../src/Boutquin.Numerics/MonteCarlo/PSquareEstimator.cs))

Function: `void Add(T observation)` / `void Add(double observation)`

#### Test #132: Table I median stream with erratum

Paper line: 175–203,280–350; Table I visually checked; PDF page(s) 3–6.

Basis: Printed input stream, explicit erratum, and exact rational evaluation of the paper equations.

**Inputs:**

- percentile: 0.5
- First five observations: [0.02,0.15,0.74,3.39,0.83] (0.15 is the explicit correction to 0.5)
- Next observations: [22.37,10.15,15.43,38.62,15.92,34.60,10.28,1.47,0.40,0.05,11.39,0.27,0.42,0.09,11.37]
- Generic double estimator consumes the complete stream

**Output / assertions:**

- Unrounded median heights after observations 6..20: [0.74,0.74,2.1783333333333332,4.752685185185185,4.752685185185185,9.274704861111111,9.274704861111111,9.274704861111111,9.274704861111111,6.297302000661376,6.297302000661376,6.297302000661376,6.297302000661376,4.440634353260337,4.440634353260337]; absolute tolerance 1e-12 at every step
- Final marker positions: exactly [1,6,10,16,20]; count: exactly 20
- Final unrounded marker heights: [0.02,0.4938954475308642,4.440634353260337,17.20390427414021,38.62]; absolute tolerance 1e-12
- Facade median equals the generic median at every step

Status: **Implemented** — [PSquarePaperReplicationTests.cs](../tests/Boutquin.Numerics.Tests.Unit/MonteCarlo/PSquarePaperReplicationTests.cs) :: `TableI_ErratumStream_MatchesUnroundedPaperEquations`.

The printed medians [0.74,0.74,2.18,4.75,4.75,9.28,9.28,9.28,9.28,6.30,6.30,6.30,6.30,4.44,4.44] and final heights [0.02,0.50,4.44,17.22,38.62] contain intermediate rounding/arithmetic differences. They are not all obtainable by rounding the unrounded recurrence at the end. The test corrects the expected values by independently evaluating Box 1's parabolic/linear equations with rational arithmetic: [generate_p2_paper_reference.py](../tests/Verification/generate_p2_paper_reference.py). Production need not round each marker update to the printed table's decimal precision.

<a id="paper-nist-nls"></a>

## Paper: NIST StRD nonlinear least squares (dataset collection; no PDF)

Source: [primary source](https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml).

### Class: `LevenbergMarquardtSolver` (filename: [LevenbergMarquardtSolver.cs](../src/Boutquin.Numerics/Solvers/LevenbergMarquardtSolver.cs))

Function: `MultivariateSolverResult Solve(Func<double[], double[]> residuals, double[] initialGuess, Func<double[], double[,]>? jacobian = null, double[]? lowerBounds = null, double[]? upperBounds = null)`

#### Test #133: Lanczos3_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Lanczos3; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Lanczos3()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Lanczos3_ConvergesToCertifiedValues`
- maxIterations: 500; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Lanczos3()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Lanczos3_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #134: Eckerle4_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Eckerle4; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Eckerle4()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Eckerle4_ConvergesToCertifiedValues`
- maxIterations: 500; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Eckerle4()`
- Parameter relative error < 1e-3 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Eckerle4_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #135: Bennett5_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Bennett5; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Bennett5()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Bennett5_ConvergesToCertifiedValues`
- maxIterations: 1000; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Bennett5()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Bennett5_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #136: Hahn1_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Hahn1; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Hahn1()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Hahn1_ConvergesToCertifiedValues`
- maxIterations: 2000; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Hahn1()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Hahn1_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #137: Kirby2_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Kirby2; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Kirby2()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Kirby2_ConvergesToCertifiedValues`
- maxIterations: 1000; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Kirby2()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Kirby2_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #138: Misra1a_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Misra1a; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Misra1a()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Misra1a_ConvergesToCertifiedValues`
- maxIterations: 500; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Misra1a()`
- Parameter relative error < 1e-3 (absolute error at certified zero); SSE relative error < 0.01; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Misra1a_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #139: Chwirut1_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Chwirut1; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Chwirut1()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Chwirut1_ConvergesToCertifiedValues`
- maxIterations: 500; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Chwirut1()`
- Parameter relative error < 1e-3 (absolute error at certified zero); SSE relative error < 0.01; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Chwirut1_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #140: Chwirut2_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Chwirut2; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Chwirut2()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Chwirut2_ConvergesToCertifiedValues`
- maxIterations: 500; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Chwirut2()`
- Parameter relative error < 1e-3 (absolute error at certified zero); SSE relative error < 0.01; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Chwirut2_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #141: Lanczos1_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Lanczos1; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Lanczos1()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Lanczos1_ConvergesToCertifiedValues`
- maxIterations: 500; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Lanczos1()`
- Parameter relative error < 1e-3 (absolute error at certified zero); SSE relative error < 0.01; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Lanczos1_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #142: Lanczos2_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Lanczos2; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Lanczos2()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Lanczos2_ConvergesToCertifiedValues`
- maxIterations: 500; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Lanczos2()`
- Parameter relative error < 1e-3 (absolute error at certified zero); SSE relative error < 0.01; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Lanczos2_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #143: Gauss1_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Gauss1; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Gauss1()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Gauss1_ConvergesToCertifiedValues`
- maxIterations: 500; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Gauss1()`
- Parameter relative error < 1e-3 (absolute error at certified zero); SSE relative error < 0.01; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Gauss1_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #144: Gauss2_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Gauss2; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Gauss2()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Gauss2_ConvergesToCertifiedValues`
- maxIterations: 500; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Gauss2()`
- Parameter relative error < 1e-3 (absolute error at certified zero); SSE relative error < 0.01; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Gauss2_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #145: DanWood_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: DanWood; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `DanWood()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `DanWood_ConvergesToCertifiedValues`
- maxIterations: 500; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `DanWood()`
- Parameter relative error < 1e-3 (absolute error at certified zero); SSE relative error < 0.01; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `DanWood_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #146: Misra1b_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Misra1b; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Misra1b()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Misra1b_ConvergesToCertifiedValues`
- maxIterations: 500; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Misra1b()`
- Parameter relative error < 1e-3 (absolute error at certified zero); SSE relative error < 0.01; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Misra1b_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #147: Nelson_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Nelson; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Nelson()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Nelson_ConvergesToCertifiedValues`
- maxIterations: 1000; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Nelson()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Nelson_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper. Nelson uses the logged-response, two-predictor model stored as a flat X slab.

#### Test #148: MGH17_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: MGH17; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `MGH17()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `MGH17_ConvergesToCertifiedValues`
- maxIterations: 2000; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `MGH17()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `MGH17_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #149: Gauss3_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Gauss3; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Gauss3()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Gauss3_ConvergesToCertifiedValues`
- maxIterations: 2000; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Gauss3()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Gauss3_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #150: Misra1c_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Misra1c; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Misra1c()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Misra1c_ConvergesToCertifiedValues`
- maxIterations: 500; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Misra1c()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Misra1c_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #151: Misra1d_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Misra1d; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Misra1d()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Misra1d_ConvergesToCertifiedValues`
- maxIterations: 500; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Misra1d()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Misra1d_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #152: Roszman1_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Roszman1; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Roszman1()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Roszman1_ConvergesToCertifiedValues`
- maxIterations: 1000; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Roszman1()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Roszman1_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #153: ENSO_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: ENSO; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `ENSO()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `ENSO_ConvergesToCertifiedValues`
- maxIterations: 2000; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `ENSO()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `ENSO_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #154: MGH09_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: MGH09; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `MGH09()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `MGH09_ConvergesToCertifiedValues`
- maxIterations: 2000; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `MGH09()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `MGH09_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #155: Thurber_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Thurber; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Thurber()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Thurber_ConvergesToCertifiedValues`
- maxIterations: 2000; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Thurber()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Thurber_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #156: BoxBOD_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: BoxBOD; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `BoxBOD()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `BoxBOD_ConvergesToCertifiedValues`
- maxIterations: 2000; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `BoxBOD()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `BoxBOD_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #157: Rat42_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Rat42; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Rat42()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Rat42_ConvergesToCertifiedValues`
- maxIterations: 1000; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Rat42()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Rat42_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #158: MGH10_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: MGH10; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `MGH10()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `MGH10_ConvergesToCertifiedValues`
- maxIterations: 2000; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `MGH10()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `MGH10_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.

#### Test #159: Rat43_ConvergesToCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Model, Starting Values, Certified Values, and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Dataset: Rat43; all observations and Start 2: [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Rat43()`
- residuals: exact model/data transform in [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Rat43_ConvergesToCertifiedValues`
- maxIterations: 2000; all other solver options: defaults; jacobian/bounds: omitted

**Output / assertions:**

- Parameters: every entry of Certified; SSE: CertifiedSse in [NistStRDData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDData.cs) :: `Rat43()`
- Parameter relative error < 1e-2 (absolute error at certified zero); SSE relative error < 0.05; if CertifiedSse < 1e-18, computed SSE < 1e-18
- Converged: true; termination: FunctionToleranceReached, ParameterToleranceReached, or GradientToleranceReached; SSE=2×FinalCost

Status: **Implemented** — [NistStRDTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistStRDTests.cs) :: `Rat43_ConvergesToCertifiedValues`.

This is a NIST Start 2 result check, not an iteration-count replication of the LM implementation paper.


<a id="paper-nist-lls"></a>

## Paper: NIST StRD linear least squares (dataset collection; no PDF)

Source: [primary source](https://www.itl.nist.gov/div898/strd/lls/lls.shtml).

### Class: `OrdinaryLeastSquares` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult Fit(double[,] x, ReadOnlySpan<double> y, bool includeIntercept = true)`

#### Test #160: Fit_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Norris; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Norris()`

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Norris()`
- Relative tolerances: coefficients 1e-12; nonzero standard errors 1e-11; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionTests.cs) :: `Fit_MatchesCertifiedValues`.

### Class: `OrdinaryLeastSquares<decimal>` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult<T> Fit(T[,] x, ReadOnlySpan<T> y, bool includeIntercept = true)`

#### Test #161: Fit_Decimal_MatchesCertifiedValues_UniformTolerance

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Norris; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Norris()` (cast input doubles to decimal)

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Norris()`
- Relative tolerances: coefficients 1e-13; nonzero standard errors 1e-7; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionDecimalTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionDecimalTests.cs) :: `Fit_Decimal_MatchesCertifiedValues_UniformTolerance`.

### Class: `OrdinaryLeastSquares` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult Fit(double[,] x, ReadOnlySpan<double> y, bool includeIntercept = true)`

#### Test #162: Fit_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Pontius; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Pontius()`

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Pontius()`
- Relative tolerances: coefficients 1e-12; nonzero standard errors 1e-11; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionTests.cs) :: `Fit_MatchesCertifiedValues`.

### Class: `OrdinaryLeastSquares<decimal>` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult<T> Fit(T[,] x, ReadOnlySpan<T> y, bool includeIntercept = true)`

#### Test #163: Fit_Decimal_MatchesCertifiedValues_UniformTolerance

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Pontius; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Pontius()` (cast input doubles to decimal)

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Pontius()`
- Relative tolerances: coefficients 1e-13; nonzero standard errors 1e-7; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionDecimalTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionDecimalTests.cs) :: `Fit_Decimal_MatchesCertifiedValues_UniformTolerance`.

Decimal Pontius SE(b2) is instead required to be nonnegative and within 5e-17 absolute of the certified value.

### Class: `OrdinaryLeastSquares` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult Fit(double[,] x, ReadOnlySpan<double> y, bool includeIntercept = true)`

#### Test #164: Fit_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: NoInt1; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `NoInt1()`

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `NoInt1()`
- Relative tolerances: coefficients 1e-12; nonzero standard errors 1e-11; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionTests.cs) :: `Fit_MatchesCertifiedValues`.

### Class: `OrdinaryLeastSquares<decimal>` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult<T> Fit(T[,] x, ReadOnlySpan<T> y, bool includeIntercept = true)`

#### Test #165: Fit_Decimal_MatchesCertifiedValues_UniformTolerance

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: NoInt1; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `NoInt1()` (cast input doubles to decimal)

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `NoInt1()`
- Relative tolerances: coefficients 1e-13; nonzero standard errors 1e-7; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionDecimalTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionDecimalTests.cs) :: `Fit_Decimal_MatchesCertifiedValues_UniformTolerance`.

### Class: `OrdinaryLeastSquares` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult Fit(double[,] x, ReadOnlySpan<double> y, bool includeIntercept = true)`

#### Test #166: Fit_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: NoInt2; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `NoInt2()`

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `NoInt2()`
- Relative tolerances: coefficients 1e-12; nonzero standard errors 1e-11; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionTests.cs) :: `Fit_MatchesCertifiedValues`.

### Class: `OrdinaryLeastSquares<decimal>` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult<T> Fit(T[,] x, ReadOnlySpan<T> y, bool includeIntercept = true)`

#### Test #167: Fit_Decimal_MatchesCertifiedValues_UniformTolerance

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: NoInt2; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `NoInt2()` (cast input doubles to decimal)

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `NoInt2()`
- Relative tolerances: coefficients 1e-13; nonzero standard errors 1e-7; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionDecimalTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionDecimalTests.cs) :: `Fit_Decimal_MatchesCertifiedValues_UniformTolerance`.

### Class: `OrdinaryLeastSquares` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult Fit(double[,] x, ReadOnlySpan<double> y, bool includeIntercept = true)`

#### Test #168: Fit_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Filip; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Filip()`

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Filip()`
- Relative tolerances: coefficients 1e-7; nonzero standard errors 1e-6; nonzero RSS/residual standard deviation 1e-5; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionTests.cs) :: `Fit_MatchesCertifiedValues`.

### Class: `OrdinaryLeastSquares<decimal>` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult<T> Fit(T[,] x, ReadOnlySpan<T> y, bool includeIntercept = true)`

#### Test #169: Fit_Decimal_MatchesCertifiedValues_UniformTolerance

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Filip; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Filip()` (cast input doubles to decimal)

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Filip()`
- Relative tolerances: coefficients 1e-6; nonzero standard errors 1e-6; nonzero RSS/residual standard deviation 1e-6; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionDecimalTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionDecimalTests.cs) :: `Fit_Decimal_MatchesCertifiedValues_UniformTolerance`.

### Class: `OrdinaryLeastSquares` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult Fit(double[,] x, ReadOnlySpan<double> y, bool includeIntercept = true)`

#### Test #170: Fit_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Longley; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Longley()`

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Longley()`
- Relative tolerances: coefficients 1e-12; nonzero standard errors 1e-11; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionTests.cs) :: `Fit_MatchesCertifiedValues`.

### Class: `OrdinaryLeastSquares<decimal>` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult<T> Fit(T[,] x, ReadOnlySpan<T> y, bool includeIntercept = true)`

#### Test #171: Fit_Decimal_MatchesCertifiedValues_UniformTolerance

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Longley; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Longley()` (cast input doubles to decimal)

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Longley()`
- Relative tolerances: coefficients 1e-13; nonzero standard errors 1e-7; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionDecimalTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionDecimalTests.cs) :: `Fit_Decimal_MatchesCertifiedValues_UniformTolerance`.

### Class: `OrdinaryLeastSquares` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult Fit(double[,] x, ReadOnlySpan<double> y, bool includeIntercept = true)`

#### Test #172: Fit_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Wampler1; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler1()`

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler1()`
- Relative tolerances: coefficients 1e-12; nonzero standard errors 1e-11; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionTests.cs) :: `Fit_MatchesCertifiedValues`.

### Class: `OrdinaryLeastSquares<decimal>` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult<T> Fit(T[,] x, ReadOnlySpan<T> y, bool includeIntercept = true)`

#### Test #173: Fit_Decimal_MatchesCertifiedValues_UniformTolerance

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Wampler1; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler1()` (cast input doubles to decimal)

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler1()`
- Relative tolerances: coefficients 1e-13; nonzero standard errors 1e-7; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionDecimalTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionDecimalTests.cs) :: `Fit_Decimal_MatchesCertifiedValues_UniformTolerance`.

### Class: `OrdinaryLeastSquares` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult Fit(double[,] x, ReadOnlySpan<double> y, bool includeIntercept = true)`

#### Test #174: Fit_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Wampler2; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler2()`

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler2()`
- Relative tolerances: coefficients 1e-12; nonzero standard errors 1e-11; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionTests.cs) :: `Fit_MatchesCertifiedValues`.

### Class: `OrdinaryLeastSquares<decimal>` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult<T> Fit(T[,] x, ReadOnlySpan<T> y, bool includeIntercept = true)`

#### Test #175: Fit_Decimal_MatchesCertifiedValues_UniformTolerance

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Wampler2; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler2()` (cast input doubles to decimal)

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler2()`
- Relative tolerances: coefficients 1e-13; nonzero standard errors 1e-7; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionDecimalTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionDecimalTests.cs) :: `Fit_Decimal_MatchesCertifiedValues_UniformTolerance`.

### Class: `OrdinaryLeastSquares` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult Fit(double[,] x, ReadOnlySpan<double> y, bool includeIntercept = true)`

#### Test #176: Fit_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Wampler3; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler3()`

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler3()`
- Relative tolerances: coefficients 1e-12; nonzero standard errors 1e-11; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionTests.cs) :: `Fit_MatchesCertifiedValues`.

### Class: `OrdinaryLeastSquares<decimal>` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult<T> Fit(T[,] x, ReadOnlySpan<T> y, bool includeIntercept = true)`

#### Test #177: Fit_Decimal_MatchesCertifiedValues_UniformTolerance

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Wampler3; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler3()` (cast input doubles to decimal)

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler3()`
- Relative tolerances: coefficients 1e-13; nonzero standard errors 1e-7; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionDecimalTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionDecimalTests.cs) :: `Fit_Decimal_MatchesCertifiedValues_UniformTolerance`.

### Class: `OrdinaryLeastSquares` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult Fit(double[,] x, ReadOnlySpan<double> y, bool includeIntercept = true)`

#### Test #178: Fit_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Wampler4; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler4()`

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler4()`
- Relative tolerances: coefficients 1e-12; nonzero standard errors 1e-11; nonzero RSS/residual standard deviation 1e-6; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionTests.cs) :: `Fit_MatchesCertifiedValues`.

### Class: `OrdinaryLeastSquares<decimal>` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult<T> Fit(T[,] x, ReadOnlySpan<T> y, bool includeIntercept = true)`

#### Test #179: Fit_Decimal_MatchesCertifiedValues_UniformTolerance

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Wampler4; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler4()` (cast input doubles to decimal)

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler4()`
- Relative tolerances: coefficients 1e-13; nonzero standard errors 1e-7; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionDecimalTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionDecimalTests.cs) :: `Fit_Decimal_MatchesCertifiedValues_UniformTolerance`.

### Class: `OrdinaryLeastSquares` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult Fit(double[,] x, ReadOnlySpan<double> y, bool includeIntercept = true)`

#### Test #180: Fit_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Wampler5; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler5()`

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler5()`
- Relative tolerances: coefficients 1e-12; nonzero standard errors 1e-11; nonzero RSS/residual standard deviation 1e-3; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionTests.cs) :: `Fit_MatchesCertifiedValues`.

### Class: `OrdinaryLeastSquares<decimal>` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult<T> Fit(T[,] x, ReadOnlySpan<T> y, bool includeIntercept = true)`

#### Test #181: Fit_Decimal_MatchesCertifiedValues_UniformTolerance

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Wampler5; x, y, includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler5()` (cast input doubles to decimal)

**Output / assertions:**

- Coefficients, StandardErrors, ResidualSumOfSquares, ResidualStandardDeviation, RSquared: certified fields in [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler5()`
- Relative tolerances: coefficients 1e-13; nonzero standard errors 1e-7; nonzero RSS/residual standard deviation 1e-7; R² 1e-9
- Zero RSS: < 1e-18; zero certified standard errors: comparison skipped; zero-target coefficients: absolute tolerance instead

Status: **Implemented** — [NistLinearRegressionDecimalTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionDecimalTests.cs) :: `Fit_Decimal_MatchesCertifiedValues_UniformTolerance`.

### Class: `OrdinaryLeastSquares` (filename: [OrdinaryLeastSquares.cs](../src/Boutquin.Numerics/Solvers/OrdinaryLeastSquares.cs))

Function: `OlsResult Fit(double[,] x, ReadOnlySpan<double> y, bool includeIntercept = true)`

#### Test #182: Longley_MeetsSupplementalCoefficientBar

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- x,y,includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Longley()`

**Output / assertions:**

- Every coefficient equals Coefficients in the same fixture within relative tolerance 1e-9

Status: **Implemented** — [NistLinearRegressionTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionTests.cs) :: `Longley_MeetsSupplementalCoefficientBar`.

#### Test #183: Wampler5_MeetsSupplementalSixDigitCoefficientBar

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- x,y,includeIntercept: [NistLinearRegressionData.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionData.cs) :: `Wampler5()`

**Output / assertions:**

- Every coefficient equals Coefficients in the same fixture within relative tolerance 1e-6

Status: **Implemented** — [NistLinearRegressionTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NistStRD/NistLinearRegressionTests.cs) :: `Wampler5_MeetsSupplementalSixDigitCoefficientBar`.


<a id="paper-nist-univ"></a>

## Paper: NIST StRD univariate statistics (dataset collection; no PDF)

Source: [primary source](https://www.itl.nist.gov/div898/strd/univ/univ.shtml).

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Add(decimal value)`

#### Test #184: WelfordMoments_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: NumAcc1; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `NumAcc1()`
- Feed Values in order to one accumulator

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `NumAcc1()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MatchesCertifiedValues`.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Merge(WelfordMoments other)`

#### Test #185: WelfordMoments_MergeRoundTripMatchesSerial

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: NumAcc1; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `NumAcc1()`
- split=floor(length/2); accumulate first/last halves independently, then left.Merge(right)

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `NumAcc1()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MergeRoundTripMatchesSerial`.

The merge recurrence is also Chan–Golub–LeVeque Equations (2.1a)–(2.1b); NIST supplies the independent numeric target.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Add(decimal value)`

#### Test #186: WelfordMoments_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: NumAcc2; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `NumAcc2()`
- Feed Values in order to one accumulator

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `NumAcc2()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MatchesCertifiedValues`.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Merge(WelfordMoments other)`

#### Test #187: WelfordMoments_MergeRoundTripMatchesSerial

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: NumAcc2; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `NumAcc2()`
- split=floor(length/2); accumulate first/last halves independently, then left.Merge(right)

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `NumAcc2()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MergeRoundTripMatchesSerial`.

The merge recurrence is also Chan–Golub–LeVeque Equations (2.1a)–(2.1b); NIST supplies the independent numeric target.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Add(decimal value)`

#### Test #188: WelfordMoments_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: NumAcc3; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `NumAcc3()`
- Feed Values in order to one accumulator

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `NumAcc3()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MatchesCertifiedValues`.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Merge(WelfordMoments other)`

#### Test #189: WelfordMoments_MergeRoundTripMatchesSerial

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: NumAcc3; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `NumAcc3()`
- split=floor(length/2); accumulate first/last halves independently, then left.Merge(right)

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `NumAcc3()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MergeRoundTripMatchesSerial`.

The merge recurrence is also Chan–Golub–LeVeque Equations (2.1a)–(2.1b); NIST supplies the independent numeric target.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Add(decimal value)`

#### Test #190: WelfordMoments_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: NumAcc4; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `NumAcc4()`
- Feed Values in order to one accumulator

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `NumAcc4()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MatchesCertifiedValues`.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Merge(WelfordMoments other)`

#### Test #191: WelfordMoments_MergeRoundTripMatchesSerial

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: NumAcc4; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `NumAcc4()`
- split=floor(length/2); accumulate first/last halves independently, then left.Merge(right)

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `NumAcc4()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MergeRoundTripMatchesSerial`.

The merge recurrence is also Chan–Golub–LeVeque Equations (2.1a)–(2.1b); NIST supplies the independent numeric target.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Add(decimal value)`

#### Test #192: WelfordMoments_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Lew; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `Lew()`
- Feed Values in order to one accumulator

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `Lew()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MatchesCertifiedValues`.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Merge(WelfordMoments other)`

#### Test #193: WelfordMoments_MergeRoundTripMatchesSerial

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Lew; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `Lew()`
- split=floor(length/2); accumulate first/last halves independently, then left.Merge(right)

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `Lew()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MergeRoundTripMatchesSerial`.

The merge recurrence is also Chan–Golub–LeVeque Equations (2.1a)–(2.1b); NIST supplies the independent numeric target.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Add(decimal value)`

#### Test #194: WelfordMoments_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Lottery; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `Lottery()`
- Feed Values in order to one accumulator

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `Lottery()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MatchesCertifiedValues`.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Merge(WelfordMoments other)`

#### Test #195: WelfordMoments_MergeRoundTripMatchesSerial

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Lottery; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `Lottery()`
- split=floor(length/2); accumulate first/last halves independently, then left.Merge(right)

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `Lottery()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MergeRoundTripMatchesSerial`.

The merge recurrence is also Chan–Golub–LeVeque Equations (2.1a)–(2.1b); NIST supplies the independent numeric target.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Add(decimal value)`

#### Test #196: WelfordMoments_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Mavro; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `Mavro()`
- Feed Values in order to one accumulator

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `Mavro()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MatchesCertifiedValues`.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Merge(WelfordMoments other)`

#### Test #197: WelfordMoments_MergeRoundTripMatchesSerial

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Mavro; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `Mavro()`
- split=floor(length/2); accumulate first/last halves independently, then left.Merge(right)

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `Mavro()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MergeRoundTripMatchesSerial`.

The merge recurrence is also Chan–Golub–LeVeque Equations (2.1a)–(2.1b); NIST supplies the independent numeric target.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Add(decimal value)`

#### Test #198: WelfordMoments_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Michelso; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `Michelso()`
- Feed Values in order to one accumulator

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `Michelso()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MatchesCertifiedValues`.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Merge(WelfordMoments other)`

#### Test #199: WelfordMoments_MergeRoundTripMatchesSerial

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: Michelso; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `Michelso()`
- split=floor(length/2); accumulate first/last halves independently, then left.Merge(right)

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `Michelso()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MergeRoundTripMatchesSerial`.

The merge recurrence is also Chan–Golub–LeVeque Equations (2.1a)–(2.1b); NIST supplies the independent numeric target.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Add(decimal value)`

#### Test #200: WelfordMoments_MatchesCertifiedValues

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: PiDigits; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `PiDigits()`
- Feed Values in order to one accumulator

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `PiDigits()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MatchesCertifiedValues`.

### Class: `WelfordMoments` (filename: [WelfordMoments.cs](../src/Boutquin.Numerics/Statistics/WelfordMoments.cs))

Function: `void Merge(WelfordMoments other)`

#### Test #201: WelfordMoments_MergeRoundTripMatchesSerial

Paper line: unavailable in the text extraction; NIST data files: Certified Values and Data sections.

Basis: Certified external dataset.

**Inputs:**

- Theory name: PiDigits; decimal input observations: Values in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `PiDigits()`
- split=floor(length/2); accumulate first/last halves independently, then left.Merge(right)

**Output / assertions:**

- Mean=CertifiedMean; StdDev=CertifiedStdDev; sample Variance=CertifiedVariance=CertifiedStdDev² in [NistUnivariateData.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistUnivariateData.cs) :: `PiDigits()`
- Relative error for all three quantities after conversion to double: ≤ 1e-14

Status: **Implemented** — [NistWelfordTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Statistics/NistStRD/NistWelfordTests.cs) :: `WelfordMoments_MergeRoundTripMatchesSerial`.

The merge recurrence is also Chan–Golub–LeVeque Equations (2.1a)–(2.1b); NIST supplies the independent numeric target.

## External comparisons and parity checks

These methods are implemented, but their current assertions do not identify a printed result in a primary paper. They remain useful numeric checks. The detailed paper entries above identify the independent formula checks even where an older test name or comment calls them a port.

| Test file | Function / status | Complete inputs and expected values | What it establishes |
| --- | --- | --- | --- |
| [BootstrapVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/BootstrapVerificationTests.cs) | Implemented — `IidBootstrap_MeanFallsWithin95ConfidenceBand` | [bootstrap.json](../tests/Verification/vectors/bootstrap.json), JSON pointer `""` (document root) (method selects its named cases) | IID/block distribution or resampling-law comparison; no printed experiment replicated |
| [BootstrapVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/BootstrapVerificationTests.cs) | Implemented — `StationaryBootstrap_MeanOfMeans_MatchesArchBand` | [bootstrap.json](../tests/Verification/vectors/bootstrap.json), JSON pointer `""` (document root) (method selects its named cases) | IID/block distribution or resampling-law comparison; no printed experiment replicated |
| [BootstrapVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/BootstrapVerificationTests.cs) | Implemented — `MovingBlockBootstrap_MeanOfMeans_MatchesArchBand` | [bootstrap.json](../tests/Verification/vectors/bootstrap.json), JSON pointer `""` (document root) (method selects its named cases) | IID/block distribution or resampling-law comparison; no printed experiment replicated |
| [BootstrapVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/BootstrapVerificationTests.cs) | Implemented — `WildRademacher_WeightDistribution_MatchesRademacherAtoms` | [bootstrap.json](../tests/Verification/vectors/bootstrap.json), JSON pointer `""` (document root) (method selects its named cases) | IID/block distribution or resampling-law comparison; no printed experiment replicated |
| [BootstrapVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/BootstrapVerificationTests.cs) | Implemented — `Subsampler_Distribution_MatchesNumpyPort` | [bootstrap.json](../tests/Verification/vectors/bootstrap.json), JSON pointer `""` (document root) (method selects its named cases) | IID/block distribution or resampling-law comparison; no printed experiment replicated |
| [CorrelationVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CorrelationVerificationTests.cs) | Implemented — `Pearson_MatchesNumpyCorrcoef` | [correlation.json](../tests/Verification/vectors/correlation.json), JSON pointer `""` (document root) (method selects its named cases) | NumPy/SciPy correlation comparison on synthetic data |
| [CorrelationVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CorrelationVerificationTests.cs) | Implemented — `Spearman_MatchesScipy` | [correlation.json](../tests/Verification/vectors/correlation.json), JSON pointer `""` (document root) (method selects its named cases) | NumPy/SciPy correlation comparison on synthetic data |
| [CorrelationVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CorrelationVerificationTests.cs) | Implemented — `KendallTauB_MatchesScipy` | [correlation.json](../tests/Verification/vectors/correlation.json), JSON pointer `""` (document root) (method selects its named cases) | NumPy/SciPy correlation comparison on synthetic data |
| [CovarianceVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CovarianceVerificationTests.cs) | Implemented — `SampleCovariance_MatchesNumpyCov` | [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `""` (document root) (method selects its named cases) | Sample formula or declared C#-pipeline port; no independent printed-result target |
| [CovarianceVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CovarianceVerificationTests.cs) | Implemented — `Denoised_MatchesPortedPipeline` | [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `""` (document root) (method selects its named cases) | Sample formula or declared C#-pipeline port; no independent printed-result target |
| [CovarianceVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CovarianceVerificationTests.cs) | Implemented — `Detoned_MatchesPortedPipeline` | [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `""` (document root) (method selects its named cases) | Sample formula or declared C#-pipeline port; no independent printed-result target |
| [CovarianceVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CovarianceVerificationTests.cs) | Implemented — `TracyWidomDenoised_MatchesPortedPipeline` | [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `""` (document root) (method selects its named cases) | Sample formula or declared C#-pipeline port; no independent printed-result target |
| [CovarianceVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/CovarianceVerificationTests.cs) | Implemented — `DoublySparse_MatchesPortedPipeline` | [covariance.json](../tests/Verification/vectors/covariance.json), JSON pointer `""` (document root) (method selects its named cases) | Sample formula or declared C#-pipeline port; no independent printed-result target |
| [DistributionVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/DistributionVerificationTests.cs) | Implemented — `NormalCdf_MatchesScipyReferenceValues` | [distributions.json](../tests/Verification/vectors/distributions.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy CDF/quantile comparison; not a published approximation-accuracy sweep |
| [DistributionVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/DistributionVerificationTests.cs) | Implemented — `NormalInverseCdf_MatchesScipyReferenceValues` | [distributions.json](../tests/Verification/vectors/distributions.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy CDF/quantile comparison; not a published approximation-accuracy sweep |
| [GarchVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/GarchVerificationTests.cs) | Implemented — `GarchFit_MatchesScipy_Parameters` | [garch.json](../tests/Verification/vectors/garch.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy fitted synthetic GARCH; not Bollerslev’s empirical table |
| [GarchVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/GarchVerificationTests.cs) | Implemented — `GarchFit_MatchesScipy_ConditionalVariances` | [garch.json](../tests/Verification/vectors/garch.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy fitted synthetic GARCH; not Bollerslev’s empirical table |
| [GraphicalLassoVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/GraphicalLassoVerificationTests.cs) | Implemented — `GraphicalLasso_3x3Sparse_BicSelectsSklearnLambda_MatchesSklearn` | [graphical_lasso.json](../tests/Verification/vectors/graphical_lasso.json), JSON pointer `""` (document root) (method selects its named cases) | scikit-learn independent comparison/negative control on the same S: off-diagonal penalty at S + λI equals the all-entry penalty at S; BIC is a library extension evaluated against original S |
| [GraphicalLassoVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/GraphicalLassoVerificationTests.cs) | Implemented — `GraphicalLasso_3x3Sparse_ZeroEntriesMatchSklearn` | [graphical_lasso.json](../tests/Verification/vectors/graphical_lasso.json), JSON pointer `""` (document root) (method selects its named cases) | scikit-learn independent comparison/negative control on the same S: off-diagonal penalty at S + λI equals the all-entry penalty at S; BIC is a library extension evaluated against original S |
| [GraphicalLassoVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/GraphicalLassoVerificationTests.cs) | Implemented — `GraphicalLasso_5x5BlockDiagonal_MatchesSklearn` | [graphical_lasso.json](../tests/Verification/vectors/graphical_lasso.json), JSON pointer `""` (document root) (method selects its named cases) | scikit-learn independent comparison/negative control on the same S: off-diagonal penalty at S + λI equals the all-entry penalty at S; BIC is a library extension evaluated against original S |
| [GraphicalLassoVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/GraphicalLassoVerificationTests.cs) | Implemented — `GraphicalLasso_5x5BlockDiagonal_CrossBlockEntriesNearZero` | [graphical_lasso.json](../tests/Verification/vectors/graphical_lasso.json), JSON pointer `""` (document root) (method selects its named cases) | scikit-learn independent comparison/negative control on the same S: off-diagonal penalty at S + λI equals the all-entry penalty at S; BIC is a library extension evaluated against original S |
| [GraphicalLassoVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/GraphicalLassoVerificationTests.cs) | Implemented — `GraphicalLasso_LargeLambda_OracleRejectsCorruptedDiagonal` | [graphical_lasso.json](../tests/Verification/vectors/graphical_lasso.json), JSON pointer `""` (document root) (method selects its named cases) | scikit-learn independent comparison/negative control on the same S: off-diagonal penalty at S + λI equals the all-entry penalty at S; BIC is a library extension evaluated against original S |
| [GraphicalLassoVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/GraphicalLassoVerificationTests.cs) | Implemented — `GraphicalLasso_LargeLambda_SklearnAgrees` | [graphical_lasso.json](../tests/Verification/vectors/graphical_lasso.json), JSON pointer `""` (document root) (method selects its named cases) | scikit-learn independent comparison/negative control on the same S: off-diagonal penalty at S + λI equals the all-entry penalty at S; BIC is a library extension evaluated against original S |
| [HaircutFisherVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/HaircutFisherVerificationTests.cs) | Implemented — `FisherZTransform_ForwardInverseRoundtrip_WithinMachineEpsilon` | [haircut_fisher.json](../tests/Verification/vectors/haircut_fisher.json), JSON pointer `""` (document root) (method selects its named cases) | Facade parity or inverse round-trip; not an independent paper-result oracle |
| [HaircutFisherVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/HaircutFisherVerificationTests.cs) | Implemented — `FisherZTransform_InverseForwardRoundtrip_WithinMachineEpsilon` | [haircut_fisher.json](../tests/Verification/vectors/haircut_fisher.json), JSON pointer `""` (document root) (method selects its named cases) | Facade parity or inverse round-trip; not an independent paper-result oracle |
| [InterpolationVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/InterpolationVerificationTests.cs) | Implemented — `ReferenceCoverage_RejectsEmptyAndMismatchedVectors` | [interpolation.json](../tests/Verification/vectors/interpolation.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy/NumPy interpolation comparison or coverage guard; PCHIP endpoint/interior slope policy differs from Fritsch–Carlson |
| [InterpolationVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/InterpolationVerificationTests.cs) | Implemented — `LinearInterpolator_MatchesScipy` | [interpolation.json](../tests/Verification/vectors/interpolation.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy/NumPy interpolation comparison or coverage guard; PCHIP endpoint/interior slope policy differs from Fritsch–Carlson |
| [InterpolationVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/InterpolationVerificationTests.cs) | Implemented — `MonotoneCubic_MatchesScipyPchip` | [interpolation.json](../tests/Verification/vectors/interpolation.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy/NumPy interpolation comparison or coverage guard; PCHIP endpoint/interior slope policy differs from Fritsch–Carlson |
| [InterpolationVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/InterpolationVerificationTests.cs) | Implemented — `NaturalCubicSpline_MatchesScipy` | [interpolation.json](../tests/Verification/vectors/interpolation.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy/NumPy interpolation comparison or coverage guard; PCHIP endpoint/interior slope policy differs from Fritsch–Carlson |
| [InterpolationVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/InterpolationVerificationTests.cs) | Implemented — `TwoPointLinearInterpolator_MatchesNumpyReference` | [interpolation.json](../tests/Verification/vectors/interpolation.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy/NumPy interpolation comparison or coverage guard; PCHIP endpoint/interior slope policy differs from Fritsch–Carlson |
| [InterpolationVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/InterpolationVerificationTests.cs) | Implemented — `MonotoneCubicSplineN2_MatchesScipyPchip` | [interpolation.json](../tests/Verification/vectors/interpolation.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy/NumPy interpolation comparison or coverage guard; PCHIP endpoint/interior slope policy differs from Fritsch–Carlson |
| [LevenbergMarquardtVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/LevenbergMarquardtVerificationTests.cs) | Implemented — `Solve_ExponentialDecayNoisy_MatchesScipy` | [lm.json](../tests/Verification/vectors/lm.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy solves of synthetic residual models; independent solver comparison |
| [LevenbergMarquardtVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/LevenbergMarquardtVerificationTests.cs) | Implemented — `Solve_SumOfSinusoids_MatchesScipy` | [lm.json](../tests/Verification/vectors/lm.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy solves of synthetic residual models; independent solver comparison |
| [LevenbergMarquardtVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/LevenbergMarquardtVerificationTests.cs) | Implemented — `Solve_GompertzGrowth_MatchesScipy` | [lm.json](../tests/Verification/vectors/lm.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy solves of synthetic residual models; independent solver comparison |
| [LevenbergMarquardtVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/LevenbergMarquardtVerificationTests.cs) | Implemented — `Solve_Logistic_MatchesScipy` | [lm.json](../tests/Verification/vectors/lm.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy solves of synthetic residual models; independent solver comparison |
| [LinearAlgebraVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/LinearAlgebraVerificationTests.cs) | Implemented — `Cholesky_ReconstructsNumpyInputToPrecisionNumeric` | [linalg.json](../tests/Verification/vectors/linalg.json), JSON pointer `""` (document root) (method selects its named cases) | NumPy LAPACK solve/eigenvalue or reconstruction comparison on synthetic matrices |
| [LinearAlgebraVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/LinearAlgebraVerificationTests.cs) | Implemented — `JacobiEigenvalues_MatchNumpyLapackToStatisticalPrecision` | [linalg.json](../tests/Verification/vectors/linalg.json), JSON pointer `""` (document root) (method selects its named cases) | NumPy LAPACK solve/eigenvalue or reconstruction comparison on synthetic matrices |
| [LinearAlgebraVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/LinearAlgebraVerificationTests.cs) | Implemented — `GaussianElimination_SolvesLinearSystemToPrecisionNumeric` | [linalg.json](../tests/Verification/vectors/linalg.json), JSON pointer `""` (document root) (method selects its named cases) | NumPy LAPACK solve/eigenvalue or reconstruction comparison on synthetic matrices |
| [LowerTailDependenceVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/LowerTailDependenceVerificationTests.cs) | Implemented — `MonotoneLambda_MatchesNumpyReference` | [lower_tail.json](../tests/Verification/vectors/lower_tail.json), JSON pointer `""` (document root) (method selects its named cases) | Synthetic finite-q ratio, not the limiting Coles tail-dependence coefficient |
| [LowerTailDependenceVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/LowerTailDependenceVerificationTests.cs) | Implemented — `IndependentLambda_MatchesNumpyReference` | [lower_tail.json](../tests/Verification/vectors/lower_tail.json), JSON pointer `""` (document root) (method selects its named cases) | Synthetic finite-q ratio, not the limiting Coles tail-dependence coefficient |
| [NelderMeadVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/NelderMeadVerificationTests.cs) | Implemented — `NelderMead_MatchesSciPyReferenceArgmin` | [nelder_mead.json](../tests/Verification/vectors/nelder_mead.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy optima on benchmark functions; no published evaluation-count target |
| [OrdinaryLeastSquaresVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/OrdinaryLeastSquaresVerificationTests.cs) | Implemented — `Fit_MatchesStatsmodels` | [ols.json](../tests/Verification/vectors/ols.json), JSON pointer `""` (document root) (method selects its named cases) | statsmodels on synthetic regression cases; certified NIST checks are separately listed above |
| [PsdVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/PsdVerificationTests.cs) | Implemented — `Higham_MatchesReferenceNearestCorrelationMatrix` | [psd.json](../tests/Verification/vectors/psd.json), JSON pointer `""` (document root) (method selects its named cases) | Independent Python projection on synthetic matrices; the printed Higham example is separately listed above |
| [PsdVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/PsdVerificationTests.cs) | Implemented — `EigenClip_MatchesReferenceNearestPsdMatrix` | [psd.json](../tests/Verification/vectors/psd.json), JSON pointer `""` (document root) (method selects its named cases) | Independent Python projection on synthetic matrices; the printed Higham example is separately listed above |
| [QpSolverVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/QpSolverVerificationTests.cs) | Implemented — `ActiveSetQpSolver_MinVar_MatchesSlsqp` | [qp_solver.json](../tests/Verification/vectors/qp_solver.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy SLSQP synthetic constrained minimization |
| [QpSolverVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/QpSolverVerificationTests.cs) | Implemented — `ActiveSetQpSolver_MeanVar_MatchesSlsqp` | [qp_solver.json](../tests/Verification/vectors/qp_solver.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy SLSQP synthetic constrained minimization |
| [SampleMomentsVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/SampleMomentsVerificationTests.cs) | Implemented — `SampleSkewness_AllCases_MatchScipy` | [sample_moments.json](../tests/Verification/vectors/sample_moments.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy moments on synthetic arrays; no underlying numeric paper example |
| [SampleMomentsVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/SampleMomentsVerificationTests.cs) | Implemented — `SampleExcessKurtosis_AllCases_MatchScipy` | [sample_moments.json](../tests/Verification/vectors/sample_moments.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy moments on synthetic arrays; no underlying numeric paper example |
| [ScalarStatisticsVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/ScalarStatisticsVerificationTests.cs) | Implemented — `PSquareEstimator_MatchesReference` | [scalar_stats.json](../tests/Verification/vectors/scalar_stats.json), JSON pointer `""` (document root) (method selects its named cases) | P² port of the same marker algorithm; paper Table I remains uncovered |
| [SolverVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/SolverVerificationTests.cs) | Implemented — `BrentSolver_MatchesReferenceRoots` | [solvers.json](../tests/Verification/vectors/solvers.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy roots of six synthetic functions; not paper step counts or published traces |
| [SolverVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/SolverVerificationTests.cs) | Implemented — `BisectionSolver_MatchesReferenceRoots` | [solvers.json](../tests/Verification/vectors/solvers.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy roots of six synthetic functions; not paper step counts or published traces |
| [SolverVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/SolverVerificationTests.cs) | Implemented — `NewtonRaphsonSolver_Bracketed_MatchesReferenceRoots` | [solvers.json](../tests/Verification/vectors/solvers.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy roots of six synthetic functions; not paper step counts or published traces |
| [SolverVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/SolverVerificationTests.cs) | Implemented — `NewtonRaphsonSolver_Unbracketed_MatchesReferenceRoots` | [solvers.json](../tests/Verification/vectors/solvers.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy roots of six synthetic functions; not paper step counts or published traces |
| [SolverVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/SolverVerificationTests.cs) | Implemented — `SecantSolver_MatchesReferenceRoots` | [solvers.json](../tests/Verification/vectors/solvers.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy roots of six synthetic functions; not paper step counts or published traces |
| [SolverVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/SolverVerificationTests.cs) | Implemented — `MullerSolver_MatchesReferenceRoots` | [solvers.json](../tests/Verification/vectors/solvers.json), JSON pointer `""` (document root) (method selects its named cases) | SciPy roots of six synthetic functions; not paper step counts or published traces |
| [WelfordPcaVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/WelfordPcaVerificationTests.cs) | Implemented — `WelfordMoments_FloatSeries_MatchesReference` | [welford_pca.json](../tests/Verification/vectors/welford_pca.json), JSON pointer `""` (document root) (method selects its named cases) | Welford translation/parity or NumPy PCA on synthetic arrays; NIST checks are separately listed above |
| [WelfordPcaVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/WelfordPcaVerificationTests.cs) | Implemented — `WelfordMoments_BivariatePearson_MatchesReference` | [welford_pca.json](../tests/Verification/vectors/welford_pca.json), JSON pointer `""` (document root) (method selects its named cases) | Welford translation/parity or NumPy PCA on synthetic arrays; NIST checks are separately listed above |
| [WelfordPcaVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/WelfordPcaVerificationTests.cs) | Implemented — `WelfordMoments_PearsonLargeDecimalTranslation_PreservesExactCovariance` | [welford_pca.json](../tests/Verification/vectors/welford_pca.json), JSON pointer `""` (document root) (method selects its named cases) | Welford translation/parity or NumPy PCA on synthetic arrays; NIST checks are separately listed above |
| [WelfordPcaVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/WelfordPcaVerificationTests.cs) | Implemented — `WelfordMoments_Merge_MatchesFullSeriesReference` | [welford_pca.json](../tests/Verification/vectors/welford_pca.json), JSON pointer `""` (document root) (method selects its named cases) | Welford translation/parity or NumPy PCA on synthetic arrays; NIST checks are separately listed above |
| [WelfordPcaVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/WelfordPcaVerificationTests.cs) | Implemented — `WelfordMoments_Decimal_SmallIntegerSeries_MatchesReference` | [welford_pca.json](../tests/Verification/vectors/welford_pca.json), JSON pointer `""` (document root) (method selects its named cases) | Welford translation/parity or NumPy PCA on synthetic arrays; NIST checks are separately listed above |
| [WelfordPcaVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/WelfordPcaVerificationTests.cs) | Implemented — `Pca_Decompose_3x3_EigenvaluesMatchReference` | [welford_pca.json](../tests/Verification/vectors/welford_pca.json), JSON pointer `""` (document root) (method selects its named cases) | Welford translation/parity or NumPy PCA on synthetic arrays; NIST checks are separately listed above |
| [WelfordPcaVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/WelfordPcaVerificationTests.cs) | Implemented — `Pca_Decompose_3x3_EigenvectorsMatchReference` | [welford_pca.json](../tests/Verification/vectors/welford_pca.json), JSON pointer `""` (document root) (method selects its named cases) | Welford translation/parity or NumPy PCA on synthetic arrays; NIST checks are separately listed above |
| [WelfordPcaVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/WelfordPcaVerificationTests.cs) | Implemented — `Pca_Decompose_3x3_ExplainedVarianceMatchesReference` | [welford_pca.json](../tests/Verification/vectors/welford_pca.json), JSON pointer `""` (document root) (method selects its named cases) | Welford translation/parity or NumPy PCA on synthetic arrays; NIST checks are separately listed above |
| [WelfordPcaVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/WelfordPcaVerificationTests.cs) | Implemented — `Pca_FromReturns_50x4_MeansMatchReference` | [welford_pca.json](../tests/Verification/vectors/welford_pca.json), JSON pointer `""` (document root) (method selects its named cases) | Welford translation/parity or NumPy PCA on synthetic arrays; NIST checks are separately listed above |
| [WelfordPcaVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/WelfordPcaVerificationTests.cs) | Implemented — `Pca_FromReturns_50x4_EigenvaluesMatchReference` | [welford_pca.json](../tests/Verification/vectors/welford_pca.json), JSON pointer `""` (document root) (method selects its named cases) | Welford translation/parity or NumPy PCA on synthetic arrays; NIST checks are separately listed above |
| [WelfordPcaVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/WelfordPcaVerificationTests.cs) | Implemented — `Pca_FromReturns_50x4_ExplainedVarianceMatchesReference` | [welford_pca.json](../tests/Verification/vectors/welford_pca.json), JSON pointer `""` (document root) (method selects its named cases) | Welford translation/parity or NumPy PCA on synthetic arrays; NIST checks are separately listed above |
| [WelfordPcaVerificationTests.cs](../tests/Boutquin.Numerics.Tests.Verification/WelfordPcaVerificationTests.cs) | Implemented — `Pca_FromReturns_50x4_ProjectTop2_MatchesReference` | [welford_pca.json](../tests/Verification/vectors/welford_pca.json), JSON pointer `""` (document root) (method selects its named cases) | Welford translation/parity or NumPy PCA on synthetic arrays; NIST checks are separately listed above |

Additional cases that should not be mistaken for printed-paper replication:

| Test file | Functions / status | Reason |
| --- | --- | --- |
| [FamaFrenchPcaTests.cs](../tests/Boutquin.Numerics.Tests.Unit/LinearAlgebra/FamaFrenchPcaTests.cs) | Implemented — `Standardized_EigenvalueSumEqualsFactorCount`, `Standardized_CumulativeVarianceReachesOne`, `Standardized_ExplainedVarianceRatiosAreDescending`, `Standardized_FirstEigenvalueReflectsFactorCorrelation`, `Standardized_ReducesFirstComponentDominanceVersusUnstandardized`, `Standardized_MeanReflectsActualColumnMeans`, `Standardized_EigenvectorsAreOrthonormal`, `Standardized_NumComponentsFor95PercentVarianceMatchesCumulative`, `Standardized_ProjectionRoundTripRecoversCenteredData` | All nine methods check PCA identities/bounds on embedded factor data. The class summary says published CAPM results, but it supplies no published numeric eigenvalue or projection target. |
| [NelderMeadComparatorCharacterizationTests.cs](../tests/Boutquin.Numerics.Tests.Unit/Solvers/NelderMeadComparatorCharacterizationTests.cs) | Implemented — `Rosenbrock_PinnedResult`, `Sphere3D_PinnedResult`, `Beale_PinnedResult` | Pinned iterates and iteration counts are implementation goldens. In particular Rosenbrock_PinnedResult checks 148 iterations, not the evaluation count reported by Nelder–Mead. |
| [Solvers_ParityTests.cs](../tests/Boutquin.Numerics.Tests.Unit/GenericParity/Solvers_ParityTests.cs) | Implemented — `BrentSolver_MatchesApprovedDoubleVector`, `BisectionSolver_MatchesHistoricalDoubleVector`, `SecantSolver_MatchesHistoricalDoubleVector`, `MullerSolver_MatchesHistoricalDoubleVector`, `NewtonRaphsonSolver_Bracketed_MatchesApprovedDoubleVector`, `NewtonRaphsonSolver_Unbracketed_MatchesHistoricalDoubleVector`, `LevenbergMarquardtSolver_MatchesHistoricalDoubleVector`, `OrdinaryLeastSquares_MatchesApprovedDoubleVector` | Historical facade vectors check compatibility with prior library behavior; they are not independently printed paper traces. |

## Paper-backed components without a verified numeric replication mapping

The following sources support an algorithm or an implementation decision, but no test in this audit could be tied to a printed numeric result from that source. This is a provenance gap, not a claim that the class is unimplemented. Some papers supply theory rather than a deterministic finite example. The comparison index above locates the closest existing numeric checks.

| Class (source file) | Paper / source filename | Replication status |
| --- | --- | --- |
| `ActiveSetQpSolver` ([ActiveSetQpSolver.cs](../src/Boutquin.Numerics/Solvers/ActiveSetQpSolver.cs)) | nocedal-wright-2006-numerical-optimization-2e.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `BisectionSolver` ([BisectionSolver.cs](../src/Boutquin.Numerics/Solvers/BisectionSolver.cs)) | press-et-al-2007-numerical-recipes-3e.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `BootstrapResampler` ([BootstrapResampler.cs](../src/Boutquin.Numerics/MonteCarlo/BootstrapResampler.cs)) | lahiri-1999-theoretical-comparisons-block-bootstrap.pdf, politis-romano-1991-circular-block-resampling-tr370.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `BrentSolver` ([BrentSolver.cs](../src/Boutquin.Numerics/Solvers/BrentSolver.cs)) | brent-1973-algorithms-minimization.pdf, press-et-al-2007-numerical-recipes-3e.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `CovarianceHelpers` ([CovarianceHelpers.cs](../src/Boutquin.Numerics/Statistics/CovarianceHelpers.cs)) | laloux-1999-noise-dressing.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `CumulativeNormal` ([CumulativeNormal.cs](../src/Boutquin.Numerics/Distributions/CumulativeNormal.cs)) | laikov-2025-erf-approximation.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `DampedLinearSolve` ([DampedLinearSolve.cs](../src/Boutquin.Numerics/Solvers/Internal/DampedLinearSolve.cs)) | press-et-al-2007-numerical-recipes-3e.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `DoublySparseEstimator` ([DoublySparseEstimator.cs](../src/Boutquin.Numerics/Statistics/DoublySparseEstimator.cs)) | kutateladze-2024-doubly-sparse-covariance.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `FiniteDifferenceJacobian` ([FiniteDifferenceJacobian.cs](../src/Boutquin.Numerics/Solvers/Internal/FiniteDifferenceJacobian.cs)) | press-et-al-2007-numerical-recipes-3e.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `GarchOneOneEstimator` ([GarchOneOneEstimator.cs](../src/Boutquin.Numerics/Statistics/GarchOneOneEstimator.cs)) | bollerslev-1986-garch.pdf, bollerslev-wooldridge-1992-quasi-maximum-likelihood-time-varying-covariances.pdf, hull-2021-options-futures-derivatives-11e-global.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `GaussianElimination` ([GaussianElimination.cs](../src/Boutquin.Numerics/LinearAlgebra/GaussianElimination.cs)) | golub-van-loan-2013-matrix-computations-4e.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `GaussianSampler` ([GaussianSampler.cs](../src/Boutquin.Numerics/Random/GaussianSampler.cs)) | press-et-al-2007-numerical-recipes-3e.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `GraphicalLassoSolver` ([GraphicalLassoSolver.cs](../src/Boutquin.Numerics/Statistics/GraphicalLassoSolver.cs)) | friedman-2008-graphical-lasso.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `HouseholderQr` ([HouseholderQr.cs](../src/Boutquin.Numerics/LinearAlgebra/Internal/HouseholderQr.cs)) | golub-van-loan-2013-matrix-computations-4e.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `IRandomSource` ([IRandomSource.cs](../src/Boutquin.Numerics/Random/IRandomSource.cs)) | lemire-2018-nearly-divisionless.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `InverseNormal` ([InverseNormal.cs](../src/Boutquin.Numerics/Distributions/InverseNormal.cs)) | acklam-2000-inverse-normal.txt | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `JacobiEigenDecomposition` ([JacobiEigenDecomposition.cs](../src/Boutquin.Numerics/LinearAlgebra/JacobiEigenDecomposition.cs)) | golub-van-loan-2013-matrix-computations-4e.pdf, press-et-al-2007-numerical-recipes-3e.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `LowerTailDependence` ([LowerTailDependence.cs](../src/Boutquin.Numerics/Statistics/LowerTailDependence.cs)) | coles-heffernan-tawn-1999-dependence-measures.pdf, nelsen-2006-introduction-to-copulas-2e.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `MovingBlockBootstrapResampler` ([MovingBlockBootstrapResampler.cs](../src/Boutquin.Numerics/MonteCarlo/MovingBlockBootstrapResampler.cs)) | kunsch-1989-jackknife-bootstrap-stationary.pdf, lahiri-1999-theoretical-comparisons-block-bootstrap.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `MullerSolver` ([MullerSolver.cs](../src/Boutquin.Numerics/Solvers/MullerSolver.cs)) | muller-1956-algebraic-equations.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `NewtonRaphsonSolver` ([NewtonRaphsonSolver.cs](../src/Boutquin.Numerics/Solvers/NewtonRaphsonSolver.cs)) | press-et-al-2007-numerical-recipes-3e.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `PrincipalComponentAnalysis` ([PrincipalComponentAnalysis.cs](../src/Boutquin.Numerics/LinearAlgebra/PrincipalComponentAnalysis.cs)) | hull-2021-options-futures-derivatives-11e-global.pdf, jolliffe-2002-principal-component-analysis-2e.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `SecantSolver` ([SecantSolver.cs](../src/Boutquin.Numerics/Solvers/SecantSolver.cs)) | press-et-al-2007-numerical-recipes-3e.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `SplineBoundary` ([SplineBoundary.cs](../src/Boutquin.Numerics/Interpolation/SplineBoundary.cs)) | jarre-2025-cubic-spline-functions-revisited.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |
| `StationaryBootstrapResampler` ([StationaryBootstrapResampler.cs](../src/Boutquin.Numerics/MonteCarlo/StationaryBootstrapResampler.cs)) | lahiri-1999-theoretical-comparisons-block-bootstrap.pdf, politis-romano-1994-stationary-bootstrap.pdf | Not yet verified against a printed numeric result; existing implementation/tests are separate. |

## Source and interpretation notes

- **Fast double bootstrap:** `rbnew3dp.pdf` is the primary standard-FDB manuscript. The locally named `davidson-2023-improved-fast-double-bootstrap.pdf` contains a different seminar abstract and cannot establish the 2023 conditional refinement. The listed tests exercise the earlier FDB, including the library’s quantile/tail conventions.
- **Haircut Sharpe:** the scalar compatibility overload assumes every other trial has p = 1; `ComputePanel` and `AdjustPValues` provide the full-panel Holm/BHY procedures. The paper’s worked independent-trial adjustment is Sidak. Its worked-example check uses deliberately broad approximate tolerances.
- **Fisher transform:** the listed numeric transform/interval checks are mapped to Numerical Recipes’ transform and standard-error equations. They do not reproduce the original Fisher (1915) finite-sample density.
- **Acklam:** `acklam-2000-inverse-normal.txt` is a repository citation record, not an original PDF or a published numeric table. Inverse-normal verification uses SciPy; it must not be cited as a literal paper example.
- **RMT and shrinkage:** tests of an independently reconstructed matrix or author formula are distinguished from a NumPy port of the library pipeline. A fallback policy, fitted noise edge, sample denominator, rank-tie convention, or scalar multiple-testing approximation is stated where it changes the claim.
- **P² and spline tables:** absence of a printed-result test is documented even when formula, between-knot, or synthetic-stream tests already exist. Random meshes and unseeded paper simulations need primary fixtures for exact replay.
- **Status maintenance:** update the Inputs, Output, tolerance and exact test method together when an assertion changes. Change a missing case to Implemented only when a test asserts that case’s independent target, and retain its source locus and basis.

## Validation of this audit

- Release build: passed, 0 warnings and 0 errors.
- Existing test suite: 1,966 passed, 0 failed, 13 existing stress tests skipped (1,791 unit, 159 verification, 16 architecture tests passed).
- `dotnet format Boutquin.Numerics.slnx --no-restore --verify-no-changes`: passed.
- Catalogue structure, production signatures, exact test methods, local links, section anchors and 154 fixture pointers: checked. Every method in the verification project appears either in the detailed catalogue or the comparison index.
- The new document contains no private-document path references. The repository-wide path scan has an existing self-reference in the policy text in CLAUDE.md; it is not a reference introduced by this document.
