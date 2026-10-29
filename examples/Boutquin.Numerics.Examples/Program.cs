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

using Boutquin.Numerics.Collections;
using Boutquin.Numerics.Distributions;
using Boutquin.Numerics.Examples;
using Boutquin.Numerics.Interpolation;
using Boutquin.Numerics.LinearAlgebra;
using Boutquin.Numerics.MonteCarlo;
using Boutquin.Numerics.Random;
using Boutquin.Numerics.Solvers;
using Boutquin.Numerics.Statistics;

Console.WriteLine("Boutquin.Numerics Examples");
Console.WriteLine(new string('─', 50));

var registry = BuildExampleRegistry();

if (args.Contains("--help") || args.Contains("-h"))
{
    PrintHelp(registry);
    return;
}

string? selectedId = ParseExampleArg(args, out string? parseError);
if (parseError is not null)
{
    Console.Error.WriteLine($"Error: {parseError}");
    Console.Error.WriteLine("Use --help to see available examples.");
    return;
}

PrintListing(registry, selectedId);
Console.WriteLine();

if (selectedId is not null)
{
    if (!registry.TryGetValue(selectedId, out var entry))
    {
        Console.Error.WriteLine($"Example '{selectedId}' not found. Use --help to see available examples.");
        return;
    }
    await entry.Run().ConfigureAwait(false);
}
else
{
    foreach (var entry in registry.Values)
    {
        await entry.Run().ConfigureAwait(false);
    }
}

// ── Registry ──────────────────────────────────────────────────────────────────

static SortedDictionary<string, ExampleEntry> BuildExampleRegistry()
{
    static ExampleEntry Sync(string id, string desc, Action run) =>
        new(id, desc, () => { run(); return Task.CompletedTask; });

    return new SortedDictionary<string, ExampleEntry>(StringComparer.OrdinalIgnoreCase)
    {
        ["01"] = Sync("01", "Root Finding", Example01RootFinding),
        ["02"] = Sync("02", "Distributions", Example02Distributions),
        ["03"] = Sync("03", "Interpolation", Example03Interpolation),
        ["04"] = Sync("04", "Linear Algebra", Example04LinearAlgebra),
        ["05"] = Sync("05", "Nonlinear Least Squares", Example05NonlinearLeastSquares),
        ["06"] = Sync("06", "Ordinary Least Squares", Example06OrdinaryLeastSquares),
        ["07"] = Sync("07", "Portfolio Optimization (QP)", Example07PortfolioQp),
        ["08"] = Sync("08", "Principal Component Analysis", Example08Pca),
        ["09"] = Sync("09", "Covariance Estimation", Example09CovarianceEstimation),
        ["10"] = Sync("10", "Correlation", Example10Correlation),
        ["11"] = Sync("11", "Online Moments", Example11OnlineMoments),
        ["12"] = Sync("12", "Higher Sample Moments", Example12HigherMoments),
        ["13"] = Sync("13", "Backtest Statistics", Example13BacktestStatistics),
        ["14"] = Sync("14", "Monte Carlo Bootstrap", Example14MonteCarloBootstrap),
        ["15"] = Sync("15", "Quasi-Monte Carlo", Example15QuasiMonteCarlo),
        ["16"] = Sync("16", "Random Number Generation", Example16RandomNumbers),
        ["17"] = Sync("17", "Rolling Window", Example17RollingWindow),
        ["18"] = Sync("18", "Generic-Math Precision", Example18GenericMath),
        ["19"] = Sync("19", "Derivative-Free Optimization and GARCH", Example19OptimizationAndGarch),
        ["20"] = Sync("20", "Sparse Precision and Tail Dependence", Example20SparsePrecisionAndTailDependence),
    };
}

static string? ParseExampleArg(string[] cmdArgs, out string? error)
{
    error = null;
    for (var i = 0; i < cmdArgs.Length; i++)
    {
        if (cmdArgs[i] == "--example")
        {
            if (i + 1 == cmdArgs.Length)
            {
                error = "--example requires a value.";
                return null;
            }

            return NormalizeId(cmdArgs[i + 1]);
        }

        if (cmdArgs[i].StartsWith("--example=", StringComparison.Ordinal))
        {
            return NormalizeId(cmdArgs[i]["--example=".Length..]);
        }
    }
    return null;
}

static string NormalizeId(string raw)
{
    if (int.TryParse(raw, out int n))
    {
        return n.ToString("D2", CultureInfo.InvariantCulture);
    }

    return raw;
}

static void PrintListing(SortedDictionary<string, ExampleEntry> reg, string? selectedId)
{
    Console.WriteLine("Usage: dotnet run -- [--example <id>] [--help]");
    Console.WriteLine();
    Console.WriteLine("Available examples:");
    foreach (var (id, entry) in reg)
    {
        string marker = string.Equals(id, selectedId, StringComparison.OrdinalIgnoreCase) ? "▶" : " ";
        Console.WriteLine($"  {marker} --example {entry.Id}  {entry.Description}");
    }
}

static void PrintHelp(SortedDictionary<string, ExampleEntry> reg) =>
    PrintListing(reg, selectedId: null);

// ── Examples ──────────────────────────────────────────────────────────────────

static void Example01RootFinding()
{
    Console.WriteLine("── 01  Root Finding ─────────────────────────────────");

    // Brent — bracketed, superlinear, with a forced-bisection safeguard.
    var brent = new BrentSolver(tolerance: 1e-12, maxIterations: 100);
    var result = brent.Solve(x => x * x - 2.0, lowerBound: 0.0, upperBound: 2.0);
    Console.WriteLine($"  Brent  √2 ≈ {result.Root:F15}  converged={result.Converged}  iters={result.Iterations}");

    // Newton-Raphson with analytical derivative — quadratic convergence.
    var newton = new NewtonRaphsonSolver(derivative: x => 2.0 * x);
    var nr = newton.Solve(x => x * x - 2.0, initialGuess: 1.0);
    Console.WriteLine($"  Newton √2 ≈ {nr.Root:F15}  converged={nr.Converged}  iters={nr.Iterations}");

    // Secant — derivative-free, superlinear; two-point overload.
    var secant = new SecantSolver();
    var sr = secant.Solve(x => x * x - 2.0, p0: 0.5, p1: 2.0);
    Console.WriteLine($"  Secant √2 ≈ {sr.Root:F15}  converged={sr.Converged}  iters={sr.Iterations}");
}

static void Example02Distributions()
{
    Console.WriteLine("── 02  Distributions ────────────────────────────────");

    double pdf = NormalDistribution.Pdf(0.0);
    double cdf = NormalDistribution.Cdf(1.96);
    double quantile = NormalDistribution.InverseCdf(0.975);
    Console.WriteLine($"  N(0,1) PDF(0.0)         = {pdf:F8}");
    Console.WriteLine($"  N(0,1) CDF(1.96)        = {cdf:F6}");
    Console.WriteLine($"  N(0,1) InverseCDF(0.975) = {quantile:F6}");

    double cdfDirect = CumulativeNormal.Evaluate(1.96);
    double inverseDirect = InverseNormal.Evaluate(0.975);
    Console.WriteLine($"  CumulativeNormal(1.96)  = {cdfDirect:F6}");
    Console.WriteLine($"  InverseNormal(0.975)    = {inverseDirect:F6}");
}

static void Example03Interpolation()
{
    Console.WriteLine("── 03  Interpolation ────────────────────────────────");

    double[] xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    double[] ys = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0];   // y = x²; Q-splines require six knots.

    double linear = LinearInterpolator.Instance.Interpolate(2.5, xs, ys);
    double mono = MonotoneCubicInterpolator.Instance.Interpolate(2.5, xs, ys);
    Console.WriteLine($"  Linear(2.5)      = {linear:F4}  (exact 6.25)");
    Console.WriteLine($"  MonotoneCubic(2.5) = {mono:F4}  (exact 6.25)");

    var spline = new CubicSplineInterpolator(xs, ys, SplineBoundary.NotAKnot);
    Console.WriteLine($"  NotAKnot(2.5)    = {spline.Interpolate(2.5):F4}");

    var modSpline = new CubicSplineInterpolator(xs, ys, SplineBoundary.ModifiedNotAKnot);
    Console.WriteLine($"  ModNotAKnot(2.5) = {modSpline.Interpolate(2.5):F4}");

    var qSpline = new CubicSplineInterpolator(xs, ys, SplineBoundary.QSpline);
    Console.WriteLine($"  QSpline(2.5)     = {qSpline.Interpolate(2.5):F4}");

    var dynamic = InterpolatorFactory.Create(InterpolatorKind.MonotoneCubic);
    Console.WriteLine($"  Factory [{dynamic.Name}](2.5) = {dynamic.Interpolate(2.5, xs, ys):F4}");

    // Hagan-West monotone-convex — xs[0]=0, ys[0]=0 is virtual origin.
    double[] curveXs = [0.0, 0.25, 1.0, 5.0, 10.0, 30.0];
    double[] curveYs =
    [
        0.0,
        -Math.Log(0.995), -Math.Log(0.97),
        -Math.Log(0.83),  -Math.Log(0.68), -Math.Log(0.25),
    ];
    double ncr2y = MonotoneConvexInterpolator.Instance.Interpolate(2.0, curveXs, curveYs);
    double df2y = Math.Exp(-ncr2y);
    Console.WriteLine($"  MonotoneConvex 2y DF = {df2y:F6}  (non-neg forward rates)");
}

static void Example04LinearAlgebra()
{
    Console.WriteLine("── 04  Linear Algebra ───────────────────────────────");

    decimal[,] spd =
    {
        {  4m, 12m, -16m },
        { 12m, 37m, -43m },
        { -16m, -43m, 98m },
    };

    // Cholesky factorisation.
    decimal[,] L = CholeskyDecomposition.Decompose(spd);
    decimal[] solution = CholeskyDecomposition.Solve(L, [1m, 2m, 3m]);
    Console.WriteLine($"  Cholesky Ax=b solution: [{string.Join(", ", solution.Select(v => v.ToString("F6", CultureInfo.InvariantCulture)))}]");

    // Pivoted Cholesky for PSD matrices.
    var pivoted = CholeskyDecomposition.DecomposePivoted(spd, tolerance: 1e-10m, maxRank: 2);
    Console.WriteLine($"  Pivoted Cholesky rank = {pivoted.Rank}");

    // Eigendecomposition.
    var eigen = JacobiEigenDecomposition.Decompose(spd);
    Console.WriteLine($"  Eigenvalues: [{string.Join(", ", eigen.Values.Select(v => v.ToString("F4", CultureInfo.InvariantCulture)))}]");

    // Gaussian elimination with randomised pivoting.
    decimal[] x = GaussianElimination.Solve(spd, [1m, 2m, 3m], seed: 42);
    Console.WriteLine($"  Gauss Ax=b solution:  [{string.Join(", ", x.Select(v => v.ToString("F6", CultureInfo.InvariantCulture)))}]");

    // Project onto PSD cone.
    decimal[,] psd = NearestPsdProjection.EigenClip(SampleData.s_maybeIndefinite);
    Console.WriteLine($"  EigenClip IsPsd = {NearestPsdProjection.IsPsd(psd)}");

    // Nearest valid correlation matrix.
    decimal[,] cleanCorr = NearestPsdProjection.Higham(SampleData.s_messyCorr);
    Console.WriteLine($"  Higham IsPsd = {NearestPsdProjection.IsPsd(cleanCorr)}");
}

static void Example05NonlinearLeastSquares()
{
    Console.WriteLine("── 05  Nonlinear Least Squares ──────────────────────");

    double[] xs = SampleData.s_lmXs;
    double[] ys = SampleData.s_lmYs;

    double[] Residuals(double[] theta)
    {
        double a = theta[0], b = theta[1];
        var r = new double[xs.Length];
        for (var i = 0; i < xs.Length; i++)
        {
            r[i] = ys[i] - a * Math.Exp(b * xs[i]);
        }

        return r;
    }

    var solver = new LevenbergMarquardtSolver();
    var result = solver.Solve(residuals: Residuals, initialGuess: [1.0, 0.1]);

    if (result.Converged)
    {
        double a = result.Parameters[0];
        double b = result.Parameters[1];
        double sigmaA = Math.Sqrt(result.ParameterCovariance![0, 0]);
        double sigmaB = Math.Sqrt(result.ParameterCovariance[1, 1]);
        Console.WriteLine($"  Basic:   a={a:F6} ± {sigmaA:F6},  b={b:F6} ± {sigmaB:F6}  (target a=2.5, b=0.3)");
    }

    // Analytic Jacobian — ∂r_i/∂a = -exp(b·x_i), ∂r_i/∂b = -a·x_i·exp(b·x_i).
    double[,] Jacobian(double[] theta)
    {
        double a = theta[0], b = theta[1];
        var j = new double[xs.Length, 2];
        for (var i = 0; i < xs.Length; i++)
        {
            double e = Math.Exp(b * xs[i]);
            j[i, 0] = -e;
            j[i, 1] = -a * xs[i] * e;
        }
        return j;
    }

    var analytic = solver.Solve(Residuals, [1.0, 0.1], Jacobian);
    Console.WriteLine($"  Analytic Jacobian: a={analytic.Parameters[0]:F6},  b={analytic.Parameters[1]:F6}");

    // Box-bounded calibration.
    var bounded = solver.Solve(
        Residuals, [1.0, 0.1],
        jacobian: null,
        lowerBounds: [0.0, -1.0],
        upperBounds: [10.0, 1.0]);
    Console.WriteLine($"  Bounded: a={bounded.Parameters[0]:F6},  b={bounded.Parameters[1]:F6}  boundsActive={bounded.BoundsActive}");

    // Buffer pool — allocation-free repeated solve.
    var buffers = new LevenbergMarquardtBuffers(parameterCount: 2, residualCount: xs.Length);
    for (var run = 0; run < 3; run++)
    {
        var pooled = solver.Solve(Residuals, [1.0, 0.1], buffers);
        Console.WriteLine($"  Pool run {run + 1}: a={pooled.Parameters[0]:F6},  b={pooled.Parameters[1]:F6}");
    }
}

static void Example06OrdinaryLeastSquares()
{
    Console.WriteLine("── 06  Ordinary Least Squares ───────────────────────");

    var fit = OrdinaryLeastSquares.Fit(SampleData.s_longleyX, SampleData.s_longleyY);
    Console.WriteLine($"  Longley intercept = {fit.Coefficients[0]:F4}");
    Console.WriteLine($"  Longley b1 (PriceDefl) = {fit.Coefficients[1]:F4}  SE={fit.StandardErrors[1]:F4}");
    Console.WriteLine($"  R² = {fit.RSquared:F6},  σ̂ = {fit.ResidualStandardDeviation:F4},  dof = {fit.DegreesOfFreedom}");

    // Fit through origin.
    var noIntercept = OrdinaryLeastSquares.Fit(SampleData.s_longleyX, SampleData.s_longleyY, includeIntercept: false);
    Console.WriteLine($"  No-intercept R² = {noIntercept.RSquared:F6}");
}

static void Example07PortfolioQp()
{
    Console.WriteLine("── 07  Portfolio Optimization (QP) ──────────────────");

    decimal[] minVarWeights = ActiveSetQpSolver.SolveMinVariance(
        SampleData.s_covariance4, minWeight: 0.0m, maxWeight: 0.4m);
    Console.WriteLine($"  MinVariance:  [{string.Join(", ", minVarWeights.Select(w => w.ToString("F4", CultureInfo.InvariantCulture)))}]  Σ={minVarWeights.Sum():F4}");

    decimal[] mvWeights = ActiveSetQpSolver.SolveMeanVariance(
        SampleData.s_covariance4, SampleData.s_expectedReturns4,
        riskAversion: 3.0m, minWeight: 0.0m, maxWeight: 0.4m);
    Console.WriteLine($"  MeanVariance: [{string.Join(", ", mvWeights.Select(w => w.ToString("F4", CultureInfo.InvariantCulture)))}]  Σ={mvWeights.Sum():F4}");

    // Generic T = double for hot-path Monte Carlo calibration.
    double[] mvDouble = ActiveSetQpSolver<double>.SolveMeanVariance(
        SampleData.s_covariance4D, SampleData.s_expectedReturns4D,
        riskAversion: 3.0, minWeight: 0.0, maxWeight: 0.4);
    Console.WriteLine($"  MeanVar<double>: [{string.Join(", ", mvDouble.Select(w => w.ToString("F4", CultureInfo.InvariantCulture)))}]  Σ={mvDouble.Sum():F4}");
}

static void Example08Pca()
{
    Console.WriteLine("── 08  Principal Component Analysis ─────────────────");

    // From raw returns — the most common path.
    decimal[,] returns = SampleData.s_assetReturns;
    var pca = PrincipalComponentAnalysis.FromReturns(returns);

    decimal level = pca.ExplainedVarianceRatio[0];
    decimal cumThree = pca.CumulativeExplainedVariance[2];
    int k95 = pca.NumComponentsForExplainedVariance(0.95);
    Console.WriteLine($"  PC1 variance share = {level:P2}");
    Console.WriteLine($"  Top-3 cumulative   = {cumThree:P2}");
    Console.WriteLine($"  k for 95% variance = {k95}");

    // Project observations onto the top-k principal components.
    decimal[,] scores = pca.Project(returns, numComponents: k95);
    Console.WriteLine($"  Scores shape: {scores.GetLength(0)}×{scores.GetLength(1)}");

    // Standardized PCA (on correlation matrix).
    var std = PrincipalComponentAnalysis.FromReturns(returns, standardize: true);
    Console.WriteLine($"  Standardized PC1 share = {std.ExplainedVarianceRatio[0]:P2}");

    // From a pre-estimated covariance.
    decimal[,] shrunk = new LedoitWolfShrinkageEstimator().Estimate(returns);
    var pcaShrunk = PrincipalComponentAnalysis.Decompose(shrunk);
    Console.WriteLine($"  From shrunk-cov PC1 share = {pcaShrunk.ExplainedVarianceRatio[0]:P2}");
}

static void Example09CovarianceEstimation()
{
    Console.WriteLine("── 09  Covariance Estimation ────────────────────────");

    decimal[,] returns = SampleData.s_assetReturns;

    var sample = new SampleCovarianceEstimator().Estimate(returns);
    var ewma = new ExponentiallyWeightedCovarianceEstimator(lambda: 0.94m).Estimate(returns);
    var lwLin = new LedoitWolfShrinkageEstimator().Estimate(returns);
    var lwCC = new LedoitWolfConstantCorrelationEstimator().Estimate(returns);
    var lwFM = new LedoitWolfSingleFactorEstimator().Estimate(returns);
    var qis = new QuadraticInverseShrinkageEstimator().Estimate(returns);
    var oas = new OracleApproximatingShrinkageEstimator().Estimate(returns);
    var denoised = new DenoisedCovarianceEstimator(applyLedoitWolfShrinkage: false).Estimate(returns);
    var detoned = new DetonedCovarianceEstimator(detoningAlpha: 1m).Estimate(returns);
    var tw = new TracyWidomDenoisedCovarianceEstimator().Estimate(returns);
    var dsce = new DoublySparseEstimator(eigenvectorThreshold: 0.1m).Estimate(returns);

    static decimal Trace(decimal[,] m)
    {
        decimal s = 0m;
        for (var i = 0; i < m.GetLength(0); i++)
        {
            s += m[i, i];
        }

        return s;
    }
    Console.WriteLine($"  Trace — Sample={Trace(sample):F6}  EWMA={Trace(ewma):F6}  LW={Trace(lwLin):F6}");
    Console.WriteLine($"          CC={Trace(lwCC):F6}  FM={Trace(lwFM):F6}  QIS={Trace(qis):F6}");
    Console.WriteLine($"          OAS={Trace(oas):F6}  Denoised={Trace(denoised):F6}  Detoned={Trace(detoned):F6}");
    Console.WriteLine($"          TW={Trace(tw):F6}  DSCE={Trace(dsce):F6}");

    // Asset-major (jagged array) input via ReturnsMatrix wrapper.
    ICovarianceEstimator estimator = new SampleCovarianceEstimator();
    decimal[,] cov = estimator.Estimate(new ReturnsMatrix(SampleData.s_assetMajorReturns));
    Console.WriteLine($"  Asset-major ReturnsMatrix Trace = {Trace(cov):F6}");

    // Enforce PSD before passing to a Cholesky-based portfolio optimiser.
    decimal[,] safeCov = NearestPsdProjection.EigenClip(detoned);
    Console.WriteLine($"  EigenClip(detoned) IsPsd = {NearestPsdProjection.IsPsd(safeCov)}");
}

static void Example10Correlation()
{
    Console.WriteLine("── 10  Correlation ──────────────────────────────────");

    decimal[] x = SampleData.s_seriesA;
    decimal[] y = SampleData.s_seriesB;

    decimal pearson = PearsonCorrelation.Compute(x, y);
    decimal[] rolling = PearsonCorrelation.Rolling(x, y, windowSize: 21);
    Console.WriteLine($"  Pearson r = {pearson:F4}  (expected ~0.7 correlation)");
    Console.WriteLine($"  Rolling[last] = {rolling[^1]:F4}  (21-period window)");

    decimal spearman = RankCorrelation.Spearman(x, y);
    decimal kendall = RankCorrelation.KendallTauB(x, y);
    Console.WriteLine($"  Spearman ρ = {spearman:F4}");
    Console.WriteLine($"  Kendall τ  = {kendall:F4}");

    double[] xd = Array.ConvertAll(x, v => (double)v);
    double[] yd = Array.ConvertAll(y, v => (double)v);
    double dcor = DistanceCorrelation.Compute(xd, yd);
    Console.WriteLine($"  Distance correlation = {dcor:F4}");

    var (lo, hi) = FisherZTransform.ConfidenceInterval(r: 0.6, n: 100, confidenceLevel: 0.95);
    Console.WriteLine($"  Fisher CI (r=0.6, n=100, 95%) = [{lo:F4}, {hi:F4}]");
}

static void Example11OnlineMoments()
{
    Console.WriteLine("── 11  Online Moments ───────────────────────────────");

    var w = new WelfordMoments();
    foreach (var v in SampleData.s_returns)
    {
        w.Add(v);
    }

    decimal mean = w.Mean;
    decimal var_ = w.Variance;
    decimal sd = w.StdDev;
    Console.WriteLine($"  Streaming  mean={mean:F6}  var={var_:F8}  sd={sd:F6}");

    // Single-pass batch convenience.
    var (batchMean, batchVar) = WelfordMoments.Compute(SampleData.s_returns);
    Console.WriteLine($"  Batch      mean={batchMean:F6}  var={batchVar:F8}");

    // Bivariate (online correlation).
    var pearson = new WelfordMoments.Pearson();
    for (var i = 0; i < SampleData.s_seriesA.Length; i++)
    {
        pearson.Add(SampleData.s_seriesA[i], SampleData.s_seriesB[i]);
    }

    decimal r = pearson.Correlation;
    decimal cov = pearson.Covariance;
    Console.WriteLine($"  Bivariate  r={r:F4}  cov={cov:F8}");
}

static void Example12HigherMoments()
{
    Console.WriteLine("── 12  Higher Sample Moments ────────────────────────");

    decimal[] returns = SampleData.s_returns;
    decimal skew = SampleSkewness.Compute(returns);
    decimal exKurt = SampleExcessKurtosis.Compute(returns);
    Console.WriteLine($"  Skewness (G1)        = {skew:F4}");
    Console.WriteLine($"  Excess kurtosis (G2) = {exKurt:F4}");

    // Generic form — double-precision.
    double[] returnsD = SampleData.s_returnsDouble;
    double skewD = SampleSkewness<double>.Compute(returnsD);
    double kurtD = SampleExcessKurtosis<double>.Compute(returnsD);
    Console.WriteLine($"  Skewness<double>     = {skewD:F4}");
    Console.WriteLine($"  ExKurtosis<double>   = {kurtD:F4}");
}

static void Example13BacktestStatistics()
{
    Console.WriteLine("── 13  Backtest Statistics ──────────────────────────");

    decimal[] dailyReturns = SampleData.s_returns;

    // Deflated Sharpe Ratio.
    var dsr = DeflatedSharpeRatio.Compute(
        observedSharpe: 1.5m, numTrials: 1000, backTestYears: 5m,
        skewness: -0.5m, kurtosis: 4.0m);
    Console.WriteLine($"  DSR={dsr.DeflatedSharpe:F4}  p-value={dsr.PValue:F4}  E[maxSR]={dsr.ExpectedMaxSharpe:F4}");

    var dsrFromReturns = DeflatedSharpeRatio.ComputeFromReturns(dailyReturns, numTrials: 100);
    Console.WriteLine($"  DSR from returns = {dsrFromReturns.DeflatedSharpe:F4}");

    // Harvey-Liu-Zhu Sharpe haircut.
    var hlz = HaircutSharpe.Compute(observedSharpe: 1.5m, numTrials: 1000, backTestYears: 5m);
    Console.WriteLine($"  Haircut SR={hlz.HaircutSharpe:F4}  Bonferroni p={hlz.BonferroniPValue:F4}");

    // Minimum Track Record Length.
    double minimumObservations = MinimumTrackRecordLength.Compute(observedSharpe: 1.5);
    Console.WriteLine($"  MinTRL observations = {minimumObservations:F2}");

    // Probability of Backtest Overfitting via CSCV.
    var pbo = ProbabilityOfBacktestOverfitting.Compute(SampleData.s_strategyPanel, splitCount: 16);
    Console.WriteLine($"  PBO = {pbo.Pbo:P1}  logit median = {pbo.LogitMedian:F3}");

    // HAC variance for autocorrelated return series.
    int lags = NeweyWestVariance.AutomaticLags(dailyReturns.Length);
    decimal hacVar = NeweyWestVariance.MeanVariance(dailyReturns, lags);
    Console.WriteLine($"  Newey-West HAC var (lags={lags}) = {hacVar:F8}");

    // Generalisation score (double-precision).
    double[] dbl = SampleData.s_returnsDouble;
    var gt = GtInspiredScore.Compute(dbl);
    Console.WriteLine($"  Generalisation score = {gt.Score:F4}");
}

static void Example14MonteCarloBootstrap()
{
    Console.WriteLine("── 14  Monte Carlo Bootstrap ────────────────────────");

    decimal[] returns = SampleData.s_returns;

    // IID bootstrap with deterministic seeded PCG64.
    var iid = BootstrapMonteCarloEngine.FromSeed(simulationCount: 5000, seed: 42);
    var iidResult = iid.Run(returns, paths =>
    {
        decimal s = 0m;
        foreach (var v in paths)
        {
            s += v;
        }

        return s / paths.Length;
    });
    Console.WriteLine($"  IID mean CI = [{iidResult.Percentile5:F4}, {iidResult.Percentile95:F4}]  median={iidResult.Median:F4}");

    // Stationary bootstrap with Politis-White auto block length.
    double blockLength = PolitisWhiteBlockLength.EstimateDouble(
        Array.ConvertAll(returns, v => (double)v));
    // The selector may return below one; the resampler needs a mean of at least one.
    double meanBlockLength = Math.Max(1.0, blockLength);
    var stationary = StationaryBootstrapResampler.FromSeed(meanBlockLength, seed: 42);
    decimal[] resampled = stationary.Resample(returns);
    Console.WriteLine($"  PW block length = {blockLength:F2}  resampler mean = {meanBlockLength:F2}  resample[0]={resampled[0]:F6}");

    // Multi-statistic bootstrap.
    var multi = iid.RunMulti(
        returns,
        statisticCount: 3,
        statisticWriter: (path, output) =>
        {
            output[0] = ComputeSharpe(path);
            output[1] = ComputeSortino(path);
            output[2] = ComputeMaxDrawdown(path);
        },
        names: ["sharpe", "sortino", "maxdd"]);
    Console.WriteLine($"  Multi Sharpe   median={multi.Medians[0]:F4}  CI=[{multi.Percentile5s[0]:F4}, {multi.Percentile95s[0]:F4}]");
    Console.WriteLine($"  Multi MaxDD    median={multi.Medians[2]:F4}");

    // Wild bootstrap for heteroskedastic residuals.
    var wild = WildBootstrapResampler.FromSeed(WildBootstrapWeights.Rademacher, seed: 42);
    decimal[] wildResampled = wild.Resample(SampleData.s_residuals);
    Console.WriteLine($"  Wild resample[0] = {wildResampled[0]:F6}");

    // Subsampling for non-stationary or extreme-value statistics.
    int b = Subsampler.SuggestSubsampleLength(returns.Length);
    decimal[] subsamples = Subsampler.Run(returns, b, span =>
    {
        decimal sum = 0m;
        for (var i = 0; i < span.Length; i++)
        {
            sum += span[i];
        }

        return sum / span.Length;
    });
    Console.WriteLine($"  Subsampler b={b}  subsamples count={subsamples.Length}");

    // Right-tailed test of H0: mean = 0. The mean is not location invariant,
    // so impose the null on both the outer and inner bootstrap populations.
    var fdb = FastDoubleBootstrap.FromSeed(outerCount: 500, seed: 42);
    static decimal Mean(ReadOnlySpan<decimal> paths)
    {
        decimal s = 0m;
        foreach (var v in paths)
        {
            s += v;
        }

        return s / paths.Length;
    }
    static void ImposeZeroMean(Span<decimal> paths)
    {
        var mean = Mean(paths);
        for (var i = 0; i < paths.Length; i++)
        {
            paths[i] -= mean;
        }
    }
    decimal pValue = fdb.PValueImposingNull(returns, observed: Mean(returns), Mean, ImposeZeroMean);
    Console.WriteLine($"  FastDoubleBootstrap p-value = {pValue:F4}");

    // Harrell-Davis percentile — variance benefits depend on the distribution.
    decimal[] sorted = iidResult.Statistics.ToArray();   // IReadOnlyList<decimal> → decimal[]
    Array.Sort(sorted);
    decimal hd05 = HarrellDavisPercentile.Compute(sorted, 0.05);
    Console.WriteLine($"  Harrell-Davis 5th pct = {hd05:F6}");
}

static void Example15QuasiMonteCarlo()
{
    Console.WriteLine("── 15  Quasi-Monte Carlo ────────────────────────────");

    // Halton — good for d ≲ 8.
    var halton = new HaltonSequence(dimension: 4, skip: 64);
    double[] h0 = halton.Next();
    double[] h1 = halton.Next();
    Console.WriteLine($"  Halton[0] = [{string.Join(", ", h0.Select(v => v.ToString("F4", CultureInfo.InvariantCulture)))}]");
    Console.WriteLine($"  Halton[1] = [{string.Join(", ", h1.Select(v => v.ToString("F4", CultureInfo.InvariantCulture)))}]");

    // Sobol — better for d > 8 (up to 32 supported).
    var sobol = new SobolSequence(dimension: 16, skip: 1024);
    double[] s0 = sobol.Next();
    Console.WriteLine($"  Sobol[0] dim0={s0[0]:F4}  dim15={s0[15]:F4}");

    // Confirm quasi-random sequence fills [0,1) uniformly over 1,000 draws.
    var halton2 = new HaltonSequence(dimension: 1, skip: 0);
    double sum = 0;
    for (var i = 0; i < 1000; i++)
    {
        sum += halton2.Next()[0];
    }

    Console.WriteLine($"  Halton d=1 mean over 1000 draws ≈ {sum / 1000:F4}  (expected ≈ 0.5)");
}

static void Example16RandomNumbers()
{
    Console.WriteLine("── 16  Random Number Generation ─────────────────────");

    // PCG-64 — independent streams via streamId.
    var rng = new Pcg64RandomSource(seed: 42UL, streamId: 0UL);
    ulong u = rng.NextULong();
    double d = ((IRandomSource)rng).NextDouble();
    int i = ((IRandomSource)rng).NextInt(upperExclusive: 100);
    Console.WriteLine($"  PCG-64 NextULong={u}  NextDouble={d:F6}  NextInt(100)={i}");

    // xoshiro256** — slightly faster, with Jump for parallel partitioning.
    var xrng = new Xoshiro256StarStarRandomSource(seed: 42UL);
    xrng.Jump();   // advance by 2^128 calls — independent stream for worker thread
    Console.WriteLine($"  Xoshiro after Jump: {xrng.NextULong()}");

    // Marsaglia polar Gaussian sampler.
    var gaussian = new GaussianSampler(rng);
    double normal = gaussian.Next();
    double scaled = gaussian.Next(mean: 0.05, stdDev: 0.20);
    double[] batch = gaussian.NextBatch(10_000);
    double batchMean = batch.Average();
    Console.WriteLine($"  GaussianSampler N(0,1)≈{normal:F4}  N(0.05,0.20)≈{scaled:F4}");
    Console.WriteLine($"  Batch 10k mean ≈ {batchMean:F4}  (expected ≈ 0.0)");
}

static void Example17RollingWindow()
{
    Console.WriteLine("── 17  Rolling Window ───────────────────────────────");

    var window = new RollingWindow<double>(capacity: 21);
    double[] stream = SampleData.s_returnsDouble;

    foreach (var observation in stream)
    {
        window.Add(observation);
        if (window.IsFull)
        {
            double latest = window[window.Count - 1];
            double rollingMean = window.ToArray().Average();
            _ = (latest, rollingMean);   // consume to avoid unused warnings
        }
    }
    Console.WriteLine($"  Window (cap=21) last 21 returns mean = {window.ToArray().Average():F6}");
    Console.WriteLine($"  Count={window.Count}  IsFull={window.IsFull}  Capacity={window.Capacity}");

    // Unconstrained T — not limited to numeric payloads.
    var eventLog = new RollingWindow<(DateTime When, string What)>(capacity: 100);
    eventLog.Add((DateTime.UtcNow, "trade executed"));
    Console.WriteLine($"  EventLog[0].What = \"{eventLog[0].What}\"");
}

static void Example18GenericMath()
{
    Console.WriteLine("── 18  Generic-Math Precision ───────────────────────");

    // Tier A + √ factorisation at T = double — curve-bootstrap pipeline.
    double[,] spdD =
    {
        {  4.0, 12.0, -16.0 },
        { 12.0, 37.0, -43.0 },
        { -16.0, -43.0, 98.0 },
    };
    double[,] Ld = CholeskyDecomposition<double>.Decompose(spdD);
    double[] xd = CholeskyDecomposition<double>.Solve(Ld, [1.0, 2.0, 3.0]);
    Console.WriteLine($"  Cholesky<double> solution: [{string.Join(", ", xd.Select(v => v.ToString("F6", CultureInfo.InvariantCulture)))}]");

    // Tier A + √ factorisation at T = decimal — NIST-bar precision.
    // y = 1 + 2x + 3x² + 4x³ + 5x⁴ + 6x⁵, x = 0..20.  cond(X) ≈ 5·10¹³.
    var fit = OrdinaryLeastSquares<decimal>.Fit(SampleData.s_wampler5X, SampleData.s_wampler5Y);
    decimal intercept = fit.Coefficients[0];
    decimal c1 = fit.Coefficients[1];
    decimal c5 = fit.Coefficients[5];
    Console.WriteLine($"  OLS<decimal> Wampler5: intercept={intercept:F9} (cert=1)");
    Console.WriteLine($"    x coeff={c1:F9} (cert=2)");
    Console.WriteLine($"    x⁵ coeff={c5:F9} (cert=6)");
    Console.WriteLine($"    R²={fit.RSquared:F9}  σ̂={fit.ResidualStandardDeviation:F9}");
}

static void Example19OptimizationAndGarch()
{
    Console.WriteLine("── 19  Derivative-Free Optimization and GARCH ───────");

    SimplexResult<double> minimum = NelderMeadMinimizer<double>.Minimize(
        point =>
        {
            double x = point[0] - 2.0;
            double y = point[1] + 1.0;
            return x * x + y * y;
        },
        initialGuess: [0.0, 0.0]);
    Console.WriteLine($"  Nelder-Mead argmin=[{minimum.Argmin[0]:F4}, {minimum.Argmin[1]:F4}]  f={minimum.MinValue:E2}");

    GarchFitResult<double> garch = GarchOneOneEstimator<double>.Fit(SampleData.s_returnsDouble);
    Console.WriteLine($"  GARCH(1,1) alpha={garch.Alpha:F4}  beta={garch.Beta:F4}  h(1)={garch.ForecastVariance(1):E4}");
}

static void Example20SparsePrecisionAndTailDependence()
{
    Console.WriteLine("── 20  Sparse Precision and Tail Dependence ─────────");

    double[,] sampleCovariance =
    {
        { 1.00, 0.35, 0.10 },
        { 0.35, 1.00, 0.25 },
        { 0.10, 0.25, 1.00 },
    };
    GraphicalLassoResult<double> sparse = GraphicalLassoSolver<double>.Solve(sampleCovariance, lambda: 0.10);
    Console.WriteLine($"  Graphical lasso precision[0,1]={sparse.Precision[0, 1]:F4}  converged={sparse.Converged}");

    double[] first = Array.ConvertAll(SampleData.s_seriesA, value => (double)value);
    double[] second = Array.ConvertAll(SampleData.s_seriesB, value => (double)value);
    TailResult<double> tail = LowerTailDependence<double>.Estimate(
        first,
        second,
        new TailOptions<double>(Quantiles: [0.05, 0.10]));
    Console.WriteLine($"  Lower-tail λL(5%)={tail.Lambda:F4}  joint count={tail.JointCount[0]}");
}

// ── Helper statistics ─────────────────────────────────────────────────────────

static decimal ComputeSharpe(ReadOnlySpan<decimal> path)
{
    if (path.Length < 2)
    {
        return 0m;
    }

    decimal sum = 0m;
    for (var i = 0; i < path.Length; i++)
    {
        sum += path[i];
    }

    decimal mean = sum / path.Length;
    decimal sumSq = 0m;
    for (var i = 0; i < path.Length; i++)
    {
        sumSq += (path[i] - mean) * (path[i] - mean);
    }

    decimal stdDev = (decimal)Math.Sqrt((double)(sumSq / (path.Length - 1)));
    return stdDev == 0m ? 0m : mean * (decimal)Math.Sqrt(252.0) / stdDev;
}

static decimal ComputeSortino(ReadOnlySpan<decimal> path)
{
    if (path.Length < 2)
    {
        return 0m;
    }

    decimal sum = 0m;
    for (var i = 0; i < path.Length; i++)
    {
        sum += path[i];
    }

    decimal mean = sum / path.Length;
    decimal downSq = 0m;
    for (var i = 0; i < path.Length; i++)
    {
        if (path[i] < 0m)
        {
            downSq += path[i] * path[i];
        }
    }

    decimal downStd = (decimal)Math.Sqrt((double)(downSq / path.Length));
    return downStd == 0m ? 0m : mean * (decimal)Math.Sqrt(252.0) / downStd;
}

static decimal ComputeMaxDrawdown(ReadOnlySpan<decimal> path)
{
    decimal nav = 1m, peak = 1m, maxDd = 0m;
    foreach (var r in path)
    {
        nav *= 1m + r;
        if (nav > peak)
        {
            peak = nav;
        }

        decimal dd = peak == 0m ? 0m : (peak - nav) / peak;
        if (dd > maxDd)
        {
            maxDd = dd;
        }
    }
    return maxDd;
}

// ── Types ─────────────────────────────────────────────────────────────────────

sealed record ExampleEntry(string Id, string Description, Func<Task> Run);
