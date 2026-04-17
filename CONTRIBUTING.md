# Contributing to Boutquin.Numerics

Thank you for considering contributing to Boutquin.Numerics! Whether it's reporting a bug, proposing a feature, or submitting a pull request, your input is welcome.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Contributing Code](#contributing-code)
- [Style Guides](#style-guides)
  - [Git Commit Messages](#git-commit-messages)
  - [C# Style Guide](#c-style-guide)
  - [Documentation Style Guide](#documentation-style-guide)
  - [Numerical Correctness](#numerical-correctness)
- [Pull Request Process](#pull-request-process)
- [License](#license)
- [Community](#community)

## Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Report unacceptable behavior through [GitHub Issues](https://github.com/boutquin/Boutquin.Numerics/issues).

## How to Contribute

### Reporting Bugs

Open an issue on the [Issues](https://github.com/boutquin/Boutquin.Numerics/issues) page with:

- A clear and descriptive title.
- Steps to reproduce the issue (ideally a minimal failing code snippet).
- Expected and actual numerical output, including tolerances where relevant.
- Reference values (from numpy / scipy / statsmodels / R / a textbook) when asserting correctness.
- Environment: OS, .NET runtime version, package version.

### Suggesting Enhancements

Open an issue describing:

- The numerical method, estimator, or algorithm you would like to see added.
- A primary reference (paper, textbook, or RFC) with the algorithm's published form.
- The intended consumer (solver caller, covariance estimator user, Monte Carlo driver, etc.).
- Any trade-offs: precision regime (`double` vs `decimal`), memory profile, parallelism.

### Contributing Code

1. **Fork the repository** and clone your fork locally.
   ```bash
   git clone https://github.com/your-username/Boutquin.Numerics.git
   cd Boutquin.Numerics
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature-or-bugfix-name
   ```

3. **Implement the change** following the style guides below.

4. **Add tests** covering the new behavior. For algorithmic code, prefer:
   - xUnit + FluentAssertions unit tests under `tests/Boutquin.Numerics.Tests.Unit/` with explicit tolerances.
   - Cross-language reference vectors under `tests/Boutquin.Numerics.Tests.Verification/` when a Python reference (numpy / scipy / statsmodels / scikit-learn) can establish ground truth.
   - Architecture assertions under `tests/Boutquin.Numerics.ArchitectureTests/` when the change introduces new namespaces, public types, or external dependencies.

5. **Record the public API surface** in `src/Boutquin.Numerics/PublicAPI.Unshipped.txt`. The `PublicAPI` analyzer enforces this at build time.

6. **Update `CHANGELOG.md`** under the `[Unreleased]` section.

7. **Run the full gate** before opening a PR:
   ```bash
   dotnet build Boutquin.Numerics.slnx --configuration Release
   dotnet test Boutquin.Numerics.slnx --configuration Release
   dotnet format Boutquin.Numerics.slnx --verify-no-changes
   ```

8. **Push and open a pull request**.

## Style Guides

### Git Commit Messages

- Use the present tense ("Add Brent solver" not "Added Brent solver").
- Use the imperative mood ("Move bracketing check to helper" not "Moves bracketing check to helper").
- Limit the first line to 72 characters.
- Reference issues and pull requests where applicable.

### C# Style Guide

- Follow the conventions documented in `CLAUDE.md` and `.editorconfig` at the repository root.
- Public types are `sealed` unless they are interfaces or records (enforced by architecture tests).
- Numerics has **zero** `Boutquin.*` dependencies — this is enforced by `Boutquin.Numerics.ArchitectureTests`. Do not add a reference to MarketData, Analytics, Trading, OptionPricing, or any other Boutquin package.
- Litmus test for any new type: "Would this code make sense in a non-financial application?" If not, it does not belong in Numerics.
- Pick the right **tier** for a new algorithm: Tier A (arithmetic-only), Tier A + √ (plus `Sqrt` via `NumericPrecision<T>`), Tier B (transcendentals — IEEE-only), Tier C (polynomial-approximation-bound). The tier drives the generic constraint on `T`. See [`docs/tier-constraints.md`](docs/tier-constraints.md) for tier definitions, per-folder assignments, the scalar-cast sub-rule for `decimal → double`, and the forbidden patterns (no `decimal → double → decimal` inside an inner loop; no `decimal` facade on a Tier B algorithm; no cast-through overloads).

### Documentation Style Guide

All public API additions must satisfy the in-code documentation bar:

- `<summary>` on every public type, constructor, method, property, and enum member.
- `<param>`, `<returns>`, and `<remarks>` per the required-elements checklist.
- No banned boilerplate phrases ("Provides the ... functionality", "Executes the ... operation", "Gets or sets the ... for this instance", "Input value for ...", "/// Executes ...", "Operation result.", "/// Gets the ...").
- Algorithmic references: name the paper, author, year, and (where applicable) arXiv number.
- Every generic public type carries its **tier label** as the first `<para>` of its `<remarks>` block — e.g. `<b>Generic-math tier:</b> A+√ — accepts double, float, Half, decimal via NumericPrecision<T>.Sqrt.` See [`docs/tier-constraints.md`](docs/tier-constraints.md).
- Elevated standards apply to root solvers, covariance estimators, linear algebra factorizations, RNGs, and bootstrap engines — each must name the algorithm, the reference, the convergence or PSD contract, and the exit-status semantics.

Validation commands:

```bash
# Enforce banned-phrase policy (must return zero matches).
rg -n "Provides the .* functionality and related domain behavior|Executes the .* operation for this component|The .* input value for the operation|Gets or sets the .* for this instance" src --glob '*.cs'

# Enforce low-signal phrase policy.
rg -n "Input value for <paramref name=|/// Executes |Operation result\." src --glob '*.cs'

# Enforce accessor-verb property doc policy.
rg -n "/// Gets the " src --glob '*.cs'
```

### Numerical Correctness

- State tolerances explicitly in docs and tests. Do not assert equality on floating-point values.
- Document convergence claims (linear, superlinear, quadratic) and any worst-case bounds.
- Document PSD contracts per-estimator — `ICovarianceEstimator.Estimate` does not promise a PSD result; callers wrap with `NearestPsdProjection.EigenClip` when hard guarantees are needed.
- Seeded deterministic output from any RNG or Monte Carlo engine must be bit-for-bit stable across runtimes.

## Pull Request Process

1. **Ensure the full gate passes**: build (warnings-as-errors), unit tests, verification tests, architecture tests, `dotnet format --verify-no-changes`.
2. **Describe your changes** in the PR body: reference the issue, summarize the algorithm or fix, link the primary reference, and note any `PublicAPI.Unshipped.txt` entries added.
3. **Review process**: maintainers will review for correctness, style, and architectural fit. You may be asked to tighten tolerances, add reference vectors, or rename symbols.
4. **Merge**: once approved, a maintainer merges the PR. Releases are cut separately via the dual-repo squash workflow on the public repository.

## License

By contributing to Boutquin.Numerics, you agree that your contributions are licensed under the Apache 2.0 License.

## Community

Join the [GitHub Discussions](https://github.com/boutquin/Boutquin.Numerics/discussions) to ask questions, propose algorithms, and share usage patterns.

---

Thank you for contributing!
