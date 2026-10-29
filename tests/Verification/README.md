# Verification artifact regeneration

Install `requirements.txt`, then run from the repository root:

```bash
python3 tests/Verification/generate_vectors.py
dotnet test tests/Boutquin.Numerics.Tests.Verification/Boutquin.Numerics.Tests.Verification.csproj -c Release
python3 -m pytest tests/Verification/test_regeneration_tools.py tests/Verification/test_c1_persistence_and_oracles.py tests/Verification/test_dependency_update_closure.py -q
```

The command regenerates all 27 computed JSON fixtures in `vectors/`, including
Jarre spline, PPW 2009, RMT, independent GARCH and converged single-factor spike
controls. The independent GARCH oracle requires the pinned `arch==8.0.0` release.
Graphical Lasso generation includes its all-entry objective provenance and BIC
fields before publishing the file. Covariance generation also refreshes the
selected OAS/NERCOME contract fields.

`jarre_author_20251029.json` is the retained upstream MATLAB/Octave replay with
its native-output hash; the Python command preserves that snapshot. The P² and
release-covariance scripts that print literal reference values do not write JSON.

QP corpus files live outside `vectors/` and have a separate exact-rational entry
point, `python3 tests/Verification/qp-corpus/generate_corpus.py [OUTDIR]`.
Each artifact is written to a sibling temporary file, flushed and synced, then
replaced atomically and its directory synced. Serialization, file-sync and replace
failures preserve the previous destination and remove the temporary file. This
is a per-file guarantee; a batch can publish earlier complete files before a later
generator fails. A failure syncing the directory after replacement is reported,
but cannot restore the prior destination automatically.

Regeneration may change last-place floating-point digits across numerical-library
or BLAS versions. Qualify the regenerated outputs with their C# consumers before
accepting new fixture bytes.

The dependency-update closure test requires the local .NET SDK and cached
`xunit.runner.visualstudio` 4.0.0/4.0.1 packages. It replays a test-package bump in
an isolated manifest-only copy, requires the stale StressRunner lock to fail with
NU1004, refreshes that direct consumer, and requires whole-solution locked restore
to pass. StressRunner declares the same dependency roots as the unit tests so
dependency updaters discover its lock too; test SDK build assets remain excluded
and its console entry point stays in control.
