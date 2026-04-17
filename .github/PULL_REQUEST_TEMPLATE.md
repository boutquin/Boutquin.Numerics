## Summary

Brief description of what this PR does.

## Changes

- ...

## Related Issues

Closes #

## Checklist

- [ ] Code compiles with zero warnings (TreatWarningsAsErrors enabled)
- [ ] All existing unit, verification, and architecture tests pass
- [ ] New tests added for new functionality (unit + reference vectors where applicable)
- [ ] `PublicAPI.Unshipped.txt` updated for any new or changed public API
- [ ] `dotnet format --verify-no-changes` produces no changes
- [ ] XML doc comments are complete (no banned phrases; algorithmic reference by author/year/arXiv; tier label in `<remarks>` for new generic types — see [CONTRIBUTING.md](../CONTRIBUTING.md#documentation-style-guide) and [docs/tier-constraints.md](../docs/tier-constraints.md))
- [ ] CHANGELOG.md updated under `[Unreleased]` (if user-facing change)
- [ ] No new `Boutquin.*` dependencies introduced (architecture tests enforce zero domain coupling)
