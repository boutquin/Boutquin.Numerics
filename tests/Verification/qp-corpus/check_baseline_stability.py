#!/usr/bin/env python3
"""Independent anchor for the pre-rewrite baseline table of the ActiveSetQpSolver corpus.

The corpus test re-records the qualification-dependent columns of baseline.json (qualified, qualified_refused,
qualified_wrong) and the rows of families added after M-1. The per-family ok / refused / wrong counts of the M-1
cases describe the pre-rewrite solver and must never change. This script compares them, independently of the
table the corpus test asserts, against the baseline.json held in git history:

  * the baseline.json stored at commit 988d658b (ANCHOR below), read with `git show`: the commit that first added
    the table. baseline-commit.txt names the earlier commit whose solver the table was recorded against (the
    reference of the test-file and PublicAPI byte-identity checks); it predates the table, so the anchor must
    descend from it, which is checked;
  * every row of that table (family, type) must still exist in the working-tree baseline.json with the same
    cases / ok / refused / wrong counts, and the per-type timing ratios recorded on the M-1 case set must be
    unchanged.

Exit 0 when every M-1 row and the timing ratios match, 1 on any difference (each one printed), 2 when the anchor
cannot be read.

usage: python3 tests/Verification/qp-corpus/check_baseline_stability.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS = ("cases", "ok", "refused", "wrong")

# The commit that first added baseline.json; its table is the anchor of the M-1 rows.
ANCHOR = "988d658bfabf1550f12e6686054a0ba180c46496"


def toplevel():
    top = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
    return top.stdout.strip() if top.returncode == 0 else None


TOP = toplevel()


def git(*args):
    # Run at the repository root, so that pathspecs and <rev>:<path> both resolve repository-relative.
    return subprocess.run(["git", "-C", TOP or HERE, *args], capture_output=True, text=True, check=False)


def repo_path(name):
    if TOP is None:
        return None
    return os.path.relpath(os.path.join(HERE, name), TOP).replace(os.sep, "/")


def anchor_table(sha, rel):
    if git("merge-base", "--is-ancestor", sha, ANCHOR).returncode != 0:
        return None, None
    shown = git("show", f"{ANCHOR}:{rel}")
    return (ANCHOR, shown.stdout) if shown.returncode == 0 else (None, None)


def main():
    try:
        with open(os.path.join(HERE, "baseline-commit.txt"), encoding="utf-8") as fh:
            sha = fh.read().strip()
        with open(os.path.join(HERE, "baseline.json"), encoding="utf-8") as fh:
            current = json.load(fh)
    except (OSError, ValueError) as exc:
        print("cannot read the working-tree baseline: %s" % exc)
        return 2

    rel = repo_path("baseline.json")
    anchor, text = (None, None) if rel is None else anchor_table(sha, rel)
    if anchor is None:
        print("cannot read baseline.json at %s, or it does not descend from the baseline commit %s" % (ANCHOR, sha))
        return 2
    anchored = json.loads(text)

    rows = {(f["family"], f["type"]): f for f in current.get("families", [])}
    differences = []
    for row in anchored["families"]:
        key = (row["family"], row["type"])
        now = rows.get(key)
        if now is None:
            differences.append("%s|%s: M-1 row missing from the working-tree baseline" % key)
            continue
        for col in COLUMNS:
            if row[col] != now.get(col):
                differences.append("%s|%s %s: anchored %s, working tree %s" % (key[0], key[1], col, row[col], now.get(col)))

    if anchored.get("timing") != current.get("timing"):
        differences.append("timing: the per-type ratios recorded on the M-1 case set changed")

    for line in differences:
        print(line)
    print("anchor %s (baseline-commit %s): %d M-1 rows checked, %d differences"
          % (anchor, sha, len(anchored["families"]), len(differences)))
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
