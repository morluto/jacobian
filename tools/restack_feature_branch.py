"""Rebuild a leaf feature branch from its parent plus unique commits.

This helper is advisory. It never force-pushes. Duplicate patches that already
exist on the declared parent are reported rather than dropped automatically.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(args: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit_subjects(cwd: Path, revision_range: str) -> list[tuple[str, str]]:
    output = _run(
        ["git", "log", "--reverse", "--format=%H%x09%s", revision_range],
        cwd=cwd,
    )
    if not output:
        return []
    rows: list[tuple[str, str]] = []
    for line in output.splitlines():
        sha, _, subject = line.partition("\t")
        rows.append((sha, subject))
    return rows


def restack(
    *,
    cwd: Path,
    parent: str,
    feature: str,
) -> int:
    parent_sha = _run(["git", "rev-parse", parent], cwd=cwd)
    feature_sha = _run(["git", "rev-parse", feature], cwd=cwd)
    unique = _commit_subjects(cwd, f"{parent_sha}..{feature_sha}")
    parent_subjects = {subject for _, subject in _commit_subjects(cwd, parent_sha)}
    duplicates = [
        (sha, subject) for sha, subject in unique if subject in parent_subjects
    ]
    print(f"parent: {parent} ({parent_sha[:12]})")
    print(f"feature: {feature} ({feature_sha[:12]})")
    print(f"unique commits: {len(unique)}")
    for sha, subject in unique:
        print(f"  {sha[:12]} {subject}")
    if duplicates:
        print("duplicate-subject patches already on parent:")
        for sha, subject in duplicates:
            print(f"  {sha[:12]} {subject}")
    else:
        print("duplicate-subject patches already on parent: none")
    print(
        "restack is advisory: rebuild the leaf from the parent plus unique "
        "commits yourself; this helper does not rewrite published branches"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="main")
    parser.add_argument("--feature", default="HEAD")
    args = parser.parse_args(argv)
    return restack(cwd=Path.cwd(), parent=args.parent, feature=args.feature)


if __name__ == "__main__":
    raise SystemExit(main())
