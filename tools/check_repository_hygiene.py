"""Reject tracked local artifacts and unresolved conflict markers."""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (".agents-tmp-", "pr-audit/")
MARKERS = ("<<<<<<< ", ">>>>>>> ", "======= ")


def tracked_paths(root: Path) -> tuple[PurePosixPath, ...]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True
    )
    return tuple(PurePosixPath(p.decode()) for p in result.stdout.split(b"\0") if p)


def check(root: Path = ROOT) -> tuple[str, ...]:
    violations: list[str] = []
    for path in tracked_paths(root):
        name = path.as_posix()
        forbidden_root_audit = (
            path.parent == PurePosixPath(".")
            and path.suffix == ".md"
            and "audit_report" in path.stem
        )
        if any(pattern in name for pattern in FORBIDDEN) or forbidden_root_audit:
            violations.append(f"{name}: forbidden local artifact")
            continue
        try:
            lines = (root / path).read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        violations.extend(
            f"{name}:{index}: unresolved conflict marker"
            for index, line in enumerate(lines, 1)
            if line.startswith(MARKERS)
        )
    return tuple(violations)


if __name__ == "__main__":
    failures = check()
    print("\n".join(failures))
    raise SystemExit(bool(failures))
