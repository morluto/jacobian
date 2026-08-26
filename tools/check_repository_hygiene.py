"""Reject tracked local artifacts and unresolved conflict markers."""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.command_runner import ToolCommandStatus, run_operator_command

_FIXTURE_PREFIXES = (
    PurePosixPath("tests/fixtures"),
    PurePosixPath("vendor"),
    PurePosixPath("vendored"),
)


def tracked_paths(root: Path) -> tuple[PurePosixPath, ...]:
    """Return tracked paths through the bounded repository-command policy."""

    result = run_operator_command(
        "git",
        ("ls-files", "-z"),
        cwd=root,
        timeout_seconds=30.0,
        stdout_limit_bytes=16 * 1024 * 1024,
        stderr_limit_bytes=64 * 1024,
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        raise RuntimeError(
            result.diagnostic or "git ls-files failed during repository hygiene check"
        )
    return tuple(PurePosixPath(p.decode()) for p in result.stdout.split(b"\0") if p)


def _is_local_artifact(path: PurePosixPath) -> bool:
    """Return whether a tracked path matches a documented local-work pattern."""
    return (
        any(part.startswith(".agents-tmp-") for part in path.parts)
        or "pr-audit" in path.parts
        or (path.parent == PurePosixPath(".") and path.match("*audit_report*.md"))
    )


def _allows_literal_conflict_markers(path: PurePosixPath) -> bool:
    """Keep vendored and deliberately adversarial fixture text out of the scan."""
    return any(path.is_relative_to(prefix) for prefix in _FIXTURE_PREFIXES)


def _is_conflict_marker(line: str) -> bool:
    """Recognize the three line forms emitted by Git's conflict renderer."""
    return line.startswith(("<<<<<<< ", ">>>>>>> ")) or line == "======="


def check(root: Path = ROOT) -> tuple[str, ...]:
    violations: list[str] = []
    for path in tracked_paths(root):
        name = path.as_posix()
        if _is_local_artifact(path):
            violations.append(f"{name}: forbidden local artifact")
            continue
        if _allows_literal_conflict_markers(path):
            continue
        try:
            lines = (root / path).read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        violations.extend(
            f"{name}:{index}: unresolved conflict marker"
            for index, line in enumerate(lines, 1)
            if _is_conflict_marker(line)
        )
    return tuple(violations)


if __name__ == "__main__":
    failures = check()
    print("\n".join(failures))
    raise SystemExit(bool(failures))
