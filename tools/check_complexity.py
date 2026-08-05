"""Ratchet Ruff C901 findings against the checked-in complexity baseline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tooling.command_runner import (  # noqa: E402
    ToolCommandStatus,
    run_operator_command,
)

DEFAULT_BASELINE = ROOT / "tools" / "c901-baseline.json"
DEFAULT_PATHS = ("src", "tests", "benchmarks", "tools", ".github/scripts")
_MESSAGE = re.compile(r"`(?P<symbol>.+)` is too complex \((?P<complexity>\d+) > \d+\)")


class ComplexityBaselineError(ValueError):
    """Raised when the baseline or Ruff output is malformed."""


@dataclass(frozen=True, order=True)
class ComplexityViolation:
    """One path-and-symbol C901 observation."""

    path: str
    symbol: str
    complexity: int

    @property
    def key(self) -> tuple[str, str]:
        return self.path, self.symbol


@dataclass(frozen=True)
class ComplexityBaseline:
    """Validated snapshot of known C901 observations."""

    max_complexity: int
    violations: tuple[ComplexityViolation, ...]


def serialize_baseline(baseline: ComplexityBaseline) -> str:
    """Serialize a baseline in the repository's canonical order and format."""

    payload = {
        "version": 1,
        "max_complexity": baseline.max_complexity,
        "violations": [
            {
                "path": violation.path,
                "symbol": violation.symbol,
                "complexity": violation.complexity,
            }
            for violation in sorted(baseline.violations)
        ],
    }
    return f"{json.dumps(payload, indent=2)}\n"


def write_baseline(path: Path, baseline: ComplexityBaseline) -> None:
    """Write a canonical complexity baseline."""

    path.write_text(serialize_baseline(baseline), encoding="utf-8")


def _violation(raw: Any, *, max_complexity: int) -> ComplexityViolation:
    if not isinstance(raw, dict) or set(raw) != {"path", "symbol", "complexity"}:
        raise ComplexityBaselineError(
            "each baseline violation must contain path, symbol, and complexity"
        )
    path = raw["path"]
    symbol = raw["symbol"]
    complexity = raw["complexity"]
    if (
        not isinstance(path, str)
        or not path
        or PurePosixPath(path).is_absolute()
        or ".." in PurePosixPath(path).parts
    ):
        raise ComplexityBaselineError("baseline paths must be relative POSIX paths")
    if not isinstance(symbol, str) or not symbol:
        raise ComplexityBaselineError("baseline symbols must be non-empty strings")
    if (
        not isinstance(complexity, int)
        or isinstance(complexity, bool)
        or complexity <= max_complexity
    ):
        raise ComplexityBaselineError(
            "baseline complexity must be an integer above max_complexity"
        )
    return ComplexityViolation(path, symbol, complexity)


def load_baseline(path: Path) -> ComplexityBaseline:
    """Parse and validate a complexity baseline."""

    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComplexityBaselineError(
            f"cannot read complexity baseline: {exc}"
        ) from exc
    if not isinstance(raw, dict) or set(raw) != {
        "version",
        "max_complexity",
        "violations",
    }:
        raise ComplexityBaselineError(
            "baseline must contain version, max_complexity, and violations"
        )
    if raw["version"] != 1:
        raise ComplexityBaselineError("complexity baseline version must be 1")
    max_complexity = raw["max_complexity"]
    if (
        not isinstance(max_complexity, int)
        or isinstance(max_complexity, bool)
        or max_complexity < 1
    ):
        raise ComplexityBaselineError("max_complexity must be a positive integer")
    raw_violations = raw["violations"]
    if not isinstance(raw_violations, list):
        raise ComplexityBaselineError("violations must be an array")
    violations = tuple(
        sorted(
            _violation(item, max_complexity=max_complexity) for item in raw_violations
        )
    )
    keys = [item.key for item in violations]
    if len(keys) != len(set(keys)):
        raise ComplexityBaselineError("baseline path and symbol keys must be unique")
    return ComplexityBaseline(max_complexity, violations)


def parse_ruff_output(
    payload: str,
    *,
    root: Path,
    max_complexity: int,
) -> tuple[ComplexityViolation, ...]:
    """Parse Ruff's JSON C901 output into stable repository-relative records."""

    try:
        raw: Any = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ComplexityBaselineError(f"Ruff returned invalid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise ComplexityBaselineError("Ruff C901 output must be an array")
    observations: list[ComplexityViolation] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ComplexityBaselineError("Ruff C901 entries must be objects")
        filename = item.get("filename")
        message = item.get("message")
        if (
            item.get("code") != "C901"
            or not isinstance(filename, str)
            or not isinstance(message, str)
        ):
            raise ComplexityBaselineError("Ruff returned a malformed C901 entry")
        match = _MESSAGE.fullmatch(message)
        if match is None:
            raise ComplexityBaselineError(f"unrecognized Ruff C901 message: {message}")
        try:
            path = Path(filename).resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ComplexityBaselineError(
                f"Ruff reported a path outside the repository: {filename}"
            ) from exc
        complexity = int(match.group("complexity"))
        if complexity <= max_complexity:
            raise ComplexityBaselineError(
                f"Ruff reported {path}:{match.group('symbol')} below the configured limit"
            )
        observations.append(
            ComplexityViolation(path, match.group("symbol"), complexity)
        )
    observations.sort()
    keys = [item.key for item in observations]
    if len(keys) != len(set(keys)):
        raise ComplexityBaselineError("Ruff returned duplicate path and symbol keys")
    return tuple(observations)


def compare_violations(
    baseline: ComplexityBaseline,
    current: tuple[ComplexityViolation, ...],
) -> tuple[str, ...]:
    """Return actionable differences between the baseline and current findings."""

    expected = {item.key: item for item in baseline.violations}
    observed = {item.key: item for item in current}
    problems: list[str] = []
    for key in sorted(observed.keys() - expected.keys()):
        item = observed[key]
        problems.append(
            f"new violation: {item.path}:{item.symbol} has complexity {item.complexity}"
        )
    for key in sorted(observed.keys() & expected.keys()):
        before = expected[key]
        after = observed[key]
        if after.complexity > before.complexity:
            problems.append(
                f"complexity increased: {after.path}:{after.symbol} "
                f"{before.complexity} -> {after.complexity}"
            )
        elif after.complexity < before.complexity:
            problems.append(
                f"baseline is stale after improvement: {after.path}:{after.symbol} "
                f"{before.complexity} -> {after.complexity}"
            )
    for key in sorted(expected.keys() - observed.keys()):
        item = expected[key]
        problems.append(
            f"remove resolved violation from baseline: {item.path}:{item.symbol}"
        )
    return tuple(problems)


def run_ruff(
    paths: tuple[str, ...],
    *,
    max_complexity: int,
) -> tuple[ComplexityViolation, ...]:
    """Run Ruff with the ratchet limit while ignoring inline suppressions."""

    arguments = (
        "check",
        *paths,
        "--select",
        "C901",
        "--ignore-noqa",
        "--config",
        f"lint.mccabe.max-complexity={max_complexity}",
        "--output-format",
        "json",
    )
    result = run_operator_command(
        "ruff",
        arguments,
        cwd=ROOT,
        timeout_seconds=300.0,
        stdout_limit_bytes=16 * 1024 * 1024,
        stderr_limit_bytes=4 * 1024 * 1024,
    )
    if result.status is not ToolCommandStatus.EXITED:
        detail = result.diagnostic or result.stderr.decode(errors="replace")[:1024]
        raise ComplexityBaselineError(f"Ruff C901 scan failed: {detail}")
    if result.exit_code not in {0, 1}:
        detail = (
            result.stderr.decode(errors="replace").strip()
            or result.stdout.decode(errors="replace").strip()
        )
        raise ComplexityBaselineError(f"Ruff C901 scan failed: {detail}")
    return parse_ruff_output(
        result.stdout.decode("utf-8", errors="strict"),
        root=ROOT,
        max_complexity=max_complexity,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--update",
        action="store_true",
        help="replace the baseline with the current canonical Ruff C901 snapshot",
    )
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)
    if args.update and args.paths:
        parser.error("--update cannot be combined with path filters")
    paths = tuple(args.paths) if args.paths else DEFAULT_PATHS
    try:
        baseline = load_baseline(args.baseline)
        current = run_ruff(paths, max_complexity=baseline.max_complexity)
        current_baseline = ComplexityBaseline(baseline.max_complexity, current)
        if args.update:
            write_baseline(args.baseline, current_baseline)
            print(
                f"Updated {args.baseline} with {len(current)} known violations, "
                f"limit {baseline.max_complexity}."
            )
            return 0
        if args.baseline.read_text(encoding="utf-8") != serialize_baseline(baseline):
            print(
                "C901 complexity baseline is not canonically formatted; "
                "run `uv run --locked python tools/check_complexity.py --update`.",
                file=sys.stderr,
            )
            return 1
        problems = compare_violations(baseline, current)
    except ComplexityBaselineError as exc:
        parser.error(str(exc))
    if problems:
        print("C901 complexity baseline mismatch:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(
        f"C901 baseline unchanged: {len(current)} known violations, "
        f"limit {baseline.max_complexity}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
