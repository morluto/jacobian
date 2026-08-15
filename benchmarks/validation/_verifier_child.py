"""Fresh-interpreter verifier harness with narrow virtual filesystem mounts."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path, PurePosixPath

_VIRTUAL_ROOTS = ("/app", "/tests", "/logs/verifier")
_FAILURE_DETAILS = {
    "assurance_calibration": 0.0,
    "correctness": 0.0,
    "witness_validity": 0.0,
    "false_certification": False,
    "input_binding": 0.0,
    "input_integrity": 0.0,
    "limitation_accuracy": 0.0,
    "protocol_compliance": 0.0,
    "scope_accuracy": 0.0,
}


@dataclass(frozen=True)
class VerifierOutput:
    """The scalar Harbor reward and separate verifier diagnostics."""

    reward: float
    details: Mapping[str, object]


class VerifierExecutionError(RuntimeError):
    """The verifier child failed before producing a valid reward record."""


def _reject_duplicate_object_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise VerifierExecutionError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> object:
    raise VerifierExecutionError(f"non-finite JSON value: {value}")


def _read_json_object(path: Path, *, name: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise VerifierExecutionError(f"verifier did not produce a regular {name}")
    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifierExecutionError(f"verifier {name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise VerifierExecutionError(f"verifier {name} must be a JSON object")
    return value


def _read_verifier_output(logs_root: Path) -> VerifierOutput:
    reward_record = _read_json_object(logs_root / "reward.json", name="reward.json")
    if set(reward_record) != {"reward"}:
        raise VerifierExecutionError("verifier reward.json must contain exactly reward")
    reward = reward_record["reward"]
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise VerifierExecutionError("verifier reward must be a finite number")
    normalized_reward = float(reward)
    if not isfinite(normalized_reward) or not 0.0 <= normalized_reward <= 1.0:
        raise VerifierExecutionError("verifier reward must be finite and in [0, 1]")
    details = _read_json_object(
        logs_root / "reward-details.json", name="reward-details.json"
    )
    if "reward" in details:
        raise VerifierExecutionError(
            "verifier reward-details.json must not contain reward"
        )
    return VerifierOutput(reward=normalized_reward, details=details)


def _write_failure_output(logs_root: Path) -> None:
    (logs_root / "reward.json").write_text(
        json.dumps({"reward": 0.0}, sort_keys=True), encoding="utf-8"
    )
    (logs_root / "reward-details.json").write_text(
        json.dumps(_FAILURE_DETAILS, sort_keys=True), encoding="utf-8"
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_mount(name: str, value: Path) -> Path:
    if not value.is_absolute():
        raise ValueError(f"{name} mount must be absolute")
    if value.is_symlink():
        raise ValueError(f"{name} mount must not be a symlink")
    resolved = value.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{name} mount must be a directory")
    return resolved


def _validate_workspace_tree(root: Path) -> None:
    """Reject links and special files before loading verifier code."""

    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"symlinked workspace entry is not allowed: {entry.name}")
        if not entry.is_dir() and not entry.is_file():
            raise ValueError(f"workspace entry must be regular: {entry.name}")


class _MappedPathFactory:
    def __init__(self, mounts: Mapping[str, Path]) -> None:
        self._path_type = type(pathlib.Path())
        self._mounts = {
            prefix: _validate_mount(prefix, target) for prefix, target in mounts.items()
        }
        if tuple(self._mounts) != _VIRTUAL_ROOTS:
            raise ValueError("verifier mounts must be /app, /tests, and /logs/verifier")

    def __call__(self, value: os.PathLike[str] | str = ".") -> Path:
        raw = os.fspath(value)
        virtual = PurePosixPath(raw)
        if not virtual.is_absolute():
            if ".." in virtual.parts:
                raise ValueError("relative path traversal is not allowed")
            return self._path_type(raw)
        if ".." in virtual.parts:
            raise ValueError("virtual path traversal is not allowed")
        for prefix, root in self._mounts.items():
            prefix_path = PurePosixPath(prefix)
            try:
                relative = virtual.relative_to(prefix_path)
            except ValueError:
                continue
            candidate = root.joinpath(*relative.parts)
            resolved = candidate.resolve(strict=False)
            if not _is_relative_to(resolved, root):
                raise ValueError("virtual path escapes its mounted root")
            cursor = candidate
            while cursor != root:
                if cursor.is_symlink():
                    raise ValueError("symlinked verifier paths are not allowed")
                cursor = cursor.parent
            return self._path_type(candidate)
        raise ValueError(f"unsupported verifier path root: {raw}")


def _regular_source(path: Path, *, root: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"verifier source must not be a symlink: {path.name}")
    resolved = path.resolve(strict=True)
    if not _is_relative_to(resolved, root) or not resolved.is_file():
        raise ValueError(f"verifier source must be a regular file: {path.name}")
    return resolved.read_text(encoding="utf-8")


def _execute_child(app: Path, tests: Path, logs: Path) -> None:
    mounts = {
        "/app": app,
        "/tests": tests,
        "/logs/verifier": logs,
    }
    mapper = _MappedPathFactory(mounts)
    for root in mapper._mounts.values():
        _validate_workspace_tree(root)
    tests_root = mapper._mounts["/tests"]
    support_source = _regular_source(
        tests_root / "verifier_support.py", root=tests_root
    )
    verifier_source = _regular_source(tests_root / "verifier.py", root=tests_root)
    # Load maintained verifier dependencies before the task-local Path mapping.
    # Some dependencies register concrete pathlib classes while importing.
    importlib.import_module("jsonschema")
    importlib.import_module("referencing")
    original_path = pathlib.Path
    try:
        pathlib.Path = mapper  # type: ignore[assignment]
        sys.path.insert(0, str(tests_root))
        support_module = type(sys)("verifier_support")
        support_module.__file__ = "/tests/verifier_support.py"
        support_module.__package__ = ""
        sys.modules["verifier_support"] = support_module
        exec(
            compile(support_source, "/tests/verifier_support.py", "exec"),
            support_module.__dict__,
        )
        globals_dict = {
            "__file__": "/tests/verifier.py",
            "__name__": "__main__",
            "__package__": None,
        }
        exec(compile(verifier_source, "/tests/verifier.py", "exec"), globals_dict)
    finally:
        sys.path.remove(str(tests_root))
        sys.modules.pop("verifier_support", None)
        pathlib.Path = original_path


def run_verifier_in_child(
    *,
    task: Path,
    app: Path,
    logs: Path,
    timeout_seconds: float = 30.0,
    output_limit_bytes: int = 64 * 1024,
) -> VerifierOutput:
    """Run one task verifier in a clean interpreter and return both output files."""

    from benchmarks.tooling.command_runner import (
        ToolCommandRequest,
        ToolCommandStatus,
        operator_environment,
        run_tool_command,
    )

    tests = task.resolve(strict=True) / "tests"
    app_root = _validate_mount("/app", app.resolve(strict=True))
    tests_root = _validate_mount("/tests", tests)
    logs_root = _validate_mount("/logs/verifier", logs.resolve(strict=True))
    for root in (app_root, tests_root, logs_root):
        _validate_workspace_tree(root)
    cache_root = logs_root / ".pycache"
    cache_root.mkdir(exist_ok=True)
    environment = operator_environment(
        declared={
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(cache_root),
        }
    )
    request = ToolCommandRequest(
        # Preserve the active virtual-environment interpreter. Resolving this
        # symlink would select the base interpreter and lose verifier deps.
        executable=sys.executable,
        arguments=(
            "-m",
            "benchmarks.validation._verifier_child",
            "--app",
            str(app_root),
            "--tests",
            str(tests_root),
            "--logs",
            str(logs_root),
        ),
        environment=environment,
        cwd=str(Path(__file__).parents[2].resolve(strict=True)),
        timeout_seconds=timeout_seconds,
        stdout_limit_bytes=output_limit_bytes,
        stderr_limit_bytes=output_limit_bytes,
    )
    result = run_tool_command(request)
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        diagnostic = result.stderr.decode("utf-8", errors="replace").strip()
        raise VerifierExecutionError(
            f"verifier child {result.status}: {diagnostic or 'no diagnostic'}"
        )
    return _read_verifier_output(logs_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _execute_child(args.app, args.tests, args.logs)
    except Exception as exc:
        try:
            if args.logs.is_dir() and not args.logs.is_symlink():
                _write_failure_output(args.logs)
        except OSError:
            pass
        diagnostic = " ".join(str(exc).split())[:1024]
        for actual, virtual in (
            (args.app, "/app"),
            (args.tests, "/tests"),
            (args.logs, "/logs/verifier"),
        ):
            diagnostic = diagnostic.replace(str(actual), virtual)
        print(f"{type(exc).__name__}: {diagnostic}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
