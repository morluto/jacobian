"""Hold an exclusive worktree lock while a command runs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.command_runner import (  # noqa: E402
    ToolCommandRequest,
    ToolCommandStatus,
    operator_environment,
    run_tool_command,
)

LOCK_NAME = ".jacobian-validation.lock"
_VALIDATION_TIMEOUT_SECONDS = 4_800.0
_VALIDATION_OUTPUT_LIMIT_BYTES = 64 * 1024 * 1024
_VALIDATION_ENVIRONMENT = (
    "PATH",
    "VIRTUAL_ENV",
    "UV_CACHE_DIR",
    "UV_PYTHON_INSTALL_DIR",
    "MAKEFLAGS",
)


def _repo_root() -> Path:
    marker = Path("Makefile")
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / marker).is_file() and (candidate / ".git").exists():
            return candidate
    return cwd


def _lock_path(root: Path) -> Path:
    return root / LOCK_NAME


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"validation lock is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("validation lock payload must be an object")
    return payload


def _stream_to(stream: Any) -> Any:
    """Return a byte sink that preserves the wrapped command's output."""

    binary = getattr(stream, "buffer", None)
    target = binary if binary is not None else stream

    def sink(block: bytes) -> None:
        target.write(block if binary is not None else block.decode("utf-8", "replace"))
        flush = getattr(target, "flush", None)
        if callable(flush):
            flush()

    return sink


def _run(target: str, command: list[str]) -> int:
    if not command:
        raise SystemExit("validation lock run requires a command")
    root = _repo_root()
    path = _lock_path(root)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            payload = _read_payload(path)
            holder = payload.get("target", "unknown")
            pid = payload.get("pid", "unknown")
            raise SystemExit(
                f"worktree already running broad validation {holder!r} (pid {pid})"
            ) from exc
        encoded = json.dumps(
            {"target": target, "pid": os.getpid(), "started": time.time()},
            sort_keys=True,
        ).encode("utf-8")
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, encoded)
        os.fsync(fd)
        executable = command[0]
        if Path(executable).is_absolute():
            resolved = str(Path(executable).resolve(strict=True))
        else:
            candidate = shutil.which(executable)
            if candidate is None:
                print(f"validation command is unavailable: {executable}", file=sys.stderr)
                return 127
            resolved = candidate
        result = run_tool_command(
            ToolCommandRequest(
                executable=resolved,
                arguments=tuple(command[1:]),
                environment=operator_environment(include=_VALIDATION_ENVIRONMENT),
                cwd=str(root.resolve(strict=True)),
                timeout_seconds=_VALIDATION_TIMEOUT_SECONDS,
                stdout_limit_bytes=_VALIDATION_OUTPUT_LIMIT_BYTES,
                stderr_limit_bytes=_VALIDATION_OUTPUT_LIMIT_BYTES,
                stdout_sink=_stream_to(sys.stdout),
                stderr_sink=_stream_to(sys.stderr),
            )
        )
        if result.status is ToolCommandStatus.EXITED and result.exit_code is not None:
            return result.exit_code
        if result.diagnostic:
            print(result.diagnostic, file=sys.stderr)
        return 124 if result.status is ToolCommandStatus.TIMED_OUT else 125
    finally:
        os.close(fd)


def _status() -> int:
    path = _lock_path(_repo_root())
    if not path.exists():
        print("validation lease: free")
        return 0
    fd = os.open(path, os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload = _read_payload(path)
            print(
                "validation lease: held "
                f"target={payload.get('target')} pid={payload.get('pid')}"
            )
            return 0
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    print("validation lease: free")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Hold the lock while a command runs")
    run_parser.add_argument("--target", required=True)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    subparsers.add_parser("status", help="Print whether this worktree holds the lock")
    args = parser.parse_args(argv)
    if args.command == "status":
        return _status()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    return _run(args.target, command)


if __name__ == "__main__":
    raise SystemExit(main())
