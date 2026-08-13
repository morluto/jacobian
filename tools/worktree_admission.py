"""OS-locked admission for exhaustive local validation in one worktree."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any

TOKEN_ENV = "JACOBIAN_WORKTREE_ADMISSION_TOKEN"
LOCK_NAME = ".jacobian-validation.lock"


def _repo_root() -> Path:
    marker = Path("Makefile")
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / marker).is_file() and (candidate / ".git").exists():
            return candidate
    return cwd


def _lock_path(root: Path) -> Path:
    return root / LOCK_NAME


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"validation lock is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("validation lock payload must be an object")
    return payload


def _assert_reentry(token: str, path: Path) -> None:
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload = _read_payload(path)
            expected = payload.get("token_hash")
            if not isinstance(expected, str) or not hmac.compare_digest(
                expected, _token_hash(token)
            ):
                raise SystemExit("invalid worktree admission token") from None
            return
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    raise SystemExit("stale worktree admission token; the exhaustive lease is free")


def _run(target: str, command: list[str]) -> int:
    if not command:
        raise SystemExit("worktree admission run requires a command")
    root = _repo_root()
    path = _lock_path(root)
    existing = os.environ.get(TOKEN_ENV)
    if existing:
        _assert_reentry(existing, path)
        return subprocess.call(command)

    token = secrets.token_hex(32)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            payload = _read_payload(path)
            holder = payload.get("target", "unknown")
            pid = payload.get("pid", "unknown")
            raise SystemExit(
                f"worktree already running exhaustive validation {holder!r} (pid {pid})"
            ) from exc
        payload = {
            "token_hash": _token_hash(token),
            "target": target,
            "pid": os.getpid(),
            "started": time.time(),
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, encoded)
        os.fsync(fd)
        env = os.environ.copy()
        env[TOKEN_ENV] = token
        return subprocess.call(command, env=env)
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
    run_parser = subparsers.add_parser("run", help="Hold the lease while a command runs")
    run_parser.add_argument("--target", required=True)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    subparsers.add_parser("status", help="Print whether this worktree holds the lease")
    args = parser.parse_args(argv)
    if args.command == "status":
        return _status()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    return _run(args.target, command)


if __name__ == "__main__":
    raise SystemExit(main())
