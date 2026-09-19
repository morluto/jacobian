"""Bounded parent adapter for the polynomial-ideal SymPy worker."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from jacobian.process import (
    ProcessPlatformTools,
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_PROTOCOL_VERSION = 1
_WORKER_PATH = Path(__file__).resolve().with_name("_sympy_worker.py")
_STDOUT_LIMIT = 8 * 1024 * 1024
_STDERR_LIMIT = 64 * 1024
_ADDRESS_SPACE_BYTES = 2 * 1024 * 1024 * 1024
_FILE_SIZE_BYTES = 1024 * 1024


class _ResultLimitExceededError(ValueError):
    """The exact backend result exceeds the declared output limits."""


class _SympyKernelTimeoutError(TimeoutError):
    """The bounded SymPy worker exceeded the request deadline."""


class _SympyKernelCancelledError(RuntimeError):
    """The bounded SymPy worker was cancelled before producing a result."""


class _SympyKernelError(RuntimeError):
    """The bounded SymPy worker failed without producing an exact result."""


def _canonical_request_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        {"protocol_version": _PROTOCOL_VERSION, "payload": payload},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _encode_request(payload: dict[str, Any]) -> tuple[str, bytes]:
    digest = hashlib.sha256(_canonical_request_bytes(payload)).hexdigest()
    request = {
        "protocol_version": _PROTOCOL_VERSION,
        "request_digest": digest,
        "payload": payload,
    }
    return digest, json.dumps(
        request,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _decode_response(output: bytes, expected_digest: str) -> dict[str, Any]:
    try:
        decoded = json.loads(output.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as error:
        raise _SympyKernelError("SymPy worker returned malformed JSON") from error
    if not isinstance(decoded, dict):
        raise _SympyKernelError("SymPy worker response must be an object")
    if decoded.get("protocol_version") != _PROTOCOL_VERSION:
        raise _SympyKernelError("SymPy worker protocol version mismatch")
    if decoded.get("request_digest") != expected_digest:
        raise _SympyKernelError("SymPy worker response is not bound to the request")
    status = decoded.get("status")
    detail = decoded.get("detail", "")
    if not isinstance(detail, str):
        raise _SympyKernelError("SymPy worker returned a malformed detail field")
    if status == "limit":
        raise _ResultLimitExceededError(detail)
    if status == "error":
        raise _SympyKernelError(detail or "SymPy worker failed")
    if status != "ok":
        raise _SympyKernelError("SymPy worker returned an unknown status")
    return decoded


def _run_sympy_kernel(
    payload: dict[str, Any],
    wall_seconds: float,
    *,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Execute one admitted payload in an isolated, killable SymPy worker."""

    if deadline is not None:
        wall_seconds = min(wall_seconds, deadline - time.monotonic())
    if wall_seconds <= 0:
        raise _SympyKernelTimeoutError()

    digest, request = _encode_request(payload)
    executable = shutil.which(sys.executable) or sys.executable
    prlimit = shutil.which("prlimit")
    if prlimit is not None:
        prlimit = str(Path(prlimit).resolve())
    try:
        with tempfile.TemporaryDirectory(prefix="jacobian-sympy-") as directory:
            if deadline is not None:
                wall_seconds = min(wall_seconds, deadline - time.monotonic())
            if wall_seconds <= 0:
                raise _SympyKernelTimeoutError()
            completed = run_bounded_process(
                [executable, "-I", str(_WORKER_PATH)],
                input_bytes=request,
                timeout_seconds=float(wall_seconds),
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_STDOUT_LIMIT,
                stderr_limit=_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(wall_seconds)),
                    address_space_bytes=_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_FILE_SIZE_BYTES,
                ),
                platform_tools=ProcessPlatformTools(prlimit_executable=prlimit),
                cwd=directory,
            )
    except OSError as error:
        raise _SympyKernelError(str(error)) from None

    if completed.cancelled:
        raise _SympyKernelCancelledError()
    if completed.timed_out:
        raise _SympyKernelTimeoutError()
    if completed.stdout_exceeded:
        raise _ResultLimitExceededError(
            "the exact kernel result exceeded the worker channel bound"
        )
    if completed.stderr_exceeded:
        raise _SympyKernelError("SymPy worker exceeded its diagnostic channel bound")
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise _SympyKernelError(detail or "SymPy worker exited abnormally")
    if completed.stderr:
        raise _SympyKernelError("SymPy worker emitted unexpected diagnostics")
    return _decode_response(completed.stdout, digest)
