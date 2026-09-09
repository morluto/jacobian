"""Bounded private frames exchanged with supervised workers."""

from __future__ import annotations

import json
from typing import Any

from jacobian._execution import BackendFailureReason


def encode_worker_progress_frame(
    progress: int, *, total: int | None = None, message: str | None = None
) -> bytes:
    """Encode one absolute-progress frame."""

    frame: dict[str, Any] = {"kind": "progress", "progress": progress}
    if total is not None:
        frame["total"] = total
    if message is not None:
        frame["message"] = message
    return json.dumps(frame, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


def encode_worker_error_frame(reason: BackendFailureReason) -> bytes:
    """Encode one worker execution-error frame."""

    return (
        json.dumps(
            {"kind": "error", "reason": reason.value}, separators=(",", ":")
        ).encode()
        + b"\n"
    )


def encode_worker_result_frame(result: Any) -> bytes:
    """Encode one final-result frame."""

    return (
        json.dumps(
            {"kind": "result", "result": result},
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        + b"\n"
    )


__all__ = [
    "encode_worker_error_frame",
    "encode_worker_progress_frame",
    "encode_worker_result_frame",
]
