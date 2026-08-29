"""Killable request-scoped process boundary for common interlacing."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pydantic import ValidationError

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.real_algebra._common_interlacing_models import (
    CommonInterlacingProfile,
    LabelledRationalPolynomial,
)

_WORKER = Path(__file__).resolve().with_name("_common_interlacing_worker.py")
_WALL_SECONDS = 180.0
_ADDRESS_SPACE_BYTES = 2 * 1024 * 1024 * 1024
_STDOUT_LIMIT = 11 * 1024 * 1024
_STDERR_LIMIT = 64 * 1024


def _domain_error(payload: dict[str, Any]) -> OperationDomainValidationError:
    errors = payload.get("errors")
    if not isinstance(errors, list) or len(errors) != 1:
        raise RuntimeError("common-interlacing worker returned malformed diagnostics")
    error = errors[0]
    if not isinstance(error, dict):
        raise RuntimeError("common-interlacing worker returned malformed diagnostics")
    location = error.get("loc")
    code = error.get("type")
    message = error.get("msg")
    if (
        not isinstance(location, list)
        or not all(isinstance(item, (str, int)) for item in location)
        or not isinstance(code, str)
        or not isinstance(message, str)
    ):
        raise RuntimeError("common-interlacing worker returned malformed diagnostics")
    return OperationDomainValidationError(
        location=tuple(location),
        code=code,
        message=message,
    )


def run_common_interlacing_profile(
    family: tuple[LabelledRationalPolynomial, ...],
) -> CommonInterlacingProfile:
    """Run factorization, isolation, and comparison in one killable worker."""

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    execution = current_request_execution()
    started = execution.started_at if execution is not None else time.monotonic()
    owner_deadline = started + _WALL_SECONDS
    deadline = min(
        owner_deadline,
        execution.deadline
        if execution is not None and execution.deadline is not None
        else owner_deadline,
    )
    bind_request_deadline(deadline)
    if deadline <= time.monotonic():
        raise OperationExecutionTimeoutError(
            "request deadline expired before common-interlacing execution"
        )

    request_bytes = json.dumps(
        [source.model_dump(mode="json") for source in family],
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        with TemporaryDirectory(prefix="jacobian-common-interlacing-") as directory:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OperationExecutionTimeoutError(
                    "request deadline expired before common-interlacing backend launch"
                )
            completed = run_bounded_process(
                [sys.executable, str(_WORKER)],
                input_bytes=request_bytes,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_STDOUT_LIMIT,
                stderr_limit=_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(remaining)),
                    address_space_bytes=_ADDRESS_SPACE_BYTES,
                    file_size_bytes=1024 * 1024,
                ),
                cwd=directory,
            )
    except OSError as exc:
        raise RuntimeError(
            "bounded common-interlacing worker could not be started"
        ) from exc

    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "request cancelled during common-interlacing execution"
        )
    if completed.timed_out or time.monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            "request deadline expired during common-interlacing execution"
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        raise RuntimeError(
            "bounded common-interlacing worker exceeded its output limit"
        )
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "bounded common-interlacing worker returned malformed output"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            "bounded common-interlacing worker returned malformed output"
        )
    if (
        completed.returncode == 0
        and payload.get("ok") is False
        and payload.get("kind") == "domain"
    ):
        raise _domain_error(payload)
    if completed.returncode != 0 or payload.get("ok") is not True:
        raise RuntimeError(
            "bounded common-interlacing worker did not establish a profile"
        )
    try:
        result = CommonInterlacingProfile.model_validate(payload.get("result"))
    except ValidationError as exc:
        raise RuntimeError(
            "bounded common-interlacing worker returned a malformed profile"
        ) from exc
    if time.monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            "request deadline expired during common-interlacing result construction"
        )
    return result


__all__ = ["run_common_interlacing_profile"]
