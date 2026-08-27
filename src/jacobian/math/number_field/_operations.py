"""Domain-owned number field operations."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_field._models import (
    NumberFieldDiscriminantResult,
    NumberFieldRequest,
)

_WORKER = Path(__file__).resolve().with_name("_worker.py")
_WORKER_TIMEOUT_SECONDS = 60.0
_WORKER_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1024 * 1024


def compute_nf_discriminant(
    request: NumberFieldRequest,
) -> NumberFieldDiscriminantResult:
    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    try:
        # The worker needs no ambient files: its request is stdin and its
        # response is bounded stdout.  A private cwd and regular-file ceiling
        # keep a native backend from using the checkout as scratch space.
        with TemporaryDirectory(prefix="jacobian-number-field-") as worker_directory:
            completed = run_bounded_process(
                [sys.executable, str(_WORKER)],
                input_bytes=json.dumps(
                    request.model_dump(mode="json"), separators=(",", ":")
                ).encode(),
                timeout_seconds=_WORKER_TIMEOUT_SECONDS,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=128 * 1024,
                stderr_limit=64 * 1024,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=math.ceil(_WORKER_TIMEOUT_SECONDS),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
            )
    except OSError:
        return NumberFieldDiscriminantResult(
            status="UNKNOWN",
            detail="the bounded number-field worker could not be started",
        )
    if (
        completed.timed_out
        or completed.cancelled
        or completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        return NumberFieldDiscriminantResult(
            status="UNKNOWN",
            detail="the bounded number-field worker did not establish a discriminant",
        )
    try:
        response = json.loads(completed.stdout.decode("utf-8"))
        if response["kind"] == "complete":
            discriminant = format_canonical_integer(
                parse_canonical_integer(response["discriminant"])
            )
            return NumberFieldDiscriminantResult(discriminant=discriminant)
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        response = None
    if isinstance(response, dict) and response.get("kind") == "invalid":
        raise OperationDomainValidationError(
            location=("coefficients_descending",),
            code="number_field.not_irreducible",
            message="number-field polynomial must be irreducible over QQ",
        )
    return NumberFieldDiscriminantResult(
        status="UNKNOWN",
        detail="the bounded number-field worker returned malformed output",
    )
