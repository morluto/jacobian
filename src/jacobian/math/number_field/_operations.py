"""Domain-owned number field operations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_field._models import (
    NumberFieldDiscriminantResult,
    NumberFieldRequest,
)

_WORKER = Path(__file__).resolve().with_name("_worker.py")


def compute_nf_discriminant(
    request: NumberFieldRequest,
) -> NumberFieldDiscriminantResult:
    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    completed = run_bounded_process(
        [sys.executable, str(_WORKER)],
        input_bytes=json.dumps(
            request.model_dump(mode="json"), separators=(",", ":")
        ).encode(),
        timeout_seconds=60.0,
        environment=worker_environment(locale="C.UTF-8"),
        stdout_limit=128 * 1024,
        stderr_limit=64 * 1024,
        resource_limits=ProcessResourceLimits(address_space_bytes=1024 * 1024 * 1024),
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
