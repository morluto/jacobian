"""Killable process boundary for exact number-field discriminants."""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.canonical import (
    CanonicalizationError,
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.number_fields._models import (
    NumberFieldDiscriminantResult,
    NumberFieldRequest,
)

_WORKER = Path(__file__).resolve().with_name("_worker.py")
_WORKER_TIMEOUT_SECONDS = 60.0
_WORKER_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1024 * 1024
_WORKER_STDERR_BYTES = 64 * 1024


def compute_nf_discriminant(
    request: NumberFieldRequest,
) -> NumberFieldDiscriminantResult:
    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    input_bytes = encode_strict_json(request.model_dump(mode="json"))
    degree = len(request.field.coefficients_descending) - 1
    coefficient_digits = max(
        len(format_canonical_integer(abs(value)))
        for value in request.field.coefficients_descending
    )
    discriminant_digits = max(1, (2 * degree - 1) * coefficient_digits + 4 * degree)
    stdout_limit = len(
        encode_strict_json(
            {
                "kind": "complete",
                "discriminant": "-" + "9" * discriminant_digits,
                "request_digest": "0" * 64,
            },
        )
    )
    try:
        # The worker needs no ambient files: its request is stdin and its
        # response is bounded stdout.  A private cwd and regular-file ceiling
        # keep a native backend from using the checkout as scratch space.
        with TemporaryDirectory(prefix="jacobian-number-field-") as worker_directory:
            completed = run_bounded_process(
                [sys.executable, str(_WORKER)],
                input_bytes=input_bytes,
                timeout_seconds=_WORKER_TIMEOUT_SECONDS,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=stdout_limit,
                stderr_limit=_WORKER_STDERR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=math.ceil(_WORKER_TIMEOUT_SECONDS),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
            )
    except OSError as exc:
        raise RuntimeError("bounded number-field worker could not be started") from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "number-field discriminant computation cancelled"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "number-field discriminant computation timed out"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(
            "bounded number-field worker did not establish a discriminant"
        )
    try:
        response = loads_strict_json(completed.stdout)
        if (
            not isinstance(response, dict)
            or response.get("request_digest") != hashlib.sha256(input_bytes).hexdigest()
        ):
            raise ValueError("worker response is not bound to its request")
        if response["kind"] == "complete":
            if set(response) != {"kind", "discriminant", "request_digest"}:
                raise ValueError("complete worker response has invalid fields")
            discriminant = parse_canonical_integer(response["discriminant"])
            return NumberFieldDiscriminantResult(
                field=request.field, discriminant=discriminant
            )
    except (KeyError, TypeError, ValueError, CanonicalizationError):
        response = None
    if (
        isinstance(response, dict)
        and set(response) == {"kind", "request_digest"}
        and response.get("kind") == "invalid"
    ):
        raise OperationDomainValidationError(
            location=("field",),
            code="number_field.not_irreducible",
            message="number-field polynomial must be irreducible over QQ",
        )
    raise RuntimeError("bounded number-field worker returned malformed output")
