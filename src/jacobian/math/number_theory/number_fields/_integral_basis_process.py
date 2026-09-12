"""Shared killable process boundary for exact integral-basis computation."""

from __future__ import annotations

import hashlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.math.number_theory.number_fields._integral_basis import (
    monicized_discriminant_digit_bound,
)
from jacobian.math.number_theory.number_fields._models import NumberFieldRequest
from jacobian.math.number_theory.number_fields.values import (
    MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
    SimpleNumberFieldPresentation,
)

_WORKER = Path(__file__).resolve().with_name("_worker.py")
_WORKER_TIMEOUT_SECONDS = 60.0
_WORKER_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1024 * 1024
_WORKER_STDERR_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class IntegralBasisWorkerResult:
    """The bounded worker projection retained by the parent process."""

    field_discriminant: int
    basis: tuple[tuple[CanonicalRational, ...], ...] | None


def run_integral_basis_worker(
    request: NumberFieldRequest,
    *,
    include_basis: bool,
) -> IntegralBasisWorkerResult | None:
    """Compute one integral basis in a request-owned killable worker."""

    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return run_integral_basis_worker(request, include_basis=include_basis)

    owner_deadline = execution.started_at + _WORKER_TIMEOUT_SECONDS
    deadline = (
        min(execution.deadline, owner_deadline)
        if execution.deadline is not None
        else owner_deadline
    )
    bind_request_deadline(deadline)
    request_checkpoint("before number-field integral-basis preparation")

    input_bytes = encode_strict_json(request.model_dump(mode="json"))
    stdout_limit = _worker_stdout_limit(request.field, include_basis=include_basis)
    command = [sys.executable, str(_WORKER)]
    if include_basis:
        command.append("--basis")
    try:
        # The worker needs no ambient files: its request is stdin and its
        # response is bounded stdout. A private cwd and regular-file ceiling
        # keep a native backend from using the checkout as scratch space.
        with TemporaryDirectory(prefix="jacobian-number-field-") as worker_directory:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OperationExecutionTimeoutError(
                    "request deadline expired before number-field worker launch"
                )
            from jacobian.process import (
                ProcessResourceLimits,
                run_bounded_process,
                worker_environment,
            )

            completed = run_bounded_process(
                command,
                input_bytes=input_bytes,
                timeout_seconds=remaining,
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
    except OperationExecutionTimeoutError:
        raise
    except OSError as exc:
        request_checkpoint("during number-field worker startup")
        raise RuntimeError("bounded number-field worker could not be started") from exc

    request_checkpoint("after number-field integral-basis worker")
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "number-field integral-basis computation cancelled"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "number-field integral-basis computation timed out"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(
            "bounded number-field worker did not establish an integral basis"
        )

    response = _decode_worker_response(
        completed.stdout,
        request=request,
        input_bytes=input_bytes,
        include_basis=include_basis,
        stdout_limit=stdout_limit,
    )
    request_checkpoint("after number-field integral-basis result construction")
    return response


def _worker_stdout_limit(
    field: SimpleNumberFieldPresentation,
    *,
    include_basis: bool,
) -> int:
    degree = field.degree
    discriminant_digits = monicized_discriminant_digit_bound(field)
    response: dict[str, object] = {
        "kind": "complete",
        "discriminant": "-" + "9" * discriminant_digits,
        "request_digest": "0" * 64,
    }
    if include_basis:
        rational = {
            "num": "-" + "9" * MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
            "den": "9" * MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
        }
        response["basis"] = [
            [dict(rational) for _ in range(degree)] for _ in range(degree)
        ]
    return len(encode_strict_json(response))


def _decode_worker_response(
    output: bytes,
    *,
    request: NumberFieldRequest,
    input_bytes: bytes,
    include_basis: bool,
    stdout_limit: int,
) -> IntegralBasisWorkerResult | None:
    try:
        response = loads_strict_json(
            output,
            limits=CanonicalLimits(
                max_input_bytes=stdout_limit,
                max_output_bytes=stdout_limit,
            ),
        )
        if not isinstance(response, dict):
            raise ValueError("worker response must be an object")
        if response.get("request_digest") != hashlib.sha256(input_bytes).hexdigest():
            raise ValueError("worker response is not bound to its request")
        if response.get("kind") == "invalid":
            if set(response) != {"kind", "request_digest"}:
                raise ValueError("invalid worker response has invalid fields")
            return None
        expected_fields = {"kind", "discriminant", "request_digest"}
        if include_basis:
            expected_fields.add("basis")
        if response.get("kind") != "complete" or set(response) != expected_fields:
            raise ValueError("complete worker response has invalid fields")
        raw_discriminant = response["discriminant"]
        if not isinstance(raw_discriminant, str):
            raise ValueError("worker discriminant must be a canonical integer")
        discriminant = parse_canonical_integer(raw_discriminant)
        if format_canonical_integer(discriminant) != raw_discriminant:
            raise ValueError("worker discriminant is not canonical")
        basis = (
            _decode_basis(response["basis"], degree=request.field.degree)
            if include_basis
            else None
        )
        return IntegralBasisWorkerResult(
            field_discriminant=discriminant,
            basis=basis,
        )
    except (CanonicalizationError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "bounded number-field worker returned malformed output"
        ) from exc


def _decode_basis(
    raw_basis: object, *, degree: int
) -> tuple[tuple[CanonicalRational, ...], ...]:
    if not isinstance(raw_basis, list) or len(raw_basis) != degree:
        raise ValueError("worker integral basis has invalid row count")
    basis: list[tuple[CanonicalRational, ...]] = []
    for raw_vector in raw_basis:
        if not isinstance(raw_vector, list) or len(raw_vector) != degree:
            raise ValueError("worker integral basis has invalid row width")
        vector: list[CanonicalRational] = []
        for raw_value in raw_vector:
            if not isinstance(raw_value, dict) or set(raw_value) != {"num", "den"}:
                raise ValueError("worker integral basis has an invalid rational")
            raw_num = raw_value["num"]
            raw_den = raw_value["den"]
            if not isinstance(raw_num, str) or not isinstance(raw_den, str):
                raise ValueError("worker rational components must be strings")
            num = parse_canonical_integer(raw_num)
            den = parse_canonical_integer(raw_den)
            if (
                format_canonical_integer(num) != raw_num
                or format_canonical_integer(den) != raw_den
            ):
                raise ValueError("worker rational component is not canonical")
            rational = CanonicalRational(num=num, den=den)
            require_bounded_rational(
                rational,
                max_digits=MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
                label="integral basis",
            )
            vector.append(rational)
        basis.append(tuple(vector))
    return tuple(basis)


__all__ = [
    "IntegralBasisWorkerResult",
    "run_integral_basis_worker",
]
