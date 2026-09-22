"""Killable process boundary for exact PPL linear optimization."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    current_request_execution,
    request_checkpoint,
)
from jacobian.canonical import (
    encode_strict_json,
    format_canonical_integer,
    parse_canonical_integer,
)
from jacobian.math.optimization._ppl import ExactLinearOutcome
from jacobian.process import (
    ProcessResourceLimits,
    run_checked_worker_process,
    worker_environment,
)

_WORKER = Path(__file__).resolve().with_name("_ppl_worker.py")
_PROTOCOL_VERSION = 1
_STDOUT_LIMIT = 8 * 1024 * 1024
_STDERR_LIMIT = 64 * 1024
_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_FILE_SIZE_BYTES = 16 * 1024 * 1024


def _encode_fraction(value: Fraction) -> list[str]:
    return [
        format_canonical_integer(value.numerator),
        format_canonical_integer(value.denominator),
    ]


def _decode_fraction(value: Any, *, maximum_digits: int) -> Fraction:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, str) for item in value)
        or any(len(item.lstrip("-")) > maximum_digits for item in value)
    ):
        raise ValueError("PPL worker returned a malformed rational")
    result = Fraction(
        parse_canonical_integer(value[0]), parse_canonical_integer(value[1])
    )
    if [
        format_canonical_integer(result.numerator),
        format_canonical_integer(result.denominator),
    ] != value:
        raise ValueError("PPL worker returned a noncanonical rational")
    return result


def _decode_vector(
    value: Any, *, length: int, maximum_digits: int
) -> tuple[Fraction, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError("PPL worker returned a malformed vector")
    return tuple(
        _decode_fraction(item, maximum_digits=maximum_digits) for item in value
    )


type _StandardFormData = tuple[
    tuple[Fraction, ...],
    tuple[tuple[Fraction, ...], ...],
    tuple[Fraction, ...],
]


def _decode_outcome(
    value: object,
    *,
    variables: int,
    equations: int,
    maximum_digits: int,
) -> ExactLinearOutcome:
    if not isinstance(value, dict) or set(value) != {
        "status",
        "point",
        "dual",
        "witness",
        "ray",
    }:
        raise ValueError("PPL worker returned malformed outcome")
    status = value["status"]
    if status == "OPTIMAL":
        return ExactLinearOutcome(
            status=status,
            point=_decode_vector(
                value["point"], length=variables, maximum_digits=maximum_digits
            ),
            dual=_decode_vector(
                value["dual"], length=equations, maximum_digits=maximum_digits
            ),
        )
    if status == "INFEASIBLE":
        return ExactLinearOutcome(
            status=status,
            witness=_decode_vector(
                value["witness"], length=equations, maximum_digits=maximum_digits
            ),
        )
    if status == "UNBOUNDED":
        return ExactLinearOutcome(
            status=status,
            point=_decode_vector(
                value["point"], length=variables, maximum_digits=maximum_digits
            ),
            ray=_decode_vector(
                value["ray"], length=variables, maximum_digits=maximum_digits
            ),
        )
    raise ValueError("PPL worker returned an invalid status")


def _decoder(
    shapes: tuple[tuple[int, int], ...], maximum_digits: int
) -> Callable[[object], tuple[ExactLinearOutcome, ...]]:
    def decode(value: object) -> tuple[ExactLinearOutcome, ...]:
        if not isinstance(value, dict) or set(value) != {
            "protocol_version",
            "outcomes",
        }:
            raise ValueError("PPL worker returned malformed output")
        if value["protocol_version"] != _PROTOCOL_VERSION:
            raise ValueError("PPL worker returned an unsupported protocol version")
        outcomes = value["outcomes"]
        if not isinstance(outcomes, list) or len(outcomes) != len(shapes):
            raise ValueError("PPL worker returned a malformed outcome batch")
        return tuple(
            _decode_outcome(
                outcome,
                variables=variables,
                equations=equations,
                maximum_digits=maximum_digits,
            )
            for outcome, (variables, equations) in zip(outcomes, shapes, strict=True)
        )

    return decode


def solve_standard_form_batch_process(
    programs: tuple[_StandardFormData, ...],
    *,
    maximum_result_digits: int,
) -> tuple[ExactLinearOutcome, ...]:
    """Solve admitted independent programs in one killable worker request."""

    if not programs:
        return ()
    execution = current_request_execution()
    if execution is None or execution.deadline is None:
        raise RuntimeError("exact LP worker requires an owner-bound request deadline")
    payload = encode_strict_json(
        {
            "protocol_version": _PROTOCOL_VERSION,
            "programs": [
                {
                    "objective": [_encode_fraction(value) for value in objective],
                    "coefficients": [
                        [_encode_fraction(value) for value in row]
                        for row in coefficients
                    ],
                    "rhs": [_encode_fraction(value) for value in rhs],
                }
                for objective, coefficients, rhs in programs
            ],
        }
    )
    shapes = tuple(
        (len(objective), len(rhs)) for objective, _coefficients, rhs in programs
    )
    try:
        with TemporaryDirectory(prefix="jacobian-ppl-") as worker_directory:
            remaining = execution.deadline - monotonic()
            if remaining <= 0:
                raise OperationExecutionTimeoutError(
                    "linear-program deadline expired before PPL execution"
                )
            result = run_checked_worker_process(
                [sys.executable, str(_WORKER)],
                input_bytes=payload,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_STDOUT_LIMIT,
                stderr_limit=_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(remaining)),
                    address_space_bytes=_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
                decode_result=_decoder(shapes, maximum_result_digits),
            )
    except (OperationExecutionCancelledError, OperationExecutionTimeoutError):
        raise
    except OSError as exc:
        raise RuntimeError("bounded PPL worker could not be started") from exc
    request_checkpoint("after exact PPL worker batch")
    return result


def solve_standard_form_process(
    objective: tuple[Fraction, ...],
    coefficients: tuple[tuple[Fraction, ...], ...],
    rhs: tuple[Fraction, ...],
    *,
    maximum_result_digits: int,
) -> ExactLinearOutcome:
    """Solve one admitted standard-form program through the bounded worker."""

    return solve_standard_form_batch_process(
        ((objective, coefficients, rhs),),
        maximum_result_digits=maximum_result_digits,
    )[0]


__all__ = ["solve_standard_form_batch_process", "solve_standard_form_process"]
