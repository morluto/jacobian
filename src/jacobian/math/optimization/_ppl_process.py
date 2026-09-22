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


def _decode_fraction(value: Any) -> Fraction:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, str) for item in value)
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


def _decode_vector(value: Any, *, length: int) -> tuple[Fraction, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError("PPL worker returned a malformed vector")
    return tuple(_decode_fraction(item) for item in value)


def _decoder(variables: int, equations: int) -> Callable[[object], ExactLinearOutcome]:
    def decode(value: object) -> ExactLinearOutcome:
        if not isinstance(value, dict) or set(value) != {
            "protocol_version",
            "status",
            "point",
            "dual",
            "witness",
            "ray",
        }:
            raise ValueError("PPL worker returned malformed output")
        if value["protocol_version"] != _PROTOCOL_VERSION:
            raise ValueError("PPL worker returned an unsupported protocol version")
        status = value["status"]
        if status == "OPTIMAL":
            return ExactLinearOutcome(
                status=status,
                point=_decode_vector(value["point"], length=variables),
                dual=_decode_vector(value["dual"], length=equations),
            )
        if status == "INFEASIBLE":
            return ExactLinearOutcome(
                status=status,
                witness=_decode_vector(value["witness"], length=equations),
            )
        if status == "UNBOUNDED":
            return ExactLinearOutcome(
                status=status,
                point=_decode_vector(value["point"], length=variables),
                ray=_decode_vector(value["ray"], length=variables),
            )
        raise ValueError("PPL worker returned an invalid status")

    return decode


def solve_standard_form_process(
    objective: tuple[Fraction, ...],
    coefficients: tuple[tuple[Fraction, ...], ...],
    rhs: tuple[Fraction, ...],
) -> ExactLinearOutcome:
    execution = current_request_execution()
    if execution is None or execution.deadline is None:
        raise RuntimeError("exact LP worker requires an owner-bound request deadline")
    remaining = execution.deadline - monotonic()
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "linear-program deadline expired before PPL execution"
        )
    payload = encode_strict_json(
        {
            "protocol_version": _PROTOCOL_VERSION,
            "objective": [_encode_fraction(value) for value in objective],
            "coefficients": [
                [_encode_fraction(value) for value in row] for row in coefficients
            ],
            "rhs": [_encode_fraction(value) for value in rhs],
        }
    )
    try:
        with TemporaryDirectory(prefix="jacobian-ppl-") as worker_directory:
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
                decode_result=_decoder(len(objective), len(rhs)),
            )
    except (OperationExecutionCancelledError, OperationExecutionTimeoutError):
        raise
    except OSError as exc:
        raise RuntimeError("bounded PPL worker could not be started") from exc
    request_checkpoint("after exact PPL worker")
    return result


__all__ = ["solve_standard_form_process"]
