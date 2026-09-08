"""Killable SymPy cancellation for admitted rational-gradient components."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from sympy import QQ, Poly, Rational

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    loads_strict_json,
)
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_WORKER_PATH = Path(__file__).resolve().with_name("_normalize_worker.py")
_STDOUT_BYTES = 8 * 1024 * 1024
_STDERR_BYTES = 64 * 1024
_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_PARENT_FINALIZATION_SECONDS = 1.0


def _poly_payload(polynomial: Any) -> list[list[Any]]:
    return [
        [*exponents, str(coefficient.p), str(coefficient.q)]
        for exponents, coefficient in polynomial.terms()
    ]


def _poly_from_payload(records: object, symbols: tuple[Any, ...]) -> Any:
    if not isinstance(records, list):
        raise ValueError("malformed cancelled polynomial")
    coefficients: dict[tuple[int, ...], Any] = {}
    variable_count = len(symbols)
    for record in records:
        if not isinstance(record, list) or len(record) != variable_count + 2:
            raise ValueError("malformed cancelled polynomial")
        exponents = tuple(record[:variable_count])
        if any(type(exponent) is not int or exponent < 0 for exponent in exponents):
            raise ValueError("malformed cancelled polynomial")
        numerator, denominator = record[-2], record[-1]
        if not isinstance(numerator, str) or not isinstance(denominator, str):
            raise ValueError("malformed cancelled polynomial")
        coefficients[exponents] = Rational(int(numerator), int(denominator))
    return Poly.from_dict(coefficients, *symbols, domain=QQ)


def cancel_fraction(
    numerator: Any, denominator: Any, *, deadline: float
) -> tuple[Any, Any]:
    """Cancel one admitted pair in a killable worker under the shared deadline."""

    remaining = deadline - monotonic() - _PARENT_FINALIZATION_SECONDS
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "rational gradient deadline expired before fraction cancellation"
        )
    payload = encode_strict_json(
        {
            "variable_count": len(numerator.gens),
            "numerator": _poly_payload(numerator),
            "denominator": _poly_payload(denominator),
        }
    )
    try:
        with TemporaryDirectory(prefix="jacobian-gradient-cancel-") as worker_directory:
            completed = run_bounded_process(
                [sys.executable, str(_WORKER_PATH)],
                input_bytes=payload,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_STDOUT_BYTES,
                stderr_limit=_STDERR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(remaining)),
                    address_space_bytes=_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_STDOUT_BYTES,
                ),
                cwd=worker_directory,
            )
    except OSError as exc:
        raise RuntimeError(
            "bounded rational-gradient cancellation worker could not be started"
        ) from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "rational gradient cancelled during fraction cancellation"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "rational gradient deadline expired during fraction cancellation"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(
            "bounded rational-gradient cancellation worker did not return a fraction"
        )
    try:
        response = loads_strict_json(
            completed.stdout,
            limits=CanonicalLimits(
                max_input_bytes=_STDOUT_BYTES,
                max_output_bytes=_STDOUT_BYTES,
            ),
        )
    except CanonicalizationError as exc:
        raise RuntimeError(
            "bounded rational-gradient cancellation worker returned malformed output"
        ) from exc
    if (
        not isinstance(response, dict)
        or response.get("status") != "ok"
        or "numerator" not in response
        or "denominator" not in response
    ):
        raise RuntimeError(
            "bounded rational-gradient cancellation worker returned malformed output"
        )
    symbols = tuple(numerator.gens)
    return (
        _poly_from_payload(response["numerator"], symbols),
        _poly_from_payload(response["denominator"], symbols),
    )
