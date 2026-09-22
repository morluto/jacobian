"""Killable SymPy cancellation for admitted curvature components."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_checkpoint,
)
from jacobian.canonical import encode_strict_json
from jacobian.process import (
    ProcessResourceLimits,
    run_checked_worker_process,
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
    from sympy import QQ, Poly, Rational

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


def _decode_normalization_result(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("normalization worker result must be an object")
    return value


def cancel_fraction(
    numerator: Any,
    denominator: Any,
    *,
    deadline: float,
    owner: str = "metric curvature",
) -> tuple[Any, Any]:
    """Cancel one admitted pair in a killable worker under the shared deadline."""

    remaining = deadline - monotonic() - _PARENT_FINALIZATION_SECONDS
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            f"{owner} deadline expired before cancellation"
        )
    request_checkpoint(f"before {owner} cancellation payload encoding")
    payload = encode_strict_json(
        {
            "variable_count": len(numerator.gens),
            "numerator": _poly_payload(numerator),
            "denominator": _poly_payload(denominator),
        }
    )
    request_checkpoint(f"after {owner} cancellation payload encoding")
    remaining = deadline - monotonic() - _PARENT_FINALIZATION_SECONDS
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            f"{owner} deadline expired after cancellation payload encoding"
        )
    try:
        with TemporaryDirectory(prefix="jacobian-metric-cancel-") as worker_directory:
            response = run_checked_worker_process(
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
                decode_result=_decode_normalization_result,
            )
    except (OperationExecutionCancelledError, OperationExecutionTimeoutError):
        raise
    except OSError as exc:
        raise RuntimeError(
            f"bounded {owner} cancellation worker could not be started"
        ) from exc
    if (
        not isinstance(response, dict)
        or response.get("status") != "ok"
        or "numerator" not in response
        or "denominator" not in response
    ):
        raise RuntimeError(
            f"bounded {owner} cancellation worker returned malformed output"
        )
    symbols = tuple(numerator.gens)
    return (
        _poly_from_payload(response["numerator"], symbols),
        _poly_from_payload(response["denominator"], symbols),
    )
