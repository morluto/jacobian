"""Killable recognition, differentiation, and cancellation for general gradients."""

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
    request_checkpoint,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._conversions import (
    sparse_rational_polynomial_from_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import RationalFunction
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_WORKER_PATH = Path(__file__).resolve().with_name("_worker.py")
_STDOUT_BYTES = 16 * 1024 * 1024
_STDERR_BYTES = 64 * 1024
_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_PARENT_FINALIZATION_SECONDS = 0.05


def _source_payload(polynomial: Any) -> list[list[object]]:
    return [
        [
            *term.exponents,
            format_canonical_integer(term.coefficient.num),
            format_canonical_integer(term.coefficient.den),
        ]
        for term in polynomial.terms
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


def evaluate_admitted_general_gradient(
    function: RationalFunction,
    *,
    deadline: float,
) -> tuple[RationalFunction, ...]:
    """Recognize and differentiate one admitted general gradient in a worker."""

    remaining = deadline - monotonic() - _PARENT_FINALIZATION_SECONDS
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "rational gradient deadline expired before general-gradient expansion"
        )
    axis = function.variables
    request_checkpoint("before general-gradient payload encoding")
    payload = encode_strict_json(
        {
            "variables": list(axis),
            "numerator": _source_payload(function.numerator),
            "denominator": _source_payload(function.denominator),
        }
    )
    request_checkpoint("after general-gradient payload encoding")
    remaining = deadline - monotonic() - _PARENT_FINALIZATION_SECONDS
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "rational gradient deadline expired after payload encoding"
        )
    try:
        with TemporaryDirectory(
            prefix="jacobian-rational-gradient-"
        ) as worker_directory:
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
            "bounded rational-gradient worker could not be started"
        ) from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "rational gradient cancelled during general-gradient expansion"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "rational gradient deadline expired during general-gradient expansion"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError("bounded rational-gradient worker returned malformed output")
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
            "bounded rational-gradient worker returned malformed output"
        ) from exc
    if not isinstance(response, dict):
        raise RuntimeError("bounded rational-gradient worker returned malformed output")
    if response.get("status") == "noncanonical":
        raise OperationDomainValidationError(
            location=(),
            code="polynomial.not_coprime",
            message="rational-function numerator and denominator must be coprime",
        )
    if (
        response.get("status") != "ok"
        or not isinstance(response.get("fractions"), list)
        or len(response["fractions"]) != len(axis)
    ):
        raise RuntimeError("bounded rational-gradient worker returned malformed output")
    symbols = symbols_for_variables(axis)
    components = []
    for record in response["fractions"]:
        request_checkpoint("before general-gradient result decoding")
        if (
            not isinstance(record, dict)
            or "numerator" not in record
            or "denominator" not in record
        ):
            raise RuntimeError(
                "bounded rational-gradient worker returned malformed output"
            )
        numerator = _poly_from_payload(record["numerator"], symbols)
        denominator = _poly_from_payload(record["denominator"], symbols)
        components.append(
            RationalFunction(
                variables=axis,
                numerator=sparse_rational_polynomial_from_sympy(
                    numerator, axis, maximum_terms=256
                ),
                denominator=sparse_rational_polynomial_from_sympy(
                    denominator, axis, maximum_terms=256
                ),
            )
        )
    return tuple(components)
