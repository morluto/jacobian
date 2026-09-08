"""Killable SymPy quotient-rule expansion and cancellation for gradients."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
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
_MAX_RESULT_TERMS = 256


def _poly_payload(polynomial: SparseRationalPolynomial) -> list[list[Any]]:
    return [
        [
            *term.exponents,
            format_canonical_integer(term.coefficient.num),
            format_canonical_integer(term.coefficient.den),
        ]
        for term in polynomial.terms
    ]


def _poly_from_payload(
    records: object, variable_count: int
) -> SparseRationalPolynomial:
    if not isinstance(records, list):
        raise ValueError("malformed cancelled polynomial")
    terms: list[RationalPolynomialTerm] = []
    for record in records:
        if not isinstance(record, list) or len(record) != variable_count + 2:
            raise ValueError("malformed cancelled polynomial")
        exponents = tuple(record[:variable_count])
        if any(type(exponent) is not int or exponent < 0 for exponent in exponents):
            raise ValueError("malformed cancelled polynomial")
        numerator, denominator = record[-2], record[-1]
        if not isinstance(numerator, str) or not isinstance(denominator, str):
            raise ValueError("malformed cancelled polynomial")
        terms.append(
            RationalPolynomialTerm(
                coefficient=CanonicalRational(num=int(numerator), den=int(denominator)),
                exponents=exponents,
            )
        )
    if len(terms) > _MAX_RESULT_TERMS:
        raise ValueError("cancelled polynomial exceeds the canonical term bound")
    return SparseRationalPolynomial(terms=tuple(terms))


def normalize_partial(
    source: RationalFunction, axis: int, *, deadline: float
) -> RationalFunction:
    """Expand and cancel one admitted partial in a killable worker."""

    remaining = deadline - monotonic() - _PARENT_FINALIZATION_SECONDS
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "rational gradient deadline expired before the gradient kernel"
        )
    variable_count = len(source.variables)
    payload = encode_strict_json(
        {
            "variable_count": variable_count,
            "axis": axis,
            "numerator": _poly_payload(source.numerator),
            "denominator": _poly_payload(source.denominator),
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
            "bounded rational-gradient kernel worker could not be started"
        ) from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "rational gradient cancelled during the gradient kernel"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "rational gradient deadline expired during the gradient kernel"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(
            "bounded rational-gradient kernel worker did not return a fraction"
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
            "bounded rational-gradient kernel worker returned malformed output"
        ) from exc
    if (
        not isinstance(response, dict)
        or response.get("status") != "ok"
        or "numerator" not in response
        or "denominator" not in response
    ):
        raise RuntimeError(
            "bounded rational-gradient kernel worker returned malformed output"
        )
    try:
        numerator = _poly_from_payload(response["numerator"], variable_count)
        denominator = _poly_from_payload(response["denominator"], variable_count)
    except ValueError as exc:
        raise RuntimeError(
            "bounded rational-gradient kernel worker returned malformed output"
        ) from exc
    return RationalFunction._from_kernel(
        variables=source.variables,
        numerator=numerator,
        denominator=denominator,
    )
