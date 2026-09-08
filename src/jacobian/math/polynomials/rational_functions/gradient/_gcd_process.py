"""Killable exact gcd(q, q') bounds for rational gradient admission."""

from __future__ import annotations

import math
import sys
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
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
)
from jacobian.math.polynomials.rational_functions._bounds import (
    PolynomialBound,
    _one_polynomial,
    _zero_polynomial,
)
from jacobian.math.polynomials.values import SparseRationalPolynomial

_WORKER_PATH = Path(__file__).resolve().with_name("_gcd_worker.py")
_GCD_STDOUT_BYTES = 64 * 1024
_GCD_STDERR_BYTES = 64 * 1024
_GCD_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024


def _polynomial_payload(polynomial: SparseRationalPolynomial) -> list[list[Any]]:
    return [
        [
            *term.exponents,
            format_canonical_integer(term.coefficient.num),
            format_canonical_integer(term.coefficient.den),
        ]
        for term in polynomial.terms
    ]


def _bound_from_payload(payload: object, variable_count: int) -> PolynomialBound:
    if not isinstance(payload, dict) or set(payload) != {
        "term_count",
        "degrees",
        "total_degree",
        "minimum_exponents",
    }:
        raise RuntimeError(
            "bounded denominator-derivative gcd worker returned malformed output"
        )
    term_count = payload["term_count"]
    degrees = payload["degrees"]
    total_degree = payload["total_degree"]
    minimum_exponents = payload["minimum_exponents"]
    if (
        type(term_count) is not int
        or term_count < 0
        or type(total_degree) is not int
        or total_degree < 0
        or not isinstance(degrees, list)
        or not isinstance(minimum_exponents, list)
        or len(degrees) != variable_count
        or len(minimum_exponents) != variable_count
        or any(type(degree) is not int or degree < 0 for degree in degrees)
        or any(
            type(exponent) is not int or exponent < 0 for exponent in minimum_exponents
        )
    ):
        raise RuntimeError(
            "bounded denominator-derivative gcd worker returned malformed output"
        )
    if term_count == 0:
        return _zero_polynomial(variable_count)
    return PolynomialBound(
        terms=term_count,
        degrees=tuple(degrees),
        total_degree=total_degree,
        minimum_exponents=tuple(minimum_exponents),
        coefficient_digits=1,
        rational_content=Fraction(1),
    )


def _request_deadline() -> float:
    execution = current_request_execution()
    if execution is None or execution.deadline is None:
        raise OperationExecutionTimeoutError(
            "rational gradient deadline expired before denominator-derivative gcd"
        )
    return execution.deadline


def forced_denominator_derivative_gcds(
    denominator: SparseRationalPolynomial,
    variable_count: int,
) -> tuple[PolynomialBound, ...]:
    """Return ``gcd(q, ∂q/∂x_i)`` bounds under the request deadline."""

    if variable_count == 0 or not denominator.terms:
        return tuple(_one_polynomial(variable_count) for _ in range(variable_count))

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    deadline = _request_deadline()
    request_checkpoint("before denominator-derivative gcd encoding")
    payload = encode_strict_json(
        {
            "variable_count": variable_count,
            "terms": _polynomial_payload(denominator),
        }
    )
    request_checkpoint("after denominator-derivative gcd encoding")
    try:
        with TemporaryDirectory(prefix="jacobian-rational-gradient-gcd-") as worker_dir:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise OperationExecutionTimeoutError(
                    "rational gradient deadline expired before "
                    "denominator-derivative gcd"
                )
            completed = run_bounded_process(
                [sys.executable, str(_WORKER_PATH)],
                input_bytes=payload,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_GCD_STDOUT_BYTES,
                stderr_limit=_GCD_STDERR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(remaining)),
                    address_space_bytes=_GCD_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_GCD_STDOUT_BYTES,
                ),
                cwd=worker_dir,
            )
    except OSError as exc:
        raise RuntimeError(
            "bounded denominator-derivative gcd worker could not be started"
        ) from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "rational gradient cancelled during denominator-derivative gcd"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "rational gradient deadline expired during denominator-derivative gcd"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(
            "bounded denominator-derivative gcd worker did not establish a factor"
        )
    request_checkpoint("after denominator-derivative gcd")
    try:
        response = loads_strict_json(
            completed.stdout,
            limits=CanonicalLimits(
                max_input_bytes=_GCD_STDOUT_BYTES,
                max_output_bytes=_GCD_STDOUT_BYTES,
            ),
        )
    except CanonicalizationError as exc:
        raise RuntimeError(
            "bounded denominator-derivative gcd worker returned malformed output"
        ) from exc
    if not isinstance(response, dict) or set(response) != {"factors"}:
        raise RuntimeError(
            "bounded denominator-derivative gcd worker returned malformed output"
        )
    factors = response["factors"]
    if not isinstance(factors, list) or len(factors) != variable_count:
        raise RuntimeError(
            "bounded denominator-derivative gcd worker returned malformed output"
        )
    request_checkpoint("after denominator-derivative gcd decoding")
    return tuple(_bound_from_payload(factor, variable_count) for factor in factors)


__all__ = ["forced_denominator_derivative_gcds"]
