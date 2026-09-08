"""Killable exact GCDs for rational-gradient recognition and normalization."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from jacobian import process
from jacobian._exact import CanonicalRational
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
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

_WORKER_PATH = Path(__file__).resolve().with_name("_gcd_worker.py")
_GCD_STDOUT_BYTES = 256 * 1024
_GCD_STDERR_BYTES = 64 * 1024
_GCD_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class DerivativeGcdFactor:
    """Exact ``gcd(q, q')`` together with the bound used at admission."""

    bound: PolynomialBound
    records: tuple[list[Any], ...]


def _polynomial_payload(polynomial: SparseRationalPolynomial) -> list[list[Any]]:
    return [
        [
            *term.exponents,
            format_canonical_integer(term.coefficient.num),
            format_canonical_integer(term.coefficient.den),
        ]
        for term in polynomial.terms
    ]


def _sympy_payload(polynomial: Any) -> list[list[Any]]:
    if polynomial.is_zero:
        return []
    return [
        [
            *exponents,
            format_canonical_integer(int(coefficient.p)),
            format_canonical_integer(int(coefficient.q)),
        ]
        for exponents, coefficient in polynomial.terms()
    ]


def _sparse_from_records(
    records: object, variable_count: int
) -> SparseRationalPolynomial:
    if not isinstance(records, list):
        raise RuntimeError(
            "bounded rational-gradient kernel worker returned malformed output"
        )
    terms: list[RationalPolynomialTerm] = []
    for record in records:
        if not isinstance(record, list) or len(record) != variable_count + 2:
            raise RuntimeError(
                "bounded rational-gradient kernel worker returned malformed output"
            )
        exponents = tuple(record[:variable_count])
        numerator = record[-2]
        denominator = record[-1]
        if (
            any(type(exponent) is not int or exponent < 0 for exponent in exponents)
            or not isinstance(numerator, str)
            or not isinstance(denominator, str)
        ):
            raise RuntimeError(
                "bounded rational-gradient kernel worker returned malformed output"
            )
        terms.append(
            RationalPolynomialTerm(
                coefficient=CanonicalRational.from_integer_ratio(
                    int(numerator), int(denominator)
                ),
                exponents=exponents,
            )
        )
    terms.sort(key=lambda term: term.exponents, reverse=True)
    return SparseRationalPolynomial(terms=tuple(terms))


def _bound_from_payload(payload: object, variable_count: int) -> DerivativeGcdFactor:
    if not isinstance(payload, dict) or set(payload) != {
        "term_count",
        "degrees",
        "total_degree",
        "minimum_exponents",
        "terms",
    }:
        raise RuntimeError(
            "bounded denominator-derivative gcd worker returned malformed output"
        )
    term_count = payload["term_count"]
    degrees = payload["degrees"]
    total_degree = payload["total_degree"]
    minimum_exponents = payload["minimum_exponents"]
    records = payload["terms"]
    if (
        type(term_count) is not int
        or term_count < 0
        or type(total_degree) is not int
        or total_degree < 0
        or not isinstance(degrees, list)
        or not isinstance(minimum_exponents, list)
        or not isinstance(records, list)
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
        bound = _zero_polynomial(variable_count)
    else:
        bound = PolynomialBound(
            terms=term_count,
            degrees=tuple(degrees),
            total_degree=total_degree,
            minimum_exponents=tuple(minimum_exponents),
            coefficient_digits=1,
            rational_content=Fraction(1),
        )
    return DerivativeGcdFactor(bound=bound, records=tuple(records))


def _request_deadline(*, stage: str) -> float:
    execution = current_request_execution()
    if execution is None or execution.deadline is None:
        raise OperationExecutionTimeoutError(
            f"rational gradient deadline expired before {stage}"
        )
    return execution.deadline


def _run_kernel_worker(payload: dict[str, Any], *, stage: str) -> dict[str, Any]:
    deadline = _request_deadline(stage=stage)
    request_checkpoint(f"before {stage} encoding")
    encoded = encode_strict_json(payload)
    request_checkpoint(f"after {stage} encoding")
    try:
        with TemporaryDirectory(prefix="jacobian-rational-gradient-gcd-") as worker_dir:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise OperationExecutionTimeoutError(
                    f"rational gradient deadline expired before {stage}"
                )
            completed = process.run_bounded_process(
                [sys.executable, str(_WORKER_PATH)],
                input_bytes=encoded,
                timeout_seconds=remaining,
                environment=process.worker_environment(locale="C.UTF-8"),
                stdout_limit=_GCD_STDOUT_BYTES,
                stderr_limit=_GCD_STDERR_BYTES,
                resource_limits=process.ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(remaining)),
                    address_space_bytes=_GCD_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_GCD_STDOUT_BYTES,
                ),
                cwd=worker_dir,
            )
    except OSError as exc:
        raise RuntimeError(
            f"bounded rational-gradient {stage} worker could not be started"
        ) from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            f"rational gradient cancelled during {stage}"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            f"rational gradient deadline expired during {stage}"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(
            f"bounded rational-gradient {stage} worker did not establish a result"
        )
    request_checkpoint(f"after {stage}")
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
            f"bounded rational-gradient {stage} worker returned malformed output"
        ) from exc
    if not isinstance(response, dict):
        raise RuntimeError(
            f"bounded rational-gradient {stage} worker returned malformed output"
        )
    request_checkpoint(f"after {stage} decoding")
    return response


def forced_denominator_derivative_gcds(
    denominator: SparseRationalPolynomial,
    variable_count: int,
    *,
    axes: tuple[int, ...] | None = None,
) -> tuple[DerivativeGcdFactor, ...]:
    """Return ``gcd(q, ∂q/∂x_i)`` under the request deadline.

    Axes whose admitted derivative bounds are identically zero skip the
    worker and receive a unit factor.
    """

    unit = DerivativeGcdFactor(bound=_one_polynomial(variable_count), records=())
    if variable_count == 0 or not denominator.terms:
        return tuple(unit for _ in range(variable_count))
    active = tuple(range(variable_count)) if axes is None else axes
    if any(axis < 0 or axis >= variable_count for axis in active):
        raise RuntimeError(
            "bounded denominator-derivative gcd worker returned malformed output"
        )
    if not active:
        return tuple(unit for _ in range(variable_count))
    response = _run_kernel_worker(
        {
            "task": "derivative_gcds",
            "variable_count": variable_count,
            "axes": list(active),
            "terms": _polynomial_payload(denominator),
        },
        stage="denominator-derivative gcd",
    )
    if set(response) != {"factors"}:
        raise RuntimeError(
            "bounded denominator-derivative gcd worker returned malformed output"
        )
    factors = response["factors"]
    if not isinstance(factors, list) or len(factors) != len(active):
        raise RuntimeError(
            "bounded denominator-derivative gcd worker returned malformed output"
        )
    result = [unit] * variable_count
    for axis, payload in zip(active, factors, strict=True):
        result[axis] = _bound_from_payload(payload, variable_count)
    return tuple(result)


def source_is_coprime(function: RationalFunction) -> bool:
    """Recognize coprimality of one non-monomial source under the deadline."""

    variable_count = len(function.variables)
    response = _run_kernel_worker(
        {
            "task": "coprime",
            "variable_count": variable_count,
            "numerator": _polynomial_payload(function.numerator),
            "denominator": _polynomial_payload(function.denominator),
        },
        stage="source coprimality recognition",
    )
    if set(response) != {"coprime"} or type(response["coprime"]) is not bool:
        raise RuntimeError(
            "bounded source-coprimality worker returned malformed output"
        )
    return response["coprime"]


def normalize_admitted_fraction(
    numerator: Any,
    denominator: Any,
    variables: tuple[str, ...],
    factor_records: tuple[list[Any], ...] = (),
) -> RationalFunction:
    """Cancel an admitted quotient using the retained derivative factor."""

    variable_count = len(variables)
    if variable_count == 0:
        raise RuntimeError("rational-gradient normalization requires a declared axis")
    response = _run_kernel_worker(
        {
            "task": "normalize",
            "variable_count": variable_count,
            "numerator": _sympy_payload(numerator),
            "denominator": _sympy_payload(denominator),
            "factor": [list(record) for record in factor_records],
        },
        stage="fraction normalization",
    )
    if set(response) != {"numerator", "denominator"}:
        raise RuntimeError(
            "bounded fraction-normalization worker returned malformed output"
        )
    return RationalFunction._from_kernel(
        variables=variables,
        numerator=_sparse_from_records(response["numerator"], variable_count),
        denominator=_sparse_from_records(response["denominator"], variable_count),
    )


def differentiate_admitted_fraction(
    source: RationalFunction,
    axis: int,
    factor_records: tuple[list[Any], ...] = (),
) -> RationalFunction:
    """Differentiate and normalize one admitted fraction in the GCD worker."""
    variable_count = len(source.variables)
    response = _run_kernel_worker(
        {
            "task": "differentiate",
            "variable_count": variable_count,
            "axis": axis,
            "numerator": _polynomial_payload(source.numerator),
            "denominator": _polynomial_payload(source.denominator),
            "factor": [list(record) for record in factor_records],
        },
        stage="gradient kernel",
    )
    if set(response) != {"numerator", "denominator"}:
        raise RuntimeError(
            "bounded rational-gradient kernel worker returned malformed output"
        )
    return RationalFunction._from_kernel(
        variables=source.variables,
        numerator=_sparse_from_records(response["numerator"], variable_count),
        denominator=_sparse_from_records(response["denominator"], variable_count),
    )


__all__ = [
    "DerivativeGcdFactor",
    "differentiate_admitted_fraction",
    "forced_denominator_derivative_gcds",
    "normalize_admitted_fraction",
    "source_is_coprime",
]
