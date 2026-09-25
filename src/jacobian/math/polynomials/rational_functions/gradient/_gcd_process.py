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
from jacobian.canonical import encode_strict_json, format_canonical_integer
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
_GRADIENT_BATCH_STDOUT_BYTES = 8 * _GCD_STDOUT_BYTES
_GCD_STDERR_BYTES = 64 * 1024
_GCD_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_COMPOSITION_BATCH_STDOUT_BYTES = 16 * _GCD_STDOUT_BYTES + 64 * 1024
_COMPOSITION_BATCH_INPUT_BYTES = 16 * 1024 * 1024


class KernelBatchInputLimitError(ValueError):
    """A proposed optional batch exceeds its aggregate input envelope."""


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


def _run_kernel_worker(
    payload: dict[str, Any],
    *,
    stage: str,
    stdout_limit: int = _GCD_STDOUT_BYTES,
    input_limit: int | None = None,
) -> dict[str, Any]:
    deadline = _request_deadline(stage=stage)
    request_checkpoint(f"before {stage} encoding")
    encoded = encode_strict_json(payload)
    request_checkpoint(f"after {stage} encoding")
    if input_limit is not None and len(encoded) > input_limit:
        raise KernelBatchInputLimitError(
            f"{stage} input exceeds its {input_limit}-byte aggregate envelope"
        )
    try:
        with TemporaryDirectory(prefix="jacobian-rational-gradient-gcd-") as worker_dir:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise OperationExecutionTimeoutError(
                    f"rational gradient deadline expired before {stage}"
                )
            response = process.run_checked_worker_process(
                [sys.executable, str(_WORKER_PATH)],
                input_bytes=encoded,
                timeout_seconds=remaining,
                environment=process.worker_environment(locale="C.UTF-8"),
                stdout_limit=stdout_limit,
                stderr_limit=_GCD_STDERR_BYTES,
                resource_limits=process.ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(remaining)),
                    address_space_bytes=_GCD_ADDRESS_SPACE_BYTES,
                    file_size_bytes=stdout_limit,
                ),
                cwd=worker_dir,
                decode_result=lambda value: value,
                checkpoint=lambda: request_checkpoint(f"during {stage} decoding"),
            )
    except OperationExecutionCancelledError as exc:
        raise OperationExecutionCancelledError(
            f"rational gradient cancelled during {stage}"
        ) from exc
    except OperationExecutionTimeoutError as exc:
        raise OperationExecutionTimeoutError(
            f"rational gradient deadline expired during {stage}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"bounded rational-gradient {stage} worker could not be started"
        ) from exc
    request_checkpoint(f"after {stage}")
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


def recognize_and_forced_denominator_derivative_gcds(
    function: RationalFunction,
    *,
    axes: tuple[int, ...],
) -> tuple[bool, tuple[DerivativeGcdFactor, ...]]:
    """Recognize one source and compute its active denominator GCDs together.

    These exact tasks consume the same source polynomials and share the same
    killable worker lifetime. The returned coprimality fact is checked before
    the caller proceeds to derivative construction; factor bounds remain
    available for the caller's post-GCD output admission.
    """

    variable_count = len(function.variables)
    if (
        variable_count == 0
        or any(axis < 0 or axis >= variable_count for axis in axes)
        or len(set(axes)) != len(axes)
    ):
        raise RuntimeError(
            "bounded rational-gradient admission worker received invalid axes"
        )
    response = _run_kernel_worker(
        {
            "task": "gradient_admission",
            "variable_count": variable_count,
            "axes": list(axes),
            "numerator": _polynomial_payload(function.numerator),
            "denominator": _polynomial_payload(function.denominator),
        },
        stage="gradient source and denominator admission",
    )
    if set(response) != {"coprime", "factors"} or type(response["coprime"]) is not bool:
        raise RuntimeError(
            "bounded rational-gradient admission worker returned malformed output"
        )
    factor_payloads = response["factors"]
    if not isinstance(factor_payloads, list):
        raise RuntimeError(
            "bounded rational-gradient admission worker returned malformed output"
        )
    if not response["coprime"]:
        if factor_payloads:
            raise RuntimeError(
                "bounded rational-gradient admission worker returned malformed output"
            )
        return False, ()
    if len(factor_payloads) != len(axes):
        raise RuntimeError(
            "bounded rational-gradient admission worker returned malformed output"
        )
    unit = DerivativeGcdFactor(bound=_one_polynomial(variable_count), records=())
    factors = [unit] * variable_count
    for axis, payload in zip(axes, factor_payloads, strict=True):
        factors[axis] = _bound_from_payload(payload, variable_count)
    return True, tuple(factors)


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


def normalize_admitted_fractions(
    pairs: tuple[tuple[Any, Any], ...],
    variables: tuple[str, ...],
) -> tuple[RationalFunction, ...]:
    """Normalize up to 16 admitted fractions in one isolated worker."""

    if not 1 <= len(pairs) <= 16:
        raise ValueError("rational normalization batch must contain 1 to 16 rows")
    variable_count = len(variables)
    if variable_count == 0:
        raise RuntimeError("rational-gradient normalization requires a declared axis")
    response = _run_kernel_worker(
        {
            "task": "normalize_batch",
            "variable_count": variable_count,
            "fractions": [
                {
                    "numerator": _sympy_payload(numerator),
                    "denominator": _sympy_payload(denominator),
                }
                for numerator, denominator in pairs
            ],
        },
        stage="composition fraction normalization batch",
        stdout_limit=_COMPOSITION_BATCH_STDOUT_BYTES,
        input_limit=_COMPOSITION_BATCH_INPUT_BYTES,
    )
    values = response.get("fractions")
    if (
        set(response) != {"fractions"}
        or not isinstance(values, list)
        or len(values) != len(pairs)
    ):
        raise RuntimeError(
            "bounded rational-composition normalization worker returned malformed output"
        )
    results: list[RationalFunction] = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
            raise RuntimeError(
                "bounded rational-composition normalization worker returned malformed output"
            )
        results.append(
            RationalFunction._from_kernel(
                variables=variables,
                numerator=_sparse_from_records(value["numerator"], variable_count),
                denominator=_sparse_from_records(value["denominator"], variable_count),
            )
        )
    return tuple(results)


def differentiate_admitted_fractions(
    source: RationalFunction,
    derivatives: tuple[tuple[int, tuple[list[Any], ...]], ...],
) -> tuple[RationalFunction, ...]:
    """Differentiate admitted axes together in one killable worker."""

    variable_count = len(source.variables)
    if (
        len(derivatives) > variable_count
        or len({axis for axis, _factor in derivatives}) != len(derivatives)
        or any(axis < 0 or axis >= variable_count for axis, _factor in derivatives)
    ):
        raise RuntimeError("rational-gradient batch has invalid derivative axes")
    if not derivatives:
        return ()
    response = _run_kernel_worker(
        {
            "task": "differentiate_batch",
            "variable_count": variable_count,
            "numerator": _polynomial_payload(source.numerator),
            "denominator": _polynomial_payload(source.denominator),
            "derivatives": [
                {
                    "axis": axis,
                    "factor": [list(record) for record in factor_records],
                }
                for axis, factor_records in derivatives
            ],
        },
        stage="gradient kernel",
        stdout_limit=_GRADIENT_BATCH_STDOUT_BYTES,
    )
    values = response.get("derivatives")
    if (
        set(response) != {"derivatives"}
        or not isinstance(values, list)
        or len(values) != len(derivatives)
    ):
        raise RuntimeError(
            "bounded rational-gradient kernel worker returned malformed output"
        )
    results: list[RationalFunction] = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
            raise RuntimeError(
                "bounded rational-gradient kernel worker returned malformed output"
            )
        results.append(
            RationalFunction._from_kernel(
                variables=source.variables,
                numerator=_sparse_from_records(value["numerator"], variable_count),
                denominator=_sparse_from_records(value["denominator"], variable_count),
            )
        )
    return tuple(results)


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
    "differentiate_admitted_fractions",
    "forced_denominator_derivative_gcds",
    "normalize_admitted_fraction",
    "normalize_admitted_fractions",
    "recognize_and_forced_denominator_derivative_gcds",
    "source_is_coprime",
]
