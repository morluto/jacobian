"""Bounded exact cyclotomic polynomials through SymPy's ZZ backend."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import NoReturn

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS
from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    request_checkpoint,
)
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._models import IntegerPolynomial
from jacobian.math.polynomials.values import MAX_POLYNOMIAL_TERMS

MAX_CYCLOTOMIC_INDEX = 100_000
MAX_CYCLOTOMIC_DEGREE = MAX_POLYNOMIAL_TERMS - 1
MAX_CYCLOTOMIC_DIVISORS = 512
MAX_CYCLOTOMIC_FACTOR_WORK = 2_000_000
MAX_CYCLOTOMIC_INTERMEDIATE_WORK = 16_000_000
MAX_CYCLOTOMIC_COEFFICIENT_DIGITS = 4_096
MAX_CYCLOTOMIC_OUTPUT_DIGITS = 8_000_000


@dataclass(frozen=True, slots=True)
class _CyclotomicAdmission:
    """One immutable preflight plan shared by backend and result construction."""

    degree: int
    factorization_work: int
    divisor_count: int
    intermediate_work: int
    coefficient_digits: int
    output_digits: int


class CyclotomicRequest(StrictModel):
    index: StrictInt = Field(ge=1, le=MAX_CYCLOTOMIC_INDEX)


class CyclotomicResult(StrictModel):
    source_index: StrictInt = Field(ge=1, le=MAX_CYCLOTOMIC_INDEX)
    totient: StrictInt = Field(ge=1, le=MAX_POLYNOMIAL_TERMS - 1)
    polynomial: IntegerPolynomial

    @model_validator(mode="after")
    def require_degree_and_monic_shape(self) -> CyclotomicResult:
        """Check only the producer's structural shape on deserialization."""

        if self.totient != len(self.polynomial.coefficients) - 1:
            raise PydanticCustomError(
                "polynomial.cyclotomic.degree_shape",
                "totient must equal the polynomial degree",
            )
        if self.polynomial.coefficients[0] != 1:
            raise PydanticCustomError(
                "polynomial.cyclotomic.monic_shape",
                "cyclotomic polynomial must be monic",
            )
        expected_constant = -1 if self.source_index == 1 else 1
        if self.polynomial.coefficients[-1] != expected_constant:
            raise PydanticCustomError(
                "polynomial.cyclotomic.constant_shape",
                "cyclotomic polynomial has an incompatible constant term",
            )
        return self


def _backend_error(reason: BackendFailureReason, exc: BaseException) -> NoReturn:
    raise OperationBackendError(reason) from exc


def _require_typed_request(request: object) -> CyclotomicRequest:
    if not isinstance(request, CyclotomicRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="polynomial.cyclotomic.request_type",
            message="cyclotomic computation requires a CyclotomicRequest value",
        )
    return request


def _admit(index: int, factorization: dict[int, int]) -> _CyclotomicAdmission:
    """Preflight factorization, divisor, intermediate, and output envelopes."""

    factorization_work = max(1, index.bit_length()) * max(
        1, sum(prime * exponent for prime, exponent in factorization.items())
    )
    if factorization_work > MAX_CYCLOTOMIC_FACTOR_WORK:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="polynomial.cyclotomic.factorization_work_bound",
            message="cyclotomic index factorization exceeds the admitted work bound",
        )

    divisor_count = prod(exponent + 1 for exponent in factorization.values())
    if divisor_count > MAX_CYCLOTOMIC_DIVISORS:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="polynomial.cyclotomic.divisor_work_bound",
            message=(
                "cyclotomic divisor enumeration exceeds the admitted bound of "
                f"{MAX_CYCLOTOMIC_DIVISORS} divisors"
            ),
        )

    degree = index
    for prime in factorization:
        degree = degree // prime * (prime - 1)
    if degree > MAX_CYCLOTOMIC_DEGREE:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="polynomial.cyclotomic.output_degree_bound",
            message=(
                f"cyclotomic degree {degree} exceeds the output degree bound "
                f"{MAX_CYCLOTOMIC_DEGREE}"
            ),
        )

    # The direct ZZ construction is charged against a coefficient-operation
    # envelope proportional to the divisor product identity. The degree-based
    # bit estimate is conservative, while still admitting prime and prime-power
    # fixtures whose actual coefficients are tiny.
    intermediate_work = (degree + 1) * divisor_count * max(1, index.bit_length())
    if intermediate_work > MAX_CYCLOTOMIC_INTERMEDIATE_WORK:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="polynomial.cyclotomic.intermediate_work_bound",
            message="cyclotomic intermediate products exceed the admitted work bound",
        )

    coefficient_bits = degree + 1
    coefficient_digits = (coefficient_bits * 30_103) // 100_000 + 1
    if coefficient_digits > MAX_CYCLOTOMIC_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="polynomial.cyclotomic.coefficient_height_bound",
            message="cyclotomic coefficient height exceeds the exact-output bound",
        )
    if coefficient_digits > MAX_CANONICAL_INTEGER_DIGITS:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="polynomial.cyclotomic.coefficient_encoding_bound",
            message="cyclotomic coefficients exceed the canonical integer bound",
        )

    # This is the semantic exact coefficient payload, not a wire-size or
    # encoded-byte estimate.  Degree and coefficient height are separate
    # limits because either can dominate a dense exact result.
    output_digits = (degree + 1) * coefficient_digits
    if output_digits > MAX_CYCLOTOMIC_OUTPUT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="polynomial.cyclotomic.output_digit_bound",
            message="cyclotomic exact coefficient payload exceeds the output bound",
        )
    return _CyclotomicAdmission(
        degree=degree,
        factorization_work=factorization_work,
        divisor_count=divisor_count,
        intermediate_work=intermediate_work,
        coefficient_digits=coefficient_digits,
        output_digits=output_digits,
    )


def _factor_index(index: int) -> dict[int, int]:
    try:
        from sympy import factorint

        factors = factorint(index)
    except Exception as exc:
        _backend_error(BackendFailureReason.INITIALIZATION, exc)
    if not isinstance(factors, dict) or any(
        type(prime) is not int or type(exponent) is not int or prime < 2 or exponent < 1
        for prime, exponent in factors.items()
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return factors


def cyclotomic(request: CyclotomicRequest) -> CyclotomicResult:
    typed_request = _require_typed_request(request)
    request_checkpoint("before cyclotomic admission")
    index = typed_request.index
    factorization = _factor_index(index)
    admission = _admit(index, factorization)
    request_checkpoint("after cyclotomic admission")
    try:
        from sympy import Symbol, cyclotomic_poly
    except Exception as exc:
        _backend_error(BackendFailureReason.INITIALIZATION, exc)
    try:
        polynomial = cyclotomic_poly(index, Symbol("x"), polys=True)
        raw_coefficients = tuple(polynomial.all_coeffs())
    except OperationBackendError:
        raise
    except Exception as exc:
        _backend_error(BackendFailureReason.INVALID_OUTPUT, exc)
    request_checkpoint("after cyclotomic backend")
    coefficients: tuple[int, ...]
    try:
        normalized_coefficients: list[int] = []
        for value in raw_coefficients:
            if (
                type(value) is not int
                and getattr(value, "is_Integer", None) is not True
            ):
                raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
            normalized_coefficients.append(int(value))
    except OperationBackendError:
        raise
    except Exception as exc:
        _backend_error(BackendFailureReason.INVALID_OUTPUT, exc)
    coefficients = tuple(normalized_coefficients)
    actual_output_digits = sum(len(str(abs(value))) for value in coefficients)
    if (
        len(coefficients) != admission.degree + 1
        or not coefficients
        or coefficients[0] != 1
        or any(
            len(str(abs(value))) > admission.coefficient_digits
            for value in coefficients
        )
        or actual_output_digits > admission.output_digits
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    request_checkpoint("before cyclotomic result construction")
    try:
        return CyclotomicResult(
            source_index=index,
            totient=admission.degree,
            polynomial=IntegerPolynomial(
                coefficients=coefficients,
            ),
        )
    except Exception as exc:
        _backend_error(BackendFailureReason.INVALID_OUTPUT, exc)
