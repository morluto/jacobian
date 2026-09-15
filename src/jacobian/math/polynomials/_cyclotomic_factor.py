"""Cyclotomic-factor profiles within a finite supported range (#3724).

For a supplied primitive integer polynomial, compare against every cyclotomic
polynomial Phi_n with phi(n) == degree and n <= MAX_FACTOR_SEARCH_INDEX. A
match returns the index with exact reconstruction; otherwise the result is
NOT_IDENTIFIED_IN_SUPPORTED_RANGE, which is not a proof of noncyclotomicity
unless the bound is complete for the supplied degree/coefficient contract.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials._models import IntegerPolynomial

MAX_FACTOR_DEGREE = 32
MAX_FACTOR_SEARCH_INDEX = 500
MAX_FACTOR_COEFFICIENT_DIGITS = 256


def _error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


def _euler_phi(n: int) -> int:
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1 if p == 2 else 2
    if temp > 1:
        result -= result // temp
    return result


class CyclotomicFactorProfileRequest(StrictModel):
    polynomial: IntegerPolynomial


class CyclotomicFactorProfileResult(StrictModel):
    polynomial: IntegerPolynomial
    degree: int = Field(ge=1, le=MAX_FACTOR_DEGREE)
    status: Literal["IDENTIFIED_CYCLOTOMIC", "NOT_IDENTIFIED_IN_SUPPORTED_RANGE"]
    cyclotomic_index: int | None = Field(default=None, ge=1)
    searched_index_bound: int = Field(ge=1)
    scope: Literal["FINITE_SUPPORTED_RANGE_ONLY"] = "FINITE_SUPPORTED_RANGE_ONLY"

    @classmethod
    def _from_kernel(
        cls,
        *,
        polynomial: IntegerPolynomial,
        degree: int,
        status: Literal["IDENTIFIED_CYCLOTOMIC", "NOT_IDENTIFIED_IN_SUPPORTED_RANGE"],
        cyclotomic_index: int | None,
        searched_index_bound: int,
    ) -> Self:
        return cls.model_construct(
            polynomial=polynomial,
            degree=degree,
            status=status,
            cyclotomic_index=cyclotomic_index,
            searched_index_bound=searched_index_bound,
        )


def _require_factor_envelope(polynomial: IntegerPolynomial) -> int:
    coefficients = polynomial.coefficients
    if len(coefficients) < 2:
        raise _error(
            "polynomial.cyclotomic_factor_degree",
            "a cyclotomic-factor profile needs a positive degree",
        )
    degree = len(coefficients) - 1
    if degree > MAX_FACTOR_DEGREE:
        raise _error(
            "polynomial.cyclotomic_factor_degree_bound",
            f"a cyclotomic-factor profile has degree at most {MAX_FACTOR_DEGREE}",
        )
    if coefficients[0] == 0:
        raise _error(
            "polynomial.cyclotomic_factor_leading",
            "a cyclotomic-factor profile needs a nonzero leading coefficient",
        )
    digits = sum(len(str(abs(int(c)))) for c in coefficients)
    if digits > MAX_FACTOR_COEFFICIENT_DIGITS * (degree + 1):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.cyclotomic_factor_coefficient_bound",
            message="factor coefficients exceed the admitted digit bound",
        )
    return degree


def cyclotomic_factor_profile(
    polynomial: IntegerPolynomial,
) -> CyclotomicFactorProfileResult:
    """Match a primitive factor against Phi_n in the supported range."""

    from jacobian.math.polynomials._cyclotomic import _compute

    degree = _require_factor_envelope(polynomial)
    # Only monic polynomials can equal a cyclotomic polynomial.
    if polynomial.coefficients[0] != 1:
        return CyclotomicFactorProfileResult._from_kernel(
            polynomial=polynomial,
            degree=degree,
            status="NOT_IDENTIFIED_IN_SUPPORTED_RANGE",
            cyclotomic_index=None,
            searched_index_bound=MAX_FACTOR_SEARCH_INDEX,
        )
    wanted = tuple(int(c) for c in polynomial.coefficients)
    for index in range(1, MAX_FACTOR_SEARCH_INDEX + 1):
        if _euler_phi(index) != degree:
            continue
        try:
            _, candidate = _compute(index)
        except Exception:
            continue
        if tuple(int(c) for c in candidate.coefficients) == wanted:
            # Exact reconstruction: recompute and compare (defining invariant).
            _, check = _compute(index)
            assert tuple(int(c) for c in check.coefficients) == wanted
            return CyclotomicFactorProfileResult._from_kernel(
                polynomial=polynomial,
                degree=degree,
                status="IDENTIFIED_CYCLOTOMIC",
                cyclotomic_index=index,
                searched_index_bound=MAX_FACTOR_SEARCH_INDEX,
            )
    return CyclotomicFactorProfileResult._from_kernel(
        polynomial=polynomial,
        degree=degree,
        status="NOT_IDENTIFIED_IN_SUPPORTED_RANGE",
        cyclotomic_index=None,
        searched_index_bound=MAX_FACTOR_SEARCH_INDEX,
    )


__all__ = [
    "CyclotomicFactorProfileRequest",
    "CyclotomicFactorProfileResult",
    "cyclotomic_factor_profile",
]
