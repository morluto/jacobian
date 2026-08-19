"""Typed wire contracts for Galois theory operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_DEGREE = 12
MAX_FIELD_ORDER = 251


class GaloisFactorRequest(StrictModel):
    """Factor a polynomial over GF(p) to study its splitting behavior."""

    field_order: int = Field(ge=2, le=251)
    coefficients: tuple[int, ...] = Field(min_length=2, max_length=MAX_DEGREE + 1)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        from sympy import isprime

        if not isprime(self.field_order):
            raise ValueError("field_order must be prime")
        if any(not 0 <= c < self.field_order for c in self.coefficients):
            raise ValueError("coefficients must be canonical field residues")
        return self


class FrobeniusCycleRequest(StrictModel):
    """Compute the Frobenius cycle type from a factorization pattern."""

    field_order: int = Field(ge=2, le=251)
    polynomial_degree: int = Field(ge=1, le=MAX_DEGREE)
    factorization_degrees: tuple[int, ...] = Field(min_length=1, max_length=MAX_DEGREE)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        from sympy import isprime

        if not isprime(self.field_order):
            raise ValueError("field_order must be prime")
        if sum(self.factorization_degrees) != self.polynomial_degree:
            raise ValueError("factorization degrees must sum to polynomial degree")
        return self


class GaloisGroupRequest(StrictModel):
    """Compute the Galois group of a polynomial over Q."""

    coefficients: tuple[int, ...] = Field(min_length=2, max_length=MAX_DEGREE + 1)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if self.coefficients[-1] == 0:
            raise ValueError("leading coefficient must be nonzero")
        return self


class SolvableRequest(StrictModel):
    """Check if a polynomial is solvable by radicals."""

    coefficients: tuple[int, ...] = Field(min_length=2, max_length=MAX_DEGREE + 1)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if self.coefficients[-1] == 0:
            raise ValueError("leading coefficient must be nonzero")
        return self


# Results


class GaloisFactorResult(StrictModel):
    factors: tuple[tuple[int, ...], ...]
    factor_count: int = Field(ge=1)
    is_irreducible: bool
    method: str = "SYMPY_FACTOR_MOD_P"


class FrobeniusCycleResult(StrictModel):
    cycle_type: tuple[int, ...]
    degree: int = Field(ge=1)
    is_irreducible: bool
    method: str = "FACTOR_DEGREE_SUMMARY"


class GaloisGroupResult(StrictModel):
    group_name: str
    order: int = Field(ge=1)
    degree: int = Field(ge=1)
    is_solvable: bool
    method: str = "SYMPY_GALOIS_GROUP"


class SolvableResult(StrictModel):
    solvable_by_radicals: bool
    method: str = "GALOIS_GROUP_SOLVABILITY"
