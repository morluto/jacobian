"""Typed wire contracts for commutative algebra operations."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel

MAX_VARS = 6
MAX_GENERATORS = 32


class IdealRequest(StrictModel):
    """An ideal in a polynomial ring Q[x1,...,xn]."""

    variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_VARS)
    generators: tuple[str, ...] = Field(min_length=1, max_length=MAX_GENERATORS)


class IdealRadicalRequest(StrictModel):
    variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_VARS)
    generators: tuple[str, ...] = Field(min_length=1, max_length=MAX_GENERATORS)


class IdealQuotientRequest(StrictModel):
    variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_VARS)
    generators_a: tuple[str, ...] = Field(min_length=1, max_length=MAX_GENERATORS)
    generators_b: tuple[str, ...] = Field(min_length=1, max_length=MAX_GENERATORS)


class IdealRadicalResult(StrictModel):
    generators: tuple[str, ...]
    method: str = "GROEBNER_BASIS"


class IdealRadicalMembershipResult(StrictModel):
    in_radical: bool
    membership_witness: str = ""
    method: str = "GROEBNER_BASIS"


class IdealQuotientResult(StrictModel):
    generators: tuple[str, ...]
    method: str = "GROEBNER_BASIS"
