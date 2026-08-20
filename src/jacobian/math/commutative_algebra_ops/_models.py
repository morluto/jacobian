"""Typed wire contracts for commutative algebra operations."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel

MAX_VARS = 6
MAX_GENERATORS = 32


class IdealRadicalRequest(StrictModel):
    """Request for radical of an ideal I = <generators> in Q[variables]."""

    variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_VARS)
    generators: tuple[str, ...] = Field(min_length=1, max_length=MAX_GENERATORS)


class IdealRadicalMembershipRequest(StrictModel):
    """Request for checking membership of a polynomial in the radical of an ideal."""

    variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_VARS)
    generators: tuple[str, ...] = Field(min_length=1, max_length=MAX_GENERATORS)
    polynomial: str = Field(min_length=1)


class IdealQuotientRequest(StrictModel):
    """Request for the ideal quotient (I : J) in Q[variables]."""

    variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_VARS)
    generators_a: tuple[str, ...] = Field(min_length=1, max_length=MAX_GENERATORS)
    generators_b: tuple[str, ...] = Field(min_length=1, max_length=MAX_GENERATORS)


class IdealRadicalResult(StrictModel):
    generators: tuple[str, ...]
    method: str = "GROEBNER_BASIS"


class IdealRadicalMembershipResult(StrictModel):
    in_radical: bool
    method: str = "RABINOWITSCH"


class IdealQuotientResult(StrictModel):
    generators: tuple[str, ...]
    method: str = "GROEBNER_BASIS"
