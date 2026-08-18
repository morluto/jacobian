"""Typed wire contracts for integer multiplicative normal-form operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer

# Public bounds
MAX_K_VALUE = 1_000


class IntegerRequest(StrictModel):
    """One canonical integer for a multiplicative normal-form operation."""

    value: CanonicalInteger


class IntegerKRequest(IntegerRequest):
    """One canonical integer and a free parameter k >= 2."""

    k: int = Field(ge=2, le=MAX_K_VALUE)


class NonnegativeIntegerRequest(IntegerRequest):
    """One canonical nonnegative integer."""

    @model_validator(mode="after")
    def require_nonnegative(self) -> Self:
        if parse_canonical_integer(self.value) < 0:
            raise ValueError("value must be nonnegative")
        return self


class PrimeExponentRow(StrictModel):
    """One prime base and its exponent in a prime factorization."""

    prime: CanonicalInteger
    power: int = Field(ge=0)


class PerfectPowerProfileResult(StrictModel):
    """The maximal perfect-power profile of one integer."""

    kind: Literal["ZERO", "POSITIVE_UNIT", "NEGATIVE_UNIT", "NONUNIT"]
    base: CanonicalInteger | None = None
    exponent: int | None = None
    is_nontrivial_perfect_power: bool = False
    factors: tuple[PrimeExponentRow, ...] = ()
    reconstruction: CanonicalInteger | None = None

    @model_validator(mode="after")
    def require_nonunit_fields(self) -> Self:
        if self.kind == "NONUNIT" and (
            self.base is None or self.exponent is None or self.reconstruction is None
        ):
            raise ValueError("NONUNIT requires base, exponent, and reconstruction")
        return self


class KFreeDecompositionResult(StrictModel):
    """The unique decomposition n = a^k * c with c k-th-power-free."""

    kind: Literal["ZERO", "NONUNIT"]
    base: CanonicalInteger | None = None
    cofactor: CanonicalInteger | None = None
    factors: tuple[PrimeExponentRow, ...] = ()
    reconstruction: CanonicalInteger | None = None

    @model_validator(mode="after")
    def require_nonunit_fields(self) -> Self:
        if self.kind == "NONUNIT" and (
            self.base is None or self.cofactor is None or self.reconstruction is None
        ):
            raise ValueError("NONUNIT requires base, cofactor, and reconstruction")
        return self


class SquarefreeDecompositionResult(StrictModel):
    """The unique decomposition n = s^2 * d with |d| squarefree."""

    kind: Literal["ZERO", "NONUNIT"]
    square_factor: CanonicalInteger | None = None
    squarefree_part: CanonicalInteger | None = None
    factors: tuple[PrimeExponentRow, ...] = ()
    reconstruction: CanonicalInteger | None = None

    @model_validator(mode="after")
    def require_nonunit_fields(self) -> Self:
        if self.kind == "NONUNIT" and (
            self.square_factor is None
            or self.squarefree_part is None
            or self.reconstruction is None
        ):
            raise ValueError(
                "NONUNIT requires square_factor, squarefree_part, and reconstruction"
            )
        return self


class NormalizedQuadraticRadicalResult(StrictModel):
    """The canonical positive sqrt(n) = s * sqrt(d) with d squarefree."""

    kind: Literal["ZERO", "RATIONAL_INTEGER", "IRRATIONAL_QUADRATIC"]
    coefficient: CanonicalInteger
    radicand: CanonicalInteger
    reconstruction: CanonicalInteger
