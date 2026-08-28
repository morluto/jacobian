"""Typed wire contracts for integer multiplicative normal-form operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer

# Public bounds
MAX_K_VALUE = 1_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by multiplicative arithmetic."""

    return PydanticCustomError(f"arithmetic.{reason}", message)


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
            raise _validation_error(
                "value_must_be_nonnegative", "value must be nonnegative"
            )
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
    def require_consistent_fields(self) -> Self:
        if self.kind == "NONUNIT":
            if (
                self.base is None
                or self.exponent is None
                or self.reconstruction is None
            ):
                raise _validation_error(
                    "perfect_power_missing_fields",
                    "NONUNIT requires base, exponent, and reconstruction",
                )
            if self.exponent < 1:
                raise _validation_error(
                    "perfect_power_exponent_invalid", "NONUNIT exponent must be >= 1"
                )
        else:
            # ZERO, POSITIVE_UNIT, NEGATIVE_UNIT must not carry NONUNIT fields
            if self.base is not None or self.exponent is not None:
                raise _validation_error(
                    "perfect_power_unexpected_base_exponent",
                    f"{self.kind} must not carry base or exponent",
                )
            if self.factors:
                raise _validation_error(
                    "perfect_power_unexpected_factors",
                    f"{self.kind} must not carry factors",
                )
            if self.is_nontrivial_perfect_power:
                raise _validation_error(
                    "perfect_power_unexpected_claim",
                    f"{self.kind} must not claim a nontrivial perfect power",
                )
        return self


class KFreeDecompositionResult(StrictModel):
    """The unique decomposition n = a^k * c with c k-th-power-free."""

    kind: Literal["ZERO", "UNIT", "NONUNIT"]
    base: CanonicalInteger | None = None
    cofactor: CanonicalInteger | None = None
    factors: tuple[PrimeExponentRow, ...] = ()
    reconstruction: CanonicalInteger | None = None

    @model_validator(mode="after")
    def require_consistent_fields(self) -> Self:
        if self.kind == "NONUNIT":
            if (
                self.base is None
                or self.cofactor is None
                or self.reconstruction is None
            ):
                raise _validation_error(
                    "k_free_missing_fields",
                    "NONUNIT requires base, cofactor, and reconstruction",
                )
        elif self.kind == "UNIT":
            if self.base is not None or self.cofactor is not None:
                raise _validation_error(
                    "k_free_unexpected_base_cofactor",
                    "UNIT must not carry base or cofactor",
                )
            if self.factors:
                raise _validation_error(
                    "k_free_unexpected_factors", "UNIT must not carry factors"
                )
        else:  # ZERO
            if self.base is not None or self.cofactor is not None:
                raise _validation_error(
                    "k_free_unexpected_base_cofactor",
                    "ZERO must not carry base or cofactor",
                )
            if self.factors:
                raise _validation_error(
                    "k_free_unexpected_factors", "ZERO must not carry factors"
                )
        return self


class SquarefreeDecompositionResult(StrictModel):
    """The unique decomposition n = s^2 * d with |d| squarefree."""

    kind: Literal["ZERO", "UNIT", "NONUNIT"]
    square_factor: CanonicalInteger | None = None
    squarefree_part: CanonicalInteger | None = None
    factors: tuple[PrimeExponentRow, ...] = ()
    reconstruction: CanonicalInteger | None = None

    @model_validator(mode="after")
    def require_consistent_fields(self) -> Self:
        if self.kind == "NONUNIT":
            if (
                self.square_factor is None
                or self.squarefree_part is None
                or self.reconstruction is None
            ):
                raise _validation_error(
                    "squarefree_missing_fields",
                    "NONUNIT requires square_factor, squarefree_part, and reconstruction",
                )
        elif self.kind == "UNIT":
            if self.square_factor is not None or self.squarefree_part is not None:
                raise _validation_error(
                    "squarefree_unexpected_parts",
                    "UNIT must not carry square_factor or squarefree_part",
                )
            if self.factors:
                raise _validation_error(
                    "squarefree_unexpected_factors", "UNIT must not carry factors"
                )
        else:  # ZERO
            if self.square_factor is not None or self.squarefree_part is not None:
                raise _validation_error(
                    "squarefree_unexpected_parts",
                    "ZERO must not carry square_factor or squarefree_part",
                )
            if self.factors:
                raise _validation_error(
                    "squarefree_unexpected_factors", "ZERO must not carry factors"
                )
        return self


class NormalizedQuadraticRadicalResult(StrictModel):
    """The canonical positive sqrt(n) = s * sqrt(d) with d squarefree."""

    kind: Literal["ZERO", "RATIONAL_INTEGER", "IRRATIONAL_QUADRATIC"]
    coefficient: CanonicalInteger
    radicand: CanonicalInteger
    reconstruction: CanonicalInteger
