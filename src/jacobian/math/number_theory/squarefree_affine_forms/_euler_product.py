"""Contracts for the finite square-free Euler-product prefix operation."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, StrictBool, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.number_theory.squarefree_affine_forms._models import (
    MAX_EULER_PRIMES,
    MAX_LOCAL_FACTOR_PRIME,
)
from jacobian.math.number_theory.squarefree_affine_forms.values import (
    SquarefreeAffineFamily,
)


def _invariant_error(message: str) -> PydanticCustomError:
    return PydanticCustomError(
        "number_theory.squarefree_affine.product_invariant", message
    )


class SquarefreeEulerProductRequest(StrictModel):
    """Compute one exact finite Euler-product prefix over a bounded prime set."""

    source: SquarefreeAffineFamily
    primes: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_EULER_PRIMES,
        description=(
            "Distinct primes in strictly increasing order, each at most "
            f"{MAX_LOCAL_FACTOR_PRIME}. Primality and the aggregate p^2 work "
            "envelope are admitted at execution."
        ),
    )


class SquarefreeLocalFactorRow(StrictModel):
    """Compact exact local-factor row for one supplied prime."""

    prime: StrictInt = Field(ge=2, le=MAX_LOCAL_FACTOR_PRIME)
    modulus: StrictInt = Field(ge=4)
    bad_count: StrictInt = Field(ge=0)
    valid_count: StrictInt = Field(ge=0)
    local_factor: CanonicalRational
    has_local_obstruction: StrictBool

    @model_validator(mode="after")
    def require_partition_and_factor(self) -> Self:
        if self.modulus != self.prime * self.prime:
            raise _invariant_error("row modulus must equal the square of its prime")
        if self.bad_count + self.valid_count != self.modulus:
            raise _invariant_error(
                "valid and bad counts must partition every residue modulo p^2"
            )
        if (
            self.local_factor.as_integer_ratio()
            != Fraction(self.valid_count, self.modulus).as_integer_ratio()
        ):
            raise _invariant_error("row factor must equal valid_count / p^2 reduced")
        if self.has_local_obstruction != (self.valid_count == 0):
            raise _invariant_error("row obstruction flag must equal valid_count == 0")
        return self


class SquarefreeEulerProductResult(StrictModel):
    """Exact finite Euler-product prefix; explicitly not an infinite product."""

    source: SquarefreeAffineFamily
    primes: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_EULER_PRIMES)
    rows: tuple[SquarefreeLocalFactorRow, ...] = Field(
        min_length=1, max_length=MAX_EULER_PRIMES
    )
    product: CanonicalRational
    has_local_obstruction: StrictBool
    first_obstructing_prime: StrictInt | None = None

    @model_validator(mode="after")
    def require_exact_finite_product(self) -> Self:
        if tuple(row.prime for row in self.rows) != self.primes:
            raise _invariant_error(
                "local-factor rows must align with the canonical prime set"
            )
        aggregate = Fraction(1, 1)
        first_obstruction: int | None = None
        for row in self.rows:
            aggregate *= Fraction(*row.local_factor.as_integer_ratio())
            if row.has_local_obstruction and first_obstruction is None:
                first_obstruction = row.prime
        if self.product.as_integer_ratio() != aggregate.as_integer_ratio():
            raise _invariant_error(
                "aggregate product must equal the exact product of the "
                "per-prime local factors"
            )
        if self.has_local_obstruction != any(
            row.has_local_obstruction for row in self.rows
        ):
            raise _invariant_error(
                "aggregate obstruction flag must equal one obstructed row"
            )
        if self.first_obstructing_prime != first_obstruction:
            raise _invariant_error(
                "first obstructing prime must be the least prime with zero "
                "valid residues"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: SquarefreeAffineFamily,
        primes: tuple[int, ...],
        rows: tuple[SquarefreeLocalFactorRow, ...],
    ) -> Self:
        aggregate = Fraction(1, 1)
        first_obstruction: int | None = None
        for row in rows:
            aggregate *= Fraction(*row.local_factor.as_integer_ratio())
            if row.valid_count == 0 and first_obstruction is None:
                first_obstruction = row.prime
        return cls.model_construct(
            source=source,
            primes=primes,
            rows=rows,
            product=CanonicalRational.from_fraction(aggregate),
            has_local_obstruction=first_obstruction is not None,
            first_obstructing_prime=first_obstruction,
        )


__all__ = [
    "SquarefreeEulerProductRequest",
    "SquarefreeEulerProductResult",
    "SquarefreeLocalFactorRow",
]
