"""Typed wire contracts for recurrence solving."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

MAX_RATIONAL_DIGITS = 256


def _require_rationals(values: tuple[CanonicalRational, ...], *, label: str) -> None:
    for value in values:
        require_bounded_rational(value, max_digits=MAX_RATIONAL_DIGITS, label=label)


class RecurrenceFindRequest(StrictModel):
    """Find the minimal linear recurrence of a sequence over QQ."""

    sequence: tuple[CanonicalRational, ...] = Field(min_length=2, max_length=256)

    @model_validator(mode="after")
    def require_rational_sequence(self) -> Self:
        _require_rationals(self.sequence, label="sequence value")
        return self


class RecurrenceFindResult(StrictModel):
    """A fitted recurrence or an explicit finite-prefix missing outcome."""

    coefficients: tuple[CanonicalRational, ...] = Field(max_length=255)
    order: int = Field(ge=0, le=255)
    status: Literal["FOUND", "NO_FITTING_RECURRENCE"]
    method: Literal["RATIONAL_INTERPOLATION"] = "RATIONAL_INTERPOLATION"

    @model_validator(mode="after")
    def require_status_consistent_coefficients(self) -> Self:
        if self.status == "FOUND":
            if self.order == 0 or len(self.coefficients) != self.order:
                raise ValueError(
                    "a found recurrence must have one coefficient per order"
                )
        elif self.order != 0 or self.coefficients:
            raise ValueError(
                "a missing recurrence must have zero order and no coefficients"
            )
        return self


class ClosedFormRequest(StrictModel):
    """Compute a SymPy-expression closed form for a recurrence of degree at most four."""

    characteristic_coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=5,
        description="Characteristic polynomial coefficients in descending order, with degree at most four.",
    )
    initial_values: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def require_initial_values_for_order(self) -> Self:
        order = len(self.characteristic_coefficients) - 1
        if order < 1:
            raise ValueError("characteristic polynomial must have positive degree")
        if len(self.initial_values) != order:
            raise ValueError("initial value count must match the recurrence order")
        _require_rationals(
            self.characteristic_coefficients, label="characteristic coefficient"
        )
        _require_rationals(self.initial_values, label="initial value")
        if self.characteristic_coefficients[0].as_fraction() == 0:
            raise ValueError(
                "characteristic polynomial must have nonzero leading coefficient"
            )
        return self


class ClosedFormResult(StrictModel):
    """The closed-form solution as a SymPy expression string."""

    expression: str
    method: Literal["SYMPY_RSOLVE"] = "SYMPY_RSOLVE"


# ---------------------------------------------------------------------------
# Berlekamp-Massey over an explicit prime field
# ---------------------------------------------------------------------------

_MAX_FIELD_SEQUENCE_LENGTH = 256
_MAX_FIELD_PRIME = 10_000


def _require_bounded_prime(prime: int) -> None:
    if not 2 <= prime <= _MAX_FIELD_PRIME:
        raise ValueError(
            f"prime must be a prime number between 2 and {_MAX_FIELD_PRIME}"
        )
    from sympy import isprime

    if not isprime(prime):
        raise ValueError("prime must be a prime integer")


def _require_canonical_residues(
    values: tuple[int, ...], prime: int, label: str
) -> None:
    for value in values:
        if type(value) is not int or not 0 <= value < prime:
            raise ValueError(f"{label} must be canonical residues modulo the prime")


class PrimeFieldRecurrenceFindRequest(StrictModel):
    """Find the minimal linear recurrence of a sequence over ``GF(p)``."""

    prime: StrictInt = Field(ge=2, le=_MAX_FIELD_PRIME)
    sequence: tuple[StrictInt, ...] = Field(
        min_length=2,
        max_length=_MAX_FIELD_SEQUENCE_LENGTH,
    )

    @model_validator(mode="after")
    def require_valid_field_sequence(self) -> Self:
        _require_bounded_prime(self.prime)
        _require_canonical_residues(self.sequence, self.prime, "sequence values")
        return self


class PrimeFieldRecurrenceFindResult(StrictModel):
    """The minimal LFSR over ``GF(p)`` found by Berlekamp-Massey."""

    prime: StrictInt = Field(ge=2, le=_MAX_FIELD_PRIME)
    sequence: tuple[StrictInt, ...] = Field(
        min_length=2,
        max_length=_MAX_FIELD_SEQUENCE_LENGTH,
    )
    coefficients: tuple[StrictInt, ...] = Field(max_length=_MAX_FIELD_SEQUENCE_LENGTH)
    order: StrictInt = Field(ge=0, le=_MAX_FIELD_SEQUENCE_LENGTH)
    status: Literal["FOUND", "NO_FITTING_RECURRENCE"]
    method: Literal["BERLEKAMP_MASSEY"] = "BERLEKAMP_MASSEY"

    @model_validator(mode="after")
    def require_status_consistent_coefficients(self) -> Self:
        _require_bounded_prime(self.prime)
        _require_canonical_residues(self.sequence, self.prime, "sequence values")
        from jacobian.math.recurrence_solving.operations import berlekamp_massey

        expected = berlekamp_massey(list(self.sequence), self.prime)
        if (
            self.prime != expected.prime
            or self.status != expected.status
            or self.order != expected.order
            or self.coefficients != expected.coefficients
        ):
            raise ValueError(
                "result must match the exact bound Berlekamp-Massey recurrence"
            )
        return self
