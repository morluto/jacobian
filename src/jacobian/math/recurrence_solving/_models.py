"""Typed wire contracts for recurrence solving."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

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
