"""Typed wire contracts for recurrence solving."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel


class RecurrenceFindRequest(StrictModel):
    """Find the minimal linear recurrence of a sequence over QQ."""

    sequence: tuple[str, ...] = Field(min_length=2, max_length=256)

    @model_validator(mode="after")
    def require_rational_sequence(self) -> Self:
        from sympy import Rational

        try:
            tuple(Rational(value) for value in self.sequence)
        except (TypeError, ValueError) as exc:
            raise ValueError("sequence values must be rational numbers") from exc
        return self


class RecurrenceFindResult(StrictModel):
    """A fitted recurrence or an explicit finite-prefix missing outcome."""

    coefficients: tuple[str, ...] = Field(max_length=255)
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

    characteristic_coefficients: tuple[str, ...] = Field(
        min_length=1,
        max_length=5,
        description="Characteristic polynomial coefficients in descending order, with degree at most four.",
    )
    initial_values: tuple[str, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def require_initial_values_for_order(self) -> Self:
        from sympy import Rational

        order = len(self.characteristic_coefficients) - 1
        if order < 1:
            raise ValueError("characteristic polynomial must have positive degree")
        if len(self.initial_values) != order:
            raise ValueError("initial value count must match the recurrence order")
        try:
            coefficients = tuple(
                Rational(value) for value in self.characteristic_coefficients
            )
            tuple(Rational(value) for value in self.initial_values)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "recurrence coefficients and initial values must be rational"
            ) from exc
        if coefficients[0] == 0:
            raise ValueError(
                "characteristic polynomial must have nonzero leading coefficient"
            )
        return self


class ClosedFormResult(StrictModel):
    """The closed-form solution as a SymPy expression string."""

    expression: str
    method: Literal["SYMPY_RSOLVE"] = "SYMPY_RSOLVE"
