"""Exact residuals for caller-supplied P-recursive tables."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.combinatorics._models import (
    MAX_COMBINATORICS_INPUT_RATIONAL_DIGITS,
    MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
    MAX_LINEAR_RECURRENCE_INDEX,
    MAX_LINEAR_RECURRENCE_ORDER,
    MAX_P_RECURSIVE_POLYNOMIAL_DEGREE,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"combinatorics.{reason}", message)


class IndexedRecurrenceResidual(StrictModel):
    index: StrictInt = Field(ge=1, le=MAX_LINEAR_RECURRENCE_INDEX)
    value: CanonicalRational


class PolynomialCoefficientRecurrenceTableRequest(StrictModel):
    """One complete finite table and one polynomial-coefficient recurrence."""

    coefficient_polynomials: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=2, max_length=MAX_LINEAR_RECURRENCE_ORDER + 1
    )
    values: tuple[CanonicalRational, ...] = Field(
        min_length=2, max_length=MAX_LINEAR_RECURRENCE_INDEX + 1
    )
    coefficient_convention: Literal[
        "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
    ]
    polynomial_convention: Literal["ASCENDING_POWERS_OF_N"]
    table_convention: Literal["VALUES_A_0_THROUGH_A_N_IN_ORDER"]

    @model_validator(mode="after")
    def require_complete_bounded_table(self) -> Self:
        order = len(self.coefficient_polynomials) - 1
        if len(self.values) <= order:
            raise ValueError(
                "values must include the initial range and at least one checked step"
            )
        for polynomial in self.coefficient_polynomials:
            if (
                not polynomial
                or len(polynomial) > MAX_P_RECURSIVE_POLYNOMIAL_DEGREE + 1
            ):
                raise ValueError("coefficient polynomial degree is outside the bound")
            if polynomial[-1].as_fraction() == 0:
                raise ValueError("coefficient polynomial must omit trailing zero terms")
            for coefficient in polynomial:
                try:
                    require_bounded_rational(
                        coefficient,
                        max_digits=MAX_COMBINATORICS_INPUT_RATIONAL_DIGITS,
                        label="recurrence polynomial coefficient",
                    )
                except ValueError as exc:
                    raise _validation_error("recurrence_invariant", str(exc)) from None
        for value in self.values:
            try:
                require_bounded_rational(
                    value,
                    max_digits=MAX_COMBINATORICS_INPUT_RATIONAL_DIGITS,
                    label="submitted recurrence table value",
                )
            except ValueError as exc:
                raise _validation_error("recurrence_invariant", str(exc)) from None
        return self


class PolynomialCoefficientRecurrenceTableResult(StrictModel):
    """Complete exact residual ledger for the supplied finite table."""

    coefficient_convention: Literal[
        "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
    ]
    polynomial_convention: Literal["ASCENDING_POWERS_OF_N"]
    table_convention: Literal["VALUES_A_0_THROUGH_A_N_IN_ORDER"]
    recurrence_order: StrictInt = Field(ge=1, le=MAX_LINEAR_RECURRENCE_ORDER)
    term_count: StrictInt = Field(ge=2, le=MAX_LINEAR_RECURRENCE_INDEX + 1)
    residuals: tuple[IndexedRecurrenceResidual, ...] = Field(
        min_length=1, max_length=MAX_LINEAR_RECURRENCE_INDEX
    )
    satisfies_recurrence: StrictBool
    first_failure_index: StrictInt | None = Field(
        default=None, ge=1, le=MAX_LINEAR_RECURRENCE_INDEX
    )

    @model_validator(mode="after")
    def require_complete_consistent_ledger(self) -> Self:
        expected = tuple(range(self.recurrence_order, self.term_count))
        if tuple(item.index for item in self.residuals) != expected:
            raise ValueError("residuals must cover every checked table index")
        failures = tuple(
            item.index for item in self.residuals if item.value.as_fraction() != 0
        )
        if self.satisfies_recurrence != (not failures):
            raise ValueError("satisfies_recurrence must agree with residuals")
        if self.first_failure_index != (failures[0] if failures else None):
            raise ValueError("first_failure_index must identify the first failure")
        return self


def _evaluate(polynomial: tuple[Fraction, ...], index: int) -> Fraction:
    return sum(
        (coefficient * index**power for power, coefficient in enumerate(polynomial)),
        Fraction(),
    )


def recurrence_table_residuals(
    request: PolynomialCoefficientRecurrenceTableRequest,
) -> PolynomialCoefficientRecurrenceTableResult:
    polynomials = tuple(
        tuple(value.as_fraction() for value in polynomial)
        for polynomial in request.coefficient_polynomials
    )
    values = tuple(value.as_fraction() for value in request.values)
    order = len(polynomials) - 1
    residuals = []
    failures = []
    for index in range(order, len(values)):
        residual = sum(
            (
                _evaluate(polynomials[offset], index) * values[index - offset]
                for offset in range(order + 1)
            ),
            Fraction(),
        )
        wire = CanonicalRational.from_fraction(residual)
        require_bounded_rational(
            wire,
            max_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
            label="submitted recurrence residual",
        )
        residuals.append(IndexedRecurrenceResidual(index=index, value=wire))
        if residual:
            failures.append(index)
    return PolynomialCoefficientRecurrenceTableResult(
        coefficient_convention=request.coefficient_convention,
        polynomial_convention=request.polynomial_convention,
        table_convention=request.table_convention,
        recurrence_order=order,
        term_count=len(values),
        residuals=tuple(residuals),
        satisfies_recurrence=not failures,
        first_failure_index=failures[0] if failures else None,
    )


__all__ = [
    "IndexedRecurrenceResidual",
    "PolynomialCoefficientRecurrenceTableRequest",
    "PolynomialCoefficientRecurrenceTableResult",
    "recurrence_table_residuals",
]
