"""Typed wire contracts for exact shift Ore operators over QQ(n)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.polynomials.values import RationalFunction

MAX_SHIFT_ORDER = 16
MAX_SHIFT_TERMS = 16
MAX_SHIFT_COEFFICIENT_TERMS = 64
MAX_SHIFT_COEFFICIENT_DEGREE = 64
MAX_SHIFT_COEFFICIENT_DIGITS = 64
MAX_SHIFT_RESULT_ORDER = 32
MAX_SHIFT_RESULT_DEGREE = 128
MAX_SHIFT_RESULT_DIGITS = 256
MAX_SHIFT_LEDGER_ROWS = 256


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by Ore-algebra contracts."""

    return PydanticCustomError(f"ore_algebra.{reason}", message)


class ShiftOreTerm(StrictModel):
    """One nonzero left-coefficient monomial p(n) S^exponent."""

    exponent: StrictInt = Field(ge=0, le=MAX_SHIFT_ORDER)
    coefficient: RationalFunction = Field(
        description=(
            "Exact coefficient in QQ(n) with variable axis ('n',); "
            "it must be a nonzero reduced presentation."
        )
    )

    @model_validator(mode="after")
    def require_shift_coefficient_shape(self) -> Self:
        if self.coefficient.variables != ("n",):
            raise _validation_error(
                "coefficient_axis",
                "shift-operator coefficients must use the variable axis ('n',)",
            )
        if not self.coefficient.numerator.terms:
            raise _validation_error(
                "zero_coefficient", "operator terms must have nonzero coefficients"
            )
        return self


class ShiftOreOperator(StrictModel):
    """One shift operator in left-coefficient normal form sum_i p_i(n) S^i."""

    variable: Literal["n"] = "n"
    terms: tuple[ShiftOreTerm, ...] = Field(
        min_length=0,
        max_length=MAX_SHIFT_TERMS,
        description=(
            "Sparse operator terms in strictly increasing exponent order; "
            "the empty family is the canonical zero operator."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_term_order(self) -> Self:
        exponents = tuple(term.exponent for term in self.terms)
        if tuple(sorted(exponents)) != exponents or len(set(exponents)) != len(
            exponents
        ):
            raise _validation_error(
                "term_order",
                "operator terms must use strictly increasing exponent order",
            )
        return self

    @property
    def order(self) -> int:
        """Highest shift exponent, or -1 for the zero operator."""

        return max((term.exponent for term in self.terms), default=-1)


class ShiftOperatorMultiplyRequest(StrictModel):
    """Two shift operators over QQ(n) to multiply noncommutatively."""

    left: ShiftOreOperator
    right: ShiftOreOperator


class ShiftMultiplyLedgerRow(StrictModel):
    """One exact (i, j) contribution of the shift product rule."""

    left_exponent: StrictInt = Field(ge=0, le=MAX_SHIFT_ORDER)
    right_exponent: StrictInt = Field(ge=0, le=MAX_SHIFT_ORDER)
    result_exponent: StrictInt = Field(ge=0, le=MAX_SHIFT_RESULT_ORDER)
    shifted_coefficient: RationalFunction = Field(
        description="Exact sigma^left_exponent applied to the right coefficient."
    )
    contribution: RationalFunction = Field(
        description="Exact left coefficient times the shifted right coefficient."
    )

    @model_validator(mode="after")
    def require_exponent_transport(self) -> Self:
        if self.result_exponent != self.left_exponent + self.right_exponent:
            raise _validation_error(
                "exponent_transport",
                "ledger result exponents must add the consumed exponents",
            )
        return self


class ShiftOperatorMultiplyResult(StrictModel):
    """The exact noncommutative shift-operator product with its ledger."""

    left: ShiftOreOperator
    right: ShiftOreOperator
    product: ShiftOreOperator
    ledger: tuple[ShiftMultiplyLedgerRow, ...] = Field(
        max_length=MAX_SHIFT_LEDGER_ROWS,
        description=(
            "One row per nonzero (left, right) term pair, in left-then-right "
            "exponent order; rows sum exactly to the product."
        ),
    )

    @model_validator(mode="after")
    def require_ledger_coverage(self) -> Self:
        expected = tuple(
            (left.exponent, right.exponent)
            for left in self.left.terms
            for right in self.right.terms
        )
        observed = tuple((row.left_exponent, row.right_exponent) for row in self.ledger)
        if observed != expected:
            raise _validation_error(
                "ledger_coverage",
                "the ledger must cover every nonzero term pair exactly once",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        left: ShiftOreOperator,
        right: ShiftOreOperator,
        *,
        product: ShiftOreOperator,
        ledger: tuple[ShiftMultiplyLedgerRow, ...],
    ) -> Self:
        return cls.model_construct(
            left=left,
            right=right,
            product=product,
            ledger=ledger,
        )
