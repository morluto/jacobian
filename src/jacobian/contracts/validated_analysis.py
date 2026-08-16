"""Typed contracts for rigorous enclosures, exact moments, and rational LPs."""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational, require_bounded_rational
from jacobian.contracts.probability import (
    FiniteDistributionAtom,
    require_input_distribution,
)

MAX_RATIONAL_DIGITS = 128
type RationalLinearProgramStatus = Literal[
    "OPTIMAL",
    "PRIMAL_FEASIBLE",
    "INFEASIBLE",
    "UNBOUNDED",
]


class RealUnaryFunction(StrEnum):
    EXP = "EXP"
    LOG = "LOG"
    SQRT = "SQRT"
    SIN = "SIN"
    COS = "COS"


class ArbPointEnclosureRequest(ContractModel):
    function: RealUnaryFunction
    argument: CanonicalRational
    precision_bits: StrictInt = Field(default=128, ge=32, le=4096)

    @model_validator(mode="after")
    def bound_argument_size(self) -> Self:
        require_bounded_rational(
            self.argument,
            max_digits=MAX_RATIONAL_DIGITS,
            label="validated-analysis rational",
        )
        return self


class ExactDyadic(ContractModel):
    """The exact value ``mantissa * 2**exponent``."""

    mantissa: str = Field(pattern=r"^-?(?:0|[1-9][0-9]*)$")
    exponent: StrictInt

    @model_validator(mode="after")
    def require_canonical_binary_form(self) -> Self:
        mantissa = int(self.mantissa)
        if mantissa == 0 and self.exponent != 0:
            raise ValueError("canonical dyadic zero must have exponent 0")
        if mantissa != 0 and mantissa % 2 == 0:
            raise ValueError("canonical nonzero dyadic mantissa must be odd")
        return self

    def as_fraction(self) -> Fraction:
        mantissa = Fraction(int(self.mantissa))
        if self.exponent >= 0:
            return mantissa * Fraction(2**self.exponent, 1)
        return mantissa / Fraction(2 ** (-self.exponent), 1)


class ArbPointEnclosureResult(ContractModel):
    status: Literal["ENCLOSED", "NONFINITE", "TIMEOUT", "BACKEND_ERROR"]
    function: RealUnaryFunction
    argument: CanonicalRational
    precision_bits: StrictInt = Field(ge=32, le=4096)
    lower: ExactDyadic | None = None
    upper: ExactDyadic | None = None
    relative_accuracy_bits: StrictInt | None = None
    exact: bool = False
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_enclosure_to_status(self) -> Self:
        enclosed = self.status == "ENCLOSED"
        if enclosed != (self.lower is not None and self.upper is not None):
            raise ValueError("only an enclosed result may carry dyadic endpoints")
        if not enclosed and (self.relative_accuracy_bits is not None or self.exact):
            raise ValueError("a non-enclosure cannot claim accuracy or exactness")
        if enclosed:
            lower = self.lower
            upper = self.upper
            if lower is None or upper is None:
                raise ValueError("only an enclosed result may carry dyadic endpoints")
            if lower.as_fraction() > upper.as_fraction():
                raise ValueError("enclosure lower endpoint exceeds upper endpoint")
            if self.exact != (self.relative_accuracy_bits is None):
                raise ValueError(
                    "exact enclosures omit relative accuracy; inexact ones report it"
                )
        return self


class FiniteRawMomentRequest(ContractModel):
    atoms: tuple[FiniteDistributionAtom, ...] = Field(
        min_length=1,
        max_length=256,
    )
    order: StrictInt = Field(ge=0, le=128)

    @model_validator(mode="after")
    def require_probability_distribution(self) -> Self:
        require_input_distribution(self.atoms, require_canonical=False)
        return self


class FiniteRawMomentContribution(ContractModel):
    value: CanonicalRational
    probability: CanonicalRational
    powered_value: CanonicalRational
    contribution: CanonicalRational


class FiniteRawMomentResult(ContractModel):
    order: StrictInt = Field(ge=0, le=128)
    moment: CanonicalRational
    contributions: tuple[FiniteRawMomentContribution, ...] = Field(
        min_length=1,
        max_length=256,
    )

    @model_validator(mode="after")
    def bind_exact_contributions(self) -> Self:
        total = Fraction()
        for item in self.contributions:
            expected_power = item.value.as_fraction() ** self.order
            if item.powered_value.as_fraction() != expected_power:
                raise ValueError("moment powered value does not match its atom")
            expected_contribution = item.probability.as_fraction() * expected_power
            if item.contribution.as_fraction() != expected_contribution:
                raise ValueError("moment contribution does not match its atom")
            total += expected_contribution
        if self.moment.as_fraction() != total:
            raise ValueError("moment does not equal the sum of atom contributions")
        return self


class StandardFormRationalLinearProgram(ContractModel):
    """Minimize ``cᵀx`` subject to ``Ax=b`` and ``x>=0``."""

    variables: tuple[str, ...] = Field(min_length=1, max_length=32)
    objective: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=32)
    coefficients: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1,
        max_length=64,
    )
    rhs: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_canonical_dimensions(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("linear-program variable names must be unique")
        if any(
            not name
            or len(name) > 64
            or not (name[0].isalpha() or name[0] == "_")
            or any(not (char.isalnum() or char == "_") for char in name)
            for name in self.variables
        ):
            raise ValueError("linear-program variable names must be identifiers")
        width = len(self.variables)
        if len(self.objective) != width:
            raise ValueError("objective length must equal the variable count")
        if len(self.coefficients) != len(self.rhs):
            raise ValueError("coefficient row count must equal the rhs length")
        if any(len(row) != width for row in self.coefficients):
            raise ValueError("every coefficient row must match the variable count")
        for value in (
            *self.objective,
            *self.rhs,
            *(item for row in self.coefficients for item in row),
        ):
            require_bounded_rational(
                value,
                max_digits=MAX_RATIONAL_DIGITS,
                label="validated-analysis rational",
            )
        return self


class RationalLinearProgramRequest(ContractModel):
    program: StandardFormRationalLinearProgram


class RationalLinearProgramResult(ContractModel):
    """The direct mathematical outcome of one rational linear program."""

    status: RationalLinearProgramStatus
    primal_candidate: tuple[CanonicalRational, ...] | None = None
    dual_candidate: tuple[CanonicalRational, ...] | None = None
    primal_objective: CanonicalRational | None = None
    dual_objective: CanonicalRational | None = None
    primal_residuals: tuple[CanonicalRational, ...] | None = None
    dual_slacks: tuple[CanonicalRational, ...] | None = None

    @model_validator(mode="after")
    def bind_result_fields(self) -> Self:
        optimal = self.status == "OPTIMAL"
        primal_fields = (
            self.primal_candidate,
            self.primal_objective,
            self.primal_residuals,
        )
        dual_fields = (
            self.dual_candidate,
            self.dual_objective,
            self.dual_slacks,
        )
        has_primal = self.status in {"OPTIMAL", "PRIMAL_FEASIBLE"}
        if has_primal and not all(value is not None for value in primal_fields):
            raise ValueError(
                "a primal result requires a candidate, objective, and residuals"
            )
        if not has_primal and any(value is not None for value in primal_fields):
            raise ValueError("an infeasible or unbounded result cannot carry a point")
        if optimal and not all(value is not None for value in dual_fields):
            raise ValueError(
                "an optimal result requires a dual candidate, objective, and slacks"
            )
        if not optimal and any(value is not None for value in dual_fields):
            raise ValueError("only an optimal result can carry dual data")
        return self
