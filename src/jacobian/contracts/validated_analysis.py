"""Typed contracts for rigorous enclosures, exact moments, and rational LPs."""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.probability import (
    FiniteDistributionAtom,
    require_input_distribution,
)
from jacobian.contracts.results import ContractModel

MAX_RATIONAL_DIGITS = 128
type RationalLinearProgramStatus = Literal[
    "CERTIFICATE_PRODUCED",
    "PRIMAL_ONLY",
    "NO_CERTIFICATE",
    "TIMEOUT",
    "BACKEND_ERROR",
]


def _require_bounded_rational(value: CanonicalRational) -> None:
    if (
        len(value.num.lstrip("-")) > MAX_RATIONAL_DIGITS
        or len(value.den) > MAX_RATIONAL_DIGITS
    ):
        raise ValueError(
            "validated-analysis rationals are limited to 128 decimal digits "
            "per numerator and denominator"
        )


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
    wall_seconds: StrictInt = Field(default=10, ge=1, le=60)

    @model_validator(mode="after")
    def bound_argument_size(self) -> Self:
        _require_bounded_rational(self.argument)
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
    conclusion: Literal["UNKNOWN"] = "UNKNOWN"
    function: RealUnaryFunction
    argument: CanonicalRational
    precision_bits: StrictInt = Field(ge=32, le=4096)
    lower: ExactDyadic | None = None
    upper: ExactDyadic | None = None
    relative_accuracy_bits: StrictInt | None = None
    exact: bool = False
    backend: Literal["python-flint"] = "python-flint"
    backend_version: Literal["0.9.0"] = "0.9.0"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_enclosure_to_status(self) -> Self:
        enclosed = self.status == "ENCLOSED"
        if enclosed != (self.lower is not None and self.upper is not None):
            raise ValueError("only an enclosed result may carry dyadic endpoints")
        if not enclosed and (self.relative_accuracy_bits is not None or self.exact):
            raise ValueError("a non-enclosure cannot claim accuracy or exactness")
        if enclosed:
            assert self.lower is not None
            assert self.upper is not None
            if self.lower.as_fraction() > self.upper.as_fraction():
                raise ValueError("enclosure lower endpoint exceeds upper endpoint")
            if self.exact != (self.relative_accuracy_bits is None):
                raise ValueError(
                    "exact enclosures omit relative accuracy; inexact ones report it"
                )
        return self


class ArbPointEnclosureObligation(ContractModel):
    obligation_type: Literal["INDEPENDENT_ENCLOSURE_REPLAY"] = (
        "INDEPENDENT_ENCLOSURE_REPLAY"
    )
    function: RealUnaryFunction
    argument: CanonicalRational
    precision_bits: StrictInt = Field(ge=32, le=4096)
    claimed_lower: ExactDyadic | None = None
    claimed_upper: ExactDyadic | None = None
    status: Literal["ENCLOSED", "NONFINITE", "TIMEOUT", "BACKEND_ERROR"]
    required_checker: Literal["AUTHORIZED_INDEPENDENT_BALL_ARITHMETIC"] = (
        "AUTHORIZED_INDEPENDENT_BALL_ARITHMETIC"
    )


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
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    backend: Literal["python-flint"] = "python-flint"
    backend_version: Literal["0.9.0"] = "0.9.0"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

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
            _require_bounded_rational(value)
        return self


class RationalLinearProgramRequest(ContractModel):
    program: StandardFormRationalLinearProgram
    wall_seconds: StrictInt = Field(default=10, ge=1, le=60)


class RationalLinearProgramResult(ContractModel):
    status: RationalLinearProgramStatus
    conclusion: Literal["UNKNOWN"] = "UNKNOWN"
    primal_candidate: tuple[CanonicalRational, ...] | None = None
    dual_candidate: tuple[CanonicalRational, ...] | None = None
    primal_objective: CanonicalRational | None = None
    dual_objective: CanonicalRational | None = None
    primal_residuals: tuple[CanonicalRational, ...] | None = None
    dual_slacks: tuple[CanonicalRational, ...] | None = None
    certificate_available: bool = False
    backend: Literal["sympy"] = "sympy"
    backend_version: Literal["1.14.0"] = "1.14.0"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_certificate_fields(self) -> Self:
        complete = self.status == "CERTIFICATE_PRODUCED"
        certificate_fields = (
            self.primal_candidate,
            self.dual_candidate,
            self.primal_objective,
            self.dual_objective,
            self.primal_residuals,
            self.dual_slacks,
        )
        if complete != (
            self.certificate_available
            and all(value is not None for value in certificate_fields)
        ):
            raise ValueError(
                "a produced certificate requires both candidates and replay data"
            )
        if self.status == "PRIMAL_ONLY":
            if (
                self.primal_candidate is None
                or self.primal_objective is None
                or self.primal_residuals is None
                or self.dual_candidate is not None
                or self.dual_objective is not None
                or self.dual_slacks is not None
            ):
                raise ValueError("a primal-only result must carry only primal data")
        elif not complete and any(value is not None for value in certificate_fields):
            raise ValueError(
                "only certificate or primal-only results may carry candidates"
            )
        return self


class RationalLinearProgramObligation(ContractModel):
    obligation_type: Literal["RATIONAL_LP_OPTIMALITY_REPLAY"] = (
        "RATIONAL_LP_OPTIMALITY_REPLAY"
    )
    program: StandardFormRationalLinearProgram
    status: RationalLinearProgramStatus
    primal_candidate: tuple[CanonicalRational, ...] | None = None
    dual_candidate: tuple[CanonicalRational, ...] | None = None
    required_checks: tuple[
        Literal[
            "PRIMAL_FEASIBILITY",
            "DUAL_FEASIBILITY",
            "OBJECTIVE_EQUALITY",
        ],
        ...,
    ] = (
        "PRIMAL_FEASIBILITY",
        "DUAL_FEASIBILITY",
        "OBJECTIVE_EQUALITY",
    )
    required_checker: Literal["AUTHORIZED_INDEPENDENT_EXACT_RATIONAL"] = (
        "AUTHORIZED_INDEPENDENT_EXACT_RATIONAL"
    )

    @model_validator(mode="after")
    def bind_candidate_dimensions_to_program(self) -> Self:
        if self.primal_candidate is not None and len(self.primal_candidate) != len(
            self.program.variables
        ):
            raise ValueError("LP primal candidate length must equal the variable count")
        if self.dual_candidate is not None and len(self.dual_candidate) != len(
            self.program.coefficients
        ):
            raise ValueError(
                "LP dual candidate length must equal the equality-constraint count"
            )
        if self.status == "CERTIFICATE_PRODUCED" and (
            self.primal_candidate is None or self.dual_candidate is None
        ):
            raise ValueError(
                "a produced LP certificate obligation requires both candidates"
            )
        if self.status not in {"CERTIFICATE_PRODUCED", "PRIMAL_ONLY"} and (
            self.primal_candidate is not None or self.dual_candidate is not None
        ):
            raise ValueError(
                "an LP non-candidate outcome cannot create candidate obligations"
            )
        return self
