"""Typed wire contracts for commutative algebra operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    require_polynomial_budget,
)

MAX_VARS = 6
MAX_GENERATORS = 16
MAX_INPUT_TERMS = 256
MAX_INPUT_EXPONENT = 12
MAX_COEFFICIENT_DIGITS = 128
MAX_OUTPUT_GENERATORS = 64
MAX_OUTPUT_TERMS = 1024


class IdealComputationBudget(StrictModel):
    """Enforced wall-time and exact-result limits for one Singular call."""

    wall_seconds: StrictInt = Field(default=10, ge=1, le=60)
    maximum_output_generators: StrictInt = Field(
        default=MAX_OUTPUT_GENERATORS,
        ge=MAX_OUTPUT_GENERATORS,
        le=MAX_OUTPUT_GENERATORS,
    )
    maximum_output_terms: StrictInt = Field(
        default=MAX_OUTPUT_TERMS,
        ge=MAX_OUTPUT_TERMS,
        le=MAX_OUTPUT_TERMS,
    )


def _require_ideal_budget(ideal: RationalPolynomialIdeal, *, label: str) -> None:
    if len(ideal.variables) > MAX_VARS:
        raise ValueError(f"{label} exceeds the {MAX_VARS}-variable operation budget")
    if len(ideal.generators) > MAX_GENERATORS:
        raise ValueError(
            f"{label} exceeds the {MAX_GENERATORS}-generator operation budget"
        )
    if (
        sum(len(generator.polynomial.terms) for generator in ideal.generators)
        > MAX_INPUT_TERMS
    ):
        raise ValueError(
            f"{label} exceeds the {MAX_INPUT_TERMS}-term aggregate input budget"
        )
    for generator in ideal.generators:
        require_polynomial_budget(
            generator,
            maximum_terms=MAX_INPUT_TERMS,
            maximum_exponent=MAX_INPUT_EXPONENT,
            maximum_coefficient_digits=MAX_COEFFICIENT_DIGITS,
            label=f"{label} generator",
        )
        if any(
            sum(term.exponents) > MAX_INPUT_EXPONENT
            for term in generator.polynomial.terms
        ):
            raise ValueError(
                f"{label} generator exceeds total degree {MAX_INPUT_EXPONENT}"
            )


class IdealRadicalRequest(StrictModel):
    """Compute ``sqrt(I)`` for a bounded ideal ``I`` in ``QQ[variables]``."""

    ideal: RationalPolynomialIdeal = Field(
        description=(
            "An ideal in at most 6 variables with at most 16 generators and "
            "256 aggregate terms; generator total degree is at most 12 and "
            "coefficient components are at most 128 digits."
        )
    )
    resource_budget: IdealComputationBudget = Field(
        default_factory=IdealComputationBudget
    )

    @model_validator(mode="after")
    def require_backend_domain(self) -> Self:
        _require_ideal_budget(self.ideal, label="ideal")
        return self


class IdealRadicalMembershipRequest(StrictModel):
    """Check membership of one polynomial in the radical of a bounded ideal."""

    ideal: RationalPolynomialIdeal = Field(
        description=(
            "An ideal in at most 6 variables with at most 16 generators and "
            "256 aggregate terms; generator total degree is at most 12 and "
            "coefficient components are at most 128 digits."
        )
    )
    polynomial: RationalPolynomial = Field(
        description=(
            "A polynomial in the ideal's exact ordered ring, with at most 256 "
            "terms, total degree at most 12, and coefficient components at most "
            "128 digits."
        )
    )

    @model_validator(mode="after")
    def require_backend_domain(self) -> Self:
        _require_ideal_budget(self.ideal, label="ideal")
        if self.polynomial.variables != self.ideal.variables:
            raise ValueError("membership polynomial must use the ideal's ordered ring")
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=MAX_INPUT_TERMS,
            maximum_exponent=MAX_INPUT_EXPONENT,
            maximum_coefficient_digits=MAX_COEFFICIENT_DIGITS,
            label="membership polynomial",
        )
        if any(
            sum(term.exponents) > MAX_INPUT_EXPONENT
            for term in self.polynomial.polynomial.terms
        ):
            raise ValueError(
                f"membership polynomial exceeds total degree {MAX_INPUT_EXPONENT}"
            )
        return self


class IdealQuotientRequest(StrictModel):
    """Compute ``(I : J)`` for bounded ideals in one ``QQ`` ring."""

    dividend: RationalPolynomialIdeal = Field(
        description=(
            "An ideal in at most 6 variables with at most 16 generators and "
            "256 aggregate terms; generator total degree is at most 12 and "
            "coefficient components are at most 128 digits."
        )
    )
    divisor: RationalPolynomialIdeal = Field(
        description=(
            "An ideal in the dividend's exact ordered ring, with the same "
            "6-variable, 16-generator, 256-term, degree-12, and 128-digit bounds."
        )
    )
    resource_budget: IdealComputationBudget = Field(
        default_factory=IdealComputationBudget
    )

    @model_validator(mode="after")
    def require_backend_domain(self) -> Self:
        _require_ideal_budget(self.dividend, label="dividend ideal")
        _require_ideal_budget(self.divisor, label="divisor ideal")
        if self.dividend.variables != self.divisor.variables:
            raise ValueError("ideal quotient operands must use the same ordered ring")
        return self


IdealExecutionOutcome = Literal[
    "COMPUTED", "UNAVAILABLE", "TIMEOUT", "LIMIT_EXCEEDED", "ERROR"
]


class IdealRadicalResult(StrictModel):
    outcome: IdealExecutionOutcome
    radical: RationalPolynomialIdeal | None = None
    method: Literal["SINGULAR_RADICAL"] = "SINGULAR_RADICAL"
    backend_version: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        if self.outcome == "COMPUTED":
            if self.radical is None or self.backend_version is None or self.detail:
                raise ValueError(
                    "computed radical requires a value and backend version"
                )
        elif (
            self.radical is not None
            or self.backend_version is not None
            or not self.detail
        ):
            raise ValueError("failed radical computation requires only a safe detail")
        return self


class IdealRadicalMembershipResult(StrictModel):
    in_radical: bool
    method: Literal["RABINOWITSCH"] = "RABINOWITSCH"


class IdealQuotientResult(StrictModel):
    outcome: IdealExecutionOutcome
    quotient: RationalPolynomialIdeal | None = None
    method: Literal["SINGULAR_QUOTIENT"] = "SINGULAR_QUOTIENT"
    backend_version: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        if self.outcome == "COMPUTED":
            if self.quotient is None or self.backend_version is None or self.detail:
                raise ValueError(
                    "computed quotient requires a value and backend version"
                )
        elif (
            self.quotient is not None
            or self.backend_version is not None
            or not self.detail
        ):
            raise ValueError("failed quotient computation requires only a safe detail")
        return self


class IdealSaturationRequest(StrictModel):
    """Compute 'I : <d>^infinity' for a bounded ideal and a polynomial."""

    ideal: RationalPolynomialIdeal = Field(
        description=(
            "An ideal in at most 6 variables with at most 16 generators and "
            "256 aggregate terms; generator total degree is at most 12 and "
            "coefficient components are at most 128 digits."
        )
    )
    saturation_polynomial: RationalPolynomial = Field(
        description=(
            "A single polynomial d in the ideal ring, with "
            "at most 256 terms, total degree at most 12, and coefficient "
            "components at most 128 digits."
        )
    )
    resource_budget: IdealComputationBudget = Field(
        default_factory=IdealComputationBudget
    )

    @model_validator(mode="after")
    def require_backend_domain(self) -> Self:
        _require_ideal_budget(self.ideal, label="ideal")
        require_polynomial_budget(
            self.saturation_polynomial,
            maximum_terms=MAX_INPUT_TERMS,
            maximum_exponent=MAX_INPUT_EXPONENT,
            maximum_coefficient_digits=MAX_COEFFICIENT_DIGITS,
            label="saturation polynomial",
        )
        if self.saturation_polynomial.variables != self.ideal.variables:
            raise ValueError("saturation polynomial must use the ideal's ordered ring")
        if any(
            sum(term.exponents) > MAX_INPUT_EXPONENT
            for term in self.saturation_polynomial.polynomial.terms
        ):
            raise ValueError(
                f"saturation polynomial exceeds total degree {MAX_INPUT_EXPONENT}"
            )
        return self


class IdealSaturationResult(StrictModel):
    outcome: IdealExecutionOutcome
    saturation: RationalPolynomialIdeal | None = None
    method: Literal["SINGULAR_SATURATION"] = "SINGULAR_SATURATION"
    backend_version: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        if self.outcome == "COMPUTED":
            if self.saturation is None or self.backend_version is None or self.detail:
                raise ValueError(
                    "computed saturation requires a value and backend version"
                )
        elif (
            self.saturation is not None
            or self.backend_version is not None
            or not self.detail
        ):
            raise ValueError(
                "failed saturation computation requires only a safe detail"
            )
        return self


__all__ = [
    "MAX_OUTPUT_GENERATORS",
    "MAX_OUTPUT_TERMS",
    "EliminationExecutionOutcome",
    "EliminationIdealRequest",
    "EliminationIdealResult",
    "GroebnerBasisRequest",
    "GroebnerBasisResult",
    "IdealComputationBudget",
    "IdealExecutionOutcome",
    "IdealNormalFormRequest",
    "IdealNormalFormResult",
    "IdealQuotientRequest",
    "IdealQuotientResult",
    "IdealRadicalMembershipRequest",
    "IdealRadicalMembershipResult",
    "IdealRadicalRequest",
    "IdealRadicalResult",
    "IdealSaturationRequest",
    "IdealSaturationResult",
    "NormalFormExecutionOutcome",
]


# ---------------------------------------------------------------------------
# Gröbner basis computation
# ---------------------------------------------------------------------------


class GroebnerBasisRequest(StrictModel):
    """Compute a reduced Gröbner basis for a bounded ideal in QQ[variables]."""

    ideal: RationalPolynomialIdeal
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex"
    resource_budget: IdealComputationBudget = Field(
        default_factory=IdealComputationBudget
    )

    @model_validator(mode="after")
    def require_backend_domain(self) -> Self:
        _require_ideal_budget(self.ideal, label="ideal")
        return self


GroebnerExecutionOutcome = Literal["COMPUTED", "TIMEOUT"]


class GroebnerBasisResult(StrictModel):
    """A reduced Gröbner basis, or a typed timeout under the enforced budget."""

    outcome: GroebnerExecutionOutcome = "COMPUTED"
    basis: RationalPolynomialIdeal | None = None
    generator_count: StrictInt = Field(default=0, ge=0, le=MAX_OUTPUT_GENERATORS)
    monomial_order: Literal["lex", "grlex", "grevlex"]
    detail: str | None = None
    backend: Literal["SYMPY"] = "SYMPY"

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        if self.outcome == "COMPUTED":
            if self.basis is None or self.detail is not None:
                raise ValueError(
                    "computed basis requires a value and no failure detail"
                )
            if (
                self.generator_count != len(self.basis.generators)
                or self.generator_count < 1
            ):
                raise ValueError("generator_count must match the basis generator count")
        elif self.basis is not None or self.detail is None:
            raise ValueError("timed-out computation carries only a safe detail")
        return self


# ---------------------------------------------------------------------------
# Normal form / ideal membership
# ---------------------------------------------------------------------------


class IdealNormalFormRequest(StrictModel):
    """Reduce one polynomial modulo an ideal's Gröbner basis."""

    ideal: RationalPolynomialIdeal
    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_backend_domain(self) -> Self:
        _require_ideal_budget(self.ideal, label="ideal")
        if self.polynomial.variables != self.ideal.variables:
            raise ValueError("polynomial must use the ideal's ordered ring")
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=MAX_INPUT_TERMS,
            maximum_exponent=MAX_INPUT_EXPONENT,
            maximum_coefficient_digits=MAX_COEFFICIENT_DIGITS,
            label="polynomial",
        )
        return self


NormalFormExecutionOutcome = Literal["COMPUTED", "TIMEOUT"]


class IdealNormalFormResult(StrictModel):
    """The exact remainder modulo an ideal, or a typed timeout under the enforced budget."""

    outcome: NormalFormExecutionOutcome = "COMPUTED"
    remainder: RationalPolynomial | None = None
    in_ideal: bool = False
    detail: str | None = None

    @model_validator(mode="after")
    def require_consistent_membership(self) -> Self:
        if self.outcome == "COMPUTED":
            if self.remainder is None or self.detail is not None:
                raise ValueError(
                    "computed normal form requires a remainder and no failure detail"
                )
            if self.in_ideal and len(self.remainder.polynomial.terms) > 0:
                raise ValueError("a polynomial in the ideal must have a zero remainder")
            if not self.in_ideal and len(self.remainder.polynomial.terms) == 0:
                raise ValueError(
                    "a polynomial not in the ideal must have a nonzero remainder"
                )
        elif self.remainder is not None or self.detail is None:
            raise ValueError("timed-out computation carries only a safe detail")
        return self


# ---------------------------------------------------------------------------
# Elimination ideal
# ---------------------------------------------------------------------------


class EliminationIdealRequest(StrictModel):
    """Compute the elimination ideal I ∩ QQ[remaining variables]."""

    ideal: RationalPolynomialIdeal
    eliminated_variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_VARS)
    resource_budget: IdealComputationBudget = Field(
        default_factory=IdealComputationBudget
    )

    @model_validator(mode="after")
    def require_backend_domain(self) -> Self:
        _require_ideal_budget(self.ideal, label="ideal")
        eliminated = set(self.eliminated_variables)
        for var in eliminated:
            if var not in self.ideal.variables:
                raise ValueError(
                    "eliminated variables must be a subset of the ideal's variables"
                )
        remaining = tuple(v for v in self.ideal.variables if v not in eliminated)
        if not remaining:
            raise ValueError(
                "elimination cannot remove every variable; at least one must remain"
            )
        return self


EliminationExecutionOutcome = Literal["COMPUTED", "TIMEOUT"]


class EliminationIdealResult(StrictModel):
    """The elimination ideal I ∩ QQ[remaining variables], or a typed timeout under the enforced budget."""

    outcome: EliminationExecutionOutcome = "COMPUTED"
    elimination_ideal: RationalPolynomialIdeal | None = None
    eliminated_variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_VARS)
    backend: Literal["SYMPY"] = "SYMPY"
    detail: str | None = None

    @model_validator(mode="after")
    def require_consistent_result(self) -> Self:
        if self.outcome == "COMPUTED":
            if self.elimination_ideal is None or self.detail is not None:
                raise ValueError(
                    "computed elimination requires an ideal and no failure detail"
                )
            for var in self.eliminated_variables:
                if var in self.elimination_ideal.variables:
                    raise ValueError(
                        "eliminated variables must not appear in the elimination ideal"
                    )
        elif self.elimination_ideal is not None or self.detail is None:
            raise ValueError("timed-out computation carries only a safe detail")
        return self
