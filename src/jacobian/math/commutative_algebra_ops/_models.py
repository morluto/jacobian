"""Typed wire contracts for commutative algebra operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials._conversions import symbols_for_variables
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


class IdealSaturationRequest(StrictModel):
    """Compute ``I : <d>^infinity`` for a bounded ideal and one polynomial."""

    ideal: RationalPolynomialIdeal = Field(
        description=(
            "An ideal in at most 6 variables with at most 16 generators and "
            "256 aggregate terms; generator total degree is at most 12 and "
            "coefficient components are at most 128 digits."
        )
    )
    denominator: RationalPolynomial = Field(
        description=(
            "A single nonzero polynomial d in the dividend's exact ordered "
            "ring, with at most 256 terms, total degree at most 12, and "
            "coefficient components at most 128 digits."
        )
    )
    resource_budget: IdealComputationBudget = Field(
        default_factory=IdealComputationBudget
    )

    @model_validator(mode="after")
    def require_backend_domain(self) -> Self:
        _require_ideal_budget(self.ideal, label="ideal")
        if self.denominator.variables != self.ideal.variables:
            raise ValueError("saturation operands must use the same ordered ring")
        if not self.denominator.polynomial.terms:
            raise ValueError("saturation denominator must be nonzero")
        require_polynomial_budget(
            self.denominator,
            maximum_terms=MAX_INPUT_TERMS,
            maximum_exponent=MAX_INPUT_EXPONENT,
            maximum_coefficient_digits=MAX_COEFFICIENT_DIGITS,
            label="saturation denominator",
        )
        if any(
            sum(term.exponents) > MAX_INPUT_EXPONENT
            for term in self.denominator.polynomial.terms
        ):
            raise ValueError(
                f"saturation denominator exceeds total degree {MAX_INPUT_EXPONENT}"
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


class IdealSaturationResult(StrictModel):
    """The exact saturation I : <d>^infinity, bound to its source request.

    A COMPUTED result retains the request it was computed from and its
    validator replays the exact defining relation in-process (Groebner
    elimination over QQ), so the authoritative derived ideal cannot detach
    from the computation it claims to represent.
    """

    outcome: IdealExecutionOutcome
    request: IdealSaturationRequest | None = None
    saturation: RationalPolynomialIdeal | None = None
    method: Literal["SINGULAR_SATURATION"] = "SINGULAR_SATURATION"
    backend_version: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        if self.outcome == "COMPUTED":
            if (
                self.saturation is None
                or self.backend_version is None
                or self.request is None
                or self.detail
            ):
                raise ValueError(
                    "computed saturation requires a value bound to its "
                    "source request and a backend version"
                )
            self._require_saturation_relation()
        elif (
            self.saturation is not None
            or self.backend_version is not None
            or not self.detail
        ):
            raise ValueError(
                "failed saturation computation requires only a safe detail"
            )
        return self

    @staticmethod
    def _require_claimed_generator_budget(ideal: RationalPolynomialIdeal) -> None:
        """Bound claimed generators before any replay conversion."""
        if len(ideal.generators) > MAX_OUTPUT_GENERATORS:
            raise ValueError(
                f"claimed saturation exceeds {MAX_OUTPUT_GENERATORS} generators"
            )
        aggregate_terms = 0
        for generator in ideal.generators:
            terms = generator.polynomial.terms
            aggregate_terms += len(terms)
            for term in terms:
                if sum(term.exponents) > 2 * MAX_INPUT_EXPONENT:
                    raise ValueError(
                        "claimed saturation generator exceeds total degree "
                        f"{2 * MAX_INPUT_EXPONENT}"
                    )
                coefficient = term.coefficient
                digits = max(len(coefficient.num.lstrip("-")), len(coefficient.den))
                if digits > 4 * MAX_COEFFICIENT_DIGITS:
                    raise ValueError(
                        "claimed saturation coefficient exceeds digit bound"
                    )
        if aggregate_terms > MAX_OUTPUT_TERMS:
            raise ValueError(
                f"claimed saturation exceeds {MAX_OUTPUT_TERMS} aggregate terms"
            )

    def _require_saturation_relation(self) -> None:
        """Replay the exact defining relation against the retained source."""
        from jacobian.math.commutative_algebra_ops._operations import (
            _groebner_signature,
            rational_expressions_of_ideal,
            replay_saturation_bounded,
        )

        if self.request is None or self.saturation is None:
            return
        self._require_claimed_generator_budget(self.saturation)
        if self.request.ideal.variables != self.saturation.variables:
            raise ValueError("saturation must use the source ideal's ordered ring")
        claimed = _groebner_signature(
            symbols_for_variables(self.saturation.variables),
            rational_expressions_of_ideal(self.saturation),
        )
        expected = replay_saturation_bounded(self.request)
        if claimed != expected:
            raise ValueError(
                "saturation must be the exact saturation of the retained source request"
            )


__all__ = [
    "MAX_OUTPUT_GENERATORS",
    "MAX_OUTPUT_TERMS",
    "IdealComputationBudget",
    "IdealExecutionOutcome",
    "IdealQuotientRequest",
    "IdealQuotientResult",
    "IdealRadicalMembershipRequest",
    "IdealRadicalMembershipResult",
    "IdealRadicalRequest",
    "IdealRadicalResult",
    "IdealSaturationRequest",
    "IdealSaturationResult",
]
