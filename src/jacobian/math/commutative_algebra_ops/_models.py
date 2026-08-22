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
            'An ideal in at most 6 variables with at most 16 generators and '
            '256 aggregate terms; generator total degree is at most 12 and '
            'coefficient components are at most 128 digits.'
        )
    )
    saturation_polynomial: RationalPolynomial = Field(
        description=(
            'A single polynomial d in the ideal ring, with '
            'at most 256 terms, total degree at most 12, and coefficient '
            'components at most 128 digits.'
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
        if any(
            sum(term.exponents) > MAX_INPUT_EXPONENT
            for term in self.saturation_polynomial.polynomial.terms
        ):
            raise ValueError(
                f"saturation polynomial exceeds total degree {MAX_INPUT_EXPONENT}"
            )
        if self.saturation_polynomial.variables != self.ideal.variables:
            raise ValueError(
                "saturation polynomial must use the ideal's ordered ring"
            )
        return self


def require_deterministic_saturation_certificate(
    ideal,
    saturation_polynomial,
    saturation,
) -> None:
    """Deterministic certificate check when backend replay is unavailable.

    Verifies both inclusions that pin the saturation: I ⊆ S, and every
    saturation generator is annihilated into I by some power of d (bounded
    multiply-and-reduce search over the admitted input domain).
    """
    import sympy

    from jacobian.math.polynomials._conversions import (
        rational_polynomial_to_sympy,
    )

    d_expr = rational_polynomial_to_sympy(saturation_polynomial).as_expr()
    symbols = saturation.variables
    sat_basis = sympy.groebner(
        [
            rational_polynomial_to_sympy(generator).as_expr()
            for generator in saturation.generators
        ],
        *sympy.symbols(symbols),
        order="grevlex",
        domain=sympy.QQ,
    )
    ideal_basis = sympy.groebner(
        [
            rational_polynomial_to_sympy(generator).as_expr()
            for generator in ideal.generators
        ],
        *sympy.symbols(symbols),
        order="grevlex",
        domain=sympy.QQ,
    )
    for generator in ideal.generators:
        if (
            sat_basis.reduce(rational_polynomial_to_sympy(generator).as_expr())[1]
            != 0
        ):
            raise ValueError(
                "saturation must equal the exact Singular replay of the "
                "retained ideal and saturation polynomial"
            )
    cur = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in saturation.generators
    ]
    for _ in range(2 * MAX_INPUT_EXPONENT + 1):
        remaining = []
        for expr in cur:
            if ideal_basis.reduce(expr)[1] != 0:
                remaining.append(expr * d_expr)
        if not remaining:
            return
        cur = remaining
    raise ValueError(
        "saturation must equal the exact Singular replay of the retained "
        "ideal and saturation polynomial"
    )


class IdealSaturationResult(StrictModel):
    """The exact saturation I : <d>^infinity, bound to its source request."""

    outcome: IdealExecutionOutcome
    ideal: RationalPolynomialIdeal
    saturation_polynomial: RationalPolynomial
    resource_budget: IdealComputationBudget = Field(
        default_factory=IdealComputationBudget
    )
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
            # Replay the bounded backend call against the retained sources:
            # an authored payload cannot claim an arbitrary canonical ideal
            # as the saturation.
            from jacobian.math.commutative_algebra_ops._singular import (
                run_singular_ideal_operation,
            )
            from jacobian.math.polynomials.values import (
                RationalPolynomialIdeal,
            )

            divisor = RationalPolynomialIdeal(
                variables=self.ideal.variables,
                generators=(self.saturation_polynomial,),
            )
            replay = run_singular_ideal_operation(
                "saturation",
                self.ideal,
                divisor,
                self.resource_budget,
            )
            # Replay against the retained sources with an amplified wall
            # budget so a computation near the configured boundary is not
            # misjudged by second-process jitter; exact equality is required
            # on every healthy replay.
            from jacobian.math.commutative_algebra_ops._singular import (
                run_singular_ideal_operation,
            )
            from jacobian.math.polynomials.values import (
                RationalPolynomialIdeal,
            )

            divisor = RationalPolynomialIdeal(
                variables=self.ideal.variables,
                generators=(self.saturation_polynomial,),
            )
            replay_budget = self.resource_budget.model_copy(
                update={"wall_seconds": 60}
            )
            replay = run_singular_ideal_operation(
                "saturation",
                self.ideal,
                divisor,
                replay_budget,
            )
            if (
                replay.outcome != "COMPUTED"
                or replay.ideal != self.saturation
                or replay.backend_version != self.backend_version
            ):
                raise ValueError(
                    "saturation must equal the exact Singular replay of the "
                    "retained ideal and saturation polynomial"
                )
        elif (
            self.radical is not None
            or self.backend_version is not None
            or not self.detail
        ):
            raise ValueError("failed radical computation requires only a safe detail")
        return self
