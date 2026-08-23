"""Typed wire contracts for commutative algebra operations."""

from __future__ import annotations

from typing import Any, Literal, Self

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
DEFAULT_WALL_SECONDS = 10.0


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


GroebnerExecutionOutcome = Literal["COMPUTED", "ERROR", "LIMIT_EXCEEDED", "TIMEOUT"]


class GroebnerBasisResult(StrictModel):
    """A reduced Gröbner basis, or a typed timeout under the enforced budget."""

    request: GroebnerBasisRequest
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
            if self.request.monomial_order != self.monomial_order:
                raise ValueError("basis must carry its request's monomial order")
            _require_source_bound_basis(
                self.basis,
                self.request.ideal,
                self.request.monomial_order,
                float(self.request.resource_budget.wall_seconds),
            )
        elif self.basis is not None or self.detail is None:
            raise ValueError("timed-out computation carries only a safe detail")
        return self


def _require_zero_free_basis(
    basis_exprs: list[Any],
    source_exprs: list[Any],
) -> list[Any] | None:
    """Gate zero entries, returning ``None`` when no replay is needed.

    A reduced Gröbner basis never contains the zero polynomial. Only the
    producer's singleton-zero representation of the zero ideal itself may
    carry one; any other zero entry silently weakens every invariant check.
    """
    if not basis_exprs:
        if any(not expr.is_zero for expr in source_exprs):
            raise ValueError("basis must contain every source-ideal generator")
        return None
    if any(expr.is_zero for expr in basis_exprs):
        if not (len(basis_exprs) == 1 and all(expr.is_zero for expr in source_exprs)):
            raise ValueError(
                "a reduced Gröbner basis contains no zero generator; only "
                "the zero ideal admits the singleton-zero representation"
            )
        return None
    return basis_exprs


def _require_source_bound_basis(
    basis: RationalPolynomialIdeal,
    source: RationalPolynomialIdeal,
    monomial_order: str,
    wall_seconds: float,
) -> None:
    """Gate cheap structural invariants, then replay the exact ones.

    Reducedness, the Buchberger criterion, and both ideal inclusions are
    exact work with unbounded intermediate growth, so they run as ONE
    bounded killable-worker pass under the declared budget instead of
    unbounded parent-process SymPy calls.
    """
    from jacobian.math.polynomials._conversions import (
        rational_polynomial_to_sympy,
    )

    if basis.variables != source.variables:
        raise ValueError("basis must use the source ideal's ordered ring")
    basis_exprs = [rational_polynomial_to_sympy(g).as_expr() for g in basis.generators]
    source_exprs = [
        rational_polynomial_to_sympy(g).as_expr() for g in source.generators
    ]
    nonzero = _require_zero_free_basis(basis_exprs, source_exprs)
    if nonzero is None:
        return
    from jacobian.math.commutative_algebra_ops._operations import (
        _run_sympy_kernel,
    )

    payload = {
        "mode": "verify_groebner_basis",
        "variables": list(source.variables),
        "order": monomial_order,
        "generators": [
            generator.model_dump(mode="json") for generator in source.generators
        ],
        "basis": [generator.model_dump(mode="json") for generator in basis.generators],
    }
    try:
        result = _run_sympy_kernel(payload, wall_seconds)
    except Exception as error:
        raise ValueError(
            "the retained sources could not be verified against this basis "
            f"within the enforced wall-time budget: {error}"
        ) from None
    if not result.get("equal"):
        raise ValueError(
            "basis and source ideals differ: "
            + str(result.get("detail", "inclusion replay failed"))
        )


# ---------------------------------------------------------------------------
# Normal form / ideal membership
# ---------------------------------------------------------------------------


NormalFormMonomialOrder = Literal["lex", "grlex", "grevlex"]


class IdealNormalFormRequest(StrictModel):
    """Reduce one polynomial modulo an ideal's Gröbner basis.

    ``monomial_order`` names the order of the Groebner basis the reduction
    uses; normal forms depend on it, so it is part of the public contract.
    """

    ideal: RationalPolynomialIdeal
    polynomial: RationalPolynomial
    monomial_order: NormalFormMonomialOrder = "grevlex"

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


NormalFormExecutionOutcome = Literal["COMPUTED", "ERROR", "LIMIT_EXCEEDED", "TIMEOUT"]


class IdealNormalFormResult(StrictModel):
    """The exact remainder modulo an ideal, or a typed incomplete outcome."""

    request: IdealNormalFormRequest
    outcome: NormalFormExecutionOutcome = "COMPUTED"
    remainder: RationalPolynomial | None = None
    in_ideal: bool | None = None
    monomial_order: NormalFormMonomialOrder = "grevlex"
    detail: str | None = None

    @model_validator(mode="after")
    def require_consistent_membership(self) -> Self:
        if self.outcome != "COMPUTED" and self.in_ideal is not None:
            raise ValueError(
                "an incomplete normal-form outcome states no membership conclusion"
            )
        if self.monomial_order != self.request.monomial_order:
            raise ValueError("monomial_order must match the retained request")
        if self.outcome == "COMPUTED":
            if self.remainder is None or self.detail is not None:
                raise ValueError(
                    "computed normal form requires a remainder and no failure detail"
                )
            # A computed outcome claims its authoritative membership
            # decision; omitting it would let the result claim success
            # while withholding the conclusion.
            if self.in_ideal is None:
                raise ValueError(
                    "a computed normal form must state its membership "
                    "conclusion in in_ideal"
                )
            if self.in_ideal and len(self.remainder.polynomial.terms) > 0:
                raise ValueError("a polynomial in the ideal must have a zero remainder")
            if not self.in_ideal and len(self.remainder.polynomial.terms) == 0:
                raise ValueError(
                    "a polynomial not in the ideal must have a nonzero remainder"
                )
            _require_source_bound_remainder(self.request, self.remainder)
        elif self.remainder is not None or self.detail is None:
            raise ValueError("timed-out computation carries only a safe detail")
        return self


def _require_source_bound_remainder(
    request: IdealNormalFormRequest,
    remainder: RationalPolynomial,
) -> None:
    """Replay the defining Gröbner reduction inside the bounded kernel.

    The exact reduction has unbounded intermediate work, so it reuses the
    producer's killable-worker mode under the declared wall budget instead
    of an unbounded parent-process SymPy call.
    """
    from jacobian.math.commutative_algebra_ops._operations import (
        _run_sympy_kernel,
    )

    payload = {
        "mode": "normal_form",
        "variables": list(request.ideal.variables),
        "order": request.monomial_order,
        "generators": [
            generator.model_dump(mode="json") for generator in request.ideal.generators
        ],
        "polynomial": request.polynomial.model_dump(mode="json"),
    }
    try:
        result_payload = _run_sympy_kernel(payload, DEFAULT_WALL_SECONDS)
    except Exception as error:
        raise ValueError(
            "the remainder could not be re-verified within the enforced "
            f"wall-time budget: {error}"
        ) from None
    expected = RationalPolynomial.model_validate(result_payload["remainder"])
    if remainder != expected:
        raise ValueError(
            "remainder must be the defining reduction of the retained "
            "polynomial modulo the retained ideal"
        )


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


EliminationExecutionOutcome = Literal["COMPUTED", "ERROR", "LIMIT_EXCEEDED", "TIMEOUT"]


class EliminationIdealResult(StrictModel):
    """The elimination ideal I ∩ QQ[remaining variables], or a typed timeout under the enforced budget."""

    request: EliminationIdealRequest
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
            if self.eliminated_variables != self.request.eliminated_variables:
                raise ValueError("eliminated_variables must match the retained request")
            for var in self.eliminated_variables:
                if var in self.elimination_ideal.variables:
                    raise ValueError(
                        "eliminated variables must not appear in the elimination ideal"
                    )
            _require_source_bound_elimination(self.request, self.elimination_ideal)
        elif self.elimination_ideal is not None or self.detail is None:
            raise ValueError("timed-out computation carries only a safe detail")
        return self


def _require_source_bound_elimination(
    request: EliminationIdealRequest,
    elimination_ideal: RationalPolynomialIdeal,
) -> None:
    """Replay the exact intersection in the bounded kernel.

    The lex Groebner intersection is unbounded exact work, so it reuses the
    producer's killable-worker elimination mode under the declared wall
    budget instead of an unbounded parent-process SymPy call.
    """
    from jacobian.math.commutative_algebra_ops._operations import (
        _run_sympy_kernel,
    )
    from jacobian.math.polynomials.values import SparseRationalPolynomial

    payload = {
        "mode": "elimination",
        "variables": list(request.ideal.variables),
        "eliminated": list(request.eliminated_variables),
        "generators": [
            generator.model_dump(mode="json") for generator in request.ideal.generators
        ],
    }
    try:
        result_payload = _run_sympy_kernel(
            payload, float(request.resource_budget.wall_seconds)
        )
    except Exception as error:
        raise ValueError(
            "the elimination ideal could not be re-verified within the "
            f"enforced wall-time budget: {error}"
        ) from None
    remaining = tuple(
        v for v in request.ideal.variables if v not in set(request.eliminated_variables)
    )
    if result_payload.get("unit_ideal"):
        from jacobian._exact import CanonicalRational
        from jacobian.math.polynomials.values import RationalPolynomialTerm

        replayed_generators = [
            RationalPolynomial(
                variables=remaining,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num="1", den="1"),
                            exponents=(0,) * len(remaining),
                        ),
                    )
                ),
            )
        ]
    elif result_payload.get("generators"):
        replayed_generators = [
            RationalPolynomial.model_validate(item)
            for item in result_payload["generators"]
        ]
    else:
        replayed_generators = [
            RationalPolynomial(
                variables=remaining,
                polynomial=SparseRationalPolynomial(terms=()),
            )
        ]
    replayed = RationalPolynomialIdeal(
        variables=remaining,
        generators=tuple(replayed_generators),
    )
    if elimination_ideal != replayed:
        raise ValueError(
            "elimination ideal must equal the exact intersection "
            "I \u2229 QQ[remaining variables] of the retained source ideal"
        )
