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
            )
        elif self.basis is not None or self.detail is None:
            raise ValueError("timed-out computation carries only a safe detail")
        return self


def _sympy_monomial(symbols, exponents):
    import sympy

    monomial = sympy.Integer(1)
    for symbol, exponent in zip(symbols, exponents, strict=True):
        if exponent:
            monomial *= symbol**exponent
    return monomial


def _require_source_bound_basis(
    basis: RationalPolynomialIdeal,
    source: RationalPolynomialIdeal,
    monomial_order: str,
) -> None:
    """Replay the declared reduced Gröbner-basis conditions against the source."""
    import sympy

    from jacobian.math.polynomials._conversions import (
        rational_polynomial_to_sympy,
        symbols_for_variables,
    )

    def to_expr(generator):
        return rational_polynomial_to_sympy(generator).as_expr()

    symbols = symbols_for_variables(basis.variables)
    basis_exprs = [to_expr(generator) for generator in basis.generators]
    source_exprs = [to_expr(generator) for generator in source.generators]
    nonzero = [expr for expr in basis_exprs if not expr.is_zero]
    if not nonzero:
        if any(not expr.is_zero for expr in source_exprs):
            raise ValueError("basis must contain every source-ideal generator")
        return
    leading_terms = [
        sympy.LT(expr, *symbols, order=monomial_order) for expr in nonzero
    ]
    leading_exps = [
        sympy.Poly(lt, *symbols, domain=sympy.QQ).monoms()[0] for lt in leading_terms
    ]
    # Reduced: unit leading coefficients and no generator term divisible by
    # another generator's leading monomial.
    for index, (expr, lt) in enumerate(zip(nonzero, leading_terms, strict=True)):
        if sympy.LC(expr, *symbols, order=monomial_order) != 1:
            raise ValueError(
                "a reduced Gröbner basis has unit leading coefficients"
            )
        others = [lt for other_index, lt in enumerate(leading_terms) if other_index != index]
        if others:
            _, remainder = sympy.reduced(
                expr, others, *symbols, order=monomial_order, domain=sympy.QQ
            )
            if remainder != expr:
                raise ValueError(
                    "reduced Gröbner basis generators must contain no other "
                    "leading monomial"
                )
    _require_buchberger_criterion(
        nonzero, leading_terms, leading_exps, symbols, monomial_order
    )
    _require_basis_ideal_equality(
        nonzero, source_exprs, source, symbols, monomial_order
    )


def _require_buchberger_criterion(
    nonzero,
    leading_terms,
    leading_exps,
    symbols,
    monomial_order: str,
) -> None:
    """Every S-polynomial must reduce to zero over the claimed basis."""
    from itertools import combinations

    import sympy

    for first, second in combinations(range(len(nonzero)), 2):
        lcm_exp = tuple(
            max(a, b)
            for a, b in zip(leading_exps[first], leading_exps[second], strict=True)
        )
        s_poly = nonzero[first] * _sympy_monomial(
            symbols,
            tuple(a - b for a, b in zip(lcm_exp, leading_exps[first], strict=True)),
        ) - nonzero[second] * _sympy_monomial(
            symbols,
            tuple(a - b for a, b in zip(lcm_exp, leading_exps[second], strict=True)),
        )
        _, remainder = sympy.reduced(
            s_poly, nonzero, *symbols, order=monomial_order, domain=sympy.QQ
        )
        if remainder != 0:
            raise ValueError(
                "basis S-polynomials must reduce to zero; the list is not a "
                "Gröbner basis of the retained ideal"
            )


def _require_basis_ideal_equality(
    nonzero,
    source_exprs,
    source: RationalPolynomialIdeal,
    symbols,
    monomial_order: str,
) -> None:
    """Membership against a verified basis is decided by reduction to zero."""
    import sympy

    # Recompute the bounded source basis once so both inclusions reduce
    # against verified Gröbner bases.
    from jacobian.math.polynomials._conversions import symbols_for_variables

    source_symbols = symbols_for_variables(source.variables)
    source_basis = sympy.groebner(
        source_exprs,
        *source_symbols,
        order=monomial_order,
        domain=sympy.QQ,
    )
    for expr in nonzero:
        _, remainder = sympy.reduced(
            expr,
            list(source_basis.exprs),
            *source_symbols,
            order=monomial_order,
            domain=sympy.QQ,
        )
        if remainder != 0:
            raise ValueError("basis generators must lie in the source ideal")
    for expr in source_exprs:
        if expr.is_zero:
            continue
        _, remainder = sympy.reduced(
            expr, nonzero, *symbols, order=monomial_order, domain=sympy.QQ
        )
        if remainder != 0:
            raise ValueError("source ideal must be contained in the basis ideal")


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

    request: IdealNormalFormRequest
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
            _require_source_bound_remainder(self.request, self.remainder)
        elif self.remainder is not None or self.detail is None:
            raise ValueError("timed-out computation carries only a safe detail")
        return self


def _require_source_bound_remainder(
    request: IdealNormalFormRequest,
    remainder: RationalPolynomial,
) -> None:
    """Replay the defining Gröbner reduction of the retained operands."""
    import sympy

    from jacobian.math.polynomials._conversions import (
        rational_polynomial_from_sympy,
        rational_polynomial_to_sympy,
        symbols_for_variables,
    )

    variables = request.ideal.variables
    symbols = symbols_for_variables(variables)
    ideal_generators = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in request.ideal.generators
    ]
    polynomial_expr = rational_polynomial_to_sympy(request.polynomial).as_expr()
    basis = sympy.groebner(
        ideal_generators,
        *symbols,
        order="grevlex",
        domain=sympy.QQ,
    )
    _, replayed = sympy.reduced(
        polynomial_expr,
        list(basis.exprs),
        *symbols,
        order="grevlex",
        domain=sympy.QQ,
    )
    expected = rational_polynomial_from_sympy(
        sympy.Poly(replayed, *symbols, domain=sympy.QQ),
        variables,
    )
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


EliminationExecutionOutcome = Literal["COMPUTED", "TIMEOUT"]


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
    """Replay the exact intersection I ∩ QQ[remaining] from the retained ideal."""
    import sympy

    from jacobian.math.polynomials._conversions import (
        rational_polynomial_from_sympy,
        rational_polynomial_to_sympy,
        symbols_for_variables,
    )
    from jacobian.math.polynomials.values import SparseRationalPolynomial

    variables = list(request.ideal.variables)
    eliminated_set = set(request.eliminated_variables)
    remaining = [v for v in variables if v not in eliminated_set]
    ordered_variables = tuple(v for v in variables if v in eliminated_set) + tuple(
        remaining
    )
    ordered_symbols = symbols_for_variables(ordered_variables)
    ideal_generators = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in request.ideal.generators
    ]
    basis = sympy.groebner(
        ideal_generators,
        *ordered_symbols,
        order="lex",
        domain=sympy.QQ,
    )
    remaining_symbols = symbols_for_variables(tuple(remaining))
    replayed_generators: list[RationalPolynomial] = []
    unit_ideal = False
    for expr in basis:
        poly = sympy.Poly(expr, *ordered_symbols, domain=sympy.QQ)
        involved = {str(s) for s in poly.free_symbols}
        if not involved:
            unit_ideal = True
            break
        if involved.issubset(set(remaining)):
            replayed_generators.append(
                rational_polynomial_from_sympy(
                    sympy.Poly(expr, *remaining_symbols, domain=sympy.QQ),
                    tuple(remaining),
                )
            )
    if unit_ideal:
        from jacobian._exact import CanonicalRational
        from jacobian.math.polynomials.values import RationalPolynomialTerm

        one = RationalPolynomial(
            variables=tuple(remaining),
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(num="1", den="1"),
                        exponents=(0,) * len(remaining),
                    ),
                )
            ),
        )
        replayed_generators = [one]
    elif not replayed_generators:
        zero = RationalPolynomial(
            variables=tuple(remaining),
            polynomial=SparseRationalPolynomial(terms=()),
        )
        replayed_generators = [zero]
    replayed = RationalPolynomialIdeal(
        variables=tuple(remaining),
        generators=tuple(replayed_generators),
    )
    if elimination_ideal != replayed:
        raise ValueError(
            "elimination ideal must equal the exact intersection "
            "I ∩ QQ[remaining variables] of the retained source ideal"
        )
