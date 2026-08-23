"""Contracts for exact multivariate polynomial operations over ``QQ``."""

from __future__ import annotations

from math import comb
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)

_MAX_MULTIVARIATE_TERMS = 512
_MAX_MULTIVARIATE_EXPONENT = 64
_MAX_MULTIVARIATE_COEFFICIENT_DIGITS = 256
_MAX_ELIMINATION_DEGREE_SUM = 64
# The result converter rejects sparse outputs above this size.  The request
# validator uses the same bound to reject large possible supports before
# SymPy expands the Sylvester determinant.
_MAX_RESULTANT_TERMS = 1_024
# Public output-term budget for one converted irreducible factor.  The
# operation converter uses this same bound; keeping it here lets the result
# validator reproduce the kernel's exact exceedance decision.
_MAX_FACTOR_OUTPUT_TERMS = 1_024

MonomialOrder = Literal["lex", "grlex", "grevlex"]
"""Declared monomial order for multivariate polynomial division."""

_MULTIVARIATE_MIN_VARIABLES = 2


def _validate_multivariate_pair(
    left: RationalPolynomial,
    right: RationalPolynomial,
) -> None:
    """Shared validation for two polynomials in the same declared ring."""

    if len(left.variables) < _MULTIVARIATE_MIN_VARIABLES:
        raise ValueError("multivariate operations require at least two variables")
    if left.variables != right.variables:
        raise ValueError("both polynomials must use the same ordered variables")


def _resultant_support_bound(
    left: RationalPolynomial,
    right: RationalPolynomial,
    elimination_index: int,
) -> int:
    """Bound the number of possible monomials in a multivariate resultant.

    If ``f`` and ``g`` have elimination degrees ``m`` and ``n``, and total
    remaining-variable degrees ``d_f`` and ``d_g``, every resultant monomial
    has total degree at most ``n*d_f + m*d_g``.  The returned binomial is the
    number of monomials up to that degree in the remaining variables.
    """

    remaining_variable_count = len(left.variables) - 1
    if remaining_variable_count == 0:
        return 1

    def degree(polynomial: RationalPolynomial, *, in_remaining: bool) -> int:
        return max(
            (
                sum(
                    exponent
                    for index, exponent in enumerate(term.exponents)
                    if (index != elimination_index) == in_remaining
                )
                for term in polynomial.polynomial.terms
            ),
            default=0,
        )

    left_elimination_degree = degree(left, in_remaining=False)
    right_elimination_degree = degree(right, in_remaining=False)
    left_remaining_degree = degree(left, in_remaining=True)
    right_remaining_degree = degree(right, in_remaining=True)
    resultant_degree_bound = (
        right_elimination_degree * left_remaining_degree
        + left_elimination_degree * right_remaining_degree
    )
    return comb(
        resultant_degree_bound + remaining_variable_count,
        remaining_variable_count,
    )


class MultivariateGcdRequest(StrictModel):
    """Two multivariate polynomials in ``QQ[x_1, ..., x_n]``."""

    left: RationalPolynomial
    right: RationalPolynomial

    @model_validator(mode="after")
    def require_multivariate_ring(self) -> Self:
        _validate_multivariate_pair(self.left, self.right)
        for polynomial in (self.left, self.right):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_MULTIVARIATE_TERMS,
                maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
                maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
            )
        return self


class MultivariateGcdResult(StrictModel):
    gcd: RationalPolynomial
    convention: Literal["MONIC_ASSOCIATE"] = "MONIC_ASSOCIATE"


class MultivariateDivisionRequest(StrictModel):
    """Divide one multivariate polynomial by another under a declared monomial order."""

    left: RationalPolynomial
    right: RationalPolynomial
    monomial_order: MonomialOrder = "lex"

    @model_validator(mode="after")
    def require_multivariate_ring(self) -> Self:
        _validate_multivariate_pair(self.left, self.right)
        if not self.right.polynomial.terms:
            raise ValueError("divisor polynomial must be nonzero")
        for polynomial in (self.left, self.right):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_MULTIVARIATE_TERMS,
                maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
                maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
            )
        return self


class MultivariateDivisionResult(StrictModel):
    quotient: RationalPolynomial
    remainder: RationalPolynomial
    monomial_order: MonomialOrder
    convention: Literal["EXACT_DIVISION_REMAINDER"] = "EXACT_DIVISION_REMAINDER"


class MultivariateResultantRequest(StrictModel):
    """Compute a bounded resultant with respect to one variable.

    The request rejects inputs whose degree envelope can produce more terms
    than the exact sparse result contract can represent.
    """

    left: RationalPolynomial
    right: RationalPolynomial
    elimination_variable: PolynomialVariable = Field(
        description="Variable eliminated by the Sylvester resultant.",
    )

    @model_validator(mode="after")
    def require_multivariate_ring(self) -> Self:
        _validate_multivariate_pair(self.left, self.right)
        if self.elimination_variable not in self.left.variables:
            raise ValueError("elimination variable must belong to the declared ring")
        for polynomial in (self.left, self.right):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_MULTIVARIATE_TERMS,
                maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
                maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
            )
        variable_index = self.left.variables.index(self.elimination_variable)
        for polynomial, label in ((self.left, "left"), (self.right, "right")):
            degree_in_variable = max(
                (
                    term.exponents[variable_index]
                    for term in polynomial.polynomial.terms
                ),
                default=0,
            )
            if degree_in_variable == 0:
                raise ValueError(
                    f"{label} polynomial has zero degree in the elimination variable"
                )
        degree_sum = max(
            (term.exponents[variable_index] for term in self.left.polynomial.terms),
            default=0,
        ) + max(
            (term.exponents[variable_index] for term in self.right.polynomial.terms),
            default=0,
        )
        if degree_sum > _MAX_ELIMINATION_DEGREE_SUM:
            raise ValueError("Sylvester degree exceeds the resultant budget")
        if (
            _resultant_support_bound(self.left, self.right, variable_index)
            > _MAX_RESULTANT_TERMS
        ):
            raise ValueError("resultant output exceeds the term budget")
        return self


class MultivariateScalarValue(StrictModel):
    kind: Literal["SCALAR"] = "SCALAR"
    value: CanonicalRational


class MultivariatePolynomialValue(StrictModel):
    kind: Literal["POLYNOMIAL"] = "POLYNOMIAL"
    value: RationalPolynomial


MultivariateInvariantValue = Annotated[
    MultivariateScalarValue | MultivariatePolynomialValue,
    Field(discriminator="kind"),
]


class MultivariateResultantResult(StrictModel):
    elimination_variable: PolynomialVariable
    resultant: MultivariateInvariantValue
    convention: Literal["SYLVESTER_DETERMINANT"] = "SYLVESTER_DETERMINANT"


class MultivariateFactorRequest(StrictModel):
    """Exact factorization request over ``QQ[variables]`` for nonzero multivariate polynomials."""

    polynomial: RationalPolynomial = Field(
        description=(
            "Nonzero multivariate polynomial in QQ[variables] with at least "
            "two variables (univariate factorization is owned by "
            "polynomial.factor.compute); terms, exponents, and coefficients "
            "must respect the operation's exact budget."
        )
    )

    @model_validator(mode="after")
    def require_factor_budget(self) -> Self:
        if len(self.polynomial.variables) < _MULTIVARIATE_MIN_VARIABLES:
            raise ValueError(
                f"multivariate factorization requires at least {_MULTIVARIATE_MIN_VARIABLES} variables; "
                "univariate polynomials are handled by polynomial.factor.compute"
            )
        if not self.polynomial.polynomial.terms:
            raise ValueError("zero polynomial has no factorization")
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_MULTIVARIATE_TERMS,
            maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
            maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
        )
        return self


class MultivariateIrreducibleFactor(StrictModel):
    factor: RationalPolynomial
    multiplicity: int = Field(ge=1, le=_MAX_MULTIVARIATE_EXPONENT)


class MultivariateFactorResult(StrictModel):
    """Exact factorization outcome over ``QQ[variables]``.

    ``FACTORIZED`` carries the full content-and-monic-irreducibles
    decomposition.  ``OUTPUT_BUDGET_EXCEEDED`` reports, as a typed bounded
    outcome, that the exact factorization contains an irreducible factor
    beyond the public output budget; ``reconstructed`` then restates the
    requested polynomial unchanged and ``factors`` is empty.
    """

    status: Literal["FACTORIZED", "OUTPUT_BUDGET_EXCEEDED"] = "FACTORIZED"
    coefficient: CanonicalRational
    factors: tuple[MultivariateIrreducibleFactor, ...] = Field(max_length=128)
    reconstructed: RationalPolynomial
    normalization: Literal["CONTENT_AND_MONIC_IRREDUCIBLES"] | None = (
        "CONTENT_AND_MONIC_IRREDUCIBLES"
    )
    product_reconstruction: Literal["EXACT"] | None = "EXACT"

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        if self.coefficient.as_fraction() == 0:
            raise ValueError("factorization coefficient must be nonzero")
        require_polynomial_budget(
            self.reconstructed,
            maximum_terms=_MAX_MULTIVARIATE_TERMS,
            maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
            maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
        )
        if self.status == "OUTPUT_BUDGET_EXCEEDED":
            if self.factors:
                raise ValueError(
                    "budget-exceeded outcomes carry no irreducible factors"
                )
            if (
                self.normalization is not None
                or self.product_reconstruction is not None
            ):
                raise ValueError(
                    "budget-exceeded outcomes declare no normalization or "
                    "product reconstruction"
                )
            _verify_output_budget_exceeded_claim(self.reconstructed)
            return self
        if not self.reconstructed.polynomial.terms:
            raise ValueError("reconstructed polynomial must be nonzero")
        _check_factor_records(self.factors, self.reconstructed.variables)
        _require_aggregate_degree_consistent(self.factors, self.reconstructed)
        _require_distinct_canonical_order(self.factors)
        _verify_monic_irreducibles(self.factors)
        _verify_exact_reconstruction(
            self.coefficient,
            self.factors,
            self.reconstructed,
        )
        return self


def _factor_content_key(record: MultivariateIrreducibleFactor) -> tuple:
    return tuple(
        (term.exponents, term.coefficient.num, term.coefficient.den)
        for term in record.factor.polynomial.terms
    )


def _factor_total_degree(record: MultivariateIrreducibleFactor) -> int:
    return max(
        (sum(term.exponents) for term in record.factor.polynomial.terms),
        default=0,
    )


def _reconstructed_total_degree(reconstructed: RationalPolynomial) -> int:
    return max(
        (sum(term.exponents) for term in reconstructed.polynomial.terms),
        default=0,
    )


def _require_aggregate_degree_consistent(
    factors: tuple[MultivariateIrreducibleFactor, ...],
    reconstructed: RationalPolynomial,
) -> None:
    """Reject aggregate degree mismatches before any product expansion.

    The exact product's total degree equals the multiplicity-weighted sum of
    factor degrees, so a prefix overshoot proves the payload cannot satisfy
    the defining invariant without expanding a prohibitive dense product.
    """

    target = _reconstructed_total_degree(reconstructed)
    aggregate = 0
    for record in factors:
        aggregate += _factor_total_degree(record) * record.multiplicity
        if aggregate > target:
            raise ValueError(
                "aggregate irreducible degree exceeds the reconstructed "
                "total degree; the factorization product cannot match"
            )


def _verify_output_budget_exceeded_claim(reconstructed: RationalPolynomial) -> None:
    """Re-derive a claimed ``OUTPUT_BUDGET_EXCEEDED`` status from its source.

    Replays the kernel's own factorization and conversion so the reported
    incompleteness is bound to the restated polynomial instead of being an
    authorable label.
    """

    from jacobian.math.polynomials._conversions import (
        rational_polynomial_from_sympy,
        rational_polynomial_to_sympy,
    )
    from jacobian.math.polynomials._sympy import _monic_decomposition

    source = rational_polynomial_to_sympy(reconstructed)
    _, raw_factors, _ = _monic_decomposition(
        source,
        source.factor_list(),
        label="multivariate factorization",
    )
    for factor, _multiplicity in raw_factors:
        try:
            rational_polynomial_from_sympy(
                factor,
                reconstructed.variables,
                maximum_terms=_MAX_FACTOR_OUTPUT_TERMS,
            )
        except ValueError as exc:
            if "term operation budget" in str(exc):
                return
            raise
    raise ValueError(
        "claimed output-budget exceedance is not reproduced by the exact "
        "factorization of the restated polynomial"
    )


def _check_factor_records(
    factors: tuple[MultivariateIrreducibleFactor, ...],
    variables: tuple[str, ...],
) -> None:
    """Enforce the reconstruction-safe envelope before any SymPy expansion."""

    for record in factors:
        if record.factor.variables != variables:
            raise ValueError("irreducible factors must use the source ring")
        require_polynomial_budget(
            record.factor,
            maximum_terms=_MAX_MULTIVARIATE_TERMS,
            maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
            maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
        )
        if _factor_total_degree(record) == 0:
            raise ValueError("irreducible factor must be non-constant")


def _require_distinct_canonical_order(
    factors: tuple[MultivariateIrreducibleFactor, ...],
) -> None:
    seen: set[tuple] = set()
    for key in (_factor_content_key(record) for record in factors):
        if key in seen:
            raise ValueError("irreducible factors must be distinct")
        seen.add(key)
    ordered = tuple(
        sorted(
            factors,
            key=lambda record: (
                record.multiplicity,
                _factor_total_degree(record),
                _factor_content_key(record),
            ),
        ),
    )
    if factors != ordered:
        raise ValueError("irreducible factors must use canonical order")


def _require_monic(poly: object, factor: RationalPolynomial) -> None:
    lc = poly.LC()
    if getattr(lc, "p", None) != 1 or getattr(lc, "q", None) != 1:
        raise ValueError(f"irreducible factor {factor} is not monic")


def _verify_monic_irreducibles(
    factors: tuple[MultivariateIrreducibleFactor, ...],
) -> None:
    """Enforce CONTENT_AND_MONIC_IRREDUCIBLES on every listed factor."""

    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy

    for record in factors:
        poly = rational_polynomial_to_sympy(record.factor)
        try:
            _require_monic(poly, record.factor)
            if not poly.is_irreducible:
                raise ValueError(f"factor {record.factor} is not irreducible")
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("invalid factor normalization check") from exc


def _verify_exact_reconstruction(
    coefficient: CanonicalRational,
    factors: tuple[MultivariateIrreducibleFactor, ...],
    reconstructed: RationalPolynomial,
) -> None:
    """Check coefficient * ∏ factor**multiplicity == reconstructed exactly."""

    from sympy import QQ, Poly
    from sympy import Rational as SymRational

    from jacobian.math.polynomials._conversions import (
        rational_polynomial_to_sympy,
        symbols_for_variables,
    )

    try:
        reconstructed_poly = rational_polynomial_to_sympy(reconstructed)
        coeff_frac = coefficient.as_fraction()
        symbols = symbols_for_variables(reconstructed.variables)
        product = Poly(
            SymRational(coeff_frac.numerator, coeff_frac.denominator),
            *symbols,
            domain=QQ,
        )
        target_degree = _reconstructed_total_degree(reconstructed)
        for record in factors:
            factor_poly = rational_polynomial_to_sympy(record.factor)
            for _ in range(record.multiplicity):
                product = product * factor_poly
                if product.total_degree() > target_degree:
                    raise ValueError(
                        "factorization product degree exceeds the "
                        "reconstructed total degree"
                    )
        if product != reconstructed_poly:
            raise ValueError(
                "factorization product does not equal reconstructed polynomial"
            )
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("invalid factorization reconstruction") from exc


__all__ = [
    "MonomialOrder",
    "MultivariateDivisionRequest",
    "MultivariateDivisionResult",
    "MultivariateFactorRequest",
    "MultivariateFactorResult",
    "MultivariateGcdRequest",
    "MultivariateGcdResult",
    "MultivariateInvariantValue",
    "MultivariateIrreducibleFactor",
    "MultivariatePolynomialValue",
    "MultivariateResultantRequest",
    "MultivariateResultantResult",
    "MultivariateScalarValue",
]
