"""Structural contracts for multivariate factorization."""

from __future__ import annotations

from fractions import Fraction
from functools import reduce
from math import gcd, lcm
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.multivariate._models import (
    _MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
    _MAX_MULTIVARIATE_EXPONENT,
    _MAX_MULTIVARIATE_TERMS,
    _MULTIVARIATE_MIN_VARIABLES,
    _validation_error,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    require_polynomial_budget,
)

# Public output-term budget for one converted irreducible factor.
_MAX_FACTOR_OUTPUT_TERMS = 1_024


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
            raise _validation_error(
                f"multivariate factorization requires at least {_MULTIVARIATE_MIN_VARIABLES} variables; "
                "univariate polynomials are handled by polynomial.factor.compute"
            )
        if not self.polynomial.polynomial.terms:
            raise _validation_error("zero polynomial has no factorization")
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_MULTIVARIATE_TERMS,
            maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
            maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
        )
        _require_representable_content(self.polynomial)
        return self


def _require_representable_content(polynomial: RationalPolynomial) -> None:
    """Bound aggregate rational content before the factor backend expands it."""

    fractions = [term.coefficient.as_fraction() for term in polynomial.polynomial.terms]
    common_denominator = reduce(lcm, (value.denominator for value in fractions), 1)
    scaled = [
        value.numerator * (common_denominator // value.denominator)
        for value in fractions
    ]
    content_numerator = gcd(*scaled)
    canonical_bound = 10**MAX_CANONICAL_RATIONAL_DIGITS
    if (
        abs(content_numerator) >= canonical_bound
        or common_denominator >= canonical_bound
    ):
        raise _validation_error(
            "aggregate rational content exceeds the "
            f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit representable bound"
        )
    primitive_bound = 10**_MAX_MULTIVARIATE_COEFFICIENT_DIGITS
    for value in scaled:
        if abs(value // content_numerator) >= primitive_bound:
            raise _validation_error(
                "primitive integer coefficients exceed the "
                f"{_MAX_MULTIVARIATE_COEFFICIENT_DIGITS}-digit operation budget"
            )


class MultivariateIrreducibleFactor(StrictModel):
    factor: RationalPolynomial
    multiplicity: int = Field(ge=1, le=_MAX_MULTIVARIATE_EXPONENT)


class MultivariateFactorResult(StrictModel):
    """Exact factorization outcome over ``QQ[variables]``.

    ``FACTORIZED`` carries the full content-and-monic-irreducibles
    decomposition.  ``OUTPUT_BUDGET_EXCEEDED`` reports, as a typed bounded
    outcome, that the exact factorization is beyond this operation's
    public output bounds: either an irreducible factor exceeds the public
    output-term budget or the serialized exact decomposition exceeded the
    declared transport bound.  ``EXECUTION_FAILED`` is not a mathematical
    conclusion: the worker was stopped by its deadline or cancellation,
    killed by an enforced resource cap such as its CPU or address-space
    budget, crashed, or could not be contained, so no factorization was
    obtained and callers may retry.
    For both non-FACTORIZED statuses ``reconstructed`` restates the
    requested polynomial unchanged, ``coefficient`` carries the exact
    positive rational content of that polynomial, and ``factors`` is empty.
    """

    status: Literal[
        "FACTORIZED",
        "OUTPUT_BUDGET_EXCEEDED",
        "EXECUTION_FAILED",
    ] = "FACTORIZED"
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
            raise _validation_error("factorization coefficient must be nonzero")
        require_polynomial_budget(
            self.reconstructed,
            maximum_terms=_MAX_MULTIVARIATE_TERMS,
            maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
            maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
        )
        if not self.reconstructed.polynomial.terms:
            raise _validation_error("reconstructed polynomial must be nonzero")
        if self.status != "FACTORIZED":
            if self.factors:
                raise _validation_error(
                    "non-FACTORIZED outcomes carry no irreducible factors"
                )
            if (
                self.normalization is not None
                or self.product_reconstruction is not None
            ):
                raise _validation_error(
                    "non-FACTORIZED outcomes declare no normalization or "
                    "product reconstruction"
                )
            if _primitive_content_fraction(self.reconstructed) != (
                self.coefficient.as_fraction()
            ):
                raise _validation_error(
                    "outcome coefficient does not match the exact content "
                    "of the restated polynomial"
                )
            return self
        if (
            self.normalization != "CONTENT_AND_MONIC_IRREDUCIBLES"
            or self.product_reconstruction != "EXACT"
        ):
            raise _validation_error(
                "FACTORIZED outcomes declare content-and-monic-irreducibles "
                "normalization and exact product reconstruction"
            )
        _check_factor_records(self.factors, self.reconstructed.variables)
        _require_aggregate_degree_consistent(self.factors, self.reconstructed)
        _require_distinct_canonical_order(self.factors)
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        status: Literal["FACTORIZED", "OUTPUT_BUDGET_EXCEEDED", "EXECUTION_FAILED"],
        coefficient: CanonicalRational,
        factors: tuple[MultivariateIrreducibleFactor, ...],
        reconstructed: RationalPolynomial,
    ) -> Self:
        """Build one result from the operation's already-checked kernel output.

        The public model intentionally retains structural validation only.
        An independently supplied claim can be checked with the explicit
        owner verifier, which is the only path that may replay the bounded
        factorization worker.
        """

        return cls(
            status=status,
            coefficient=coefficient,
            factors=factors,
            reconstructed=reconstructed,
            normalization=(
                "CONTENT_AND_MONIC_IRREDUCIBLES" if status == "FACTORIZED" else None
            ),
            product_reconstruction="EXACT" if status == "FACTORIZED" else None,
        )


_FactorContentKey = tuple[tuple[tuple[int, ...], str, str], ...]


def _factor_content_key(record: MultivariateIrreducibleFactor) -> _FactorContentKey:
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
    """Reject aggregate degree mismatches before any product expansion."""

    target = _reconstructed_total_degree(reconstructed)
    aggregate = 0
    for record in factors:
        aggregate += _factor_total_degree(record) * record.multiplicity
        if aggregate > target:
            raise _validation_error(
                "aggregate irreducible degree exceeds the reconstructed "
                "total degree; the factorization product cannot match"
            )


def _primitive_content_fraction(polynomial: RationalPolynomial) -> Fraction:
    """Return exact positive rational content without entering a backend."""

    fractions = [term.coefficient.as_fraction() for term in polynomial.polynomial.terms]
    common_denominator = reduce(lcm, (value.denominator for value in fractions), 1)
    scaled = [
        value.numerator * (common_denominator // value.denominator)
        for value in fractions
    ]
    return Fraction(gcd(*scaled), common_denominator)


def _check_factor_records(
    factors: tuple[MultivariateIrreducibleFactor, ...],
    variables: tuple[str, ...],
) -> None:
    """Enforce the reconstruction-safe envelope before any SymPy expansion."""

    for record in factors:
        if record.factor.variables != variables:
            raise _validation_error("irreducible factors must use the source ring")
        require_polynomial_budget(
            record.factor,
            maximum_terms=_MAX_FACTOR_OUTPUT_TERMS,
            maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
            maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
        )
        if _factor_total_degree(record) == 0:
            raise _validation_error("irreducible factor must be non-constant")


def _require_distinct_canonical_order(
    factors: tuple[MultivariateIrreducibleFactor, ...],
) -> None:
    seen: set[_FactorContentKey] = set()
    for key in (_factor_content_key(record) for record in factors):
        if key in seen:
            raise _validation_error("irreducible factors must be distinct")
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
        raise _validation_error("irreducible factors must use canonical order")


__all__ = [
    "MultivariateFactorRequest",
    "MultivariateFactorResult",
    "MultivariateIrreducibleFactor",
]
