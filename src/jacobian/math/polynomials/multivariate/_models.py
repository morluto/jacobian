"""Contracts for exact multivariate polynomial operations over ``QQ``."""

from __future__ import annotations

from fractions import Fraction
from functools import reduce
from math import comb, gcd, lcm
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
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
        _require_representable_content(self.polynomial)
        return self


def _require_representable_content(polynomial: RationalPolynomial) -> None:
    """Bound the aggregate rational content before any backend expansion.

    Clearing denominators to the least common multiple of all term
    denominators produces the primitive integer representative, so per-term
    digit budgets bound neither the rational content every result carries as
    one canonical rational nor the primitive coefficients the reconstructed
    polynomial and its monic factors must publish.  Wire terms are reduced,
    so the cleared content is already reduced, and both derived envelopes
    are checked here with exact integer arithmetic on the admitted terms.
    """

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
        raise ValueError(
            "aggregate rational content exceeds the "
            f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit representable bound"
        )
    primitive_bound = 10**_MAX_MULTIVARIATE_COEFFICIENT_DIGITS
    for value in scaled:
        if abs(value // content_numerator) >= primitive_bound:
            raise ValueError(
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
            raise ValueError("factorization coefficient must be nonzero")
        require_polynomial_budget(
            self.reconstructed,
            maximum_terms=_MAX_MULTIVARIATE_TERMS,
            maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
            maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
        )
        if self.status != "FACTORIZED":
            from jacobian.math.polynomials.multivariate import _factor_backend

            if self.factors:
                raise ValueError("non-FACTORIZED outcomes carry no irreducible factors")
            if (
                self.normalization is not None
                or self.product_reconstruction is not None
            ):
                raise ValueError(
                    "non-FACTORIZED outcomes declare no normalization or "
                    "product reconstruction"
                )
            if _factor_backend.primitive_content_fraction(self.reconstructed) != (
                self.coefficient.as_fraction()
            ):
                raise ValueError(
                    "outcome coefficient does not match the exact content "
                    "of the restated polynomial"
                )
            if self.status == "OUTPUT_BUDGET_EXCEEDED":
                # A capacity claim IS a mathematical claim about the exact
                # output, so replay it.  An interrupted replay establishes
                # nothing and must not authenticate the claim; a replayed
                # resource/output exceedance does.
                _verify_output_budget_exceeded_claim(
                    self.coefficient, self.reconstructed
                )
            return self
        if (
            self.normalization != "CONTENT_AND_MONIC_IRREDUCIBLES"
            or self.product_reconstruction != "EXACT"
        ):
            raise ValueError(
                "FACTORIZED outcomes declare content-and-monic-irreducibles "
                "normalization and exact product reconstruction"
            )
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


_FactorContentKey = tuple[tuple[tuple[int, ...], str, str], ...]
_SympyFactorKey = tuple[tuple[tuple[int, ...], int, int], ...]


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


def _monic_content_fraction(content: Any) -> Fraction:
    """Extract the exact rational content returned by ``_monic_decomposition``."""

    leading = getattr(content, "LC", None)
    value = leading() if callable(leading) else content
    return Fraction(int(value.p), int(value.q))


def _verify_output_budget_exceeded_claim(
    coefficient: CanonicalRational,
    reconstructed: RationalPolynomial,
) -> None:
    """Re-derive a claimed ``OUTPUT_BUDGET_EXCEEDED`` status from its source.

    Replays the kernel's own bounded factorization so the reported
    incompleteness is bound to the restated polynomial instead of being an
    authorable label: the exact rational content is recomputed cheaply and
    compared, and the claim is reproduced only when a replayed factor
    conversion exceeds the output-term budget or the replayed run again
    exceeds the declared transport bound on the serialized decomposition.
    An interrupted replay — deadline, cancellation, or resource-cap kill
    such as worker memory exhaustion — establishes nothing about output
    size and fails closed instead of authenticating the claim.
    """

    from jacobian.math.polynomials.multivariate import _factor_backend
    from jacobian.math.polynomials.multivariate._factor_backend import (
        FactorBackendExhaustedError,
        FactorBackendInterruptedError,
    )

    if _factor_backend.primitive_content_fraction(reconstructed) != (
        coefficient.as_fraction()
    ):
        raise ValueError(
            "budget-exceeded outcome coefficient does not match the exact "
            "content of the restated polynomial"
        )
    try:
        decomposition = _factor_backend.run_bounded_factorization(
            reconstructed,
            wall_seconds=_factor_backend.FACTOR_VERIFY_WALL_SECONDS,
        )
    except FactorBackendExhaustedError:
        # The replay hit the same declared transport bound on the
        # serialized decomposition: the claimed beyond-bounds behavior of
        # this exact source is reproduced.  An interrupted replay —
        # deadline, cancellation, or a resource-cap kill such as worker
        # memory exhaustion — proves nothing and must not validate the
        # claim, so only exhaustion returns here.
        return
    except FactorBackendInterruptedError as exc:
        raise ValueError(
            "budget-exceeded outcome could not be re-derived because the "
            "verification replay was itself stopped before completing"
        ) from exc
    from jacobian.math.polynomials._conversions import (
        rational_polynomial_from_sympy,
        rational_polynomial_to_sympy,
    )
    from jacobian.math.polynomials._sympy import _monic_decomposition

    source = rational_polynomial_to_sympy(reconstructed)
    _content, raw_factors, _reconstructed = _monic_decomposition(
        source,
        decomposition,
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
    """Enforce the reconstruction-safe envelope before any SymPy expansion.

    Factor records are kernel outputs, so their term budget is the output
    conversion budget rather than the request envelope.
    """

    for record in factors:
        if record.factor.variables != variables:
            raise ValueError("irreducible factors must use the source ring")
        require_polynomial_budget(
            record.factor,
            maximum_terms=_MAX_FACTOR_OUTPUT_TERMS,
            maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
            maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
        )
        if _factor_total_degree(record) == 0:
            raise ValueError("irreducible factor must be non-constant")


def _require_distinct_canonical_order(
    factors: tuple[MultivariateIrreducibleFactor, ...],
) -> None:
    seen: set[_FactorContentKey] = set()
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


def _require_monic(poly: Any, factor: RationalPolynomial) -> None:
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


def _sympy_factor_key(poly: Any) -> _SympyFactorKey:
    """Return the canonical hashable form of one monic QQ ``Poly``."""

    return tuple(
        sorted(
            (tuple(monom), int(coeff.p), int(coeff.q)) for monom, coeff in poly.terms()
        )
    )


def _verify_exact_reconstruction(
    coefficient: CanonicalRational,
    factors: tuple[MultivariateIrreducibleFactor, ...],
    reconstructed: RationalPolynomial,
) -> None:
    """Check coefficient * ∏ factor**multiplicity == reconstructed exactly.

    The replay recomputes the exact content and the canonical monic
    irreducible multiset of the retained source polynomial with the same
    bounded, killable ``factor_list`` invocation the operation itself
    performs, then compares it against the claimed decomposition.  Monic
    irreducible factorization over ``QQ[variables]`` is unique, so matching
    multisets establish the product identity without expanding any
    intermediate product; partial products of admitted factorizations can be
    exponentially denser than their source (paired cyclotomic sums reach
    4^7 * 2 = 32,768 terms for an 8-variable input of 256 terms), so a
    division replay cannot carry a cofactor bound that covers every
    admitted factorization.  The verification cost is one killable worker
    call on the already-admitted source envelope.
    """

    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
    from jacobian.math.polynomials._sympy import _monic_decomposition
    from jacobian.math.polynomials.multivariate import _factor_backend
    from jacobian.math.polynomials.multivariate._factor_backend import (
        FactorBackendExhaustedError,
        FactorBackendInterruptedError,
    )

    try:
        decomposition = _factor_backend.run_bounded_factorization(
            reconstructed,
            wall_seconds=_factor_backend.FACTOR_VERIFY_WALL_SECONDS,
        )
        source = rational_polynomial_to_sympy(reconstructed)
        content, raw_factors, _ = _monic_decomposition(
            source,
            decomposition,
            label="multivariate factorization",
        )
        claimed: dict[_SympyFactorKey, int] = {}
        for record in factors:
            key = _sympy_factor_key(rational_polynomial_to_sympy(record.factor))
            claimed[key] = claimed.get(key, 0) + record.multiplicity
        replayed: dict[_SympyFactorKey, int] = {}
        for factor, multiplicity in raw_factors:
            key = _sympy_factor_key(factor)
            replayed[key] = replayed.get(key, 0) + multiplicity
        if (
            _monic_content_fraction(content) != coefficient.as_fraction()
            or claimed != replayed
        ):
            raise ValueError(
                "factorization product does not equal reconstructed polynomial"
            )
    except ValueError:
        raise
    except (FactorBackendExhaustedError, FactorBackendInterruptedError) as exc:
        raise ValueError(
            "factorization verification could not reproduce the exact "
            "factorization within the declared work budget"
        ) from exc
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
