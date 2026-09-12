"""Exact bracket-algebra kernels: canonical atoms, relations, and residuals."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Literal

from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS, CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids.oriented._bracket_models import (
    MAX_BRACKET_COEFFICIENT_DIGITS,
    MAX_BRACKET_CONTRIBUTIONS,
    MAX_BRACKET_FACTORS,
    MAX_BRACKET_GROUND_SIZE,
    MAX_BRACKET_OUTPUT_CELLS,
    MAX_BRACKET_RESULT_ALLOCATION_UNITS,
    MAX_BRACKET_TERMS,
    BracketMonomial,
    BracketPolynomial,
    BracketPolynomialTerm,
    CanonicalBracket,
    GrassmannPlueckerRelation,
    GrassmannPlueckerRelationResult,
    ordered_bracket,
)

__all__ = [
    "bracket_polynomial_from_terms",
    "bracket_syzygy_residual",
    "grassmann_pluecker_relation",
]

_RelationTerm = tuple[int, tuple[tuple[int, int, int], ...]]
_MonomialKey = tuple[tuple[tuple[int, int, int], int], ...]
_CoefficientComponent = tuple[Fraction, tuple[int, int]]

# Cache the canonical-integer envelope once. Constructing
# ``10 ** MAX_CANONICAL_INTEGER_DIGITS`` on every pair reduction recreates a
# 32,769-digit integer and dominates otherwise cheap residual admission.
_CANONICAL_INTEGER_LIMIT = 10**MAX_CANONICAL_INTEGER_DIGITS


def _combined_coefficients(
    contributions: list[tuple[Fraction, tuple[tuple[CanonicalBracket, int], ...]]],
) -> dict[_MonomialKey, Fraction]:
    """Combine signed bracket monomials into exact sparse coefficients."""

    combined: dict[_MonomialKey, Fraction] = {}
    for coefficient, brackets in contributions:
        if coefficient == 0:
            continue
        multiplicities: dict[tuple[int, int, int], int] = {}
        for bracket, multiplicity in brackets:
            multiplicities[bracket.indices] = (
                multiplicities.get(bracket.indices, 0) + multiplicity
            )
        canonical_key = tuple(sorted(multiplicities.items()))
        combined[canonical_key] = combined.get(canonical_key, Fraction(0)) + coefficient
    return {
        key: coefficient for key, coefficient in combined.items() if coefficient != 0
    }


def _polynomial_from_coefficients(
    coefficients: dict[_MonomialKey, Fraction], ground_size: int
) -> BracketPolynomial:
    """Build a canonical sparse polynomial from already grouped coefficients."""

    if len(coefficients) > MAX_BRACKET_TERMS:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_output_term_bound",
            message="the formal residual has too many sparse output terms",
        )
    terms = []
    for monomial_key in sorted(coefficients):
        coefficient = coefficients[monomial_key]
        terms.append(
            BracketPolynomialTerm(
                coefficient=CanonicalRational(
                    num=coefficient.numerator, den=coefficient.denominator
                ),
                monomial=BracketMonomial(
                    factors=tuple(
                        (CanonicalBracket(indices=triples), multiplicity)
                        for triples, multiplicity in monomial_key
                    )
                ),
            )
        )
    return BracketPolynomial(ground_size=ground_size, terms=tuple(terms))


def _combine(
    contributions: list[tuple[Fraction, tuple[tuple[CanonicalBracket, int], ...]]],
    ground_size: int,
) -> BracketPolynomial:
    """Combine signed bracket monomials into a canonical sparse polynomial."""

    return _polynomial_from_coefficients(
        _combined_coefficients(contributions), ground_size
    )


def bracket_polynomial_from_terms(
    ground_size: int,
    terms: list[tuple[Fraction, tuple[tuple[int, int, int], ...]]],
) -> BracketPolynomial:
    """Build a canonical bracket polynomial from signed ordered triples."""

    contributions: list[tuple[Fraction, tuple[tuple[CanonicalBracket, int], ...]]] = []
    for coefficient, ordered_triples in terms:
        brackets: list[CanonicalBracket] = []
        sign = 1
        for ordered in ordered_triples:
            bracket, parity = ordered_bracket(ordered)
            brackets.append(bracket)
            sign *= parity
        contributions.append(
            (coefficient * sign, tuple((bracket, 1) for bracket in brackets))
        )
    return _combine(contributions, ground_size)


def _four_term_relation(indices: tuple[int, ...]) -> list[_RelationTerm]:
    """The four-term rank-3 Grassmann-Pluecker relation on six indices.

    ``[a b c][d e f] - [a b d][c e f] + [a b e][c d f] - [a b f][c d e] = 0``
    holds for every 3xN matrix; it is the Pluecker relation on the six columns.
    """

    a, b, c, d, e, f = indices
    return [
        (1, ((a, b, c), (d, e, f))),
        (-1, ((a, b, d), (c, e, f))),
        (1, ((a, b, e), (c, d, f))),
        (-1, ((a, b, f), (c, d, e))),
    ]


def _shared_index_three_term_relation(indices: tuple[int, ...]) -> list[_RelationTerm]:
    """The three-term Grassmann-Pluecker relation sharing one index.

    For five distinct indices ``(a,b,c,d,e)``, the bisyllabic relation
    ``[a b c][a d e] - [a b d][a c e] + [a b e][a c d] = 0`` holds for every
    3xN matrix: the shared index ``a`` makes it the Pluecker relation on the
    two-dimensional quotient by the column ``a``.
    """

    a, b, c, d, e = indices
    return [
        (1, ((a, b, c), (a, d, e))),
        (-1, ((a, b, d), (a, c, e))),
        (1, ((a, b, e), (a, c, d))),
    ]


def _require_grassmann_pluecker_relation(
    ground_size: int,
    indices: tuple[int, ...],
    family: Literal["FOUR_TERM", "SHARED_INDEX_THREE_TERM"],
) -> None:
    """Admit one rank-3 Grassmann-Pluecker pattern without a wire request."""

    if family == "FOUR_TERM":
        expected = 6
    elif family == "SHARED_INDEX_THREE_TERM":
        expected = 5
    else:
        raise OperationDomainValidationError(
            location=("family",),
            code="bracket.relation_family",
            message="a rank-3 Grassmann-Pluecker relation is four-term or shared-index",
        )
    if type(ground_size) is not int or not (
        3 <= ground_size <= MAX_BRACKET_GROUND_SIZE
    ):
        raise OperationDomainValidationError(
            location=("ground_size",),
            code="bracket.relation_ground_size",
            message="the declared ground range must lie inside the admitted envelope",
        )
    if not isinstance(indices, tuple) or len(indices) != expected:
        raise OperationDomainValidationError(
            location=("indices",),
            code="bracket.relation_index_count",
            message=f"{family} requires exactly {expected} indices",
        )
    if ground_size < expected:
        raise OperationDomainValidationError(
            location=("ground_size",),
            code="bracket.relation_ground_too_small",
            message="the declared ground range must contain every relation index",
        )
    if any(
        type(index) is not int or index < 0 or index >= ground_size for index in indices
    ):
        raise OperationDomainValidationError(
            location=("indices",),
            code="bracket.relation_index_outside_ground",
            message="every relation index must lie inside the declared ground range",
        )
    if len(set(indices)) != len(indices):
        raise OperationDomainValidationError(
            location=("indices",),
            code="bracket.relation_indices_not_distinct",
            message="a Grassmann-Pluecker relation requires distinct indices",
        )


def _relation_polynomial(
    ground_size: int,
    indices: tuple[int, ...],
    family: Literal["FOUR_TERM", "SHARED_INDEX_THREE_TERM"],
) -> BracketPolynomial:
    """Build the one canonical polynomial for a validated relation pattern."""

    ordered_terms = (
        _four_term_relation(indices)
        if family == "FOUR_TERM"
        else _shared_index_three_term_relation(indices)
    )
    contributions: list[tuple[Fraction, tuple[tuple[CanonicalBracket, int], ...]]] = []
    for coefficient, ordered_triples in ordered_terms:
        brackets: list[CanonicalBracket] = []
        sign = coefficient
        for ordered in ordered_triples:
            try:
                bracket, parity = ordered_bracket(ordered)
            except PydanticCustomError as error:
                raise OperationDomainValidationError(
                    location=("indices",),
                    code="bracket.relation_degenerate_indices",
                    message=(
                        "a Grassmann-Pluecker relation requires distinct indices: "
                        f"{error.message()}"
                    ),
                ) from error
            brackets.append(bracket)
            sign *= parity
        contributions.append(
            (Fraction(sign), tuple((bracket, 1) for bracket in brackets))
        )
    return _combine(contributions, ground_size)


def _admit_source_relation(relation: GrassmannPlueckerRelation) -> None:
    """Admit the caller's source-bound relation claim once per request."""

    expected = _relation_polynomial(
        relation.ground_size,
        relation.indices,
        relation.family,
    )
    if relation.polynomial != expected:
        raise OperationDomainValidationError(
            location=("terms",),
            code="bracket.syzygy_source_relation_mismatch",
            message=(
                "the supplied polynomial does not match its Grassmann-Pluecker "
                "source metadata"
            ),
        )


def _integer_digit_upper_bound(value: int) -> int:
    """Return a cheap sound decimal-digit upper bound without formatting it."""

    if value == 0:
        return 1
    # log10(2) < 30103 / 100000, so this rounds upwards for every bit length.
    return min(
        (abs(value).bit_length() * 30103 + 99_999) // 100_000,
        MAX_BRACKET_COEFFICIENT_DIGITS,
    )


def _rational_component_digit_upper_bound(value: CanonicalRational) -> tuple[int, int]:
    return (
        _integer_digit_upper_bound(value.num),
        _integer_digit_upper_bound(value.den),
    )


def _rational_product_digit_upper_bound(
    left: CanonicalRational, right: CanonicalRational
) -> tuple[int, int]:
    """Bound component widths for a rational product without charging units."""

    left_num_digits, left_den_digits = _rational_component_digit_upper_bound(left)
    right_num_digits, right_den_digits = _rational_component_digit_upper_bound(right)
    if right.den == 1 and abs(right.num) == 1:
        return left_num_digits, left_den_digits
    if left.den == 1 and abs(left.num) == 1:
        return right_num_digits, right_den_digits
    return left_num_digits + right_num_digits, left_den_digits + right_den_digits


def _cancel_opposite_components(
    components: list[_CoefficientComponent],
) -> list[_CoefficientComponent]:
    """Remove equal and opposite exact coefficient components."""

    pending: dict[Fraction, list[tuple[int, int]]] = {}
    for value, widths in components:
        opposite = -value
        opposite_widths = pending.get(opposite)
        if opposite_widths:
            opposite_widths.pop()
            if not opposite_widths:
                del pending[opposite]
        else:
            pending.setdefault(value, []).append(widths)
    return [
        (value, widths)
        for value, component_widths in pending.items()
        for widths in component_widths
    ]


def _exceeds_canonical_integer_bound(value: int) -> bool:
    """Check a nonnegative output integer without formatting huge values."""

    if value.bit_length() <= 3 * MAX_CANONICAL_INTEGER_DIGITS:
        return False
    return bool(value >= _CANONICAL_INTEGER_LIMIT)


def _integer_product_exceeds_canonical(left: int, right: int) -> bool:
    """Return whether ``|left| * |right|`` exceeds the canonical integer envelope."""

    left_abs = abs(left)
    right_abs = abs(right)
    if left_abs == 0 or right_abs == 0:
        return False
    if left_abs == 1:
        return _exceeds_canonical_integer_bound(right_abs)
    if right_abs == 1:
        return _exceeds_canonical_integer_bound(left_abs)
    return bool(left_abs > (_CANONICAL_INTEGER_LIMIT - 1) // right_abs)


def _bounded_component_sum(
    components: list[_CoefficientComponent],
) -> tuple[Fraction, int]:
    """Sum cancelled components and admit the exact canonical coefficient.

    Pairwise merge search is not an admission strategy: the mathematical
    coefficient is the exact rational sum. Denominator LCMs are bounded
    through successive gcd cofactors before any ``Fraction`` addition, so
    pairwise-coprime 32,768-digit denominators reject without constructing
    a million-digit common denominator.
    """

    pending = _cancel_opposite_components(components)
    total = Fraction(0)
    work_digit_bound = 0
    denominator_lcm = 1
    for value, widths in pending:
        denominator = value.denominator
        shared = gcd(denominator_lcm, denominator)
        cofactor = denominator // shared
        if _integer_product_exceeds_canonical(denominator_lcm, cofactor):
            raise OperationResourceAdmissionError(
                location=("terms",),
                code="bracket.syzygy_coefficient_digit_bound",
                message=(
                    "exact residual coefficient growth exceeds the supported "
                    "digit bound"
                ),
            )
        denominator_lcm *= cofactor
        total += value
        work_digit_bound = max(work_digit_bound, widths[0], widths[1])
    if total:
        work_digit_bound = max(
            work_digit_bound,
            _integer_digit_upper_bound(total.numerator),
            _integer_digit_upper_bound(total.denominator),
        )
        if _exceeds_canonical_integer_bound(
            abs(total.numerator)
        ) or _exceeds_canonical_integer_bound(total.denominator):
            raise OperationResourceAdmissionError(
                location=("terms",),
                code="bracket.syzygy_coefficient_digit_bound",
                message=(
                    "exact residual coefficient growth exceeds the supported "
                    "digit bound"
                ),
            )
    return total, work_digit_bound


def _admit_result_allocation(
    coefficients: dict[_MonomialKey, Fraction],
    ground_size: int,
) -> None:
    """Admit retained scalar digits and structural slots for the residual."""

    index_digits = _integer_digit_upper_bound(ground_size - 1)
    term_cells = len(coefficients)
    factor_cells = sum(len(key) for key in coefficients)
    output_cells = term_cells + factor_cells
    allocation_units = 256 + sum(
        256
        + 2
        * max(
            _integer_digit_upper_bound(coefficient.numerator),
            _integer_digit_upper_bound(coefficient.denominator),
        )
        for coefficient in coefficients.values()
    )
    allocation_units += sum(
        128 + 3 * index_digits + _integer_digit_upper_bound(multiplicity)
        for key in coefficients
        for _, multiplicity in key
    )
    # Keep the cell count explicit: this prevents a future change to the
    # carrier shape from silently dropping a factor from the estimate.
    if output_cells > MAX_BRACKET_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_output_cell_bound",
            message="the exact sparse residual output exceeds the supported cell bound",
        )
    if allocation_units > MAX_BRACKET_RESULT_ALLOCATION_UNITS:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_result_allocation_bound",
            message="the exact residual exceeds its retained allocation bound",
        )


def _admit_surviving_components(
    coefficient_components: dict[_MonomialKey, list[_CoefficientComponent]],
) -> tuple[dict[_MonomialKey, Fraction], int]:
    """Return an admitted grouped coefficient plan and its width bound."""

    coefficients: dict[_MonomialKey, Fraction] = {}
    coefficient_digit_bound = 0
    for key, components in coefficient_components.items():
        total, local_digit_bound = _bounded_component_sum(components)
        if total == 0:
            coefficient_digit_bound = max(coefficient_digit_bound, local_digit_bound)
            continue

        if _exceeds_canonical_integer_bound(
            abs(total.numerator)
        ) or _exceeds_canonical_integer_bound(total.denominator):
            raise OperationResourceAdmissionError(
                location=("terms",),
                code="bracket.syzygy_coefficient_digit_bound",
                message="exact residual coefficient growth exceeds the supported digit bound",
            )
        coefficient_digit_bound = max(
            coefficient_digit_bound,
            local_digit_bound,
            _integer_digit_upper_bound(total.numerator),
            _integer_digit_upper_bound(total.denominator),
        )
        coefficients[key] = total
    if len(coefficients) > MAX_BRACKET_TERMS:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_output_term_bound",
            message="the formal residual has too many sparse output terms",
        )
    return coefficients, coefficient_digit_bound


def _require_syzygy_terms(
    target: BracketPolynomial,
    terms: object,
) -> tuple[tuple[CanonicalRational, BracketMonomial, GrassmannPlueckerRelation], ...]:
    """Admit native syzygy term shape without a wire request model."""

    if not isinstance(terms, tuple):
        raise OperationDomainValidationError(
            location=("terms",),
            code="bracket.syzygy_terms_type",
            message="syzygy terms must be a tuple of scalar, multiplier, and relation",
        )
    if len(terms) > 128:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_term_count",
            message="a syzygy residual admits at most 128 combination terms",
        )
    admitted: list[
        tuple[CanonicalRational, BracketMonomial, GrassmannPlueckerRelation]
    ] = []
    for term in terms:
        if not isinstance(term, tuple) or len(term) != 3:
            raise OperationDomainValidationError(
                location=("terms",),
                code="bracket.syzygy_term_shape",
                message=(
                    "each syzygy term is a scalar, a multiplier monomial, and a "
                    "source-bound Grassmann-Pluecker relation"
                ),
            )
        scalar, multiplier, relation = term
        if (
            not isinstance(scalar, CanonicalRational)
            or not isinstance(multiplier, BracketMonomial)
            or not isinstance(relation, GrassmannPlueckerRelation)
        ):
            raise OperationDomainValidationError(
                location=("terms",),
                code="bracket.syzygy_term_shape",
                message=(
                    "each syzygy term is a scalar, a multiplier monomial, and a "
                    "source-bound Grassmann-Pluecker relation"
                ),
            )
        if relation.ground_size != target.ground_size:
            raise OperationDomainValidationError(
                location=("terms",),
                code="bracket.syzygy_ground_mismatch",
                message="target and every supplied relation must share one ground range",
            )
        if any(
            factor.indices[2] >= target.ground_size for factor, _ in multiplier.factors
        ):
            raise OperationDomainValidationError(
                location=("terms",),
                code="bracket.syzygy_multiplier_index_outside_ground",
                message="every multiplier bracket index must lie in the target ground range",
            )
        admitted.append((scalar, multiplier, relation))
    return tuple(admitted)


def _admit_residual_envelope(
    target: BracketPolynomial,
    terms: tuple[
        tuple[CanonicalRational, BracketMonomial, GrassmannPlueckerRelation], ...
    ],
) -> dict[_MonomialKey, Fraction]:
    """Admit source claims, sparse output, and exact coefficient growth up front."""

    terms = _require_syzygy_terms(target, terms)

    active_terms = tuple(
        (scalar, multiplier, relation)
        for scalar, multiplier, relation in terms
        if scalar.num != 0
    )
    contribution_count = len(target.terms) + sum(
        len(relation.polynomial.terms) for _, _, relation in active_terms
    )
    if contribution_count > MAX_BRACKET_CONTRIBUTIONS:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_contribution_bound",
            message="the formal residual contribution count exceeds the bound",
        )

    for _, _, relation in active_terms:
        _admit_source_relation(relation)

    def admit_monomial(
        factors: tuple[tuple[CanonicalBracket, int], ...],
    ) -> _MonomialKey:
        multiplicities: dict[tuple[int, int, int], int] = {}
        for factor, multiplicity in factors:
            multiplicities[factor.indices] = (
                multiplicities.get(factor.indices, 0) + multiplicity
            )
        if any(
            _exceeds_canonical_integer_bound(value) for value in multiplicities.values()
        ):
            raise OperationResourceAdmissionError(
                location=("terms",),
                code="bracket.syzygy_multiplicity_digit_bound",
                message=(
                    "assembled bracket-factor multiplicity exceeds the canonical "
                    "integer representation envelope"
                ),
            )
        if len(multiplicities) > MAX_BRACKET_FACTORS:
            raise OperationDomainValidationError(
                location=("terms",),
                code="bracket.syzygy_monomial_factors",
                message="every assembled monomial must fit the bracket-factor bound",
            )
        return tuple(sorted(multiplicities.items()))

    coefficient_components: dict[_MonomialKey, list[_CoefficientComponent]] = {}
    for term in target.terms:
        key = admit_monomial(term.monomial.factors)
        coefficient_components.setdefault(key, []).append(
            (
                term.coefficient.as_fraction(),
                _rational_component_digit_upper_bound(term.coefficient),
            )
        )
    for scalar, multiplier, relation in active_terms:
        scalar_value = scalar.as_fraction()
        for term in relation.polynomial.terms:
            key = admit_monomial(multiplier.factors + term.monomial.factors)
            if term.coefficient.num == 1:
                coefficient = -scalar_value
            elif term.coefficient.num == -1:
                coefficient = scalar_value
            else:
                coefficient = -scalar_value * term.coefficient.as_fraction()
            coefficient_components.setdefault(key, []).append(
                (
                    coefficient,
                    _rational_product_digit_upper_bound(scalar, term.coefficient),
                )
            )

    if not coefficient_components:
        return {}
    coefficients, coefficient_digit_bound = _admit_surviving_components(
        coefficient_components
    )
    if coefficient_digit_bound > MAX_BRACKET_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_coefficient_digit_bound",
            message="exact residual coefficient growth exceeds the supported digit bound",
        )
    _admit_result_allocation(coefficients, target.ground_size)
    return coefficients


def grassmann_pluecker_relation(
    ground_size: int,
    indices: tuple[int, ...],
    family: Literal["FOUR_TERM", "SHARED_INDEX_THREE_TERM"],
) -> GrassmannPlueckerRelationResult:
    """Return the canonical formal expression of one GP relation."""

    _require_grassmann_pluecker_relation(ground_size, indices, family)
    polynomial = _relation_polynomial(ground_size, indices, family)
    return GrassmannPlueckerRelationResult(
        ground_size=ground_size,
        indices=indices,
        family=family,
        polynomial=polynomial,
    )


def bracket_syzygy_residual(
    target: BracketPolynomial,
    terms: tuple[
        tuple[CanonicalRational, BracketMonomial, GrassmannPlueckerRelation], ...
    ] = (),
) -> BracketPolynomial:
    """Return ``target - sum(scalar * multiplier * relation)`` exactly.

    The arithmetic is in the free commutative polynomial algebra on canonical
    bracket atoms.  Each supplied relation is admitted against its retained
    indexed source, but this operation does not infer that a nonzero residual
    vanishes on minors.
    """

    coefficients = _admit_residual_envelope(target, terms)
    return _polynomial_from_coefficients(coefficients, target.ground_size)
