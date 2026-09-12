"""Exact bracket-algebra kernels: canonical atoms, relations, and residuals."""

from __future__ import annotations

from fractions import Fraction

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids.oriented._bracket_models import (
    MAX_BRACKET_COEFFICIENT_DIGITS,
    MAX_BRACKET_CONTRIBUTIONS,
    MAX_BRACKET_FACTORS,
    MAX_BRACKET_OUTPUT_CELLS,
    MAX_BRACKET_TERMS,
    BracketMonomial,
    BracketPolynomial,
    BracketPolynomialTerm,
    BracketSyzygyResidualRequest,
    CanonicalBracket,
    GrassmannPlueckerRelation,
    GrassmannPlueckerRelationRequest,
    GrassmannPlueckerRelationResult,
    ordered_bracket,
)

__all__ = [
    "bracket_polynomial_from_terms",
    "bracket_syzygy_residual",
    "grassmann_pluecker_relation",
]

_RelationTerm = tuple[int, tuple[tuple[int, int, int], ...]]


def _combine(
    contributions: list[tuple[Fraction, tuple[tuple[CanonicalBracket, int], ...]]],
    ground_size: int,
) -> BracketPolynomial:
    """Combine signed bracket monomials into a canonical sparse polynomial."""

    combined: dict[tuple[tuple[tuple[int, int, int], int], ...], Fraction] = {}
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
    if len(combined) > MAX_BRACKET_TERMS:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_output_term_bound",
            message="the formal residual has too many sparse output terms",
        )
    terms = []
    for monomial_key in sorted(combined):
        coefficient = combined[monomial_key]
        if coefficient == 0:
            continue
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


def _relation_polynomial(
    request: GrassmannPlueckerRelationRequest,
) -> BracketPolynomial:
    """Build the one canonical polynomial for a validated relation request."""

    ordered_terms = (
        _four_term_relation(request.indices)
        if request.family == "FOUR_TERM"
        else _shared_index_three_term_relation(request.indices)
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
    return _combine(contributions, request.ground_size)


def _admit_source_relation(relation: GrassmannPlueckerRelation) -> None:
    """Admit the caller's source-bound relation claim once per request."""

    expected = _relation_polynomial(
        GrassmannPlueckerRelationRequest(
            ground_size=relation.ground_size,
            indices=relation.indices,
            family=relation.family,
        )
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


def _admit_residual_envelope(request: BracketSyzygyResidualRequest) -> None:
    """Admit source claims, sparse output, and exact coefficient growth up front."""

    contribution_count = len(request.target.terms) + sum(
        len(relation.polynomial.terms) for _, _, relation in request.terms
    )
    if contribution_count > MAX_BRACKET_CONTRIBUTIONS:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_contribution_bound",
            message="the formal residual contribution count exceeds the bound",
        )

    for _, _, relation in request.terms:
        _admit_source_relation(relation)

    output_keys: set[tuple[tuple[tuple[int, int, int], int], ...]] = set()

    def admit_monomial(
        factors: tuple[tuple[CanonicalBracket, int], ...],
    ) -> tuple[tuple[tuple[int, int, int], int], ...]:
        multiplicities: dict[tuple[int, int, int], int] = {}
        for factor, multiplicity in factors:
            multiplicities[factor.indices] = (
                multiplicities.get(factor.indices, 0) + multiplicity
            )
        if len(multiplicities) > MAX_BRACKET_FACTORS:
            raise OperationDomainValidationError(
                location=("terms",),
                code="bracket.syzygy_monomial_factors",
                message="every assembled monomial must fit the bracket-factor bound",
            )
        output_keys.add(tuple(sorted(multiplicities.items())))
        if len(output_keys) > MAX_BRACKET_TERMS:
            raise OperationResourceAdmissionError(
                location=("terms",),
                code="bracket.syzygy_output_term_bound",
                message="the formal residual has too many sparse output terms",
            )
        return tuple(sorted(multiplicities.items()))

    coefficient_components: dict[
        tuple[tuple[tuple[int, int, int], int], ...], list[tuple[int, int]]
    ] = {}
    for term in request.target.terms:
        key = admit_monomial(term.monomial.factors)
        coefficient_components.setdefault(key, []).append(
            _rational_component_digit_upper_bound(term.coefficient)
        )
    for scalar, multiplier, relation in request.terms:
        scalar_num_digits, scalar_den_digits = _rational_component_digit_upper_bound(
            scalar
        )
        for term in relation.polynomial.terms:
            key = admit_monomial(multiplier.factors + term.monomial.factors)
            term_num_digits, term_den_digits = _rational_component_digit_upper_bound(
                term.coefficient
            )
            coefficient_components.setdefault(key, []).append(
                (
                    scalar_num_digits + term_num_digits,
                    scalar_den_digits + term_den_digits,
                )
            )

    if not coefficient_components:
        return
    coefficient_digit_bound = 0
    for components in coefficient_components.values():
        denominator_digit_bound = sum(
            denominator_digits for _, denominator_digits in components
        )
        numerator_digit_bound = max(
            numerator_digits + denominator_digit_bound - denominator_digits
            for numerator_digits, denominator_digits in components
        )
        sum_digit_overhead = (
            0
            if len(components) <= 1
            else _integer_digit_upper_bound(len(components))
        )
        coefficient_digit_bound = max(
            coefficient_digit_bound,
            denominator_digit_bound,
            numerator_digit_bound + sum_digit_overhead,
        )
    if coefficient_digit_bound > MAX_BRACKET_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_coefficient_digit_bound",
            message="exact residual coefficient growth exceeds the supported digit bound",
        )
    output_cells = len(output_keys) * (MAX_BRACKET_FACTORS + 1)
    if output_cells > MAX_BRACKET_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_output_cell_bound",
            message="the exact sparse residual output exceeds the supported cell bound",
        )


def grassmann_pluecker_relation(
    request: GrassmannPlueckerRelationRequest,
) -> GrassmannPlueckerRelationResult:
    """Return the canonical formal expression of one GP relation."""

    polynomial = _relation_polynomial(request)
    return GrassmannPlueckerRelationResult(
        ground_size=request.ground_size,
        indices=request.indices,
        family=request.family,
        polynomial=polynomial,
    )


def bracket_syzygy_residual(request: BracketSyzygyResidualRequest) -> BracketPolynomial:
    """Return ``target - sum(scalar * multiplier * relation)`` exactly.

    The arithmetic is in the free commutative polynomial algebra on canonical
    bracket atoms.  Each supplied relation is admitted against its retained
    indexed source, but this operation does not infer that a nonzero residual
    vanishes on minors.
    """

    _admit_residual_envelope(request)
    contributions: list[tuple[Fraction, tuple[tuple[CanonicalBracket, int], ...]]] = []
    for term in request.target.terms:
        contributions.append((term.coefficient.as_fraction(), term.monomial.factors))
    for scalar, multiplier, relation in request.terms:
        for term in relation.polynomial.terms:
            contributions.append(
                (
                    -scalar.as_fraction() * term.coefficient.as_fraction(),
                    multiplier.factors + term.monomial.factors,
                )
            )
    return _combine(contributions, request.target.ground_size)
