"""Exact bracket-algebra kernels: canonical atoms, relations, and residuals."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids.oriented._bracket_models import (
    MAX_BRACKET_FACTORS,
    BracketMonomial,
    BracketPolynomial,
    BracketPolynomialTerm,
    BracketSyzygyResidualRequest,
    CanonicalBracket,
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
    contributions: list[tuple[Fraction, tuple[CanonicalBracket, ...]]],
    ground_size: int,
) -> BracketPolynomial:
    """Combine signed bracket monomials into a canonical sparse polynomial."""

    combined: dict[tuple[tuple[tuple[int, int, int], int], ...], Fraction] = {}
    for coefficient, brackets in contributions:
        if coefficient == 0:
            continue
        factor_key = cast(
            tuple[tuple[int, int, int], ...],
            tuple(sorted(bracket.indices for bracket in brackets)),
        )
        # Group repeated factors into multiplicities.
        multiplicities: dict[tuple[int, int, int], int] = {}
        for triples in factor_key:
            multiplicities[triples] = multiplicities.get(triples, 0) + 1
        canonical_key = tuple(sorted(multiplicities.items()))
        combined[canonical_key] = combined.get(canonical_key, Fraction(0)) + coefficient
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

    contributions: list[tuple[Fraction, tuple[CanonicalBracket, ...]]] = []
    for coefficient, ordered_triples in terms:
        brackets: list[CanonicalBracket] = []
        sign = 1
        for ordered in ordered_triples:
            bracket, parity = ordered_bracket(ordered)
            brackets.append(bracket)
            sign *= parity
        contributions.append((coefficient * sign, tuple(brackets)))
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


def grassmann_pluecker_relation(
    request: GrassmannPlueckerRelationRequest,
) -> GrassmannPlueckerRelationResult:
    """Return the canonical formal expression of one GP relation."""

    ordered_terms = (
        _four_term_relation(request.indices)
        if request.family == "FOUR_TERM"
        else _shared_index_three_term_relation(request.indices)
    )
    contributions: list[tuple[Fraction, tuple[CanonicalBracket, ...]]] = []
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
        contributions.append((Fraction(sign), tuple(brackets)))
    polynomial = _combine(contributions, request.ground_size)
    return GrassmannPlueckerRelationResult(
        ground_size=request.ground_size,
        indices=request.indices,
        family=request.family,
        polynomial=polynomial,
    )


def bracket_syzygy_residual(request: BracketSyzygyResidualRequest) -> BracketPolynomial:
    """Return ``target - sum(scalar * multiplier * relation)`` exactly.

    The arithmetic is in the free commutative polynomial algebra on canonical
    bracket atoms.  Supplied relation polynomials are ordinary formal values;
    this operation does not infer that a nonzero residual vanishes on minors.
    """

    contribution_count = len(request.target.terms) + sum(
        len(relation.terms) for _, _, relation in request.terms
    )
    if contribution_count > 65_536:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_contribution_bound",
            message="the formal residual contribution count exceeds the bound",
        )
    for _, multiplier, relation in request.terms:
        relation_factor_sets = [
            {factor.indices for factor, _ in term.monomial.factors}
            for term in relation.terms
        ]
        multiplier_set = {factor.indices for factor, _ in multiplier.factors}
        if any(
            len(multiplier_set | relation_factors) > MAX_BRACKET_FACTORS
            for relation_factors in relation_factor_sets
        ):
            raise OperationDomainValidationError(
                location=("terms",),
                code="bracket.syzygy_monomial_factors",
                message="every assembled monomial must fit the bracket-factor bound",
            )
    contributions: list[tuple[Fraction, tuple[CanonicalBracket, ...]]] = []
    for term in request.target.terms:
        factors = tuple(
            factor
            for factor, multiplicity in term.monomial.factors
            for _ in range(multiplicity)
        )
        contributions.append((term.coefficient.as_fraction(), factors))
    for scalar, multiplier, relation in request.terms:
        multiplier_factors = tuple(
            factor
            for factor, multiplicity in multiplier.factors
            for _ in range(multiplicity)
        )
        for term in relation.terms:
            relation_factors = tuple(
                factor
                for factor, multiplicity in term.monomial.factors
                for _ in range(multiplicity)
            )
            contributions.append(
                (
                    -scalar.as_fraction() * term.coefficient.as_fraction(),
                    multiplier_factors + relation_factors,
                )
            )
    return _combine(contributions, request.target.ground_size)
