"""Canonical bracket atoms and rank-3 Pluecker relations (#2775)."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.math.combinatorics.matroids.oriented._bracket_kernel import (
    bracket_polynomial_from_terms,
    bracket_syzygy_residual,
    grassmann_pluecker_relation,
)
from jacobian.math.combinatorics.matroids.oriented._bracket_models import (
    BracketMonomial,
    BracketPolynomial,
    BracketSyzygyResidualRequest,
    CanonicalBracket,
    GrassmannPlueckerRelationRequest,
    GrassmannPlueckerRelationResult,
    ordered_bracket,
)


def _bracket_value(
    result: GrassmannPlueckerRelationResult, columns: list[list[Fraction]]
) -> Fraction:
    """Evaluate the formal relation through literal 3xN determinants."""

    def minor(triple: tuple[int, int, int]) -> Fraction:
        i, j, k = triple
        a = [[columns[row][col] for col in (i, j, k)] for row in range(3)]
        return (
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        )

    total = Fraction(0)
    for term in result.polynomial.terms:
        value = term.coefficient.as_fraction()
        for factor, multiplicity in term.monomial.factors:
            value *= minor(factor.indices) ** multiplicity
        total += value
    return total


def _random_columns(width: int, seed: int) -> list[list[Fraction]]:
    generator = random.Random(seed)
    return [
        [Fraction(generator.randint(-6, 6)) for _ in range(width)] for _ in range(3)
    ]


def test_ordered_bracket_records_the_permutation_parity() -> None:
    """An ordered triple normalizes to its increasing form with a parity sign."""
    bracket, sign = ordered_bracket((2, 0, 1))
    assert bracket.indices == (0, 1, 2)
    assert sign == 1
    bracket, sign = ordered_bracket((0, 2, 1))
    assert bracket.indices == (0, 1, 2)
    assert sign == -1


def test_repeated_index_is_a_rejected_zero_symbol() -> None:
    """A repeated index is the zero alternating symbol, never silently dropped."""
    from pydantic_core import PydanticCustomError

    with pytest.raises(PydanticCustomError) as error:
        ordered_bracket((0, 1, 1))
    assert error.value.type == "bracket.duplicate_index_zero"


def test_canonical_bracket_requires_increasing_triple() -> None:
    """The published atom is always the increasing triple."""
    with pytest.raises(ValidationError):
        CanonicalBracket(indices=(1, 0, 2))


def test_four_term_relation_vanishes_on_real_minors() -> None:
    """The four-term Pluecker relation is a real identity, not just formal."""
    result = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=6, indices=(0, 1, 2, 3, 4, 5), family="FOUR_TERM"
        )
    )
    assert len(result.polynomial.terms) == 4
    for seed in range(4):
        assert _bracket_value(result, _random_columns(6, seed)) == 0


def test_shared_index_three_term_relation_vanishes_on_real_minors() -> None:
    """The shared-index three-term relation is a real identity as well."""
    result = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=6,
            indices=(0, 1, 2, 3, 4),
            family="SHARED_INDEX_THREE_TERM",
        )
    )
    assert len(result.polynomial.terms) == 3
    for seed in range(4):
        assert _bracket_value(result, _random_columns(6, 100 + seed)) == 0


def test_relation_output_is_canonical_and_combined() -> None:
    """Coefficients combine and terms are ordered so equal inputs compare equal."""
    first = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=6, indices=(0, 1, 2, 3, 4, 5), family="FOUR_TERM"
        )
    )
    second = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=6, indices=(0, 1, 2, 3, 4, 5), family="FOUR_TERM"
        )
    )
    assert first == second
    keys = [
        tuple(factor.indices for factor, _ in term.monomial.factors)
        for term in first.polynomial.terms
    ]
    assert keys == sorted(keys)


def test_relation_rejects_duplicate_indices() -> None:
    """A relation needs six distinct indices."""
    with pytest.raises(ValidationError) as error:
        grassmann_pluecker_relation(
            GrassmannPlueckerRelationRequest(
                ground_size=6, indices=(0, 1, 2, 3, 4, 4), family="FOUR_TERM"
            )
        )
    assert "bracket.relation_indices_not_distinct" in str(error.value)


def test_relation_rejects_indices_outside_the_ground_range() -> None:
    """Every relation index lies inside the declared ground range."""
    with pytest.raises(ValidationError):
        GrassmannPlueckerRelationRequest(
            ground_size=6, indices=(0, 1, 2, 3, 4, 9), family="FOUR_TERM"
        )


def test_normalized_relation_agrees_with_the_permuted_presentation() -> None:
    """Permuting the presented indices still yields a canonical polynomial."""
    base = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=6, indices=(0, 1, 2, 3, 4, 5), family="FOUR_TERM"
        )
    )
    permuted = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=6, indices=(1, 0, 2, 3, 4, 5), family="FOUR_TERM"
        )
    )
    assert isinstance(permuted.polynomial, BracketPolynomial)
    assert len(permuted.polynomial.terms) == len(base.polynomial.terms)


def test_result_round_trips_through_strict_json() -> None:
    """The formal relation survives strict JSON serialization unchanged."""
    result = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=6, indices=(0, 1, 2, 3, 4, 5), family="FOUR_TERM"
        )
    )
    restored = GrassmannPlueckerRelationResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result


def test_repeated_bracket_factors_retain_their_exponent() -> None:
    polynomial = bracket_polynomial_from_terms(
        6, [(Fraction(1), ((0, 1, 2), (0, 1, 2)))]
    )
    assert polynomial.terms[0].monomial.factors[0][0].indices == (0, 1, 2)
    assert polynomial.terms[0].monomial.factors[0][1] == 2


def test_syzygy_residual_cancels_a_supplied_relation() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=5,
            indices=(0, 1, 2, 3, 4),
            family="SHARED_INDEX_THREE_TERM",
        )
    ).polynomial
    residual = bracket_syzygy_residual(
        BracketSyzygyResidualRequest(
            target=relation,
            terms=(
                (
                    # The empty multiplier is the multiplicative unit.
                    CanonicalRational(num=1, den=1),
                    BracketMonomial(factors=()),
                    relation,
                ),
            ),
        )
    )
    assert residual.terms == ()
