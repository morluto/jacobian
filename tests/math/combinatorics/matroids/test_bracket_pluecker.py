"""Canonical bracket atoms and rank-3 Pluecker relations (#2775)."""

from __future__ import annotations

import random
from fractions import Fraction
from itertools import combinations, islice
from typing import cast

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS, CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids.oriented._bracket_kernel import (
    bracket_polynomial_from_terms,
    bracket_syzygy_residual,
    grassmann_pluecker_relation,
)
from jacobian.math.combinatorics.matroids.oriented._bracket_models import (
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
    """Relabelling indices preserves each bracket's alternating parity."""
    base = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=6, indices=(0, 1, 2, 3, 4, 5), family="FOUR_TERM"
        )
    )
    relabelling = (2, 0, 5, 1, 4, 3)
    permuted = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=6, indices=relabelling, family="FOUR_TERM"
        )
    )
    assert isinstance(permuted.polynomial, BracketPolynomial)
    assert len(permuted.polynomial.terms) == len(base.polynomial.terms)
    expected: dict[tuple[tuple[tuple[int, int, int], int], ...], Fraction] = {}
    for term in base.polynomial.terms:
        coefficient = term.coefficient.as_fraction()
        factors = []
        for factor, multiplicity in term.monomial.factors:
            relabelled = (
                relabelling[factor.indices[0]],
                relabelling[factor.indices[1]],
                relabelling[factor.indices[2]],
            )
            inversions = sum(
                left > right for left, right in combinations(relabelled, 2)
            )
            coefficient *= -1 if inversions % 2 else 1
            factors.append(
                (cast(tuple[int, int, int], tuple(sorted(relabelled))), multiplicity)
            )
        expected[tuple(sorted(factors))] = coefficient
    permuted_coefficients = {
        tuple(
            (factor.indices, multiplicity)
            for factor, multiplicity in term.monomial.factors
        ): term.coefficient.as_fraction()
        for term in permuted.polynomial.terms
    }
    assert permuted_coefficients == expected


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


def test_result_rejects_serialized_relation_index_outside_ground() -> None:
    result = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=6, indices=(0, 1, 2, 3, 4, 5), family="FOUR_TERM"
        )
    )
    payload = result.model_dump(mode="json")
    payload["indices"] = [0, 1, 2, 3, 4, 6]
    with pytest.raises(ValidationError, match="relation_index_outside_ground"):
        GrassmannPlueckerRelationResult.model_validate_json(
            encode_strict_json(payload), strict=True
        )


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
    )
    residual = bracket_syzygy_residual(
        BracketSyzygyResidualRequest(
            target=relation.polynomial,
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


def test_syzygy_skips_zero_scaled_oversized_multiplier() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=12,
            indices=(0, 1, 2, 3, 4, 5),
            family="FOUR_TERM",
        )
    )
    multiplier = BracketMonomial(
        factors=(
            (CanonicalBracket(indices=(6, 7, 8)), 1),
            (CanonicalBracket(indices=(6, 7, 9)), 1),
            (CanonicalBracket(indices=(6, 10, 11)), 1),
        )
    )
    residual = bracket_syzygy_residual(
        BracketSyzygyResidualRequest(
            target=relation.polynomial,
            terms=((CanonicalRational(num=0, den=1), multiplier, relation),),
        )
    )
    assert residual == relation.polynomial


def test_syzygy_residual_retains_nonzero_scalar_and_multiplier() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=5,
            indices=(0, 1, 2, 3, 4),
            family="SHARED_INDEX_THREE_TERM",
        )
    )
    residual = bracket_syzygy_residual(
        BracketSyzygyResidualRequest(
            target=relation.polynomial,
            terms=(
                (
                    CanonicalRational(num=2, den=1),
                    BracketMonomial(
                        factors=((CanonicalBracket(indices=(0, 1, 2)), 1),)
                    ),
                    relation,
                ),
            ),
        )
    )
    expected = bracket_polynomial_from_terms(
        5,
        [
            (Fraction(1), ((0, 1, 2), (0, 3, 4))),
            (Fraction(-1), ((0, 1, 3), (0, 2, 4))),
            (Fraction(1), ((0, 1, 4), (0, 2, 3))),
            (Fraction(-2), ((0, 1, 2), (0, 1, 2), (0, 3, 4))),
            (Fraction(2), ((0, 1, 2), (0, 1, 3), (0, 2, 4))),
            (Fraction(-2), ((0, 1, 2), (0, 1, 4), (0, 2, 3))),
        ],
    )
    assert residual == expected
    assert residual.terms


def test_syzygy_rejects_anonymous_polynomial_sources() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=5,
            indices=(0, 1, 2, 3, 4),
            family="SHARED_INDEX_THREE_TERM",
        )
    )
    with pytest.raises(ValidationError):
        BracketSyzygyResidualRequest(
            target=relation.polynomial,
            terms=(
                (
                    CanonicalRational(num=1, den=1),
                    BracketMonomial(factors=()),
                    cast(GrassmannPlueckerRelation, relation.polynomial),
                ),
            ),
        )


def test_syzygy_rejects_a_forged_source_relation() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=5,
            indices=(0, 1, 2, 3, 4),
            family="SHARED_INDEX_THREE_TERM",
        )
    )
    forged = GrassmannPlueckerRelation(
        ground_size=relation.ground_size,
        indices=relation.indices,
        family=relation.family,
        polynomial=bracket_polynomial_from_terms(5, [(Fraction(1), ((0, 1, 2),))]),
    )
    with pytest.raises(OperationDomainValidationError, match="source metadata"):
        bracket_syzygy_residual(
            BracketSyzygyResidualRequest(
                target=relation.polynomial,
                terms=(
                    (
                        CanonicalRational(num=1, den=1),
                        BracketMonomial(factors=()),
                        forged,
                    ),
                ),
            )
        )


def test_syzygy_rejects_multiplier_outside_target_ground() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=5,
            indices=(0, 1, 2, 3, 4),
            family="SHARED_INDEX_THREE_TERM",
        )
    )
    outside = CanonicalBracket(indices=(0, 1, 5))
    with pytest.raises(ValidationError, match="multiplier_index_outside_ground"):
        BracketSyzygyResidualRequest(
            target=relation.polynomial,
            terms=(
                (
                    CanonicalRational(num=1, den=1),
                    BracketMonomial(factors=((outside, 1),)),
                    relation,
                ),
            ),
        )


def test_syzygy_rejects_513_distinct_output_terms_before_expansion() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=12,
            indices=(0, 1, 2, 3, 4, 5),
            family="FOUR_TERM",
        )
    )
    relation_keys = {
        tuple(
            (factor.indices, multiplicity)
            for factor, multiplicity in term.monomial.factors
        )
        for term in relation.polynomial.terms
    }
    brackets = [
        CanonicalBracket(indices=triple) for triple in combinations(range(12), 3)
    ]
    target_terms = []
    for left, right in combinations(brackets, 2):
        factors = tuple(
            sorted(((left, 1), (right, 1)), key=lambda item: item[0].indices)
        )
        key = tuple((factor.indices, multiplicity) for factor, multiplicity in factors)
        if key in relation_keys:
            continue
        target_terms.append(
            BracketPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=1),
                monomial=BracketMonomial(factors=factors),
            )
        )
        if len(target_terms) == 512:
            break
    target = BracketPolynomial(
        ground_size=12,
        terms=tuple(
            sorted(
                target_terms,
                key=lambda term: tuple(
                    (factor.indices, multiplicity)
                    for factor, multiplicity in term.monomial.factors
                ),
            )
        ),
    )
    with pytest.raises(
        OperationResourceAdmissionError, match="too many sparse output terms"
    ):
        bracket_syzygy_residual(
            BracketSyzygyResidualRequest(
                target=target,
                terms=(
                    (
                        CanonicalRational(num=1, den=1),
                        BracketMonomial(factors=()),
                        relation,
                    ),
                ),
            )
        )


def test_syzygy_accepts_512_terms_after_two_relation_cancellations() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=12,
            indices=(0, 1, 2, 3, 4, 5),
            family="FOUR_TERM",
        )
    )
    relation_keys = {
        tuple(
            (factor.indices, multiplicity)
            for factor, multiplicity in term.monomial.factors
        )
        for term in relation.polynomial.terms
    }
    target_terms = list(relation.polynomial.terms[:2])
    brackets = [
        CanonicalBracket(indices=triple) for triple in combinations(range(12), 3)
    ]
    unrelated_keys = set(relation_keys)
    for left, right in combinations(brackets, 2):
        factors = tuple(
            sorted(((left, 1), (right, 1)), key=lambda item: item[0].indices)
        )
        key = tuple((factor.indices, multiplicity) for factor, multiplicity in factors)
        if key in unrelated_keys:
            continue
        target_terms.append(
            BracketPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=1),
                monomial=BracketMonomial(factors=factors),
            )
        )
        unrelated_keys.add(key)
        if len(target_terms) == 512:
            break
    assert len(target_terms) == 512
    target = BracketPolynomial(
        ground_size=12,
        terms=tuple(
            sorted(
                target_terms,
                key=lambda term: tuple(
                    (factor.indices, multiplicity)
                    for factor, multiplicity in term.monomial.factors
                ),
            )
        ),
    )
    residual = bracket_syzygy_residual(
        BracketSyzygyResidualRequest(
            target=target,
            terms=(
                (
                    CanonicalRational(num=1, den=1),
                    BracketMonomial(factors=()),
                    relation,
                ),
            ),
        )
    )
    assert len(residual.terms) == 512


def test_syzygy_rejects_unbounded_intermediate_coefficient_digits() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=5,
            indices=(0, 1, 2, 3, 4),
            family="SHARED_INDEX_THREE_TERM",
        )
    )
    large_base = 10**10_000
    terms = tuple(
        (
            CanonicalRational(num=1, den=large_base + offset),
            BracketMonomial(factors=((CanonicalBracket(indices=(0, 1, 2)), 1),)),
            relation,
        )
        for offset in (1, 3, 7, 9)
    )
    with pytest.raises(OperationResourceAdmissionError, match="digit bound"):
        bracket_syzygy_residual(
            BracketSyzygyResidualRequest(target=relation.polynomial, terms=terms)
        )


def test_syzygy_rejects_assembled_multiplicity_overflow_before_combine() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=5,
            indices=(0, 1, 2, 3, 4),
            family="SHARED_INDEX_THREE_TERM",
        )
    )
    multiplier = BracketMonomial(
        factors=(
            (
                CanonicalBracket(indices=(0, 1, 2)),
                10**MAX_CANONICAL_INTEGER_DIGITS - 1,
            ),
        )
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        bracket_syzygy_residual(
            BracketSyzygyResidualRequest(
                target=BracketPolynomial(ground_size=5, terms=()),
                terms=((CanonicalRational(num=1, den=1), multiplier, relation),),
            )
        )
    assert error.value.errors()[0]["type"] == "bracket.syzygy_multiplicity_digit_bound"


def test_syzygy_accepts_maximum_assembled_multiplicity() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=5,
            indices=(0, 1, 2, 3, 4),
            family="SHARED_INDEX_THREE_TERM",
        )
    )
    maximum = 10**MAX_CANONICAL_INTEGER_DIGITS - 1
    multiplier = BracketMonomial(
        factors=((CanonicalBracket(indices=(0, 1, 2)), maximum - 1),)
    )
    residual = bracket_syzygy_residual(
        BracketSyzygyResidualRequest(
            target=BracketPolynomial(ground_size=5, terms=()),
            terms=((CanonicalRational(num=1, den=1), multiplier, relation),),
        )
    )
    assert any(
        multiplicity == maximum
        for term in residual.terms
        for _factor, multiplicity in term.monomial.factors
    )


def test_syzygy_rejects_result_allocation_growth_before_combine() -> None:
    atoms = [CanonicalBracket(indices=triple) for triple in combinations(range(12), 3)]
    coefficient = CanonicalRational(
        num=10 ** (MAX_CANONICAL_INTEGER_DIGITS - 1) - 1,
        den=1,
    )
    target_terms = tuple(
        BracketPolynomialTerm(
            coefficient=coefficient,
            monomial=BracketMonomial(
                factors=tuple(
                    (atom, 10**MAX_CANONICAL_INTEGER_DIGITS - 1) for atom in factors
                )
            ),
        )
        for factors in islice(combinations(atoms, 4), 512)
    )
    target = BracketPolynomial(
        ground_size=12,
        terms=tuple(
            sorted(
                target_terms,
                key=lambda term: tuple(
                    (factor.indices, multiplicity)
                    for factor, multiplicity in term.monomial.factors
                ),
            )
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        bracket_syzygy_residual(BracketSyzygyResidualRequest(target=target, terms=()))
    assert error.value.errors()[0]["type"] == "bracket.syzygy_result_allocation_bound"


def test_syzygy_allocation_charges_each_surviving_coefficient_width() -> None:
    atoms = [CanonicalBracket(indices=triple) for triple in combinations(range(12), 3)]
    maximum_width = 10 ** (MAX_CANONICAL_INTEGER_DIGITS - 1)
    target_terms = tuple(
        BracketPolynomialTerm(
            coefficient=CanonicalRational(
                num=maximum_width if index == 0 else 1,
                den=1,
            ),
            monomial=BracketMonomial(
                factors=tuple((factor, maximum_width) for factor in factors)
            ),
        )
        for index, factors in enumerate(islice(combinations(atoms, 2), 512))
    )
    target = BracketPolynomial(
        ground_size=12,
        terms=tuple(
            sorted(
                target_terms,
                key=lambda term: tuple(
                    (factor.indices, multiplicity)
                    for factor, multiplicity in term.monomial.factors
                ),
            )
        ),
    )
    residual = bracket_syzygy_residual(
        BracketSyzygyResidualRequest(target=target, terms=())
    )
    assert residual == target


def test_syzygy_cancels_opposite_large_coefficients_before_bound() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=5,
            indices=(0, 1, 2, 3, 4),
            family="SHARED_INDEX_THREE_TERM",
        )
    )
    denominator = 10**20_000
    residual = bracket_syzygy_residual(
        BracketSyzygyResidualRequest(
            target=BracketPolynomial(ground_size=5, terms=()),
            terms=(
                (
                    CanonicalRational(num=1, den=denominator),
                    BracketMonomial(factors=()),
                    relation,
                ),
                (
                    CanonicalRational(num=-1, den=denominator),
                    BracketMonomial(factors=()),
                    relation,
                ),
            ),
        )
    )
    assert residual.terms == ()


def test_syzygy_cancels_equal_denominator_non_pair_components_before_bound() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=5,
            indices=(0, 1, 2, 3, 4),
            family="SHARED_INDEX_THREE_TERM",
        )
    )
    denominator = 10**20_000 + 1
    residual = bracket_syzygy_residual(
        BracketSyzygyResidualRequest(
            target=BracketPolynomial(ground_size=5, terms=()),
            terms=(
                (
                    CanonicalRational(num=1, den=denominator),
                    BracketMonomial(factors=()),
                    relation,
                ),
                (
                    CanonicalRational(num=1, den=denominator),
                    BracketMonomial(factors=()),
                    relation,
                ),
                (
                    CanonicalRational(num=-2, den=denominator),
                    BracketMonomial(factors=()),
                    relation,
                ),
            ),
        )
    )
    assert residual.terms == ()


def test_syzygy_bounds_same_denominator_reduction_before_accumulation() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=5,
            indices=(0, 1, 2, 3, 4),
            family="SHARED_INDEX_THREE_TERM",
        )
    )
    maximum = 10**MAX_CANONICAL_INTEGER_DIGITS - 1
    residual = bracket_syzygy_residual(
        BracketSyzygyResidualRequest(
            target=BracketPolynomial(ground_size=5, terms=()),
            terms=tuple(
                (
                    CanonicalRational(num=scalar, den=1),
                    BracketMonomial(factors=()),
                    relation,
                )
                for scalar in (maximum, maximum, -maximum, -maximum)
            ),
        )
    )
    assert residual.terms == ()


def test_syzygy_cancels_distinct_denominators_before_bound() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=5,
            indices=(0, 1, 2, 3, 4),
            family="SHARED_INDEX_THREE_TERM",
        )
    )
    first_denominator = 10**12_000 + 1
    second_denominator = 10**12_000 + 3
    residual = bracket_syzygy_residual(
        BracketSyzygyResidualRequest(
            target=BracketPolynomial(ground_size=5, terms=()),
            terms=(
                (
                    CanonicalRational(num=1, den=first_denominator),
                    BracketMonomial(factors=()),
                    relation,
                ),
                (
                    CanonicalRational(num=1, den=second_denominator),
                    BracketMonomial(factors=()),
                    relation,
                ),
                (
                    CanonicalRational(
                        num=-(first_denominator + second_denominator),
                        den=first_denominator * second_denominator,
                    ),
                    BracketMonomial(factors=()),
                    relation,
                ),
            ),
        )
    )
    assert residual.terms == ()


def test_syzygy_cancels_shared_denominator_factors_before_bound() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=5,
            indices=(0, 1, 2, 3, 4),
            family="SHARED_INDEX_THREE_TERM",
        )
    )
    common_factor = 10**20_000 + 1
    residual = bracket_syzygy_residual(
        BracketSyzygyResidualRequest(
            target=BracketPolynomial(ground_size=5, terms=()),
            terms=(
                (
                    CanonicalRational(num=1, den=2 * common_factor),
                    BracketMonomial(factors=()),
                    relation,
                ),
                (
                    CanonicalRational(num=1, den=3 * common_factor),
                    BracketMonomial(factors=()),
                    relation,
                ),
                (
                    CanonicalRational(num=-5, den=6 * common_factor),
                    BracketMonomial(factors=()),
                    relation,
                ),
            ),
        )
    )
    assert residual.terms == ()


def test_syzygy_rejects_cross_products_before_oversized_fraction() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=5,
            indices=(0, 1, 2, 3, 4),
            family="SHARED_INDEX_THREE_TERM",
        )
    )
    first_denominator = 10**20_000 + 1
    second_denominator = 10**20_000 + 3
    with pytest.raises(OperationResourceAdmissionError, match="digit bound"):
        bracket_syzygy_residual(
            BracketSyzygyResidualRequest(
                target=BracketPolynomial(ground_size=5, terms=()),
                terms=(
                    (
                        CanonicalRational(num=1, den=first_denominator),
                        BracketMonomial(factors=()),
                        relation,
                    ),
                    (
                        CanonicalRational(num=1, den=second_denominator),
                        BracketMonomial(factors=()),
                        relation,
                    ),
                ),
            )
        )


def test_syzygy_constructs_from_cancellation_admission_plan() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=5,
            indices=(0, 1, 2, 3, 4),
            family="SHARED_INDEX_THREE_TERM",
        )
    )
    first_denominator = 10**20_000 + 1
    surviving_denominator = 10**20_000 + 3
    residual = bracket_syzygy_residual(
        BracketSyzygyResidualRequest(
            target=BracketPolynomial(ground_size=5, terms=()),
            terms=(
                (
                    CanonicalRational(num=1, den=first_denominator),
                    BracketMonomial(factors=()),
                    relation,
                ),
                (
                    CanonicalRational(num=1, den=surviving_denominator),
                    BracketMonomial(factors=()),
                    relation,
                ),
                (
                    CanonicalRational(num=-1, den=first_denominator),
                    BracketMonomial(factors=()),
                    relation,
                ),
            ),
        )
    )
    actual = {
        tuple(
            (factor.indices, multiplicity)
            for factor, multiplicity in term.monomial.factors
        ): term.coefficient.as_fraction()
        for term in residual.terms
    }
    expected = {
        tuple(
            (factor.indices, multiplicity)
            for factor, multiplicity in term.monomial.factors
        ): -Fraction(1, surviving_denominator) * term.coefficient.as_fraction()
        for term in relation.polynomial.terms
    }
    assert actual == expected


def test_syzygy_unit_relation_coefficients_preserve_maximum_scalar_width() -> None:
    relation = grassmann_pluecker_relation(
        GrassmannPlueckerRelationRequest(
            ground_size=6,
            indices=(0, 1, 2, 3, 4, 5),
            family="FOUR_TERM",
        )
    )
    scalar = 10**32_767
    residual = bracket_syzygy_residual(
        BracketSyzygyResidualRequest(
            target=BracketPolynomial(ground_size=6, terms=()),
            terms=(
                (
                    CanonicalRational(num=scalar, den=1),
                    BracketMonomial(factors=()),
                    relation,
                ),
            ),
        )
    )
    assert len(residual.terms) == 4
    assert {abs(term.coefficient.num) for term in residual.terms} == {scalar}
    assert {term.coefficient.den for term in residual.terms} == {1}


def test_compressed_large_multiplicity_survives_residual_and_json() -> None:
    from jacobian.math.combinatorics.matroids.oriented._bracket_models import (
        BracketPolynomialTerm,
    )

    target = BracketPolynomial(
        ground_size=3,
        terms=(
            BracketPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=1),
                monomial=BracketMonomial(
                    factors=((CanonicalBracket(indices=(0, 1, 2)), 10**20),)
                ),
            ),
        ),
    )
    result = bracket_syzygy_residual(
        BracketSyzygyResidualRequest(target=target, terms=())
    )
    assert result == target
    restored = BracketPolynomial.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert (
        bracket_syzygy_residual(BracketSyzygyResidualRequest(target=restored, terms=()))
        == target
    )
