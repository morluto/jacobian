"""Canonical bracket atoms and rank-3 Pluecker relations (#2775)."""

from __future__ import annotations

import math
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
    _bounded_component_sum,
    bracket_polynomial_from_terms,
    bracket_syzygy_residual,
    grassmann_pluecker_relation,
)
from jacobian.math.combinatorics.matroids.oriented._bracket_models import (
    BracketMonomial,
    BracketPolynomial,
    BracketPolynomialTerm,
    CanonicalBracket,
    GrassmannPlueckerRelation,
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
    result = grassmann_pluecker_relation(6, (0, 1, 2, 3, 4, 5), "FOUR_TERM")
    assert len(result.polynomial.terms) == 4
    for seed in range(4):
        assert _bracket_value(result, _random_columns(6, seed)) == 0


def test_shared_index_three_term_relation_vanishes_on_real_minors() -> None:
    """The shared-index three-term relation is a real identity as well."""
    result = grassmann_pluecker_relation(6, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM")
    assert len(result.polynomial.terms) == 3
    for seed in range(4):
        assert _bracket_value(result, _random_columns(6, 100 + seed)) == 0


def test_relation_output_is_canonical_and_combined() -> None:
    """Coefficients combine and terms are ordered so equal inputs compare equal."""
    first = grassmann_pluecker_relation(6, (0, 1, 2, 3, 4, 5), "FOUR_TERM")
    second = grassmann_pluecker_relation(6, (0, 1, 2, 3, 4, 5), "FOUR_TERM")
    assert first == second
    keys = [
        tuple((factor.indices for factor, _ in term.monomial.factors))
        for term in first.polynomial.terms
    ]
    assert keys == sorted(keys)


def test_relation_rejects_duplicate_indices() -> None:
    """A relation needs six distinct indices."""
    with pytest.raises(OperationDomainValidationError) as error:
        grassmann_pluecker_relation(6, (0, 1, 2, 3, 4, 4), "FOUR_TERM")
    assert error.value.errors()[0]["type"] == "bracket.relation_indices_not_distinct"


def test_relation_rejects_indices_outside_the_ground_range() -> None:
    """Every relation index lies inside the declared ground range."""
    with pytest.raises(OperationDomainValidationError) as error:
        grassmann_pluecker_relation(6, (0, 1, 2, 3, 4, 9), "FOUR_TERM")
    assert error.value.errors()[0]["type"] == "bracket.relation_index_outside_ground"


def test_normalized_relation_agrees_with_the_permuted_presentation() -> None:
    """Relabelling indices preserves each bracket's alternating parity."""
    base = grassmann_pluecker_relation(6, (0, 1, 2, 3, 4, 5), "FOUR_TERM")
    relabelling = (2, 0, 5, 1, 4, 3)
    permuted = grassmann_pluecker_relation(6, relabelling, "FOUR_TERM")
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
                (left > right for left, right in combinations(relabelled, 2))
            )
            coefficient *= -1 if inversions % 2 else 1
            factors.append(
                (cast(tuple[int, int, int], tuple(sorted(relabelled))), multiplicity)
            )
        expected[tuple(sorted(factors))] = coefficient
    permuted_coefficients = {
        tuple(
            (
                (factor.indices, multiplicity)
                for factor, multiplicity in term.monomial.factors
            )
        ): term.coefficient.as_fraction()
        for term in permuted.polynomial.terms
    }
    assert permuted_coefficients == expected


def test_result_round_trips_through_strict_json() -> None:
    """The formal relation survives strict JSON serialization unchanged."""
    result = grassmann_pluecker_relation(6, (0, 1, 2, 3, 4, 5), "FOUR_TERM")
    restored = GrassmannPlueckerRelationResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result


def test_result_rejects_serialized_relation_index_outside_ground() -> None:
    result = grassmann_pluecker_relation(6, (0, 1, 2, 3, 4, 5), "FOUR_TERM")
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
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    residual = bracket_syzygy_residual(
        relation.polynomial,
        ((CanonicalRational(num=1, den=1), BracketMonomial(factors=()), relation),),
    )
    assert residual.terms == ()


def test_syzygy_skips_zero_scaled_oversized_multiplier() -> None:
    relation = grassmann_pluecker_relation(12, (0, 1, 2, 3, 4, 5), "FOUR_TERM")
    multiplier = BracketMonomial(
        factors=(
            (CanonicalBracket(indices=(6, 7, 8)), 1),
            (CanonicalBracket(indices=(6, 7, 9)), 1),
            (CanonicalBracket(indices=(6, 10, 11)), 1),
        )
    )
    residual = bracket_syzygy_residual(
        relation.polynomial, ((CanonicalRational(num=0, den=1), multiplier, relation),)
    )
    assert residual == relation.polynomial


def test_syzygy_residual_retains_nonzero_scalar_and_multiplier() -> None:
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    residual = bracket_syzygy_residual(
        relation.polynomial,
        (
            (
                CanonicalRational(num=2, den=1),
                BracketMonomial(factors=((CanonicalBracket(indices=(0, 1, 2)), 1),)),
                relation,
            ),
        ),
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
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    with pytest.raises(OperationDomainValidationError, match="source-bound"):
        bracket_syzygy_residual(
            relation.polynomial,
            (
                (
                    CanonicalRational(num=1, den=1),
                    BracketMonomial(factors=()),
                    cast(GrassmannPlueckerRelation, relation.polynomial),
                ),
            ),
        )


def test_syzygy_rejects_a_forged_source_relation() -> None:
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    forged = GrassmannPlueckerRelation(
        ground_size=relation.ground_size,
        indices=relation.indices,
        family=relation.family,
        polynomial=bracket_polynomial_from_terms(5, [(Fraction(1), ((0, 1, 2),))]),
    )
    with pytest.raises(OperationDomainValidationError, match="source metadata"):
        bracket_syzygy_residual(
            relation.polynomial,
            ((CanonicalRational(num=1, den=1), BracketMonomial(factors=()), forged),),
        )


def test_syzygy_rejects_multiplier_outside_target_ground() -> None:
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    outside = CanonicalBracket(indices=(0, 1, 5))
    with pytest.raises(OperationDomainValidationError) as error:
        bracket_syzygy_residual(
            relation.polynomial,
            (
                (
                    CanonicalRational(num=1, den=1),
                    BracketMonomial(factors=((outside, 1),)),
                    relation,
                ),
            ),
        )
    assert (
        error.value.errors()[0]["type"]
        == "bracket.syzygy_multiplier_index_outside_ground"
    )


def test_syzygy_rejects_513_distinct_output_terms_before_expansion() -> None:
    relation = grassmann_pluecker_relation(12, (0, 1, 2, 3, 4, 5), "FOUR_TERM")
    relation_keys = {
        tuple(
            (
                (factor.indices, multiplicity)
                for factor, multiplicity in term.monomial.factors
            )
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
        key = tuple(
            ((factor.indices, multiplicity) for factor, multiplicity in factors)
        )
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
                    (
                        (factor.indices, multiplicity)
                        for factor, multiplicity in term.monomial.factors
                    )
                ),
            )
        ),
    )
    with pytest.raises(
        OperationResourceAdmissionError, match="too many sparse output terms"
    ):
        bracket_syzygy_residual(
            target,
            ((CanonicalRational(num=1, den=1), BracketMonomial(factors=()), relation),),
        )


def test_syzygy_accepts_512_terms_after_two_relation_cancellations() -> None:
    relation = grassmann_pluecker_relation(12, (0, 1, 2, 3, 4, 5), "FOUR_TERM")
    relation_keys = {
        tuple(
            (
                (factor.indices, multiplicity)
                for factor, multiplicity in term.monomial.factors
            )
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
        key = tuple(
            ((factor.indices, multiplicity) for factor, multiplicity in factors)
        )
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
                    (
                        (factor.indices, multiplicity)
                        for factor, multiplicity in term.monomial.factors
                    )
                ),
            )
        ),
    )
    residual = bracket_syzygy_residual(
        target,
        ((CanonicalRational(num=1, den=1), BracketMonomial(factors=()), relation),),
    )
    assert len(residual.terms) == 512


def test_syzygy_rejects_unbounded_intermediate_coefficient_digits() -> None:
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    large_base = 10**10000
    terms = tuple(
        (
            CanonicalRational(num=1, den=large_base + offset),
            BracketMonomial(factors=((CanonicalBracket(indices=(0, 1, 2)), 1),)),
            relation,
        )
        for offset in (1, 3, 7, 9)
    )
    with pytest.raises(OperationResourceAdmissionError, match="digit bound"):
        bracket_syzygy_residual(relation.polynomial, terms)


def test_syzygy_rejects_assembled_multiplicity_overflow_before_combine() -> None:
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    multiplier = BracketMonomial(
        factors=(
            (CanonicalBracket(indices=(0, 1, 2)), 10**MAX_CANONICAL_INTEGER_DIGITS - 1),
        )
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        bracket_syzygy_residual(
            BracketPolynomial(ground_size=5, terms=()),
            ((CanonicalRational(num=1, den=1), multiplier, relation),),
        )
    assert error.value.errors()[0]["type"] == "bracket.syzygy_multiplicity_digit_bound"


def test_syzygy_accepts_maximum_assembled_multiplicity() -> None:
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    maximum = 10**MAX_CANONICAL_INTEGER_DIGITS - 1
    multiplier = BracketMonomial(
        factors=((CanonicalBracket(indices=(0, 1, 2)), maximum - 1),)
    )
    residual = bracket_syzygy_residual(
        BracketPolynomial(ground_size=5, terms=()),
        ((CanonicalRational(num=1, den=1), multiplier, relation),),
    )
    assert any(
        (
            multiplicity == maximum
            for term in residual.terms
            for _factor, multiplicity in term.monomial.factors
        )
    )


def test_syzygy_rejects_result_allocation_growth_before_combine() -> None:
    atoms = [CanonicalBracket(indices=triple) for triple in combinations(range(12), 3)]
    coefficient = CanonicalRational(
        num=10 ** (MAX_CANONICAL_INTEGER_DIGITS - 1) - 1, den=1
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
                    (
                        (factor.indices, multiplicity)
                        for factor, multiplicity in term.monomial.factors
                    )
                ),
            )
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        bracket_syzygy_residual(target, ())
    assert error.value.errors()[0]["type"] == "bracket.syzygy_result_allocation_bound"


def test_syzygy_allocation_charges_each_surviving_coefficient_width() -> None:
    atoms = [CanonicalBracket(indices=triple) for triple in combinations(range(12), 3)]
    maximum_width = 10 ** (MAX_CANONICAL_INTEGER_DIGITS - 1)
    target_terms = tuple(
        (
            BracketPolynomialTerm(
                coefficient=CanonicalRational(
                    num=maximum_width if index == 0 else 1, den=1
                ),
                monomial=BracketMonomial(
                    factors=tuple(((factor, maximum_width) for factor in factors))
                ),
            )
            for index, factors in enumerate(islice(combinations(atoms, 2), 512))
        )
    )
    target = BracketPolynomial(
        ground_size=12,
        terms=tuple(
            sorted(
                target_terms,
                key=lambda term: tuple(
                    (
                        (factor.indices, multiplicity)
                        for factor, multiplicity in term.monomial.factors
                    )
                ),
            )
        ),
    )
    residual = bracket_syzygy_residual(target, ())
    assert residual == target


def test_syzygy_cancels_opposite_large_coefficients_before_bound() -> None:
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    denominator = 10**20000
    residual = bracket_syzygy_residual(
        BracketPolynomial(ground_size=5, terms=()),
        (
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
    assert residual.terms == ()


def test_syzygy_cancels_equal_denominator_non_pair_components_before_bound() -> None:
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    denominator = 10**20000 + 1
    residual = bracket_syzygy_residual(
        BracketPolynomial(ground_size=5, terms=()),
        (
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
    assert residual.terms == ()


def test_syzygy_bounds_same_denominator_reduction_before_accumulation() -> None:
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    maximum = 10**MAX_CANONICAL_INTEGER_DIGITS - 1
    residual = bracket_syzygy_residual(
        BracketPolynomial(ground_size=5, terms=()),
        tuple(
            (
                CanonicalRational(num=scalar, den=1),
                BracketMonomial(factors=()),
                relation,
            )
            for scalar in (maximum, maximum, -maximum, -maximum)
        ),
    )
    assert residual.terms == ()


def test_syzygy_cancels_distinct_denominators_before_bound() -> None:
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    first_denominator = 10**12000 + 1
    second_denominator = 10**12000 + 3
    residual = bracket_syzygy_residual(
        BracketPolynomial(ground_size=5, terms=()),
        (
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
    assert residual.terms == ()


def test_syzygy_cancels_shared_denominator_factors_before_bound() -> None:
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    common_factor = 10**20000 + 1
    residual = bracket_syzygy_residual(
        BracketPolynomial(ground_size=5, terms=()),
        (
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
    assert residual.terms == ()


def test_syzygy_rejects_cross_products_before_oversized_fraction() -> None:
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    first_denominator = 10**20000 + 1
    second_denominator = 10**20000 + 3
    with pytest.raises(OperationResourceAdmissionError, match="digit bound"):
        bracket_syzygy_residual(
            BracketPolynomial(ground_size=5, terms=()),
            (
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


def test_syzygy_constructs_from_cancellation_admission_plan() -> None:
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    first_denominator = 10**20000 + 1
    surviving_denominator = 10**20000 + 3
    residual = bracket_syzygy_residual(
        BracketPolynomial(ground_size=5, terms=()),
        (
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
    actual = {
        tuple(
            (
                (factor.indices, multiplicity)
                for factor, multiplicity in term.monomial.factors
            )
        ): term.coefficient.as_fraction()
        for term in residual.terms
    }
    expected = {
        tuple(
            (
                (factor.indices, multiplicity)
                for factor, multiplicity in term.monomial.factors
            )
        ): -Fraction(1, surviving_denominator) * term.coefficient.as_fraction()
        for term in relation.polynomial.terms
    }
    assert actual == expected


def test_syzygy_unit_relation_coefficients_preserve_maximum_scalar_width() -> None:
    relation = grassmann_pluecker_relation(6, (0, 1, 2, 3, 4, 5), "FOUR_TERM")
    scalar = 10**32767
    residual = bracket_syzygy_residual(
        BracketPolynomial(ground_size=6, terms=()),
        (
            (
                CanonicalRational(num=scalar, den=1),
                BracketMonomial(factors=()),
                relation,
            ),
        ),
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
    result = bracket_syzygy_residual(target, ())
    assert result == target
    restored = BracketPolynomial.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert bracket_syzygy_residual(restored, ()) == target


def test_oversized_repeated_coefficient_rejects_without_partition_search() -> None:
    """An unrepresentable total is rejected in linear arithmetic, not by partitions."""
    width = (10**MAX_CANONICAL_INTEGER_DIGITS - 1) // 64
    with pytest.raises(OperationResourceAdmissionError, match="digit bound"):
        _bounded_component_sum([(Fraction(width), (0, 0))] * 128)


def test_coprime_denominator_product_rejects_before_lcm_construction() -> None:
    """32 near-bound coprime denominators must not form a million-digit LCM."""

    digits = MAX_CANONICAL_INTEGER_DIGITS
    base = 10 ** (digits - 1)
    components = [(Fraction(1, base + index), (1, digits)) for index in range(32)]
    with pytest.raises(OperationResourceAdmissionError, match="digit bound"):
        _bounded_component_sum(components)


def test_syzygy_admits_near_bound_denominator_products() -> None:
    """Products 3N and 2N that still have 32,768 digits remain representable."""
    modulus = 10 ** (MAX_CANONICAL_INTEGER_DIGITS - 1) + 1
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    residual = bracket_syzygy_residual(
        BracketPolynomial(ground_size=5, terms=()),
        (
            (
                CanonicalRational(num=-modulus, den=2),
                BracketMonomial(factors=()),
                relation,
            ),
            (
                CanonicalRational(num=modulus, den=3),
                BracketMonomial(factors=()),
                relation,
            ),
        ),
    )
    masses = {term.coefficient.as_fraction() for term in residual.terms}
    assert masses <= {Fraction(modulus, 6), Fraction(-modulus, 6)}
    assert masses


def test_syzygy_retries_a_bounded_reduction_order() -> None:
    """A locally cheapest pair must not reject an otherwise representable sum."""
    bound = 10 ** (MAX_CANONICAL_INTEGER_DIGITS - 1)
    width = Fraction(bound + 1)
    total, _ = _bounded_component_sum(
        [
            (Fraction(7) * width / 3, (0, 0)),
            (-width / 2, (0, 0)),
            (Fraction(-3) * width / 2, (0, 0)),
            (width / 3, (0, 0)),
        ]
    )
    assert total == Fraction(2) * width / 3
    relation = grassmann_pluecker_relation(6, (0, 1, 2, 3, 4, 5), "FOUR_TERM")
    residual = bracket_syzygy_residual(
        BracketPolynomial(ground_size=6, terms=()),
        (
            (
                CanonicalRational(num=-7 * (bound + 1), den=3),
                BracketMonomial(factors=()),
                relation,
            ),
            (
                CanonicalRational(num=bound + 1, den=2),
                BracketMonomial(factors=()),
                relation,
            ),
            (
                CanonicalRational(num=3 * (bound + 1), den=2),
                BracketMonomial(factors=()),
                relation,
            ),
            (
                CanonicalRational(num=-(bound + 1), den=3),
                BracketMonomial(factors=()),
                relation,
            ),
        ),
    )
    expected = Fraction(2 * (bound + 1), 3)
    masses = {abs(term.coefficient.as_fraction()) for term in residual.terms}
    assert masses == {expected}


def test_syzygy_backtracks_later_coefficient_merges() -> None:
    """A later merge choice must be retried, not only the first pair."""
    limit = 10**MAX_CANONICAL_INTEGER_DIGITS
    width = (limit - 1) // 85
    total, _ = _bounded_component_sum(
        [
            (Fraction(-7 * width, 4), (0, 0)),
            (Fraction(14 * width, 3), (0, 0)),
            (Fraction(-85 * width, 18), (0, 0)),
            (Fraction(-13 * width, 9), (0, 0)),
            (Fraction(28 * width, 5), (0, 0)),
        ]
    )
    assert total == Fraction(47 * width, 20)
    relation = grassmann_pluecker_relation(6, (0, 1, 2, 3, 4, 5), "FOUR_TERM")
    residual = bracket_syzygy_residual(
        BracketPolynomial(ground_size=6, terms=()),
        (
            (
                CanonicalRational(num=-7 * width, den=4),
                BracketMonomial(factors=()),
                relation,
            ),
            (
                CanonicalRational(num=14 * width, den=3),
                BracketMonomial(factors=()),
                relation,
            ),
            (
                CanonicalRational(num=-85 * width, den=18),
                BracketMonomial(factors=()),
                relation,
            ),
            (
                CanonicalRational(num=-13 * width, den=9),
                BracketMonomial(factors=()),
                relation,
            ),
            (
                CanonicalRational(num=28 * width, den=5),
                BracketMonomial(factors=()),
                relation,
            ),
        ),
    )
    expected = Fraction(47 * width, 20)
    masses = {abs(term.coefficient.as_fraction()) for term in residual.terms}
    assert masses == {expected}


def test_syzygy_rejects_coprime_wide_denominators_before_fraction_sum() -> None:
    """Pairwise-coprime 32,768-digit dens reject before an LCM product is built."""

    primes = (
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
    )
    upper = 10**MAX_CANONICAL_INTEGER_DIGITS
    components: list[tuple[Fraction, tuple[int, int]]] = []
    terms = []
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    for prime in primes:
        exponent = (
            math.floor(MAX_CANONICAL_INTEGER_DIGITS * math.log(10) / math.log(prime))
            - 1
        )
        denominator = pow(prime, max(exponent, 1))
        while denominator >= upper:
            denominator //= prime
        # Distinct prime powers remain pairwise coprime; two such widths already
        # exceed the canonical integer envelope under LCM.
        components.append((Fraction(1, denominator), (1, MAX_CANONICAL_INTEGER_DIGITS)))
        terms.append(
            (
                CanonicalRational(num=1, den=denominator),
                BracketMonomial(factors=()),
                relation,
            )
        )
    with pytest.raises(OperationResourceAdmissionError) as error:
        _bounded_component_sum(components)
    assert error.value.errors()[0]["type"] == "bracket.syzygy_coefficient_digit_bound"
    with pytest.raises(OperationResourceAdmissionError) as error:
        bracket_syzygy_residual(
            BracketPolynomial(ground_size=5, terms=()), tuple(terms)
        )
    assert error.value.errors()[0]["type"] == "bracket.syzygy_coefficient_digit_bound"


def test_related_denominators_cancel_before_the_cap() -> None:
    """A zero residual that reduces below the LCM cap is admitted."""
    limit = 10**MAX_CANONICAL_INTEGER_DIGITS
    shared = limit // 20 + 1
    total, _ = _bounded_component_sum(
        [
            (Fraction(1, 6 * shared), (0, 0)),
            (Fraction(1, 10 * shared), (0, 0)),
            (Fraction(-4, 15 * shared), (0, 0)),
        ]
    )
    assert total == 0


def test_syzygy_rejects_a_non_polynomial_target() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        bracket_syzygy_residual({}, ())  # type: ignore[arg-type]
    assert error.value.errors()[0]["type"] == "bracket.syzygy_target_type"


def test_syzygy_revalidates_a_forged_relation() -> None:
    from jacobian.math.combinatorics.matroids.oriented._bracket_kernel import (
        _admit_source_relation,
    )

    forged = GrassmannPlueckerRelation.model_construct(
        ground_size=6,
        indices=(0,),
        family="FOUR_TERM",
        polynomial=BracketPolynomial(ground_size=6, terms=()),
    )
    with pytest.raises(OperationDomainValidationError):
        _admit_source_relation(forged)


def test_syzygy_forged_relation_never_unpacks_malformed_indices() -> None:
    from jacobian.math.combinatorics.matroids.oriented._bracket_kernel import (
        _admit_source_relation,
    )

    forged = GrassmannPlueckerRelation.model_construct(
        ground_size=6,
        indices=(0, 1),
        family="SHARED_INDEX_THREE_TERM",
        polynomial=BracketPolynomial(ground_size=6, terms=()),
    )
    with pytest.raises(OperationDomainValidationError):
        _admit_source_relation(forged)


def test_syzygy_admits_a_forged_relation_before_reading_its_polynomial() -> None:
    """A relation with a non-polynomial carrier is a typed domain error.

    ``model_construct`` bypasses the model validator, so the contribution
    count must not dereference ``relation.polynomial`` before the relation is
    structurally admitted.
    """
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    forged = GrassmannPlueckerRelation.model_construct(
        ground_size=relation.ground_size,
        indices=relation.indices,
        family=relation.family,
        polynomial=object(),
    )
    with pytest.raises(OperationDomainValidationError):
        bracket_syzygy_residual(
            relation.polynomial,
            ((CanonicalRational(num=1, den=1), BracketMonomial(factors=()), forged),),
        )


def test_syzygy_revalidates_forged_native_value_carriers() -> None:
    """Bypass-constructed native values must not leak raw field-access errors."""
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    target = relation.polynomial
    scalar = CanonicalRational(num=1, den=1)
    multiplier = BracketMonomial(factors=())

    forged_targets = (
        BracketPolynomial.model_construct(ground_size=5, terms=None),
        BracketPolynomial.model_construct(ground_size="wide", terms=()),
    )
    for forged in forged_targets:
        with pytest.raises(OperationDomainValidationError):
            bracket_syzygy_residual(forged, ())

    with pytest.raises(OperationDomainValidationError):
        bracket_syzygy_residual(
            target,
            (
                (
                    CanonicalRational.model_construct(num="bad", den=1),
                    multiplier,
                    relation,
                ),
            ),
        )

    with pytest.raises(OperationDomainValidationError):
        bracket_syzygy_residual(
            target,
            ((scalar, BracketMonomial.model_construct(factors=None), relation),),
        )


def test_syzygy_revalidates_forged_nested_polynomial_terms() -> None:
    """A forged nested term's coefficient cannot reach exact arithmetic."""
    term = BracketPolynomialTerm.model_construct(
        coefficient=object(), monomial=BracketMonomial(factors=())
    )
    forged = BracketPolynomial.model_construct(ground_size=5, terms=(term,))
    with pytest.raises(OperationDomainValidationError):
        bracket_syzygy_residual(forged, ())


def test_syzygy_forged_none_indices_is_a_typed_domain_error() -> None:
    """A relation with ``indices=None`` is validated, not tuple-unpacked."""
    from jacobian.math.combinatorics.matroids.oriented._bracket_kernel import (
        _admit_source_relation,
    )

    forged = GrassmannPlueckerRelation.model_construct(
        ground_size=5,
        indices=None,
        family="SHARED_INDEX_THREE_TERM",
        polynomial=BracketPolynomial(ground_size=5, terms=()),
    )
    with pytest.raises(OperationDomainValidationError):
        _admit_source_relation(forged)


def test_equal_denominator_components_are_grouped_before_the_lcm_guard() -> None:
    """Equal-denominator cancellations are found before unrelated LCM growth."""
    from fractions import Fraction

    from jacobian.math.combinatorics.matroids.oriented._bracket_kernel import (
        _bounded_component_sum,
    )

    left_denominator = 10**20000 + 7
    right_denominator = 10**20000 + 9
    width = (1, 20001)
    components = [
        (Fraction(1, left_denominator), width),
        (Fraction(1, right_denominator), width),
        (Fraction(2, left_denominator), width),
        (Fraction(2, right_denominator), width),
        (Fraction(-3, left_denominator), width),
        (Fraction(-3, right_denominator), width),
    ]
    total, _bound = _bounded_component_sum(components)
    assert total == 0


def test_syzygy_retains_the_admitted_relation_copy() -> None:
    """A relation whose polynomial is a valid wire dict is retained validated."""
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    forged = GrassmannPlueckerRelation.model_construct(
        ground_size=relation.ground_size,
        indices=relation.indices,
        family=relation.family,
        polynomial=relation.polynomial.model_dump(),
    )
    result = bracket_syzygy_residual(
        relation.polynomial,
        ((CanonicalRational(num=1, den=1), BracketMonomial(factors=()), forged),),
    )
    assert isinstance(result, BracketPolynomial)


def test_syzygy_revalidates_nested_bracket_atoms() -> None:
    """A forged bracket atom nested in a monomial is a typed domain error."""
    forged_bracket = CanonicalBracket.model_construct(indices=(5, 4, 3))
    monomial = BracketMonomial.model_construct(factors=((forged_bracket, 1),))
    term = BracketPolynomialTerm.model_construct(
        coefficient=CanonicalRational(num=1, den=1), monomial=monomial
    )
    target = BracketPolynomial.model_construct(ground_size=5, terms=(term,))
    with pytest.raises(OperationDomainValidationError):
        bracket_syzygy_residual(target, ())


def test_syzygy_revalidates_relation_before_ground_access() -> None:
    """A forged relation missing ground_size is a typed domain error."""
    relation = grassmann_pluecker_relation(
        5, (0, 1, 2, 3, 4), "SHARED_INDEX_THREE_TERM"
    )
    forged = GrassmannPlueckerRelation.model_construct(
        indices=relation.indices,
        family=relation.family,
        polynomial=relation.polynomial,
    )
    with pytest.raises(OperationDomainValidationError):
        bracket_syzygy_residual(
            relation.polynomial,
            ((CanonicalRational(num=1, den=1), BracketMonomial(factors=()), forged),),
        )


def test_bracket_monomial_schema_requires_positive_multiplicity() -> None:
    """The wire schema advertises the positive multiplicity constraint."""
    schema = BracketMonomial.model_json_schema()
    multiplicity = schema["properties"]["factors"]["items"]["prefixItems"][1]
    assert multiplicity.get("ge") == 1
    assert multiplicity.get("pattern") == "^[1-9][0-9]*$"


def test_syzygy_request_schema_publishes_assembled_factor_limit() -> None:
    """The terms description advertises the assembled-factor envelope."""
    from jacobian.math.combinatorics.matroids.oriented._bracket_models import (
        BracketSyzygyResidualRequest,
    )

    schema = BracketSyzygyResidualRequest.model_json_schema()
    assert "4 distinct atoms" in schema["properties"]["terms"]["description"]
