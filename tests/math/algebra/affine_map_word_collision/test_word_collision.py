from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.algebra.affine_map_word_collision._models import (
    AffineMapFamily,
    AffineMapSpec,
    WordCollisionProfileRequest,
)
from jacobian.math.algebra.affine_map_word_collision.operations import (
    compute_word_collision_profile,
    verify_word_collision_profile,
)


def test_two_identity_maps() -> None:
    """Two copies of x->x+1 at depth 1: one collision class with 2 words."""
    result = compute_word_collision_profile(
        [(Fraction(1), Fraction(1)), (Fraction(1), Fraction(1))], 1
    )
    assert len(result.rows) == 1
    assert result.rows[0].multiplicity == 2
    assert result.rows[0].map.slope.as_fraction() == Fraction(1)
    assert result.rows[0].map.intercept.as_fraction() == Fraction(1)


def test_distinct_maps_no_collision() -> None:
    """Two distinct maps at depth 1: no collision."""
    result = compute_word_collision_profile(
        [(Fraction(2), Fraction(0)), (Fraction(3), Fraction(1))], 1
    )
    assert len(result.rows) == 2
    assert all(r.multiplicity == 1 for r in result.rows)


def test_one_generator() -> None:
    """One generator at depth 2: only one word, no collision."""
    result = compute_word_collision_profile([(Fraction(2), Fraction(1))], 2)
    assert len(result.rows) == 1
    assert result.rows[0].multiplicity == 1


def test_word_replay() -> None:
    """Replay: each word composes to its claimed (slope, intercept)."""
    gens = [(Fraction(2), Fraction(1)), (Fraction(3), Fraction(5))]
    result = compute_word_collision_profile(gens, 2)
    for row in result.rows:
        for word in row.words:
            a, b = Fraction(1), Fraction(0)
            for idx in word:
                ga, gb = gens[idx]
                a, b = ga * a, ga * b + gb
            assert a == row.map.slope.as_fraction()
            assert b == row.map.intercept.as_fraction()


def test_prefix_traversal_matches_independent_full_word_composition() -> None:
    generators = (
        (Fraction(2), Fraction(1)),
        (Fraction(-1), Fraction(3)),
        (Fraction(0), Fraction(5)),
    )
    result = compute_word_collision_profile(generators, 4)
    observed = {
        word: (row.map.slope.as_fraction(), row.map.intercept.as_fraction())
        for row in result.rows
        for word in row.words
    }
    expected: dict[tuple[int, ...], tuple[Fraction, Fraction]] = {}
    for word in product(range(len(generators)), repeat=4):
        slope, intercept = Fraction(1), Fraction(0)
        for index in word:
            generator_slope, generator_intercept = generators[index]
            slope, intercept = (
                generator_slope * slope,
                generator_slope * intercept + generator_intercept,
            )
        expected[word] = (slope, intercept)

    assert observed == expected


def test_multiplicity_sum() -> None:
    """Sum of multiplicities equals r^d."""
    gens = [(Fraction(2), Fraction(1)), (Fraction(3), Fraction(5))]
    d = 3
    result = compute_word_collision_profile(gens, d)
    total = sum(r.multiplicity for r in result.rows)
    assert total == len(gens) ** d


def test_partition_completeness() -> None:
    """Every word appears exactly once across all rows."""
    gens = [(Fraction(2), Fraction(1)), (Fraction(1), Fraction(3))]
    result = compute_word_collision_profile(gens, 2)
    all_words = set()
    for row in result.rows:
        for word in row.words:
            assert word not in all_words
            all_words.add(word)
    assert len(all_words) == 4  # 2^2


def test_rational_coefficients() -> None:
    """Test with non-integer rational coefficients."""
    result = compute_word_collision_profile([(Fraction(1, 2), Fraction(1, 3))], 1)
    assert len(result.rows) == 1
    assert result.rows[0].map.slope.as_fraction() == Fraction(1, 2)
    assert result.rows[0].map.intercept.as_fraction() == Fraction(1, 3)


def test_rejects_depth_zero() -> None:
    with pytest.raises(ValidationError):
        WordCollisionProfileRequest(
            family=AffineMapFamily(
                generators=(
                    AffineMapSpec(
                        slope=CanonicalRational(num="1", den="1"),
                        intercept=CanonicalRational(num="1", den="1"),
                    ),
                )
            ),
            depth=0,
        )


def test_profile_source_round_trip_and_forged_multiplicity() -> None:
    result = compute_word_collision_profile(
        ((Fraction(1), Fraction(1)), (Fraction(1), Fraction(1))), 2
    )
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert decoded.family.generators == result.family.generators
    assert verify_word_collision_profile(decoded)
    forged = decoded.model_dump(mode="json")
    forged["rows"][0]["multiplicity"] += 1
    assert not verify_word_collision_profile(type(result).model_validate(forged))


def test_non_commuting_maps() -> None:
    """Non-commuting maps produce different results in different word orders."""
    f = (Fraction(2), Fraction(0))
    g = (Fraction(3), Fraction(1))
    result = compute_word_collision_profile([f, g], 2)
    # Words (0,1) and (1,0) should generally produce different maps
    assert len(result.rows) >= 2


def test_word_order_matches_documented_composition() -> None:
    """The first generator is applied first: word (0, 1) is f_1 o f_0."""
    f = (Fraction(2), Fraction(1))
    g = (Fraction(3), Fraction(5))
    result = compute_word_collision_profile((f, g), 2)
    row = next(row for row in result.rows if (0, 1) in row.words)
    assert (row.map.slope.as_fraction(), row.map.intercept.as_fraction()) == (
        Fraction(6),
        Fraction(8),
    )


def test_native_admission_rejects_excessive_word_enumeration() -> None:
    generators = tuple((Fraction(1), Fraction(index)) for index in range(20))
    with pytest.raises(OperationDomainValidationError, match="composition work limit"):
        compute_word_collision_profile(generators, 5)


def test_duplicate_maps_can_exceed_legacy_word_count() -> None:
    result = compute_word_collision_profile(((Fraction(1), Fraction(0)),) * 2, 14)
    assert result.rows[0].multiplicity == 2**14


def test_mixed_constant_family_preserves_reset_bound() -> None:
    huge = Fraction(10**32767)
    result = compute_word_collision_profile(
        ((Fraction(0), huge), (Fraction(1), Fraction(0))), 1
    )
    assert len(result.rows) == 2


def test_canonical_map_rows_compose_as_native_input() -> None:
    result = compute_word_collision_profile(((Fraction(1), Fraction(2)),), 1)
    replayed = compute_word_collision_profile((result.rows[0].map,), 1)
    assert replayed.rows[0].map == result.rows[0].map


def test_identity_after_constant_reset_preserves_bound() -> None:
    huge = Fraction(10**32767)
    result = compute_word_collision_profile(
        ((Fraction(0), huge), (Fraction(1), Fraction(0))), 2
    )
    assert len(result.rows) == 2


def test_native_admission_rejects_rational_growth_before_enumeration() -> None:
    huge = 10**32_767
    with pytest.raises(OperationDomainValidationError, match="rational digit limit"):
        compute_word_collision_profile(((Fraction(huge), Fraction(0)),), 2)


def test_zero_slope_at_canonical_intercept_boundary_is_admitted() -> None:
    intercept = Fraction(10**32767)
    result = compute_word_collision_profile(((Fraction(0), intercept),), 1)
    assert result.rows[0].map.intercept.as_fraction() == intercept


def test_single_generator_depth_above_legacy_cap_is_admitted() -> None:
    result = compute_word_collision_profile(((Fraction(1), Fraction(0)),), 11)
    assert result.depth == 11
