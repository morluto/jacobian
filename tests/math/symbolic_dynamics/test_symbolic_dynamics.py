"""Correctness and contract tests for exact symbolic dynamics operations."""

from __future__ import annotations

import itertools
import random

import pytest
from pydantic import ValidationError

from jacobian.math.symbolic_dynamics import (
    AdjacencyShift,
    ForbiddenBlockShift,
    adjacency_shift,
    block_language,
    finite_type_presentation,
    normalize_forbidden_blocks,
    periodic_point_profile,
)
from jacobian.math.symbolic_dynamics._models import (
    BlockLanguageRequest,
    BlockLanguageResult,
    FiniteTypeShiftRequest,
    FiniteTypeShiftResult,
    HigherBlockRequest,
    HigherBlockResult,
    PeriodicPointProfileRequest,
    PeriodicPointProfileResult,
)
from jacobian.math.symbolic_dynamics._operations import (
    compute_block_language,
    compute_higher_block,
    compute_periodic_point_profile,
    construct_finite_type_shift,
)
from jacobian.math.symbolic_dynamics._tools import TOOLS


def _golden_mean() -> ForbiddenBlockShift:
    return ForbiddenBlockShift(alphabet=("0", "1"), forbidden_blocks=(("1", "1"),))


def test_public_surface_excludes_adjacency_carrier_invariants() -> None:
    assert tuple(tool.operation_id for tool in TOOLS) == (
        "symbolic_dynamics.finite_type_shift.construct",
        "symbolic_dynamics.block_language.compute",
        "symbolic_dynamics.periodic_point_profile.compute",
        "symbolic_dynamics.higher_block.compute",
    )
    carrier = adjacency_shift(((1, 1), (1, 0)))
    assert carrier == AdjacencyShift(matrix=((1, 1), (1, 0)))
    assert not hasattr(carrier, "is_mixing")


def test_golden_mean_finite_type_presentation_is_exact() -> None:
    result = construct_finite_type_shift(FiniteTypeShiftRequest(shift=_golden_mean()))
    assert result.presentation.memory == 1
    assert result.presentation.state_blocks == (("0",), ("1",))
    assert result.presentation.adjacency_matrix == ((1, 1), (1, 0))
    assert tuple(
        (edge.source, edge.target, edge.appended_symbol)
        for edge in result.presentation.transitions
    ) == ((0, 0, "0"), (0, 1, "1"), (1, 0, "0"))
    assert result.complete is True

    payload = result.model_dump()
    payload["presentation"]["adjacency_matrix"] = ((1, 0), (1, 1))
    with pytest.raises(ValidationError):
        FiniteTypeShiftResult.model_validate(payload)


def test_shorter_forbidden_factors_are_enforced_in_long_memory_presentation() -> None:
    shift = ForbiddenBlockShift(
        alphabet=("0", "1"),
        forbidden_blocks=(("0",), ("1", "1")),
    )
    presentation = finite_type_presentation(shift)
    assert presentation.state_blocks == (("1",),)
    assert presentation.transitions == ()
    assert presentation.adjacency_matrix == ((0,),)


def test_forbidden_family_normalization_and_empty_block() -> None:
    redundant = ForbiddenBlockShift(
        alphabet=("0", "1"),
        forbidden_blocks=(("1", "1"), ("1",), ("1",), ("0", "1", "0")),
    )
    assert normalize_forbidden_blocks(redundant) == (("1",),)
    empty = ForbiddenBlockShift(alphabet=("0",), forbidden_blocks=((),))
    assert finite_type_presentation(empty).state_blocks == ()
    assert block_language(empty, 0) == ()


def test_complete_block_language_includes_empty_word_convention() -> None:
    result = compute_block_language(
        BlockLanguageRequest(shift=_golden_mean(), block_length=3)
    )
    assert result.allowed_blocks == (
        ("0", "0", "0"),
        ("0", "0", "1"),
        ("0", "1", "0"),
        ("1", "0", "0"),
        ("1", "0", "1"),
    )
    assert result.count == 5
    empty_word = compute_block_language(
        BlockLanguageRequest(shift=_golden_mean(), block_length=0)
    )
    assert empty_word.allowed_blocks == ((),)

    payload = result.model_dump()
    payload["count"] = 4
    with pytest.raises(ValidationError, match="not bound"):
        BlockLanguageResult.model_validate(payload)


def test_oversized_enumerations_fail_before_computation() -> None:
    alphabet = tuple(chr(ord("a") + index) for index in range(16))
    shift = ForbiddenBlockShift(alphabet=alphabet, forbidden_blocks=())
    with pytest.raises(ValidationError, match="work bound"):
        BlockLanguageRequest(shift=shift, block_length=5)
    with pytest.raises(ValidationError, match="work bound"):
        FiniteTypeShiftRequest(
            shift=ForbiddenBlockShift(
                alphabet=alphabet,
                forbidden_blocks=(("a", "a", "a", "a", "a"),),
            )
        )


def test_higher_block_presentation_uses_allowed_overlap_edges() -> None:
    result = compute_higher_block(
        HigherBlockRequest(shift=_golden_mean(), block_length=2)
    )
    assert result.presentation.state_blocks == (
        ("0", "0"),
        ("0", "1"),
        ("1", "0"),
    )
    assert result.presentation.adjacency_matrix == (
        (1, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
    )
    assert len(result.presentation.transitions) == 5

    payload = result.model_dump()
    payload["presentation"]["transitions"] = ()
    with pytest.raises(ValidationError):
        HigherBlockResult.model_validate(payload)


def test_higher_block_requires_enough_memory_for_exact_sft_presentation() -> None:
    shift = ForbiddenBlockShift(
        alphabet=("0", "1"), forbidden_blocks=(("1", "0", "1", "0"),)
    )
    with pytest.raises(ValidationError, match="at least"):
        HigherBlockRequest(shift=shift, block_length=2)


def test_periodic_profile_handles_square_mobius_factor() -> None:
    request = PeriodicPointProfileRequest(
        shift=AdjacencyShift(matrix=((2,),)), max_period=4
    )
    result = compute_periodic_point_profile(request)
    assert result.fixed_point_counts == (2, 4, 8, 16)
    assert result.least_period_point_counts == (2, 2, 6, 12)
    assert result.primitive_orbit_counts == (2, 1, 2, 3)
    assert result.complete_through_period == 4

    payload = result.model_dump()
    payload["primitive_orbit_counts"] = (2, 1, 2, 4)
    with pytest.raises(ValidationError, match="not bound"):
        PeriodicPointProfileResult.model_validate(payload)


def test_value_models_reject_ambiguous_and_invalid_carriers() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        ForbiddenBlockShift(alphabet=("0", "0"), forbidden_blocks=())
    with pytest.raises(ValidationError, match="outside"):
        ForbiddenBlockShift(alphabet=("0",), forbidden_blocks=(("1",),))
    with pytest.raises(ValidationError, match="square"):
        AdjacencyShift(matrix=((1, 0), (1,)))
    with pytest.raises(ValidationError, match="supported bounds"):
        AdjacencyShift(matrix=((-1,),))


def test_random_block_languages_match_independent_filter_oracle() -> None:
    random_source = random.Random(1968)
    alphabet = ("0", "1")
    candidate_forbidden = tuple(
        itertools.chain(
            itertools.product(alphabet, repeat=1),
            itertools.product(alphabet, repeat=2),
            itertools.product(alphabet, repeat=3),
        )
    )
    for _ in range(120):
        forbidden = tuple(
            block for block in candidate_forbidden if random_source.random() < 0.18
        )
        shift = ForbiddenBlockShift(alphabet=alphabet, forbidden_blocks=forbidden)
        for length in range(5):
            expected = tuple(
                word
                for word in itertools.product(alphabet, repeat=length)
                if all(
                    not any(
                        word[start : start + len(block)] == block
                        for start in range(len(word) - len(block) + 1)
                    )
                    for block in forbidden
                )
            )
            assert block_language(shift, length) == expected


def test_periodic_profiles_match_closed_walk_and_divisor_oracles() -> None:
    matrices = (
        ((0,),),
        ((3,),),
        ((0, 1), (1, 0)),
        ((1, 1), (1, 0)),
        ((1, 2), (0, 1)),
    )
    for matrix in matrices:
        fixed, exact, orbits = periodic_point_profile(AdjacencyShift(matrix=matrix), 6)
        size = len(matrix)
        expected_fixed = []
        for period in range(1, 7):
            count = 0
            for walk in itertools.product(range(size), repeat=period):
                weight = 1
                for index in range(period):
                    weight *= matrix[walk[index]][walk[(index + 1) % period]]
                count += weight
            expected_fixed.append(count)
        expected_exact: list[int] = []
        for period, count in enumerate(expected_fixed, 1):
            expected_exact.append(
                count
                - sum(
                    expected_exact[divisor - 1]
                    for divisor in range(1, period)
                    if period % divisor == 0
                )
            )
        assert fixed == tuple(expected_fixed)
        assert exact == tuple(expected_exact)
        assert orbits == tuple(
            count // period for period, count in enumerate(expected_exact, 1)
        )
