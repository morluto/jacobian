"""Correctness and contract tests for exact symbolic dynamics operations."""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.math.symbolic_dynamics import (
    AdjacencyShift,
    ForbiddenBlockShift,
    adjacency_shift,
    artin_mazur_zeta,
    block_language,
    finite_type_presentation,
    normalize_forbidden_blocks,
    periodic_point_profile,
)
from jacobian.math.symbolic_dynamics._models import (
    ArtinMazurZetaRequest,
    ArtinMazurZetaResult,
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
    compute_artin_mazur_zeta,
    compute_block_language,
    compute_higher_block,
    compute_periodic_point_profile,
    construct_finite_type_shift,
)
from jacobian.math.symbolic_dynamics._tools import TOOLS


def _golden_mean() -> ForbiddenBlockShift:
    return ForbiddenBlockShift(alphabet=("0", "1"), forbidden_blocks=(("1", "1"),))


def test_golden_mean_artin_mazur_zeta_is_bound_to_periodic_traces() -> None:
    request = ArtinMazurZetaRequest(
        shift=AdjacencyShift(matrix=((1, 1), (1, 0))),
        replay_period=5,
    )

    result = compute_artin_mazur_zeta(request)

    assert tuple(
        (term.coefficient.as_fraction(), term.exponents)
        for term in result.determinant_polynomial.polynomial.terms
    ) == ((-1, (2,)), (-1, (1,)), (1, (0,)))
    assert tuple(
        (term.coefficient.as_fraction(), term.exponents)
        for term in result.zeta_function.denominator.terms
    ) == ((1, (2,)), (1, (1,)), (-1, (0,)))
    assert result.zeta_function.numerator.terms[0].coefficient.as_fraction() == -1
    assert tuple(row.trace_fixed_points for row in result.replay) == (
        "1",
        "3",
        "4",
        "7",
        "11",
    )
    assert tuple(row.logarithmic_derivative_coefficient for row in result.replay) == (
        "1",
        "3",
        "4",
        "7",
        "11",
    )
    assert ArtinMazurZetaResult.model_validate(result.model_dump(mode="json")) == result

    payload = result.model_dump()
    payload["replay"][2]["logarithmic_derivative_coefficient"] = "5"
    with pytest.raises(ValidationError) as exc_info:
        ArtinMazurZetaResult.model_validate(payload)
    assert (
        exc_info.value.errors()[0]["type"]
        == "symbolic_dynamics.artin_mazur_zeta_not_bound"
    )

    payload = result.model_dump()
    payload["determinant_polynomial"]["polynomial"]["terms"][0]["coefficient"][
        "num"
    ] = "-2"
    with pytest.raises(ValidationError) as exc_info:
        ArtinMazurZetaResult.model_validate(payload)
    assert (
        exc_info.value.errors()[0]["type"]
        == "symbolic_dynamics.artin_mazur_zeta_not_bound"
    )


def test_zeta_distinguishes_full_shift_cycle_and_disjoint_components() -> None:
    empty = compute_artin_mazur_zeta(
        ArtinMazurZetaRequest(shift=AdjacencyShift(matrix=((0,),)), replay_period=4)
    )
    full_shift = compute_artin_mazur_zeta(
        ArtinMazurZetaRequest(shift=AdjacencyShift(matrix=((2,),)), replay_period=4)
    )
    cycle = compute_artin_mazur_zeta(
        ArtinMazurZetaRequest(
            shift=AdjacencyShift(matrix=((0, 1), (1, 0))), replay_period=4
        )
    )
    disjoint = compute_artin_mazur_zeta(
        ArtinMazurZetaRequest(
            shift=AdjacencyShift(matrix=((2, 0), (0, 3))), replay_period=4
        )
    )

    assert tuple(row.trace_fixed_points for row in empty.replay) == (
        "0",
        "0",
        "0",
        "0",
    )
    assert (
        empty.determinant_polynomial.polynomial.terms[0].coefficient.as_fraction() == 1
    )
    assert empty.zeta_function.numerator == empty.zeta_function.denominator
    assert tuple(row.trace_fixed_points for row in full_shift.replay) == (
        "2",
        "4",
        "8",
        "16",
    )
    assert tuple(row.trace_fixed_points for row in cycle.replay) == (
        "0",
        "2",
        "0",
        "2",
    )
    assert tuple(
        (term.coefficient.as_fraction(), term.exponents)
        for term in disjoint.determinant_polynomial.polynomial.terms
    ) == ((6, (2,)), (-5, (1,)), (1, (0,)))
    assert disjoint.zeta_function.numerator.terms[0].coefficient.as_fraction() == (
        Fraction(1, 6)
    )


def test_zeta_is_invariant_under_state_relabeling() -> None:
    original = compute_artin_mazur_zeta(
        ArtinMazurZetaRequest(
            shift=AdjacencyShift(matrix=((1, 2, 0), (0, 1, 1), (1, 0, 0))),
            replay_period=6,
        )
    )
    relabeled = compute_artin_mazur_zeta(
        ArtinMazurZetaRequest(
            shift=AdjacencyShift(matrix=((0, 1, 0), (0, 1, 2), (1, 0, 1))),
            replay_period=6,
        )
    )

    assert relabeled.determinant_polynomial == original.determinant_polynomial
    assert relabeled.zeta_function == original.zeta_function
    assert relabeled.replay == original.replay


def test_zeta_admission_bounds_coefficient_growth_before_backend() -> None:
    identity = tuple(
        tuple(1 if row == column else 0 for column in range(50)) for row in range(50)
    )
    boundary = compute_artin_mazur_zeta(
        ArtinMazurZetaRequest(shift=AdjacencyShift(matrix=identity), replay_period=50)
    )
    assert len(boundary.determinant_polynomial.polynomial.terms) == 51
    assert len(boundary.replay) == 50

    dense_large = (tuple((1_000_000,) * 50),) * 50
    with pytest.raises(ValidationError) as exc_info:
        ArtinMazurZetaRequest(shift=AdjacencyShift(matrix=dense_large), replay_period=1)
    assert (
        exc_info.value.errors()[0]["type"]
        == "symbolic_dynamics.zeta_coefficient_digit_bound"
    )
    with pytest.raises(ValueError, match="coefficient digit bound"):
        artin_mazur_zeta(AdjacencyShift(matrix=dense_large), 1)


def test_public_surface_excludes_adjacency_carrier_invariants() -> None:
    assert tuple(tool.operation_id for tool in TOOLS) == (
        "symbolic_dynamics.finite_type_shift.construct",
        "symbolic_dynamics.block_language.compute",
        "symbolic_dynamics.periodic_point_profile.compute",
        "symbolic_dynamics.artin_mazur_zeta.compute",
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
    with pytest.raises(ValidationError) as exc_info:
        FiniteTypeShiftResult.model_validate(payload)
    assert (
        exc_info.value.errors()[0]["type"]
        == "symbolic_dynamics.presentation_adjacency_transition_count"
    )


def test_shorter_forbidden_factors_are_enforced_in_long_memory_presentation() -> None:
    shift = ForbiddenBlockShift(
        alphabet=("0", "1"),
        forbidden_blocks=(("0",), ("1", "1")),
    )
    presentation = finite_type_presentation(shift)
    assert presentation.state_blocks == ()
    assert presentation.transitions == ()
    assert presentation.adjacency_matrix == ()


def test_two_sided_language_removes_states_without_left_extension() -> None:
    two_sided = ForbiddenBlockShift(
        alphabet=("0", "1"),
        forbidden_blocks=(("0", "0"), ("1", "0")),
        two_sided=True,
    )
    one_sided = two_sided.model_copy(update={"two_sided": False})
    assert block_language(two_sided, 1) == (("1",),)
    assert block_language(one_sided, 1) == (("0",), ("1",))
    result = compute_block_language(
        BlockLanguageRequest(shift=two_sided, block_length=1)
    )
    assert result.scope == "ALL_OCCURRING_BLOCKS_OF_REQUESTED_LENGTH"
    assert "occurs in an infinite shift point" in next(
        tool.description
        for tool in TOOLS
        if tool.operation_id == "symbolic_dynamics.block_language.compute"
    )


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
    with pytest.raises(ValidationError) as exc_info:
        BlockLanguageResult.model_validate(payload)
    assert (
        exc_info.value.errors()[0]["type"]
        == "symbolic_dynamics.block_language_not_bound"
    )


def test_oversized_enumerations_fail_before_computation() -> None:
    alphabet = tuple(chr(ord("a") + index) for index in range(16))
    shift = ForbiddenBlockShift(alphabet=alphabet, forbidden_blocks=())
    with pytest.raises(ValidationError) as exc_info:
        BlockLanguageRequest(shift=shift, block_length=5)
    assert (
        exc_info.value.errors()[0]["type"]
        == "symbolic_dynamics.block_language_work_bound"
    )
    with pytest.raises(ValidationError) as exc_info:
        FiniteTypeShiftRequest(
            shift=ForbiddenBlockShift(
                alphabet=alphabet,
                forbidden_blocks=(("a", "a", "a", "a", "a"),),
            )
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "symbolic_dynamics.block_language_work_bound"
    )
    ten_symbols = alphabet[:10]
    with pytest.raises(ValidationError) as exc_info:
        FiniteTypeShiftRequest(
            shift=ForbiddenBlockShift(
                alphabet=ten_symbols,
                forbidden_blocks=(("a", "a", "a", "a", "a"),),
            )
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "symbolic_dynamics.finite_type_presentation_result_bound"
    )
    with pytest.raises(ValidationError) as exc_info:
        HigherBlockRequest(
            shift=ForbiddenBlockShift(
                alphabet=ten_symbols,
                forbidden_blocks=(),
            ),
            block_length=4,
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "symbolic_dynamics.finite_type_presentation_result_bound"
    )
    oversized_support = ForbiddenBlockShift(
        alphabet=alphabet,
        forbidden_blocks=(("a",) * 20,),
    )
    with pytest.raises(ValidationError) as exc_info:
        BlockLanguageRequest(shift=oversized_support, block_length=1)
    assert (
        exc_info.value.errors()[0]["type"]
        == "symbolic_dynamics.block_language_work_bound"
    )
    with pytest.raises(ValueError, match="work bound"):
        block_language(oversized_support, 1)


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
    with pytest.raises(ValidationError) as exc_info:
        HigherBlockResult.model_validate(payload)
    assert (
        exc_info.value.errors()[0]["type"]
        == "symbolic_dynamics.presentation_adjacency_transition_count"
    )


def test_higher_block_requires_enough_memory_for_exact_sft_presentation() -> None:
    shift = ForbiddenBlockShift(
        alphabet=("0", "1"), forbidden_blocks=(("1", "0", "1", "0"),)
    )
    with pytest.raises(ValidationError) as exc_info:
        HigherBlockRequest(shift=shift, block_length=2)
    assert (
        exc_info.value.errors()[0]["type"]
        == "symbolic_dynamics.higher_block_memory_bound"
    )


def test_periodic_profile_handles_square_mobius_factor() -> None:
    request = PeriodicPointProfileRequest(
        shift=AdjacencyShift(matrix=((2,),)), max_period=4
    )
    result = compute_periodic_point_profile(request)
    assert result.fixed_point_counts == ("2", "4", "8", "16")
    assert result.least_period_point_counts == ("2", "2", "6", "12")
    assert result.primitive_orbit_counts == ("2", "1", "2", "3")
    assert result.complete_through_period == 4

    payload = result.model_dump()
    payload["primitive_orbit_counts"] = ("2", "1", "2", "4")
    with pytest.raises(ValidationError) as exc_info:
        PeriodicPointProfileResult.model_validate(payload)
    assert (
        exc_info.value.errors()[0]["type"]
        == "symbolic_dynamics.periodic_point_profile_not_bound"
    )


def test_value_models_reject_ambiguous_and_invalid_carriers() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ForbiddenBlockShift(alphabet=("0", "0"), forbidden_blocks=())
    assert (
        exc_info.value.errors()[0]["type"]
        == "symbolic_dynamics.alphabet_symbols_not_distinct"
    )
    with pytest.raises(ValidationError) as exc_info:
        ForbiddenBlockShift(alphabet=("0",), forbidden_blocks=(("1",),))
    assert (
        exc_info.value.errors()[0]["type"]
        == "symbolic_dynamics.forbidden_block_symbol_outside_alphabet"
    )
    with pytest.raises(ValidationError) as exc_info:
        AdjacencyShift(matrix=((1, 0), (1,)))
    assert (
        exc_info.value.errors()[0]["type"]
        == "symbolic_dynamics.adjacency_matrix_not_square"
    )
    with pytest.raises(ValidationError) as exc_info:
        AdjacencyShift(matrix=((-1,),))
    assert (
        exc_info.value.errors()[0]["type"] == "symbolic_dynamics.adjacency_entry_bound"
    )


def test_random_block_languages_match_bounded_extension_oracle() -> None:
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
            memory = max((len(block) - 1 for block in forbidden), default=0)
            state_count = len(alphabet) ** memory
            padding = state_count + memory
            extended = (
                word
                for word in itertools.product(alphabet, repeat=length + 2 * padding)
                if all(
                    not any(
                        word[start : start + len(block)] == block
                        for start in range(len(word) - len(block) + 1)
                    )
                    for block in forbidden
                )
            )
            expected = tuple(
                sorted({word[padding : padding + length] for word in extended})
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


def test_periodic_profile_returns_large_counts_as_canonical_integers() -> None:
    result = compute_periodic_point_profile(
        PeriodicPointProfileRequest(
            shift=AdjacencyShift(matrix=((1_000_000,),)), max_period=3
        )
    )
    assert result.fixed_point_counts[-1] == "1000000000000000000"
