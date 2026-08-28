"""Tests for fixed-arity unordered multiset-sum profiles."""

import tracemalloc
from collections import Counter
from itertools import combinations, combinations_with_replacement

import pytest
from pydantic import ValidationError
from sympy import primerange

from jacobian.canonical import CanonicalLimits, canonicalize_json
from jacobian.math.combinatorics.additive._models import (
    _MAX_SET_SIZE,
    FiniteIntegerSet,
    MultisetSumRepresentationProfileRequest,
    MultisetSumRepresentationProfileResult,
    RepresentationProfileRequest,
)
from jacobian.math.combinatorics.additive._multiset_sum import (
    MAX_ARITY,
    MAX_ELEMENT_DIGITS,
    MAX_ENUMERATION_WORK,
    MAX_INTEGER_LENGTH,
    MAX_RESULT_DIGITS,
    MAX_SUPPORT_SIZE,
    RESULT_BUDGET_BYTES,
    _bar_position_tuples,
)
from jacobian.math.combinatorics.additive._operations import (
    compute_multiset_sum_representation_profile,
    compute_representation_profile,
)
from jacobian.math.combinatorics.finite_structures.sets._models import (
    MAX_FINITE_INTEGER_SET_ELEMENTS,
)


def _request(
    source: tuple[int, ...],
    arity: int,
    window: tuple[int, int] | None = None,
) -> MultisetSumRepresentationProfileRequest:
    return MultisetSumRepresentationProfileRequest.model_validate(
        {
            "source": {"elements": [str(value) for value in source]},
            "arity": arity,
            "window": (
                None
                if window is None
                else {"lower": str(window[0]), "upper": str(window[1])}
            ),
        }
    )


def _profile(
    source: tuple[int, ...],
    arity: int,
    window: tuple[int, int] | None = None,
) -> dict[int, int]:
    result = compute_multiset_sum_representation_profile(
        _request(source, arity, window)
    )
    return {int(entry.sum): entry.multiplicity for entry in result.entries}


def test_complete_pair_multiset_profile_distinguishes_unordered_semantics() -> None:
    assert _profile((0, 1, 2), 2) == {0: 1, 1: 1, 2: 2, 3: 1, 4: 1}


def test_three_term_binary_source_has_one_representation_per_sum() -> None:
    assert _profile((0, 1), 3) == {0: 1, 1: 1, 2: 1, 3: 1}


@pytest.mark.parametrize(
    ("source", "arity", "expected"),
    [
        ((), 0, {0: 1}),
        ((4, 9), 0, {0: 1}),
        ((), 3, {}),
        ((-7,), 6, {-42: 1}),
    ],
)
def test_empty_zero_and_singleton_conventions(
    source: tuple[int, ...], arity: int, expected: dict[int, int]
) -> None:
    assert _profile(source, arity) == expected


def test_zero_arity_uses_derived_admission_not_the_legacy_source_cap() -> None:
    source = tuple(range(_MAX_SET_SIZE + 1))

    assert _profile(source, 0) == {0: 1}


def test_closed_window_is_complete_only_for_its_declared_scope() -> None:
    result = compute_multiset_sum_representation_profile(_request((0, 1, 2), 3, (2, 3)))
    assert result.window is not None
    assert (result.window.lower, result.window.upper) == ("2", "3")
    assert {int(entry.sum): entry.multiplicity for entry in result.entries} == {
        2: 2,
        3: 2,
    }


def test_translation_translates_sums_by_arity_times_offset() -> None:
    source = (-2, 0, 5)
    arity = 4
    offset = 11
    base = _profile(source, arity)
    translated = _profile(tuple(value + offset for value in source), arity)
    assert translated == {
        value + arity * offset: multiplicity for value, multiplicity in base.items()
    }


@pytest.mark.exhaustive
def test_bounded_exhaustive_profiles_match_itertools_oracle() -> None:
    universe = (-2, 0, 3, 7)
    for source_size in range(len(universe) + 1):
        for source in combinations(universe, source_size):
            for arity in range(5):
                expected = Counter(
                    sum(terms) for terms in combinations_with_replacement(source, arity)
                )
                assert _profile(source, arity) == dict(expected)


def test_result_retains_source_for_unchanged_recomposition() -> None:
    result = compute_multiset_sum_representation_profile(_request((-3, 1, 8), 3))
    recomputed = compute_multiset_sum_representation_profile(
        MultisetSumRepresentationProfileRequest(
            source=result.source,
            arity=result.arity,
            window=result.window,
        )
    )
    assert recomputed == result


def test_result_round_trip_preserves_the_profile() -> None:
    result = compute_multiset_sum_representation_profile(
        _request((-5, 0, 9), 5, (-10, 20))
    )
    decoded = MultisetSumRepresentationProfileResult.model_validate(result.model_dump())
    assert decoded == result


def test_request_requires_canonical_source_order() -> None:
    with pytest.raises(ValueError):
        _profile((1, -2, 3), 2)


def test_request_rejects_oversized_source_integer_before_parsing() -> None:
    oversized = "9" * (MAX_ELEMENT_DIGITS + 1)
    with pytest.raises(ValueError):
        _profile((int(oversized),), 2)


def test_request_schema_exposes_collection_and_scalar_bounds() -> None:
    schema = MultisetSumRepresentationProfileRequest.model_json_schema()
    source_schema = schema["$defs"]["FiniteIntegerSet"]
    assert (
        source_schema["properties"]["elements"]["maxItems"]
        == MAX_FINITE_INTEGER_SET_ELEMENTS
    )
    assert schema["properties"]["arity"]["maximum"] == MAX_ARITY
    window_schema = schema["$defs"]["MultisetSumWindow"]
    assert window_schema["properties"]["lower"]["maxLength"] == MAX_INTEGER_LENGTH


def test_singleton_source_admits_maximum_arity_without_tuple_expansion() -> None:
    assert _profile((2,), MAX_ARITY) == {2 * MAX_ARITY: 1}


def test_arity_above_schema_bound_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _request((2,), MAX_ARITY + 1)


@pytest.mark.scale
def test_compact_singleton_admits_arity_beyond_the_former_fixed_ceiling() -> None:
    # Reported failure mode: one candidate, one coordinate step, one result
    # row; the derived work and support preflights admit it at any arity.
    assert _profile((0,), 10_000_000) == {0: 1}
    assert _profile((3,), 10**15) == {3 * 10**15: 1}


def test_empty_source_admits_large_arity_with_zero_candidates() -> None:
    assert _profile((), 10**15) == {}


def test_singleton_sum_digits_stay_within_the_derived_result_bound() -> None:
    value = 10 ** (MAX_ELEMENT_DIGITS - 1) + 19
    entries = _profile((value,), MAX_ARITY)
    assert len(entries) == 1
    ((sum_value, multiplicity),) = entries.items()
    assert multiplicity == 1
    assert sum_value == value * MAX_ARITY
    assert len(str(sum_value)) <= MAX_RESULT_DIGITS


def test_costly_large_arity_is_rejected_by_work_not_by_an_arity_cap() -> None:
    with pytest.raises(ValueError):
        _profile((0, 1), 10**15)


def test_bar_positions_match_the_itertools_combinations_oracle() -> None:
    for pool_size in range(7):
        for bars in range(pool_size + 2):
            assert list(_bar_position_tuples(pool_size, bars)) == list(
                combinations(range(pool_size), bars)
            )


@pytest.mark.scale
def test_large_slot_pool_is_never_materialized_during_iteration() -> None:
    # Before the lazy iterator, combinations(range(slots), bars) snapshotted
    # the whole admitted slot range into memory (about 400 MB at the work
    # boundary) before yielding a single candidate.
    slots = MAX_ENUMERATION_WORK // 2 + 1
    tracemalloc.start()
    try:
        positions = _bar_position_tuples(slots, 2)
        assert next(positions) == (0, 1)
        assert next(positions) == (0, 2)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak_bytes < 1024 * 1024


@pytest.mark.parametrize(
    "window",
    [(10**15, 10**15 + 7), (-(10**15), -(10**15) + 7)],
)
@pytest.mark.scale
def test_disjoint_window_over_admitted_two_element_source_is_exactly_empty(
    window: tuple[int, int],
) -> None:
    # Reported failure mode: an admitted two-element request whose window
    # misses every attainable sum must return the exact empty profile without
    # enumerating the candidate family.
    arity = MAX_ENUMERATION_WORK // 2 - 1
    assert _profile((0, 1), arity, window) == {}


@pytest.mark.parametrize(
    "window",
    [(-10, -1), (10**15 + 1, 10**15 + 7)],
)
@pytest.mark.scale
def test_disjoint_window_charges_zero_work_above_the_enumeration_cap(
    window: tuple[int, int],
) -> None:
    # Reported failure mode: source {0,1} at arity 10^15 attains only [0,10^15],
    # so a window missing that interval proves an exactly empty profile without
    # inspecting any candidate; admission charges zero work instead of
    # rejecting candidate_count * 2 as excessive.
    assert _profile((0, 1), 10**15, window) == {}


@pytest.mark.parametrize(
    "window",
    [(0, 0), (0, 3)],
)
@pytest.mark.scale
def test_intersecting_window_still_pays_full_enumeration_work(
    window: tuple[int, int],
) -> None:
    # Boundary of the shortcut: a window sharing any point with the attainable
    # interval requires real candidate inspection, so the work preflight still
    # rejects it even though the declared span is small.
    with pytest.raises(ValueError):
        _profile((0, 1), 10**15, window)


def test_narrow_window_over_large_slot_family_matches_closed_form() -> None:
    arity = 100_000
    assert _profile((0, 1), arity, (0, 3)) == {0: 1, 1: 1, 2: 1, 3: 1}


def test_reversed_sum_window_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _request((0, 1), 2, (3, 2))


def test_sum_window_endpoint_digit_bound_is_enforced() -> None:
    with pytest.raises(ValidationError):
        _request((0, 1), 2, (0, int("9" * (MAX_RESULT_DIGITS + 1))))


def test_full_profile_rejects_worst_case_support_above_result_bound() -> None:
    source_size = 361
    assert source_size * (source_size + 1) // 2 > MAX_SUPPORT_SIZE
    spacing = 2 * source_size * source_size
    offset = 10**63
    source = tuple(offset + spacing * i + i * i for i in range(source_size))
    with pytest.raises(ValueError):
        _profile(source, 2)


@pytest.mark.scale
def test_dense_full_profile_uses_the_attainable_sum_range_bound() -> None:
    source = tuple(range(_MAX_SET_SIZE))
    result = compute_multiset_sum_representation_profile(_request(source, 2))

    assert len(result.entries) == 2 * _MAX_SET_SIZE - 1
    assert sum(entry.multiplicity for entry in result.entries) == (
        _MAX_SET_SIZE * (_MAX_SET_SIZE + 1) // 2
    )


@pytest.mark.scale
def test_narrow_window_admits_large_candidate_family_with_small_output() -> None:
    source = tuple(range(327))
    result = compute_multiset_sum_representation_profile(_request(source, 3, (0, 0)))
    assert [(entry.sum, entry.multiplicity) for entry in result.entries] == [("0", 1)]


def test_request_rejects_enumeration_above_work_bound() -> None:
    _request(tuple(range(340)), 3, (0, 0))
    with pytest.raises(ValueError):
        _profile(tuple(range(341)), 3, (0, 0))


def test_widened_source_axis_preserves_cartesian_pair_bound() -> None:
    left = FiniteIntegerSet(elements=tuple(str(i) for i in range(257)))
    right = FiniteIntegerSet(elements=tuple(str(i) for i in range(256)))
    request = RepresentationProfileRequest(left=left, right=right)
    with pytest.raises(ValueError):
        compute_representation_profile(request)


@pytest.mark.scale
def test_near_maximal_full_profile_stays_inside_owner_result_budget() -> None:
    source_size = 360
    # With spacing above every i^2+j^2, a pair sum encodes i+j and i^2+j^2
    # without carry; those two symmetric values determine the unordered pair.
    spacing = 2 * source_size * source_size
    offset = 10**63
    source = tuple(offset + spacing * i + i * i for i in range(source_size))
    result = compute_multiset_sum_representation_profile(_request(source, 2))
    assert len(result.entries) == source_size * (source_size + 1) // 2
    encoded = canonicalize_json(
        result.model_dump(mode="json"),
        limits=CanonicalLimits(max_output_bytes=RESULT_BUDGET_BYTES),
    )
    assert len(encoded) <= RESULT_BUDGET_BYTES


@pytest.mark.scale
def test_oeis_prime_cube_targets_have_published_multiplicities() -> None:
    # https://oeis.org/A385316 publishes these five values and uses the first
    # 327 primes for the fifth. The pinned Atlas certificate records the same
    # fixture: https://github.com/techno-optimist/erdos-frontier-atlas/tree/0394e3d3b249439ffabec7d96a3311aa441651b8/certificates/erdos-979
    targets = (24, 185527, 8627527, 999979163, 10588881419)
    prime_cubes = tuple(prime**3 for prime in primerange(2, 2200))
    assert len(prime_cubes) == 327
    for multiplicity, target in enumerate(targets, start=1):
        complete_source = tuple(value for value in prime_cubes if value <= target)
        result = compute_multiset_sum_representation_profile(
            _request(complete_source, 3, (target, target))
        )
        assert [(entry.sum, entry.multiplicity) for entry in result.entries] == [
            (str(target), multiplicity)
        ]
