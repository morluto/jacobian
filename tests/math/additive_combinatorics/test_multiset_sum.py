"""Tests for fixed-arity unordered multiset-sum profiles."""

import tracemalloc
from collections import Counter
from itertools import combinations, combinations_with_replacement

import pytest
from pydantic import ValidationError
from sympy import primerange

from jacobian.canonical import CanonicalLimits, canonicalize_json
from jacobian.math.additive_combinatorics._models import (
    _MAX_MULTISET_SUM_ARITY,
    _MAX_MULTISET_SUM_ELEMENT_DIGITS,
    _MAX_MULTISET_SUM_ENUMERATION_WORK,
    _MAX_MULTISET_SUM_INTEGER_LENGTH,
    _MAX_MULTISET_SUM_RESULT_DIGITS,
    _MAX_MULTISET_SUM_SUPPORT_SIZE,
    _MAX_SET_SIZE,
    FiniteIntegerSet,
    MultisetSumRepresentationProfileRequest,
    MultisetSumRepresentationProfileResult,
    RepresentationProfileRequest,
)
from jacobian.math.additive_combinatorics._multiset_sum import (
    MAX_ARITY,
    RESULT_BUDGET_BYTES,
    _bar_position_tuples,
)
from jacobian.math.additive_combinatorics._operations import (
    compute_multiset_sum_representation_profile,
)


def _request(
    source: tuple[int, ...],
    arity: int,
    window: tuple[int, int] | None = None,
) -> MultisetSumRepresentationProfileRequest:
    return MultisetSumRepresentationProfileRequest(
        source={"elements": [str(value) for value in source]},
        arity=arity,
        window=(
            None
            if window is None
            else {"lower": str(window[0]), "upper": str(window[1])}
        ),
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


def test_result_rejects_forged_source_bound_profile() -> None:
    result = compute_multiset_sum_representation_profile(_request((0, 1, 2), 2))
    payload = result.model_dump(mode="json")
    payload["source"]["elements"] = ["0", "1", "3"]
    with pytest.raises(ValidationError, match="exact source-bound"):
        MultisetSumRepresentationProfileResult.model_validate(payload)


def test_result_rejects_forged_multiplicity() -> None:
    result = compute_multiset_sum_representation_profile(_request((0, 1, 2), 2))
    payload = result.model_dump(mode="json")
    payload["entries"][2]["multiplicity"] = 1
    with pytest.raises(ValidationError, match="exact source-bound"):
        MultisetSumRepresentationProfileResult.model_validate(payload)


def test_result_rejects_mutated_window() -> None:
    result = compute_multiset_sum_representation_profile(_request((0, 1, 2), 2, (2, 2)))
    payload = result.model_dump(mode="json")
    payload["window"] = {"lower": "1", "upper": "2"}
    with pytest.raises(ValidationError, match="exact source-bound"):
        MultisetSumRepresentationProfileResult.model_validate(payload)


def test_result_round_trip_replays_the_defining_invariant() -> None:
    result = compute_multiset_sum_representation_profile(
        _request((-5, 0, 9), 5, (-10, 20))
    )
    assert (
        MultisetSumRepresentationProfileResult.model_validate(result.model_dump())
        == result
    )


def test_request_requires_canonical_source_order() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        _request((1, -2, 3), 2)


def test_request_rejects_oversized_source_integer_before_parsing() -> None:
    oversized = "9" * (_MAX_MULTISET_SUM_ELEMENT_DIGITS + 1)
    with pytest.raises(ValidationError, match="at most 64 digits"):
        MultisetSumRepresentationProfileRequest(
            source={"elements": [oversized]}, arity=2
        )


def test_request_schema_exposes_collection_and_scalar_bounds() -> None:
    schema = MultisetSumRepresentationProfileRequest.model_json_schema()
    source_schema = schema["$defs"]["FiniteIntegerSet"]
    assert source_schema["properties"]["elements"]["maxItems"] == _MAX_SET_SIZE
    assert schema["properties"]["arity"]["maximum"] == _MAX_MULTISET_SUM_ARITY
    window_schema = schema["$defs"]["MultisetSumWindow"]
    assert (
        window_schema["properties"]["lower"]["maxLength"]
        == _MAX_MULTISET_SUM_INTEGER_LENGTH
    )


def test_singleton_source_admits_maximum_arity_without_tuple_expansion() -> None:
    assert _profile((2,), _MAX_MULTISET_SUM_ARITY) == {2 * _MAX_MULTISET_SUM_ARITY: 1}


def test_arity_above_schema_bound_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _request((2,), _MAX_MULTISET_SUM_ARITY + 1)


def test_compact_singleton_admits_arity_beyond_the_former_fixed_ceiling() -> None:
    # Reported failure mode: one candidate, one coordinate step, one result
    # row; the derived work and support preflights admit it at any arity.
    assert _profile((0,), 10_000_000) == {0: 1}
    assert _profile((3,), 10**15) == {3 * 10**15: 1}


def test_empty_source_admits_large_arity_with_zero_candidates() -> None:
    assert _profile((), 10**15) == {}


def test_singleton_sum_digits_stay_within_the_derived_result_bound() -> None:
    value = 10 ** (_MAX_MULTISET_SUM_ELEMENT_DIGITS - 1) + 19
    entries = _profile((value,), MAX_ARITY)
    assert len(entries) == 1
    ((sum_value, multiplicity),) = entries.items()
    assert multiplicity == 1
    assert sum_value == value * MAX_ARITY
    assert len(str(sum_value)) <= _MAX_MULTISET_SUM_RESULT_DIGITS


def test_costly_large_arity_is_rejected_by_work_not_by_an_arity_cap() -> None:
    with pytest.raises(ValidationError, match="coordinate steps"):
        _request((0, 1), 10**15)


def test_bar_positions_match_the_itertools_combinations_oracle() -> None:
    for pool_size in range(7):
        for bars in range(pool_size + 2):
            assert list(_bar_position_tuples(pool_size, bars)) == list(
                combinations(range(pool_size), bars)
            )


def test_large_slot_pool_is_never_materialized_during_iteration() -> None:
    # Before the lazy iterator, combinations(range(slots), bars) snapshotted
    # the whole admitted slot range into memory (about 400 MB at the work
    # boundary) before yielding a single candidate.
    slots = _MAX_MULTISET_SUM_ENUMERATION_WORK // 2 + 1
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
def test_disjoint_window_over_admitted_two_element_source_is_exactly_empty(
    window: tuple[int, int],
) -> None:
    # Reported failure mode: an admitted two-element request whose window
    # misses every attainable sum must return the exact empty profile without
    # enumerating the candidate family.
    arity = _MAX_MULTISET_SUM_ENUMERATION_WORK // 2 - 1
    assert _profile((0, 1), arity, window) == {}


@pytest.mark.parametrize(
    "window",
    [(-10, -1), (10**15 + 1, 10**15 + 7)],
)
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
def test_intersecting_window_still_pays_full_enumeration_work(
    window: tuple[int, int],
) -> None:
    # Boundary of the shortcut: a window sharing any point with the attainable
    # interval requires real candidate inspection, so the work preflight still
    # rejects it even though the declared span is small.
    with pytest.raises(ValidationError, match="coordinate steps"):
        _request((0, 1), 10**15, window)


def test_disjoint_window_result_rejects_forged_entries() -> None:
    result = compute_multiset_sum_representation_profile(
        _request((0, 1), 10**15, (-10, -1))
    )
    assert result.entries == ()
    payload = result.model_dump(mode="json")
    payload["entries"] = [{"sum": "5", "multiplicity": 1}]
    with pytest.raises(ValidationError, match="exact source-bound"):
        MultisetSumRepresentationProfileResult.model_validate(payload)


def test_narrow_window_over_large_slot_family_matches_closed_form() -> None:
    arity = 100_000
    assert _profile((0, 1), arity, (0, 3)) == {0: 1, 1: 1, 2: 1, 3: 1}


def test_reversed_sum_window_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        _request((0, 1), 2, (3, 2))


def test_sum_window_endpoint_digit_bound_is_enforced() -> None:
    with pytest.raises(ValidationError, match="at most 82 digits"):
        _request((0, 1), 2, (0, int("9" * (_MAX_MULTISET_SUM_RESULT_DIGITS + 1))))


def test_full_profile_rejects_worst_case_support_above_result_bound() -> None:
    source_size = 361
    assert source_size * (source_size + 1) // 2 > _MAX_MULTISET_SUM_SUPPORT_SIZE
    spacing = 2 * source_size * source_size
    offset = 10**63
    source = tuple(offset + spacing * i + i * i for i in range(source_size))
    with pytest.raises(ValidationError, match="row result bound"):
        _request(source, 2)


def test_dense_full_profile_uses_the_attainable_sum_range_bound() -> None:
    source = tuple(range(_MAX_SET_SIZE))
    result = compute_multiset_sum_representation_profile(_request(source, 2))

    assert len(result.entries) == 2 * _MAX_SET_SIZE - 1
    assert sum(entry.multiplicity for entry in result.entries) == (
        _MAX_SET_SIZE * (_MAX_SET_SIZE + 1) // 2
    )


def test_narrow_window_admits_large_candidate_family_with_small_output() -> None:
    source = tuple(range(327))
    result = compute_multiset_sum_representation_profile(_request(source, 3, (0, 0)))
    assert [(entry.sum, entry.multiplicity) for entry in result.entries] == [("0", 1)]


def test_request_rejects_enumeration_above_work_bound() -> None:
    _request(tuple(range(340)), 3, (0, 0))
    with pytest.raises(ValidationError, match="coordinate steps"):
        _request(tuple(range(341)), 3, (0, 0))


def test_widened_source_axis_preserves_cartesian_pair_bound() -> None:
    left = FiniteIntegerSet(elements=tuple(str(i) for i in range(257)))
    right = FiniteIntegerSet(elements=tuple(str(i) for i in range(256)))
    with pytest.raises(ValidationError, match="Cartesian product"):
        RepresentationProfileRequest(left=left, right=right)


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
