"""Independent exact profiles and deterministic single-enumeration evidence."""

import time
from collections import Counter
from collections.abc import Iterator
from itertools import combinations
from math import comb
from sys import int_info
from threading import Event
from types import SimpleNamespace

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.designs.incidence_structures import (
    IncidenceStructure,
    _kernel,
    _models,
    containment_profile,
)
from jacobian.math.combinatorics.designs.incidence_structures._models import (
    _MAX_CONTAINMENT_TOTAL_WORK_UNITS,
    _MAX_TRADE_TOTAL_WORK_UNITS,
    MAX_BLOCKS,
    MAX_POINTS,
    MAX_SUBSETS,
    MAX_T,
    _profile_work_limit,
    _profile_work_units,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs import FiniteHypergraph
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    MAX_VERTICES,
)


def _oracle(
    points: tuple[str, ...], blocks: tuple[tuple[str, ...], ...], order: int
) -> _kernel.ContainmentProfileData:
    members = tuple(set(block) for block in blocks)
    profile = tuple(
        (subset, sum(set(subset).issubset(block) for block in members))
        for subset in combinations(points, order)
    )
    counts = tuple(count for _, count in profile)
    low, high = min(counts, default=0), max(counts, default=0)
    return (
        profile,
        tuple(sorted(Counter(counts).items())),
        sum(counts),
        low,
        high,
        low == high,
        low if low == high else None,
    )


@pytest.mark.parametrize("kind", ["incidence", "hypergraph"])
def test_every_small_block_and_order_matches_set_containment(kind: str) -> None:
    points = ("z", "a", "é", "p10", "p2")
    blocks = tuple(
        subset for order in range(6) for subset in combinations(points, order)
    )
    # Repeat both empty and nonempty blocks with independent identities.
    blocks += ((), ("z", "é"), ("z", "é"), points)
    source = (
        IncidenceStructure(
            points=points,
            block_ids=tuple(f"b{i}" for i in range(len(blocks))),
            blocks=blocks,
        )
        if kind == "incidence"
        else FiniteHypergraph(
            vertices=points,
            edges=tuple(
                (f"b{i}", tuple(reversed(block))) for i, block in enumerate(blocks)
            ),
        )
    )
    for order in range(8):
        assert _kernel.containment_profile_data(source, order) == _oracle(
            points, blocks, order
        )
        result = containment_profile(source, order)
        assert result.subset_profile == _oracle(points, blocks, order)[0]
        assert type(result).model_validate_json(result.model_dump_json()) == result


def test_dense_profile_materializes_each_output_combination_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = tuple(f"p{i}" for i in range(100))
    source = IncidenceStructure(
        points=points,
        block_ids=tuple(f"b{i}" for i in range(100)),
        blocks=(points,) * 100,
    )
    calls: list[tuple[tuple[str, ...], int]] = []
    emitted = 0

    def observed(axis: tuple[str, ...], order: int) -> Iterator[tuple[str, ...]]:
        nonlocal emitted
        calls.append((axis, order))
        for subset in combinations(axis, order):
            emitted += 1
            yield subset

    # Observation wraps the real iterator; mathematical expectations below do
    # not use the producer kernel or a replacement counting algorithm.
    monkeypatch.setattr(_kernel, "combinations", observed)
    result = containment_profile(source, 2)
    assert calls == [(points, 2)]
    assert emitted == len(result.subset_profile) == 4_950
    assert all(count == 100 for _, count in result.subset_profile)
    assert result.histogram == ((100, 4_950),)
    assert result.total_multiplicity == 495_000


def test_previously_rejected_dense_hypergraph_is_cheap_and_exact() -> None:
    points = tuple(f"p{i}" for i in range(14))
    source = FiniteHypergraph(
        vertices=points, edges=tuple((f"b{i}", points) for i in range(2_500))
    )
    result = containment_profile(source, 7)
    assert result.subset_profile == tuple(
        (subset, 2_500) for subset in combinations(points, 7)
    )
    assert result.histogram == ((2_500, 3_432),)
    assert result.total_multiplicity == 8_580_000
    assert result.constant_lambda == 2_500
    assert type(result).model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("order", [0, 1, 3, 10])
def test_twelve_thousand_indexed_edges_preserve_multiplicity(order: int) -> None:
    points = tuple(f"p{i}" for i in range(15))
    source = FiniteHypergraph(
        vertices=points, edges=tuple((f"e{i}", points[:3]) for i in range(12_000))
    )
    result = containment_profile(source, order)
    expected = tuple(
        (subset, 12_000 if set(subset) <= set(points[:3]) else 0)
        for subset in combinations(points, order)
    )
    assert result.subset_profile == expected
    assert result.total_multiplicity == (12_000 * comb(3, order) if order <= 3 else 0)


def test_subset_refusal_stays_at_the_real_complete_output_boundary() -> None:
    points = tuple(f"p{i}" for i in range(101))
    with pytest.raises(
        OperationDomainValidationError, match="complete subset-count budget"
    ):
        containment_profile(FiniteHypergraph(vertices=points, edges=()), 2)


def test_work_estimate_charges_word_width_and_skips_short_blocks() -> None:
    points = tuple(f"p{i}" for i in range(32))
    source = FiniteHypergraph(
        vertices=points, edges=tuple((f"b{i}", points[:3]) for i in range(12_000))
    )
    words = (12_000 + int_info.bits_per_digit - 1) // int_info.bits_per_digit
    subsets = comb(32, 3)
    expected = (
        32
        + 12_000
        + 36_000
        + 32 * (1_500 + words)
        + 36_000
        + 12_000
        + subsets * (3 * (1 + words) + 6)
        + subsets * subsets.bit_length()
    )
    assert _profile_work_units(source, 3) == expected
    # No index or nonzero-width intersections for edges too short to contribute.
    assert _profile_work_units(source, 4) == 48_032 + comb(32, 4) * 10 + 1


@pytest.mark.parametrize("limb_bits", [15, 30])
def test_current_carrier_and_subset_bounds_fit_new_word_work_envelope(
    monkeypatch: pytest.MonkeyPatch, limb_bits: int
) -> None:
    monkeypatch.setattr(_models, "int_info", SimpleNamespace(bits_per_digit=limb_bits))
    assert _profile_work_limit(_MAX_CONTAINMENT_TOTAL_WORK_UNITS) == (
        16_000_000 if limb_bits == 15 else 8_000_000
    )
    # A complete upper bound, not a sampled benchmark: eligible edge count E
    # satisfies t*E <= 36,000. CPython's actual integer limb width is charged.
    word_bound = MAX_TOTAL_INCIDENCES // limb_bits + MAX_T
    source_bound = MAX_VERTICES + MAX_EDGES + MAX_TOTAL_INCIDENCES
    index_bound = (
        MAX_VERTICES * ((MAX_EDGES + 7) // 8 + (MAX_EDGES + limb_bits - 1) // limb_bits)
        + MAX_TOTAL_INCIDENCES
        + MAX_EDGES
    )
    output_bound = MAX_SUBSETS * (word_bound + MAX_T + 6 + MAX_SUBSETS.bit_length())
    assert source_bound + index_bound + output_bound <= _profile_work_limit(
        _MAX_CONTAINMENT_TOTAL_WORK_UNITS
    )
    # Trade values have the narrower 100-point/100-block carrier. Enumerating
    # the 100 possible axes proves preservation across every admitted order.
    for n in range(1, MAX_POINTS + 1):
        source = IncidenceStructure(
            points=tuple(f"p{i}" for i in range(n)),
            block_ids=tuple(f"b{i}" for i in range(MAX_BLOCKS)),
            blocks=(tuple(f"p{i}" for i in range(n)),) * MAX_BLOCKS,
        )
        total = 0
        for order in range(1, MAX_T + 1):
            if comb(n, order) > MAX_SUBSETS:
                break
            total += 2 * _profile_work_units(source, order)
            assert total <= _profile_work_limit(_MAX_TRADE_TOTAL_WORK_UNITS)


@pytest.mark.parametrize("phase", ["indexing", "enumeration"])
def test_cancellation_is_checked_inside_both_kernel_phases(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    points = tuple(f"p{i}" for i in range(20))
    source = IncidenceStructure(
        points=points, block_ids=("b1", "b2"), blocks=(points, points)
    )
    signal = Event()
    seen = 0

    def checkpoint(message: str) -> None:
        nonlocal seen
        if message == f"during containment profile {phase}":
            seen += 1
            if seen == 2:
                signal.set()
        request_checkpoint(message)

    monkeypatch.setattr(_kernel, "request_checkpoint", checkpoint)
    with (
        request_execution(time.monotonic(), cancellation_signal=signal),
        pytest.raises(OperationExecutionCancelledError),
    ):
        containment_profile(source, 2)
    assert seen == 2
