"""Minimum generalized exact-cover tests."""

import time
from typing import Any, NoReturn

import pytest

import jacobian.math.combinatorics.exact_cover as exact_cover_module
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.exact_cover import (
    ExactCoverRow,
    GeneralizedExactCoverInstance,
    MinimumGeneralizedExactCoverResult,
    minimum_generalized_exact_cover,
)


def _instance(
    rows: tuple[tuple[str, tuple[str, ...]], ...],
) -> GeneralizedExactCoverInstance:
    return GeneralizedExactCoverInstance(
        primary_items=("p", "q"),
        secondary_items=(),
        rows=tuple(ExactCoverRow(row_id=name, items=items) for name, items in rows),
    )


def test_minimum_cover_beats_first_larger_cover() -> None:
    result = minimum_generalized_exact_cover(
        _instance(
            (
                ("a-p", ("p",)),
                ("b-q", ("q",)),
                ("z-both", ("p", "q")),
            )
        )
    )
    assert result.status == "EXACT"
    assert result.selected_row_ids == ("z-both",)
    assert result.lower_bound == result.upper_bound == 1


def test_tied_minima_use_lexicographically_canonical_row_ids() -> None:
    result = minimum_generalized_exact_cover(
        _instance(
            (
                ("a-p", ("p",)),
                ("b-q", ("q",)),
                ("c-p", ("p",)),
                ("d-q", ("q",)),
            )
        )
    )
    assert result.status == "EXACT"
    assert result.selected_row_ids == ("a-p", "b-q")


def test_tied_minima_remain_canonical_after_branching_heuristic() -> None:
    result = minimum_generalized_exact_cover(
        GeneralizedExactCoverInstance(
            primary_items=("p0", "p1", "p2"),
            secondary_items=(),
            rows=tuple(
                ExactCoverRow(row_id=row_id, items=items)
                for row_id, items in (
                    ("a", ("p2",)),
                    ("b", ("p0",)),
                    ("c", ("p2",)),
                    ("d", ("p0", "p2")),
                    ("e", ("p0", "p2")),
                    ("f", ("p1", "p2")),
                    ("g", ("p0", "p1")),
                )
            ),
        )
    )
    assert result.status == "EXACT"
    assert result.selected_row_ids == ("a", "g")


def test_minimum_cover_is_available_from_native_combinatorics_api() -> None:
    from jacobian.math import combinatorics

    result = combinatorics.minimum_generalized_exact_cover(
        _instance((("p", ("p",)), ("q", ("q",))))
    )
    assert result.status == "EXACT"


def test_infeasibility_requires_exhaustion() -> None:
    instance = GeneralizedExactCoverInstance(
        primary_items=("p",), secondary_items=(), rows=()
    )
    assert minimum_generalized_exact_cover(instance).status == "INFEASIBLE"


def test_empty_primary_axis_has_empty_minimum_even_with_secondary_rows() -> None:
    result = minimum_generalized_exact_cover(
        GeneralizedExactCoverInstance(
            primary_items=(),
            secondary_items=("s",),
            rows=(ExactCoverRow(row_id="unused", items=("s",)),),
        )
    )
    assert result.status == "EXACT"
    assert result.selected_row_ids == ()
    assert result.lower_bound == result.upper_bound == 0
    assert result.item_multiplicities is not None
    assert result.item_multiplicities[0].item_id == "s"
    assert result.item_multiplicities[0].multiplicity == 0


def test_single_row_cover_avoids_generic_search_admission() -> None:
    primary = tuple(f"p{index:03d}" for index in range(256))
    instance = GeneralizedExactCoverInstance(
        primary_items=primary,
        secondary_items=(),
        rows=(
            ExactCoverRow(row_id="r0000", items=primary),
            *(
                ExactCoverRow(row_id=f"r{index:04d}", items=())
                for index in range(1, 4096)
            ),
        ),
    )

    result = minimum_generalized_exact_cover(instance)

    assert result.status == "EXACT"
    assert result.selected_row_ids == ("r0000",)
    assert result.lower_bound == result.upper_bound == 1


def test_single_row_shortcut_charges_source_row_traversals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exact_cover_module, "_MINIMUM_EXACT_COVER_WORK_LIMIT", 6_000)
    instance = GeneralizedExactCoverInstance(
        primary_items=("p",),
        secondary_items=(),
        rows=(
            ExactCoverRow(row_id="r0000", items=("p",)),
            *(
                ExactCoverRow(row_id=f"r{index:04d}", items=())
                for index in range(1, 4096)
            ),
        ),
    )

    with pytest.raises(OperationResourceAdmissionError):
        minimum_generalized_exact_cover(instance)


def test_single_row_cover_retains_the_full_secondary_axis() -> None:
    secondary = tuple(f"s{index:04d}" for index in range(4_095))
    result = minimum_generalized_exact_cover(
        GeneralizedExactCoverInstance(
            primary_items=("p",),
            secondary_items=secondary,
            rows=(ExactCoverRow(row_id="cover", items=("p",)),),
        )
    )

    assert result.status == "EXACT"
    assert result.selected_row_ids == ("cover",)
    assert result.item_multiplicities is not None
    assert len(result.item_multiplicities) == 4_096
    assert result.item_multiplicities[-1].item_id == secondary[-1]
    assert result.item_multiplicities[-1].multiplicity == 0


def test_sparse_two_row_cover_avoids_empty_row_search_admission() -> None:
    primary = tuple(f"p{index:03d}" for index in range(256))
    instance = GeneralizedExactCoverInstance(
        primary_items=primary,
        secondary_items=(),
        rows=(
            ExactCoverRow(row_id="r0000", items=primary[:128]),
            ExactCoverRow(row_id="r0001", items=primary[128:]),
            *(
                ExactCoverRow(row_id=f"r{index:04d}", items=())
                for index in range(2, 4096)
            ),
        ),
    )

    result = minimum_generalized_exact_cover(instance)

    assert result.status == "EXACT"
    assert result.selected_row_ids == ("r0000", "r0001")
    assert result.lower_bound == result.upper_bound == 2
    assert result.searched_node_count == 1


def test_secondary_conflict_is_infeasible() -> None:
    result = minimum_generalized_exact_cover(
        GeneralizedExactCoverInstance(
            primary_items=("p", "q"),
            secondary_items=("s",),
            rows=(
                ExactCoverRow(row_id="p-s", items=("p", "s")),
                ExactCoverRow(row_id="q-s", items=("q", "s")),
            ),
        )
    )
    assert result.status == "INFEASIBLE"


def test_minimum_result_round_trip_does_not_replay_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = minimum_generalized_exact_cover(_instance((("both", ("p", "q")),)))
    payload = result.model_dump_json()

    def fail_if_replayed(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("result parsing must not replay coverage")

    monkeypatch.setattr(exact_cover_module, "_expected_coverage", fail_if_replayed)
    restored = MinimumGeneralizedExactCoverResult.model_validate_json(payload)
    assert restored.selected_row_ids == result.selected_row_ids
    assert restored.item_multiplicities == result.item_multiplicities


def test_minimum_search_honors_request_deadline() -> None:
    with (
        request_execution(time.monotonic(), outer_deadline=time.monotonic() - 1),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        minimum_generalized_exact_cover(_instance((("both", ("p", "q")),)))


def test_node_limit_without_incumbent_is_an_operational_failure() -> None:
    rows = (
        *(ExactCoverRow(row_id=f"p0-{index:04d}", items=("p0",)) for index in range(8)),
        ExactCoverRow(row_id="p1-a", items=("p1",)),
        ExactCoverRow(row_id="p1-b", items=("p1",)),
    )
    instance = GeneralizedExactCoverInstance(
        primary_items=("p0", "p1"),
        secondary_items=(),
        rows=rows,
    )
    with pytest.raises(OperationResourceAdmissionError):
        minimum_generalized_exact_cover(instance, search_node_limit=1)


def test_node_limit_with_incumbent_returns_witness_backed_bounds() -> None:
    result = minimum_generalized_exact_cover(
        _instance(
            (
                ("a-p", ("p",)),
                ("b-q", ("q",)),
                ("c-p", ("p",)),
                ("d-q", ("q",)),
            )
        ),
        search_node_limit=3,
    )
    assert result.status == "BOUNDED"
    assert result.selected_row_ids == ("a-p", "b-q")
    assert result.lower_bound == result.upper_bound == 2


def test_large_node_by_item_scan_is_rejected_before_search() -> None:
    primary = tuple(f"p{i:04}" for i in range(2_048))
    instance = GeneralizedExactCoverInstance(
        primary_items=primary,
        secondary_items=(),
        rows=tuple(
            ExactCoverRow(row_id=f"r{copy}-{index:04}", items=(item,))
            for copy in range(2)
            for index, item in enumerate(primary)
        ),
    )
    with pytest.raises(OperationResourceAdmissionError):
        minimum_generalized_exact_cover(instance)


def test_sparse_high_degree_instance_is_admitted_after_unit_forcing() -> None:
    rows = (
        ExactCoverRow(row_id="pair", items=("p1", "p2")),
        *(
            ExactCoverRow(row_id=f"solo-{index:04d}", items=("p0",))
            for index in range(2_047)
        ),
    )
    instance = GeneralizedExactCoverInstance(
        primary_items=("p0", "p1", "p2"),
        secondary_items=(),
        rows=rows,
    )
    result = minimum_generalized_exact_cover(instance)
    assert result.status == "EXACT"
    assert result.selected_row_ids is not None
    assert "pair" in result.selected_row_ids
    assert result.lower_bound == result.upper_bound == 2


def test_deserialized_exact_result_rejects_zero_primary_multiplicity() -> None:
    genuine = minimum_generalized_exact_cover(
        _instance((("a-p", ("p",)), ("b-q", ("q",))))
    )
    payload = genuine.model_dump()
    assert payload["item_multiplicities"] is not None
    payload["item_multiplicities"][0]["multiplicity"] = 0
    with pytest.raises(ValueError, match="primary item"):
        MinimumGeneralizedExactCoverResult.model_validate(payload)


def test_two_primary_min_degree_branching_is_admitted() -> None:
    rows = (
        *(
            ExactCoverRow(row_id=f"p0-{index:04d}", items=("p0",))
            for index in range(2_047)
        ),
        ExactCoverRow(row_id="p1-a", items=("p1",)),
        ExactCoverRow(row_id="p1-b", items=("p1",)),
    )
    instance = GeneralizedExactCoverInstance(
        primary_items=("p0", "p1"),
        secondary_items=(),
        rows=rows,
    )
    result = minimum_generalized_exact_cover(instance)
    assert result.status == "EXACT"
    assert result.selected_row_ids == ("p0-0000", "p1-a")
    assert result.lower_bound == result.upper_bound == 2


def test_forced_secondary_conflict_is_infeasible_without_full_scan_charge() -> None:
    primary = tuple(f"p{index:03d}" for index in range(256))
    forced = ExactCoverRow(row_id="forced", items=(*primary[:-1], "s"))
    conflicting = tuple(
        ExactCoverRow(row_id=f"last-{index:04d}", items=(primary[-1], "s"))
        for index in range(4_095)
    )
    instance = GeneralizedExactCoverInstance(
        primary_items=primary,
        secondary_items=("s",),
        rows=(forced, *conflicting),
    )
    result = minimum_generalized_exact_cover(instance)
    assert result.status == "INFEASIBLE"
    assert result.selected_row_ids is None


def test_shared_secondary_on_every_row_is_infeasible_without_scan_charge() -> None:
    primary = tuple(f"p{index:03d}" for index in range(256))
    rows = tuple(
        ExactCoverRow(row_id=f"r{item}-{copy:02d}", items=(item, "s"))
        for item in primary
        for copy in range(16)
    )
    instance = GeneralizedExactCoverInstance(
        primary_items=primary,
        secondary_items=("s",),
        rows=rows,
    )
    result = minimum_generalized_exact_cover(instance)
    assert result.status == "INFEASIBLE"
    assert result.searched_node_count == 1


def test_exponential_near_universal_search_is_refused() -> None:
    primary = tuple(f"p{index:04d}" for index in range(2_048))
    rows = tuple(
        ExactCoverRow(
            row_id=f"{item}-{suffix}",
            items=(item,) if index < 50 else (item, "s"),
        )
        for index, item in enumerate(primary)
        for suffix in ("a", "b")
    )
    instance = GeneralizedExactCoverInstance(
        primary_items=primary,
        secondary_items=("s",),
        rows=rows,
    )
    with pytest.raises(OperationResourceAdmissionError, match="work"):
        minimum_generalized_exact_cover(instance)


def test_unit_forcing_honors_an_expired_deadline() -> None:
    primary = tuple(f"p{index:04d}" for index in range(4_096))
    rows = tuple(ExactCoverRow(row_id=f"r{item}", items=(item,)) for item in primary)
    instance = GeneralizedExactCoverInstance(
        primary_items=primary,
        secondary_items=(),
        rows=rows,
    )
    with request_execution(time.monotonic()):
        bind_request_deadline(time.monotonic() - 1)
        with pytest.raises(OperationExecutionTimeoutError, match="deadline expired"):
            minimum_generalized_exact_cover(instance)


def test_secondary_presolve_observes_request_cancellation() -> None:
    """The secondary scan checkpoints so cancellation is observed promptly."""

    class Cancelled:
        def is_set(self) -> bool:
            return True

    primary = ("p0", "p1")
    secondaries = tuple(f"s{index:04d}" for index in range(4_094))
    rows = tuple(
        ExactCoverRow(
            row_id=f"r{index:04d}",
            items=tuple(
                sorted(
                    (
                        primary[index % 2],
                        *(
                            secondaries[(index + step) % len(secondaries)]
                            for step in range(15)
                        ),
                    )
                )
            ),
        )
        for index in range(4_096)
    )
    instance = GeneralizedExactCoverInstance(
        primary_items=primary,
        secondary_items=secondaries,
        rows=rows,
    )
    with (
        request_execution(0.0, cancellation_signal=Cancelled()),
        pytest.raises(OperationExecutionCancelledError),
    ):
        minimum_generalized_exact_cover(instance)


def test_secondary_presolve_is_linear_in_source_incidences() -> None:
    """Indexing secondaries once avoids a quadratic, uncheckpointed scan.

    The same request is rejected by the inclusive work bound, but the scan is
    now linear: the pre-fix quadratic membership scan took several seconds.
    """
    primary = ("p0", "p1")
    secondaries = tuple(f"s{index:04d}" for index in range(4_094))
    rows = tuple(
        ExactCoverRow(
            row_id=f"r{index:04d}",
            items=tuple(
                sorted(
                    {
                        primary[index % 2],
                        *(
                            secondaries[(index + step) % len(secondaries)]
                            for step in range(15)
                        ),
                    }
                )
            ),
        )
        for index in range(4_096)
    )
    instance = GeneralizedExactCoverInstance(
        primary_items=primary,
        secondary_items=secondaries,
        rows=rows,
    )
    started = time.monotonic()
    with pytest.raises(OperationResourceAdmissionError):
        minimum_generalized_exact_cover(instance)
    assert time.monotonic() - started < 1.0


def test_near_universal_branch_is_bound_by_the_full_ceiling() -> None:
    """A missing count matching the min degree does not bound the node search."""
    primary = tuple(f"p{index:03d}" for index in range(256))
    exceptional = ExactCoverRow(row_id="p000-ex", items=(primary[0],))
    rows = (
        exceptional,
        *tuple(
            ExactCoverRow(row_id=f"r{item}-{copy:02d}", items=(item, "s"))
            for item in primary
            for copy in range(16 if item != primary[0] else 15)
        ),
    )
    instance = GeneralizedExactCoverInstance(
        primary_items=primary,
        secondary_items=("s",),
        rows=rows,
    )
    with pytest.raises(OperationResourceAdmissionError):
        minimum_generalized_exact_cover(instance)


def test_mandatory_secondary_with_covering_row_keeps_the_full_ceiling() -> None:
    """A mandatory secondary plus a covering row is not a proven linear search.

    Every row contains ``s`` and one row covers all primaries, but after
    selecting it the DFS can still enumerate the residual ``8^8`` combinations
    of non-``s`` rows, so the request is refused at the full work envelope.
    """
    primary = tuple(f"p{index:04d}" for index in range(300))
    others = primary[1:]
    rows: list[ExactCoverRow] = []
    for item in primary[:8]:
        for copy in range(8):
            rows.append(ExactCoverRow(row_id=f"{item}-{copy}", items=(item,)))
    for copy in range(64):
        rows.append(ExactCoverRow(row_id=f"single-{copy:03d}", items=(primary[8], "s")))
    for copy in range(64):
        rows.append(
            ExactCoverRow(
                row_id=f"bulk-{copy:03d}", items=tuple(sorted(("s", *others)))
            )
        )
    index = 0
    while len(rows) < 4_096:
        rows.append(ExactCoverRow(row_id=f"pad-{index:04d}", items=(primary[0], "s")))
        index += 1
    instance = GeneralizedExactCoverInstance(
        primary_items=primary,
        secondary_items=("s",),
        rows=tuple(sorted(rows, key=lambda row: row.row_id)),
    )
    with pytest.raises(OperationResourceAdmissionError, match="work"):
        minimum_generalized_exact_cover(instance)


# ---------------------------------------------------------------------------
# Enumeration laws: minimum-cover partitions (Workstream C)
# ---------------------------------------------------------------------------


def _minimum_primary_blocks(
    instance: GeneralizedExactCoverInstance, selected_row_ids: tuple[str, ...]
) -> list[frozenset[str]]:
    rows_by_id = {row.row_id: row for row in instance.rows}
    primary = set(instance.primary_items)
    return [
        frozenset(item for item in rows_by_id[row_id].items if item in primary)
        for row_id in selected_row_ids
    ]


def _assert_exact_partition(
    blocks: list[frozenset[str]], universe: tuple[str, ...]
) -> None:
    for block in blocks:
        assert block, "partition blocks must be nonempty"
    seen: set[str] = set()
    for block in blocks:
        assert not seen.intersection(block), "partition blocks must be disjoint"
        seen.update(block)
    assert seen == set(universe), "partition blocks must cover the universe"


def test_exact_minimum_partitions_the_primary_universe() -> None:
    result = minimum_generalized_exact_cover(
        _instance(
            (
                ("a-p", ("p",)),
                ("b-q", ("q",)),
                ("z-both", ("p", "q")),
            )
        )
    )
    assert result.status == "EXACT"
    assert result.selected_row_ids == ("z-both",)
    assert result.lower_bound == result.upper_bound == 1
    assert result.selected_row_ids is not None
    _assert_exact_partition(
        _minimum_primary_blocks(result.instance, result.selected_row_ids),
        result.instance.primary_items,
    )


def test_empty_universe_empty_family_minimum_is_exact_empty() -> None:
    result = minimum_generalized_exact_cover(
        GeneralizedExactCoverInstance(primary_items=(), secondary_items=(), rows=())
    )
    assert result.status == "EXACT"
    assert result.selected_row_ids is not None and result.selected_row_ids == ()
    assert result.lower_bound == result.upper_bound == 0
    _assert_exact_partition(
        _minimum_primary_blocks(result.instance, result.selected_row_ids),
        result.instance.primary_items,
    )


def test_nonminimum_cover_is_not_the_exact_minimum() -> None:
    instance = _instance(
        (
            ("a-p", ("p",)),
            ("b-q", ("q",)),
            ("z-both", ("p", "q")),
        )
    )
    genuine = minimum_generalized_exact_cover(instance)
    assert genuine.status == "EXACT"
    assert genuine.selected_row_ids == ("z-both",)
    # A valid but non-optimal cover family passes shape validation, yet the
    # minimum operation still distinguishes it by cardinality.
    weakened = MinimumGeneralizedExactCoverResult.model_validate(
        {
            "instance": instance.model_dump(mode="json"),
            "status": "EXACT",
            "selected_row_ids": ["a-p", "b-q"],
            "item_multiplicities": [
                {"item_id": "p", "kind": "PRIMARY", "multiplicity": 1},
                {"item_id": "q", "kind": "PRIMARY", "multiplicity": 1},
            ],
            "lower_bound": 2,
            "upper_bound": 2,
            "searched_node_count": 1,
        }
    )
    assert weakened.selected_row_ids != genuine.selected_row_ids
    assert weakened.upper_bound is not None and genuine.upper_bound is not None
    assert weakened.upper_bound > genuine.upper_bound
    assert minimum_generalized_exact_cover(instance) == genuine
