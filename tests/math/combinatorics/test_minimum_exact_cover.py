"""Minimum generalized exact-cover tests."""

import time
from typing import Any, NoReturn

import pytest

import jacobian.math.combinatorics.exact_cover as exact_cover_module
from jacobian._execution import OperationExecutionTimeoutError, request_execution
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
    rows = tuple(
        ExactCoverRow(row_id=f"p0-{index:04d}", items=("p0",)) for index in range(8)
    ) + (
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
    rows = (ExactCoverRow(row_id="pair", items=("p1", "p2")),) + tuple(
        ExactCoverRow(row_id=f"solo-{index:04d}", items=("p0",))
        for index in range(2_047)
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
    rows = tuple(
        ExactCoverRow(row_id=f"p0-{index:04d}", items=("p0",)) for index in range(2_047)
    ) + (
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
