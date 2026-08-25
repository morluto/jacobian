"""Contract tests for exact rational box-union volume."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.math.geometry.boxes import (
    BoxIntersectionLedgerEntry,
    BoxUnionVolumeRequest,
    BoxUnionVolumeResult,
    RationalAxisAlignedBox,
    RationalClosedInterval,
    compute_box_union_volume,
)
from jacobian.math.geometry.boxes._models import MAX_BOX_UNION_RESULT_BYTES
from jacobian.math.geometry.boxes.values import MAX_CANONICAL_BOX_DIMENSION


def _rational(value: str | int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _box(*bounds: tuple[str | int, str | int]) -> RationalAxisAlignedBox:
    return RationalAxisAlignedBox(
        dimension=len(bounds),
        intervals=tuple(
            RationalClosedInterval(lower=_rational(lower), upper=_rational(upper))
            for lower, upper in bounds
        ),
    )


def _empty_box(dimension: int) -> RationalAxisAlignedBox:
    return RationalAxisAlignedBox(dimension=dimension, intervals=None)


def _volume_by_integer_cells(boxes: tuple[RationalAxisAlignedBox, ...]) -> int:
    """Independent oracle for boxes with integer endpoints."""

    cells: set[tuple[int, ...]] = set()
    for box in boxes:
        if box.intervals is None:
            continue
        ranges = tuple(
            range(
                int(interval.lower.as_fraction()),
                int(interval.upper.as_fraction()),
            )
            for interval in box.intervals
        )
        cells.update(product(*ranges))
    return len(cells)


def _three_boxes() -> tuple[RationalAxisAlignedBox, ...]:
    return (
        _box((0, 2), (0, 1), (0, 1)),
        _box((1, 3), (0, 1), (0, 1)),
        _box((0, 3), (0, 1), (Fraction(1, 2), Fraction(3, 2))),
    )


def _three_box_source() -> BoxUnionVolumeRequest:
    return BoxUnionVolumeRequest(boxes=_three_boxes())


def test_three_box_inclusion_exclusion_fixture() -> None:
    result = compute_box_union_volume(_three_boxes())

    assert result.union_volume.as_fraction() == Fraction(9, 2)
    assert tuple(entry.box_indices for entry in result.intersections) == (
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    )
    assert tuple(entry.volume.as_fraction() for entry in result.intersections) == (
        Fraction(2),
        Fraction(2),
        Fraction(3),
        Fraction(1),
        Fraction(1),
        Fraction(1),
        Fraction(1, 2),
    )


@pytest.mark.parametrize(
    "boxes",
    [
        (_box((0, 2), (0, 1)), _box((1, 3), (0, 2))),
        (_box((0, 4), (0, 3)), _box((1, 2), (1, 2))),
        (_box((0, 1), (0, 1)), _box((2, 4), (1, 3))),
        (
            _box((0, 3), (0, 2)),
            _box((1, 4), (1, 3)),
            _box((2, 5), (0, 1)),
        ),
    ],
)
def test_union_volume_matches_independent_cell_oracle(
    boxes: tuple[RationalAxisAlignedBox, ...],
) -> None:
    result = compute_box_union_volume(boxes)
    assert result.union_volume.as_fraction() == _volume_by_integer_cells(boxes)


def test_disjoint_intersections_are_omitted() -> None:
    boxes = (_box((0, 1)), _box((2, 4)))
    result = compute_box_union_volume(boxes)

    assert result.source == BoxUnionVolumeRequest(boxes=boxes)
    assert result.union_volume.as_fraction() == 3
    assert tuple(entry.box_indices for entry in result.intersections) == ((0,), (1,))


def test_duplicate_boxes_replay_by_source_index() -> None:
    box = _box((0, 2), (0, 1))
    result = compute_box_union_volume((box, box))

    assert result.union_volume.as_fraction() == 2
    assert tuple(entry.box_indices for entry in result.intersections) == (
        (0,),
        (1,),
        (0, 1),
    )
    assert all(entry.volume.as_fraction() == 2 for entry in result.intersections)


def test_empty_boxes_are_pruned_without_losing_source_indices() -> None:
    boxes = (_empty_box(1), _box((0, 2)), _empty_box(1), _box((1, 3)))
    request = BoxUnionVolumeRequest(boxes=boxes)
    result = compute_box_union_volume(boxes)

    assert result.source == request
    assert result.union_volume.as_fraction() == 3
    assert tuple(entry.box_indices for entry in result.intersections) == (
        (1,),
        (3,),
        (1, 3),
    )


def test_all_empty_boxes_have_empty_ledger_and_zero_volume() -> None:
    result = compute_box_union_volume(tuple(_empty_box(3) for _ in range(65)))

    assert result.intersections == ()
    assert result.union_volume.as_fraction() == 0


def test_one_nonempty_box_with_empties_beyond_fixed_cap_is_admitted() -> None:
    boxes = (_box((0, 2)), *(_empty_box(1) for _ in range(64)))
    result = compute_box_union_volume(boxes)

    assert len(result.source.boxes) == 65
    assert result.union_volume.as_fraction() == 2
    assert tuple(entry.box_indices for entry in result.intersections) == ((0,),)
    assert result.intersections[0].volume.as_fraction() == 2


def test_empty_heavy_result_serialization_stays_within_budget() -> None:
    boxes = tuple(_empty_box(1) for _ in range(65))
    result = compute_box_union_volume(boxes)

    assert (
        len(encode_strict_json(result.model_dump(mode="json")))
        <= MAX_BOX_UNION_RESULT_BYTES
    )


def test_touching_and_degenerate_boxes_remain_nonempty() -> None:
    result = compute_box_union_volume((_box((0, 1)), _box((1, 2)), _box((3, 3))))
    ledger = {entry.box_indices: entry for entry in result.intersections}

    assert result.union_volume.as_fraction() == 2
    assert ledger[(0, 1)].intersection == _box((1, 1))
    assert ledger[(0, 1)].volume.as_fraction() == 0
    assert ledger[(2,)].volume.as_fraction() == 0
    assert (0, 2) not in ledger


def test_input_order_changes_indices_but_not_union_volume() -> None:
    boxes = _three_boxes()

    assert (
        compute_box_union_volume(boxes).union_volume
        == compute_box_union_volume(tuple(reversed(boxes))).union_volume
    )


def test_result_rejects_wrong_union_volume() -> None:
    result = compute_box_union_volume(_three_boxes())
    payload = result.model_dump(mode="json")
    payload["union_volume"] = {"num": "5", "den": "1"}

    with pytest.raises(ValidationError):
        BoxUnionVolumeResult.model_validate(payload)


def test_result_rejects_omitted_intersection() -> None:
    result = compute_box_union_volume(_three_boxes())
    payload = result.model_dump(mode="json")
    payload["intersections"] = payload["intersections"][:-1]

    with pytest.raises(ValidationError):
        BoxUnionVolumeResult.model_validate(payload)


def test_result_rejects_corrupted_intersection() -> None:
    result = compute_box_union_volume(_three_boxes())
    payload = result.model_dump(mode="json")
    payload["intersections"][3]["intersection"] = _box(
        (0, 1), (0, 1), (0, 1)
    ).model_dump(mode="json")
    payload["intersections"][3]["volume"] = {"num": "1", "den": "1"}

    with pytest.raises(ValidationError):
        BoxUnionVolumeResult.model_validate(payload)


def test_result_rejects_intersection_index_mutation() -> None:
    result = compute_box_union_volume(_three_boxes())
    payload = result.model_dump(mode="json")
    payload["intersections"][3]["box_indices"] = [0, 2]

    with pytest.raises(ValidationError):
        BoxUnionVolumeResult.model_validate(payload)


def test_result_rejects_source_mutation() -> None:
    result = compute_box_union_volume(_three_boxes())
    payload = result.model_dump(mode="json")
    payload["source"]["boxes"][0] = _box((0, 1), (0, 1), (0, 1)).model_dump(mode="json")

    with pytest.raises(ValidationError):
        BoxUnionVolumeResult.model_validate(payload)


def test_entry_rejects_volume_not_matching_intersection() -> None:
    with pytest.raises(ValidationError):
        BoxIntersectionLedgerEntry(
            box_indices=(0,),
            intersection=_box((0, 2)),
            volume=_rational(3),
        )


def test_native_api_accepts_canonical_box_tuple_without_request_wrapper() -> None:
    boxes = (_box((0, 2)), _box((1, 3)))

    result = compute_box_union_volume(boxes)

    assert result.source == BoxUnionVolumeRequest(boxes=boxes)
    assert result.union_volume.as_fraction() == 3
    assert tuple(entry.box_indices for entry in result.intersections) == (
        (0,),
        (1,),
        (0, 1),
    )


def test_native_call_admits_the_family_before_the_kernel() -> None:
    with pytest.raises(ValidationError):
        compute_box_union_volume((_box((0, 1)), _box((0, 1), (0, 1))))


def test_math_tool_consumes_parsed_request_without_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.geometry.boxes._operations as operations
    from jacobian.math.geometry.boxes._tools import TOOLS

    tool = next(
        entry
        for entry in TOOLS
        if entry.operation_id == "geometry.box_union.volume.compute"
    )
    request = _three_box_source()

    constructions = 0
    request_type = operations.BoxUnionVolumeRequest

    def counting_request(*args: object, **kwargs: object) -> BoxUnionVolumeRequest:
        nonlocal constructions
        constructions += 1
        return request_type(*args, **kwargs)

    monkeypatch.setattr(operations, "BoxUnionVolumeRequest", counting_request)
    tool_result = tool.run(request)
    assert constructions == 0

    native_result = compute_box_union_volume(request.boxes)
    assert constructions == 1
    monkeypatch.undo()

    assert tool_result == native_result
    assert native_result.source == request


def test_returned_intersections_compose_unchanged_as_box_inputs() -> None:
    result = compute_box_union_volume(_three_boxes())
    pair_intersections = tuple(
        entry.intersection
        for entry in result.intersections
        if len(entry.box_indices) == 2
    )

    replay = compute_box_union_volume(pair_intersections)

    assert replay.source.boxes == pair_intersections
    assert replay.union_volume.as_fraction() == 2


def test_rejects_malformed_interval_and_dimension_mismatch() -> None:
    with pytest.raises(ValidationError):
        RationalClosedInterval(lower=_rational(2), upper=_rational(1))

    with pytest.raises(ValidationError):
        BoxUnionVolumeRequest(boxes=(_box((0, 1)), _box((0, 1), (0, 1))))


def test_single_box_with_257_digit_endpoint_is_admitted_with_exact_volume() -> None:
    result = compute_box_union_volume((_box((0, Fraction(10**256))),))

    assert result.union_volume.as_fraction() == Fraction(10**256)
    assert tuple(entry.box_indices for entry in result.intersections) == ((0,),)
    assert result.intersections[0].volume.as_fraction() == Fraction(10**256)


def test_endpoint_at_derived_growth_boundary_is_admitted() -> None:
    result = compute_box_union_volume((_box((0, Fraction(10**16_376 - 1))),))

    assert result.union_volume.as_fraction() == Fraction(10**16_376 - 1)
    assert tuple(entry.box_indices for entry in result.intersections) == ((0,),)


def test_endpoint_beyond_derived_growth_budget_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BoxUnionVolumeRequest(boxes=(_box((0, Fraction(10**16_377 - 1))),))


def test_rejects_complete_replay_work_before_expansion() -> None:
    box = _box((0, 1), (0, 1))
    with pytest.raises(ValidationError):
        BoxUnionVolumeRequest(boxes=(box,) * 16)


def test_accepts_immediately_below_small_coordinate_result_boundary() -> None:
    box = _box((0, 1))
    request = BoxUnionVolumeRequest(boxes=(box,) * 15)

    assert len(request.boxes) == 15


def test_rejects_next_small_coordinate_result_boundary() -> None:
    box = _box((0, 1))
    with pytest.raises(ValidationError):
        BoxUnionVolumeRequest(boxes=(box,) * 16)


def test_rejects_worst_case_ledger_bytes_before_expansion() -> None:
    endpoint = Fraction(10**255, 10**255 + 1)
    box = _box((0, endpoint))
    with pytest.raises(ValidationError):
        BoxUnionVolumeRequest(boxes=(box,) * 14)


def test_rejects_nonempty_candidate_limit() -> None:
    with pytest.raises(ValidationError):
        BoxUnionVolumeRequest(boxes=(_box((0, 1)),) * 17)


def test_high_dimension_single_box_is_admitted_by_scaled_budgets() -> None:
    result = compute_box_union_volume((_box(*((0, 1),) * 9),))

    assert result.union_volume.as_fraction() == 1
    assert tuple(entry.box_indices for entry in result.intersections) == ((0,),)


def test_boxes_admit_the_full_canonical_dimension_range_and_no_more() -> None:
    result = compute_box_union_volume((_box(*((0, 1),) * MAX_CANONICAL_BOX_DIMENSION),))

    assert result.union_volume.as_fraction() == 1
    assert result.source.boxes[0].dimension == MAX_CANONICAL_BOX_DIMENSION

    with pytest.raises(ValidationError):
        _empty_box(MAX_CANONICAL_BOX_DIMENSION + 1)


def test_high_dimension_families_remain_bounded_by_derived_budgets() -> None:
    endpoint = Fraction(10**255, 10**255 + 1)
    wide_axis = (0, endpoint)
    mixed_box = _box(wide_axis, *((0, 1),) * (MAX_CANONICAL_BOX_DIMENSION - 1))
    result = compute_box_union_volume((mixed_box,))

    assert result.union_volume.as_fraction() == endpoint
    assert tuple(entry.box_indices for entry in result.intersections) == ((0,),)
    assert result.intersections[0].volume.as_fraction() == endpoint

    wide_box = _box(*((0, endpoint),) * MAX_CANONICAL_BOX_DIMENSION)
    with pytest.raises(ValidationError):
        compute_box_union_volume((wide_box,))

    unit_box = _box(*((0, 1),) * MAX_CANONICAL_BOX_DIMENSION)
    with pytest.raises(ValidationError):
        BoxUnionVolumeRequest(boxes=(unit_box,) * 16)


def test_schema_explains_empty_boxes_and_coupled_bounds() -> None:
    box_schema = RationalAxisAlignedBox.model_json_schema()
    request_schema = BoxUnionVolumeRequest.model_json_schema()

    assert "intervals=null" in box_schema["description"]
    assert "same dimension" in request_schema["description"]
    assert (
        f"canonical {MAX_CANONICAL_RATIONAL_DIGITS:,}-digit"
        in request_schema["description"]
    )
    assert "256 digits" not in request_schema["description"]
    boxes_property = request_schema["properties"]["boxes"]
    assert "maxItems" not in boxes_property
    assert "published operation budgets" in boxes_property["description"]
