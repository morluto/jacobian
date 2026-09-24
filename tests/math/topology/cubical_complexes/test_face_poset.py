from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.cubical_complexes._models import (
    MAX_POSET_ELEMENTS,
    CubicalCell,
    CubicalComplexRequest,
)
from jacobian.math.topology.cubical_complexes.operations import face_poset

_OPERATION_ID = "topology.cubical_complex.face_poset.compute"


def _oracle_face_closure(cells: tuple[CubicalCell, ...]) -> tuple[CubicalCell, ...]:
    faces: set[tuple[tuple[int, int], ...]] = set()
    for cell in cells:
        coordinate_choices = tuple(
            ((start, end),)
            if start == end
            else ((start, start), (start, end), (end, end))
            for start, end in cell.intervals
        )
        faces.update(product(*coordinate_choices))
    return tuple(CubicalCell(intervals=face) for face in sorted(faces))


def _oracle_order(
    cells: tuple[CubicalCell, ...],
) -> set[tuple[CubicalCell, CubicalCell]]:
    return {
        (lower, upper)
        for lower in cells
        for upper in cells
        if lower != upper
        and all(
            outer_start <= inner_start and inner_end <= outer_end
            for (inner_start, inner_end), (outer_start, outer_end) in zip(
                lower.intervals, upper.intervals, strict=True
            )
        )
    }


def _assert_matches_independent_oracle(request_cells: tuple[CubicalCell, ...]) -> None:
    result = face_poset(CubicalComplexRequest(cells=request_cells))
    expected_cells = _oracle_face_closure(request_cells)
    expected_order = _oracle_order(expected_cells)
    label_to_cell = {entry.element: entry.cell for entry in result.cell_elements}
    actual_order = {
        (label_to_cell[pair.lower], label_to_cell[pair.upper])
        for pair in result.poset.strict_order_pairs
    }
    actual_covers = {
        (label_to_cell[pair.lower], label_to_cell[pair.upper])
        for pair in result.poset.cover_relations
    }
    expected_covers = {
        pair
        for pair in expected_order
        if not any(
            (pair[0], middle) in expected_order and (middle, pair[1]) in expected_order
            for middle in expected_cells
            if middle != pair[0] and middle != pair[1]
        )
    }

    assert result.complex.cells == expected_cells
    assert tuple(entry.cell for entry in result.cell_elements) == expected_cells
    assert tuple(entry.dimension for entry in result.cell_elements) == tuple(
        cell.dimension for cell in expected_cells
    )
    assert actual_order == expected_order
    assert actual_covers == expected_covers


def test_square_face_poset_matches_product_face_and_inclusion_oracle() -> None:
    square = CubicalCell(intervals=((0, 1), (0, 1)))
    _assert_matches_independent_oracle((square,))


def test_point_cell_has_empty_order_relations_and_void_input_is_unavailable() -> None:
    point = CubicalCell(intervals=((4, 4), (9, 9)))
    result = face_poset(CubicalComplexRequest(cells=(point,)))
    assert result.complex.cells == (point,)
    assert result.poset.elements == ("c00",)
    assert result.poset.strict_order_pairs == ()
    assert result.poset.cover_relations == ()
    assert result.poset.incomparable_pairs == ()
    assert result.poset.ranks is not None
    assert result.poset.ranks[0].rank == 0
    _assert_matches_independent_oracle((point,))

    with pytest.raises(ValidationError):
        CubicalComplexRequest(cells=())


def test_nonpure_face_poset_retains_cell_dimensions_without_poset_ranks() -> None:
    edge = CubicalCell(intervals=((0, 1),))
    isolated_point = CubicalCell(intervals=((4, 4),))
    result = face_poset(CubicalComplexRequest(cells=(isolated_point, edge)))
    assert result.poset.graded is False
    assert result.poset.ranks is None
    assert tuple(entry.dimension for entry in result.cell_elements) == (0, 1, 0, 0)
    _assert_matches_independent_oracle((isolated_point, edge))


def test_face_poset_catalog_example_executes_and_round_trips() -> None:
    operation = next(
        tool for tool in BUILTIN_TOOLS if tool.operation_id == _OPERATION_ID
    )
    request = operation.request_type.model_validate(operation.examples[0].input)
    result = operation.run(request)
    assert len(result.cell_elements) == 9
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_face_poset_preflights_existing_cell_count_and_single_cell_closure() -> None:
    at_element_bound = tuple(
        CubicalCell(intervals=((index, index),)) for index in range(MAX_POSET_ELEMENTS)
    )
    admitted = face_poset(CubicalComplexRequest(cells=at_element_bound))
    assert len(admitted.poset.elements) == MAX_POSET_ELEMENTS
    assert admitted.poset.strict_order_pairs == ()
    assert len(admitted.poset.incomparable_pairs) == (
        MAX_POSET_ELEMENTS * (MAX_POSET_ELEMENTS - 1) // 2
    )

    too_many_points = tuple(
        CubicalCell(intervals=((index, index),))
        for index in range(MAX_POSET_ELEMENTS + 1)
    )
    with pytest.raises(OperationResourceAdmissionError):
        face_poset(CubicalComplexRequest(cells=too_many_points))

    four_cube = CubicalCell(intervals=((0, 1),) * 4)
    with pytest.raises(OperationResourceAdmissionError):
        face_poset(CubicalComplexRequest(cells=(four_cube,)))
