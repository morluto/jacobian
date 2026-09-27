"""Exact cubical one-skeleton projection to the indexed graph value."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.cubical_complexes import one_skeleton, operations
from jacobian.math.topology.cubical_complexes._models import (
    CubicalCell,
    CubicalOneSkeletonResult,
)


def _square(left: int) -> CubicalCell:
    return CubicalCell(intervals=((left, left + 1), (0, 1)))


def test_square_projects_to_cycle_with_coordinate_axis_map() -> None:
    result = one_skeleton((_square(0),))

    assert result.complex.ambient_dimension == 2
    assert len(result.vertex_cells) == result.graph.vertex_count == 4
    assert len(result.graph.edges) == 4
    assert result.vertex_cells == tuple(
        sorted(result.vertex_cells, key=lambda cell: cell.intervals)
    )
    expected_coordinate_edges = {
        (
            tuple(low),
            tuple(high),
        )
        for low, high in (
            (((0, 0), (0, 0)), ((1, 1), (0, 0))),
            (((0, 0), (1, 1)), ((1, 1), (1, 1))),
            (((0, 0), (0, 0)), ((0, 0), (1, 1))),
            (((1, 1), (0, 0)), ((1, 1), (1, 1))),
        )
    }
    actual_coordinate_edges = {
        tuple(result.vertex_cells[index].intervals for index in edge)
        for edge in result.graph.edges
    }
    assert actual_coordinate_edges == expected_coordinate_edges
    assert (
        CubicalOneSkeletonResult.model_validate_json(result.model_dump_json()) == result
    )


def test_adjacent_squares_share_one_edge_and_canonicalize_generator_order() -> None:
    first = one_skeleton((_square(0), _square(1)))
    second = one_skeleton((_square(1), _square(0), _square(0)))

    assert first == second
    assert first.graph.vertex_count == 6
    assert len(first.graph.edges) == 7


def test_face_work_is_rejected_before_closure_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = tuple(
        CubicalCell(intervals=((offset, offset + 1),) * 10) for offset in range(5_000)
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("face closure ran before work admission")

    monkeypatch.setattr(operations, "_canonical_complex", forbidden)
    with pytest.raises(OperationResourceAdmissionError, match="face-generation work"):
        one_skeleton(cells)
