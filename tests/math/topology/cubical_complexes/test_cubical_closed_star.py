"""Exact source-bound cubical closed stars and preflight admission."""

import json

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.cubical_complexes import operations
from jacobian.math.topology.cubical_complexes._models import (
    CubicalCell,
    CubicalClosedStarRequest,
    CubicalClosedStarResult,
)
from jacobian.math.topology.cubical_complexes._tools import TOOLS


def _cell(*intervals):
    return CubicalCell(intervals=intervals)


def test_endpoint_closed_star_in_a_path_is_one_edge_closure():
    request = CubicalClosedStarRequest(
        cells=(_cell((0, 1)), _cell((1, 2))), cell=_cell((0, 0))
    )
    result = operations.closed_star(request)
    assert result.complex.cells == (
        _cell((0, 0)),
        _cell((0, 1)),
        _cell((1, 1)),
        _cell((1, 2)),
        _cell((2, 2)),
    )
    assert result.closed_star.cells == (
        _cell((0, 0)),
        _cell((0, 1)),
        _cell((1, 1)),
    )
    assert (
        CubicalClosedStarResult.model_validate_json(result.model_dump_json()) == result
    )


def test_star_in_square_and_shared_face_of_adjacent_squares():
    square = _cell((0, 1), (0, 1))
    one_square = operations.closed_star(
        CubicalClosedStarRequest(cells=(square,), cell=_cell((0, 0), (0, 0)))
    )
    assert one_square.closed_star.cells == one_square.complex.cells

    left = _cell((0, 1), (0, 1))
    right = _cell((1, 2), (0, 1))
    shared_edge = _cell((1, 1), (0, 1))
    adjacent = operations.closed_star(
        CubicalClosedStarRequest(cells=(left, right), cell=shared_edge)
    )
    assert adjacent.closed_star.cells == adjacent.complex.cells


def test_disconnected_component_does_not_enter_the_selected_cell_star():
    request = CubicalClosedStarRequest(
        cells=(_cell((0, 1)), _cell((10, 10))), cell=_cell((0, 0))
    )
    result = operations.closed_star(request)
    assert result.closed_star.cells == (
        _cell((0, 0)),
        _cell((0, 1)),
        _cell((1, 1)),
    )


def test_absent_cell_and_wrong_axis_are_rejected():
    with pytest.raises(OperationDomainValidationError) as absent:
        operations.closed_star(
            CubicalClosedStarRequest(cells=(_cell((0, 1)),), cell=_cell((3, 3)))
        )
    assert absent.value.errors()[0]["type"] == "cubical_complex.closed_star_cell_absent"

    with pytest.raises(OperationDomainValidationError) as wrong_axis:
        operations.closed_star(
            CubicalClosedStarRequest(cells=(_cell((0, 1)),), cell=_cell((0, 0), (0, 0)))
        )
    assert (
        wrong_axis.value.errors()[0]["type"]
        == "cubical_complex.closed_star_axis_mismatch"
    )


def test_digit_and_output_bounds_reject_before_face_expansion(monkeypatch):
    def fail_if_expanded(*_args, **_kwargs):
        raise AssertionError("face expansion ran before admission")

    monkeypatch.setattr(operations, "_canonical_complex", fail_if_expanded)
    huge_edge = _cell((10**70, 10**70 + 1))
    with pytest.raises(OperationResourceAdmissionError) as coordinate_bound:
        operations.closed_star(
            CubicalClosedStarRequest(cells=(huge_edge,), cell=_cell((10**70, 10**70)))
        )
    assert coordinate_bound.value.errors()[0]["type"] == (
        "cubical_complex.closed_star_coordinate_bound"
    )

    top_cube = _cell(*((0, 1) for _ in range(10)))
    corner = _cell(*((0, 0) for _ in range(10)))
    with pytest.raises(OperationResourceAdmissionError) as output_bound:
        operations.closed_star(CubicalClosedStarRequest(cells=(top_cube,), cell=corner))
    assert (
        output_bound.value.errors()[0]["type"] == "cubical_complex.closed_star_bounds"
    )


def test_maximum_one_dimensional_generator_family_stays_bounded():
    cells = tuple(_cell((index, index + 1)) for index in range(5000))
    result = operations.closed_star(
        CubicalClosedStarRequest(cells=cells, cell=_cell((2500, 2500)))
    )
    assert len(result.complex.cells) == 10001
    assert result.closed_star.cells == (
        _cell((2499, 2499)),
        _cell((2499, 2500)),
        _cell((2500, 2500)),
        _cell((2500, 2501)),
        _cell((2501, 2501)),
    )


def test_admitted_nine_cube_star_retains_the_complete_source_closure():
    top_cube = _cell(*((0, 1) for _ in range(9)))
    corner = _cell(*((0, 0) for _ in range(9)))
    result = operations.closed_star(
        CubicalClosedStarRequest(cells=(top_cube,), cell=corner)
    )
    assert len(result.complex.cells) == 3**9
    assert result.closed_star.cells == result.complex.cells


def test_published_closed_star_tool_runs_its_example():
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "topology.cubical_complex.closed_star.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert isinstance(result, CubicalClosedStarResult)
    assert result.closed_star.cells == (
        _cell((0, 0)),
        _cell((0, 1)),
        _cell((1, 1)),
    )
