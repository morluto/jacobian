from collections import Counter

import pytest

from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.topology.cubical_complexes import (
    CubicalCellBoundaryRequest,
    cell_boundary,
)
from jacobian.math.topology.cubical_complexes._models import CubicalCell
from jacobian.math.topology.cubical_complexes._tools import TOOLS


def _direct_oriented_boundary(intervals):
    """Test oracle from the signed endpoint formula, independent of the kernel."""
    faces = []
    active_axes = [axis for axis, (lo, hi) in enumerate(intervals) if lo < hi]
    for position, axis in enumerate(active_axes):
        sign = 1 if position % 2 == 0 else -1
        lo, hi = intervals[axis]
        for endpoint, coefficient in ((hi, sign), (lo, -sign)):
            face = list(intervals)
            face[axis] = (endpoint, endpoint)
            faces.append((tuple(face), coefficient, axis, endpoint))
    return faces


def test_cell_boundary_keeps_ambient_axis_signs_and_squares_to_zero():
    cell = CubicalCell(intervals=((2, 2), (0, 1), (-1, 0), (8, 9)))
    result = cell_boundary(cell)

    # Independent hand-computed endpoint formula: active axes 1, 2, 3 get
    # alternating orientation signs even with degenerate axes at 0 and 3.
    expected = [
        (((2, 2), (1, 1), (-1, 0), (8, 9)), 1, 1, 1),
        (((2, 2), (0, 0), (-1, 0), (8, 9)), -1, 1, 0),
        (((2, 2), (0, 1), (0, 0), (8, 9)), -1, 2, 0),
        (((2, 2), (0, 1), (-1, -1), (8, 9)), 1, 2, -1),
        (((2, 2), (0, 1), (-1, 0), (9, 9)), 1, 3, 9),
        (((2, 2), (0, 1), (-1, 0), (8, 8)), -1, 3, 8),
    ]
    assert [
        (term.face.intervals, term.coefficient, term.ambient_axis, term.endpoint)
        for term in result.terms
    ] == expected
    assert [
        (term.face.intervals, term.coefficient, term.ambient_axis, term.endpoint)
        for term in result.terms
    ] == _direct_oriented_boundary(cell.intervals)

    second_boundary = Counter()
    for face, coefficient, _, _ in _direct_oriented_boundary(cell.intervals):
        for subface, subcoefficient, _, _ in _direct_oriented_boundary(face):
            second_boundary[subface] += coefficient * subcoefficient
    assert all(coefficient == 0 for coefficient in second_boundary.values())


def test_point_cell_has_empty_boundary():
    point = CubicalCell(intervals=((3, 3), (-4, -4)))
    assert cell_boundary(point).terms == ()


def test_cell_boundary_is_a_published_typed_operation():
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "topology.cubical.cell.boundary.compute"
    )
    request = CubicalCellBoundaryRequest.model_validate(
        {"cell": {"intervals": [[0, 1], [0, 1]]}}
    )
    result = tool.run(request)
    assert len(result.terms) == 4
    wire_value = result.model_dump(mode="json")
    restored = type(result).model_validate_json(encode_strict_json(wire_value))
    assert restored == result

    public_result = invoke_operation(
        tool.operation_id,
        request.model_dump(mode="json"),
        Catalog.open(),
    )
    assert public_result.operation_id == tool.operation_id
    assert type(result).model_validate(public_result.output) == result


def test_cell_boundary_rejects_overlong_coordinates_before_face_expansion():
    cell = CubicalCell(intervals=((10**64, 10**64 + 1),))
    with pytest.raises(OperationResourceAdmissionError) as error:
        cell_boundary(cell)
    assert (
        error.value.errors()[0]["type"]
        == "cubical_complex.cell_boundary_coordinate_budget"
    )
