import pytest

import jacobian.math.topology.cubical_complexes._cell_vertices as cell_vertices_module
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.cubical_complexes import (
    CubicalCellVerticesResult,
    cell_vertices,
)
from jacobian.math.topology.cubical_complexes._models import CubicalCell


def _direct_cartesian_vertices(intervals):
    active_axes = tuple(
        axis for axis, (lower, upper) in enumerate(intervals) if lower != upper
    )
    points = []
    for corner in range(1 << len(active_axes)):
        active_bit = 0
        coordinates = []
        for lower, upper in intervals:
            if lower == upper:
                coordinates.append(lower)
            else:
                coordinates.append((lower, upper)[(corner >> active_bit) & 1])
                active_bit += 1
        points.append(tuple((coordinate, coordinate) for coordinate in coordinates))
    return tuple(sorted(points))


def test_vertices_match_independent_cartesian_product_and_keep_axis_order():
    cell = CubicalCell(intervals=((2, 3), (-4, -4), (8, 9)))
    result = cell_vertices(cell)

    assert tuple(vertex.intervals for vertex in result.vertices) == (
        _direct_cartesian_vertices(cell.intervals)
    )
    assert all(
        len(vertex.intervals) == len(cell.intervals) for vertex in result.vertices
    )
    assert len(result.vertices) == 4


def test_point_cell_has_itself_as_its_single_vertex():
    point = CubicalCell(intervals=((7, 7), (-3, -3), (0, 0)))
    result = cell_vertices(point)
    assert result.cell == point
    assert tuple(vertex.intervals for vertex in result.vertices) == (point.intervals,)


def test_native_result_round_trips_through_strict_json():
    cell = CubicalCell(intervals=((0, 1), (2, 2)))
    result = cell_vertices(cell)

    restored = CubicalCellVerticesResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )

    assert restored == result


def test_vertex_output_byte_bound_is_checked_before_enumeration(monkeypatch):
    cell = CubicalCell(intervals=((10**63, 10**63 + 1),) * 10)

    def forbidden_product(*_args, **_kwargs):
        pytest.fail("vertices must not be enumerated before byte admission")

    monkeypatch.setattr(cell_vertices_module, "product", forbidden_product)
    with pytest.raises(OperationResourceAdmissionError) as error:
        cell_vertices(cell)
    assert error.value.errors()[0]["type"] == (
        "cubical_complex.cell_vertices_result_bytes"
    )


def test_vertex_coordinate_digit_bound_is_checked():
    cell = CubicalCell(intervals=((10**64, 10**64 + 1),))
    with pytest.raises(OperationResourceAdmissionError) as error:
        cell_vertices(cell)
    assert error.value.errors()[0]["type"] == (
        "cubical_complex.cell_vertices_coordinate_digits"
    )


def test_negative_boundary_coordinate_is_admitted():
    cell = CubicalCell(intervals=((-(10**63), -(10**63) + 1),))
    result = cell_vertices(cell)
    assert tuple(vertex.intervals for vertex in result.vertices) == (
        ((-(10**63), -(10**63)),),
        ((-(10**63) + 1, -(10**63) + 1),),
    )


def test_huge_native_endpoint_is_rejected_without_string_conversion(monkeypatch):
    def forbidden_digit_width(*_args, **_kwargs):
        pytest.fail("oversized native endpoints must be rejected before formatting")

    monkeypatch.setattr(
        cell_vertices_module, "decimal_digit_width", forbidden_digit_width
    )
    huge = 1 << 100_000
    forged = CubicalCell.model_construct(intervals=((huge, huge + 1),))

    with pytest.raises(OperationResourceAdmissionError) as error:
        cell_vertices(forged)

    assert error.value.errors()[0]["type"] == (
        "cubical_complex.cell_vertices_coordinate_digits"
    )
