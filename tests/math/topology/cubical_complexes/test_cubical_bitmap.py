"""Exact binary-bitmap to cubical-complex conversion."""

from itertools import product as cartesian_product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationResourceAdmissionError,
)
from jacobian.math.topology.cubical_complexes._models import (
    CubicalBitmapRequest,
    CubicalBitmapResult,
)
from jacobian.math.topology.cubical_complexes._tools import TOOLS
from jacobian.math.topology.cubical_complexes.extensions import bitmap_to_complex
from jacobian.math.topology.cubical_complexes.operations import f_vector


def _brute_face_closure(pixels: tuple[tuple[bool, ...], ...]):
    cells = set()
    for row, values in enumerate(pixels):
        for column, foreground in enumerate(values):
            if not foreground:
                continue
            choices = (
                ((column, column + 1), (column, column), (column + 1, column + 1)),
                ((row, row + 1), (row, row), (row + 1, row + 1)),
            )
            for intervals in cartesian_product(*choices):
                cells.add(intervals)
    return tuple(sorted(cells))


def _mask(rows: int, columns: int, bits: int) -> tuple[tuple[bool, ...], ...]:
    return tuple(
        tuple(bool(bits & (1 << (row * columns + column))) for column in range(columns))
        for row in range(rows)
    )


@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_every_tiny_bitmap_matches_independent_face_enumeration(shape) -> None:
    rows, columns = shape
    for bits in range(1 << (rows * columns)):
        pixels = _mask(rows, columns, bits)
        request = CubicalBitmapRequest(pixels=pixels)
        result = bitmap_to_complex(request)
        assert tuple(
            cell.intervals for cell in result.complex.cells
        ) == _brute_face_closure(pixels)
        assert (result.row_count, result.column_count) == shape
        assert tuple(
            (entry.row, entry.column) for entry in result.pixel_to_cell
        ) == tuple(
            (row, column)
            for row, values in enumerate(pixels)
            for column, foreground in enumerate(values)
            if foreground
        )
        if bits == 0:
            assert result.complex.cells == ()
            assert result.complex.ambient_dimension == 2
            assert CubicalBitmapResult.model_validate_json(
                result.model_dump_json()
            ) == result


def test_pixel_axes_closure_and_shared_faces_are_exact() -> None:
    result = bitmap_to_complex(CubicalBitmapRequest(pixels=((True, True),)))
    assert result.complex.ambient_dimension == 2
    assert result.complex.cells == tuple(
        sorted(result.complex.cells, key=lambda cell: cell.intervals)
    )
    assert f_vector(result.complex.cells).f_vector.counts == (6, 7, 2)
    assert result.row_count == 1
    assert result.column_count == 2


def test_bitmap_tool_is_published_with_a_valid_example() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "topology.cubical_complex.from_binary_bitmap_2d.compute"
    )
    request = CubicalBitmapRequest.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert result.complex.cells
    assert result.row_count == result.column_count == 1


def test_bitmap_requires_rectangular_strict_boolean_rows() -> None:
    with pytest.raises(ValidationError) as error:
        CubicalBitmapRequest(pixels=((True,), (False, True)))
    assert error.value.errors()[0]["type"] == "cubical_complex.bitmap_not_rectangular"

    with pytest.raises(ValidationError):
        CubicalBitmapRequest.model_validate({"pixels": [[1]]})


def test_bitmap_admits_side_lengths_and_foreground_cell_count() -> None:
    with pytest.raises(ValidationError):
        CubicalBitmapRequest(pixels=(tuple(True for _ in range(257)),))

    too_many_foreground = CubicalBitmapRequest(
        pixels=tuple(tuple(True for _ in range(71)) for _ in range(71))
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        bitmap_to_complex(too_many_foreground)
    assert error.value.errors()[0]["type"] == "cubical_complex.bitmap_top_cell_budget"


def test_bitmap_output_preflight_keeps_the_result_f_vector_composable():
    below_limit = CubicalBitmapRequest(
        pixels=tuple(tuple(True for _ in range(34)) for _ in range(34))
    )
    result = bitmap_to_complex(below_limit)
    assert f_vector(result.complex.cells).f_vector.counts == (1225, 2380, 1156)

    above_limit = CubicalBitmapRequest(
        pixels=tuple(tuple(True for _ in range(35)) for _ in range(35))
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        bitmap_to_complex(above_limit)
    assert error.value.errors()[0]["type"] == "cubical_complex.bitmap_face_budget"
