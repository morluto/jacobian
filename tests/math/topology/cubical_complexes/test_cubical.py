"""Tests for cubical complex operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.topology.cubical_complexes._models import (
    CubicalCell,
    CubicalComplexRequest,
    FaceClosureRequest,
)
from jacobian.math.topology.cubical_complexes._operations import (
    compute_f_vector,
    compute_face_closure,
)


class TestFaceClosure:
    def test_single_2d_cell(self) -> None:
        result = compute_face_closure(
            FaceClosureRequest(cells=(CubicalCell(intervals=((0, 1), (0, 1))),))
        )
        assert result.original_cells == 1
        # 4 vertices + 4 edges + 1 square = 9 cells
        assert result.total_cells == 9
        assert result.cells_by_dimension == (4, 4, 1)

    def test_single_1d_cell(self) -> None:
        result = compute_face_closure(
            FaceClosureRequest(cells=(CubicalCell(intervals=((0, 1),)),))
        )
        assert result.original_cells == 1
        # 2 vertices + 1 edge = 3 cells
        assert result.total_cells == 3
        assert result.cells_by_dimension == (2, 1)


class TestFVector:
    def test_single_square(self) -> None:
        result = compute_f_vector(
            CubicalComplexRequest(cells=(CubicalCell(intervals=((0, 1), (0, 1))),))
        )
        assert result.dimension == 2
        # 4 vertices + 4 edges + 1 square
        assert result.f_vector == (4, 4, 1)
        assert result.euler_characteristic == 1


class TestValidation:
    def test_invalid_interval_non_unit(self) -> None:
        with pytest.raises(ValidationError) as error:
            CubicalCell(intervals=((0, 2),))
        assert error.value.errors()[0]["type"] == "cubical_complex.interval_length"

    def test_invalid_interval_order(self) -> None:
        with pytest.raises(ValidationError) as error:
            CubicalCell(intervals=((1, 0),))
        assert error.value.errors()[0]["type"] == "cubical_complex.interval_order"
