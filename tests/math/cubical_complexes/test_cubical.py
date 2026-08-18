"""Tests for cubical complex operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.cubical_complexes._models import (
    CubicalCell,
    CubicalComplexRequest,
    FaceClosureRequest,
)
from jacobian.math.cubical_complexes._operations import (
    compute_f_vector,
    compute_face_closure,
)


class TestFaceClosure:
    def test_single_2d_cell(self) -> None:
        result = compute_face_closure(
            FaceClosureRequest(cells=[CubicalCell(intervals=((0, 1), (0, 1)))])
        )
        assert result.original_cells == 1
        assert result.total_cells == 4
        assert result.cells_by_dimension == (1, 2, 1)

    def test_single_1d_cell(self) -> None:
        result = compute_face_closure(
            FaceClosureRequest(cells=[CubicalCell(intervals=((0, 1),))])
        )
        assert result.original_cells == 1
        assert result.total_cells == 2
        assert result.cells_by_dimension == (1, 1)


class TestFVector:
    def test_single_square(self) -> None:
        result = compute_f_vector(
            CubicalComplexRequest(cells=[CubicalCell(intervals=((0, 1), (0, 1)))])
        )
        assert result.dimension == 2
        assert result.f_vector == (0, 0, 1)
        assert result.euler_characteristic == 1


class TestValidation:
    def test_invalid_interval_non_unit(self) -> None:
        with pytest.raises(ValidationError, match="unit length"):
            CubicalCell(intervals=((0, 2),))

    def test_invalid_interval_order(self) -> None:
        with pytest.raises(ValidationError, match="a < b"):
            CubicalCell(intervals=((1, 0),))
