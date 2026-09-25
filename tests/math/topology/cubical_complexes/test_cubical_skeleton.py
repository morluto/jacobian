"""Exact cubical skeleton operations and independent combinatorial oracles."""

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.cubical_complexes._models import (
    CubicalCell,
    CubicalSkeletonRequest,
    CubicalSkeletonResult,
)
from jacobian.math.topology.cubical_complexes._tools import TOOLS
from jacobian.math.topology.cubical_complexes.operations import skeleton


def _intervals(result: CubicalSkeletonResult) -> set[tuple[tuple[int, int], ...]]:
    return {cell.intervals for cell in result.skeleton.cells}


def test_square_one_skeleton_matches_explicit_boundary_cell_oracle() -> None:
    source = (CubicalCell(intervals=((0, 1), (0, 1))),)
    result = skeleton(source, 1)

    expected = {
        ((0, 0), (0, 0)),
        ((0, 0), (1, 1)),
        ((1, 1), (0, 0)),
        ((1, 1), (1, 1)),
        ((0, 1), (0, 0)),
        ((0, 1), (1, 1)),
        ((0, 0), (0, 1)),
        ((1, 1), (0, 1)),
    }
    assert _intervals(result) == expected
    assert len(result.complex.cells) == 9
    assert result.dimension_bound == 1
    restored = CubicalSkeletonResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_skeleton_is_idempotent_under_minimum_dimension_bound() -> None:
    source = (
        CubicalCell(intervals=((0, 1), (0, 1))),
        CubicalCell(intervals=((1, 2), (0, 1))),
    )
    for i, j in ((0, 1), (1, 2), (2, 0), (2, 2)):
        outer = skeleton(source, j)
        nested = skeleton(outer.skeleton.cells, i)
        direct = skeleton(source, min(i, j))
        assert nested.skeleton == direct.skeleton


def test_bound_above_top_dimension_returns_source_complex() -> None:
    source = (CubicalCell(intervals=((3, 3), (4, 4))),)
    result = skeleton(source, 10)
    assert result.skeleton == result.complex
    assert result.skeleton.cells == source


def test_request_bounds_dimension_before_operation() -> None:
    with pytest.raises(ValidationError):
        CubicalSkeletonRequest(
            cells=(CubicalCell(intervals=((0, 1),)),), dimension_bound=11
        )
    assert any(
        tool.operation_id == "topology.cubical_complex.skeleton.compute"
        for tool in TOOLS
    )


def test_native_skeleton_rejects_invalid_dimension_before_closure() -> None:
    source = (CubicalCell(intervals=((0, 1),)),)
    for invalid in (-1, 1.5, True):
        with pytest.raises(OperationDomainValidationError):
            skeleton(source, invalid)  # type: ignore[arg-type]


def test_skeleton_preflights_duplicate_complex_output_with_large_coordinates() -> None:
    source = (CubicalCell(intervals=((10**100, 10**100 + 1),) * 10),)
    with pytest.raises(OperationResourceAdmissionError):
        skeleton(source, 10)
