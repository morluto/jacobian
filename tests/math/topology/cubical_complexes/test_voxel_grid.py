"""Independent cell-set oracles for three-dimensional voxel grids."""

from __future__ import annotations

import json
from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.topology.cubical_complexes._models import (
    CubicalComplex,
)
from jacobian.math.topology.cubical_complexes.voxel_grid import (
    BinaryVoxels3DRequest,
    binary_voxels_to_complex,
    operations,
)


def _oracle(voxels: tuple[tuple[tuple[bool, ...], ...], ...]):
    """Enumerate every closed face of every occupied unit cube directly."""
    faces = set()
    for z, plane in enumerate(voxels):
        for y, row in enumerate(plane):
            for x, occupied in enumerate(row):
                if not occupied:
                    continue
                for intervals in product(
                    ((x, x), (x, x + 1), (x + 1, x + 1)),
                    ((y, y), (y, y + 1), (y + 1, y + 1)),
                    ((z, z), (z, z + 1), (z + 1, z + 1)),
                ):
                    faces.add(intervals)
    return tuple(sorted(faces))


def test_all_two_by_two_by_two_grids_match_independent_face_enumeration() -> None:
    for bits in product((False, True), repeat=8):
        voxels = tuple(
            tuple(tuple(bits[(z * 2 + y) * 2 + x] for x in range(2)) for y in range(2))
            for z in range(2)
        )
        result = binary_voxels_to_complex(BinaryVoxels3DRequest(voxels=voxels))
        assert tuple(cell.intervals for cell in result.cells) == _oracle(voxels)
        assert result.ambient_dimension == 3


def test_empty_grid_retains_the_three_dimensional_axis_after_json_round_trip() -> None:
    request = BinaryVoxels3DRequest(voxels=(((False, False),),))
    result = binary_voxels_to_complex(request)
    decoded = CubicalComplex.model_validate_json(result.model_dump_json())
    assert decoded == CubicalComplex(ambient_dimension=3, cells=())


def test_two_adjacent_voxels_share_faces_and_keep_xyz_axis_order() -> None:
    voxels = (((True, True),),)
    result = binary_voxels_to_complex(BinaryVoxels3DRequest(voxels=voxels))
    assert tuple(cell.intervals for cell in result.cells) == _oracle(voxels)
    assert len(result.cells) == 45
    cell_intervals = {cell.intervals for cell in result.cells}
    assert ((0, 2), (0, 1), (0, 1)) not in cell_intervals
    assert ((0, 1), (0, 1), (0, 1)) in cell_intervals
    assert ((1, 2), (0, 1), (0, 1)) in cell_intervals


def test_full_grid_near_canonical_cell_limit_is_accepted() -> None:
    voxels = tuple(
        tuple(tuple(True for _ in range(18)) for _ in range(18)) for _ in range(18)
    )
    result = binary_voxels_to_complex(BinaryVoxels3DRequest(voxels=voxels))
    assert len(result.cells) == 37**3 == 50_653
    assert ((17, 18), (17, 18), (17, 18)) in {cell.intervals for cell in result.cells}


def test_full_grid_above_canonical_cell_limit_is_refused_before_cell_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voxels = tuple(
        tuple(tuple(True for _ in range(19)) for _ in range(19)) for _ in range(19)
    )

    def unexpected_construct(**_kwargs: object) -> CubicalComplex:
        raise AssertionError("oversized output reached canonical value construction")

    monkeypatch.setattr(operations, "CubicalComplex", unexpected_construct)
    with pytest.raises(OperationResourceAdmissionError) as error:
        binary_voxels_to_complex(BinaryVoxels3DRequest(voxels=voxels))
    assert error.value.errors()[0]["type"] == (
        "cubical_complex.voxel_grid_face_bound_exceeded"
    )


def test_face_candidate_bound_rejects_before_face_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_expand(*_args: object, **_kwargs: object) -> set[object]:
        raise AssertionError("candidate-overflow request reached face expansion")

    monkeypatch.setattr(operations, "MAX_CUBICAL_VOXEL_FACE_CANDIDATES", 26)
    monkeypatch.setattr(operations, "_face_keys", unexpected_expand)
    request = BinaryVoxels3DRequest(voxels=(((True,),),))
    with pytest.raises(OperationResourceAdmissionError) as error:
        binary_voxels_to_complex(request)
    assert error.value.errors()[0]["type"] == (
        "cubical_complex.voxel_grid_candidate_bound_exceeded"
    )


def test_ragged_and_non_boolean_grids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="voxel_grid_not_rectangular"):
        BinaryVoxels3DRequest(voxels=(((True,),), ((True, False),)))
    with pytest.raises(ValidationError):
        BinaryVoxels3DRequest.model_validate_json('{"voxels": [[[1]]]}')


def test_model_construct_forged_grid_is_checked_without_pydantic_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = BinaryVoxels3DRequest.model_construct(voxels=(((1,),),))

    def unexpected_replay(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("kernel replayed request through Pydantic")

    monkeypatch.setattr(BinaryVoxels3DRequest, "model_dump", unexpected_replay)
    with pytest.raises(OperationDomainValidationError) as error:
        binary_voxels_to_complex(request)
    assert error.value.errors()[0]["type"] == (
        "cubical_complex.voxel_grid_carrier_shape"
    )


def test_forged_ragged_grid_is_rejected_without_model_validation() -> None:
    request = BinaryVoxels3DRequest.model_construct(
        voxels=(((True,),), ((True, False),))
    )
    with pytest.raises(OperationDomainValidationError) as error:
        binary_voxels_to_complex(request)
    assert error.value.errors()[0]["type"] == (
        "cubical_complex.voxel_grid_not_rectangular"
    )


def test_tool_example_executes_through_catalog() -> None:
    operation_id = "topology.cubical_complex.from_binary_voxels_3d.compute"
    catalog = Catalog.open()
    tool = catalog.operation(operation_id)
    assert tool is not None
    result = invoke_operation(operation_id, tool.examples[0].input, catalog)
    complex_ = tool.result_type.model_validate_json(json.dumps(result.output))
    assert isinstance(complex_, CubicalComplex)
    assert complex_.ambient_dimension == 3
    assert len(complex_.cells) == 27
