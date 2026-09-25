"""Exact construction of a cubical complex from a finite voxel grid."""

from __future__ import annotations

from itertools import product
from math import ceil, log2
from typing import NoReturn, cast

from jacobian._execution import OperationWorkLedger, request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.cubical_complexes._models import (
    MAX_FACE_CELLS,
    CubicalCell,
    CubicalComplex,
)
from jacobian.math.topology.cubical_complexes.voxel_grid._models import (
    MAX_CUBICAL_VOXEL_GRID_SIDE,
    MAX_CUBICAL_VOXEL_GRID_VOLUME,
    BinaryVoxels3DRequest,
)

MAX_CUBICAL_VOXEL_FACE_CANDIDATES = 200_000
MAX_CUBICAL_VOXEL_WORK = 2_500_000
CellKey = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


def _reject(code: str, message: str, *location: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code=f"cubical_complex.{code}",
        message=message,
    )


def _validate_raw_shape(
    request: object,
    ledger: OperationWorkLedger,
) -> tuple[BinaryVoxels3DRequest, int, int, int, int]:
    if type(request) is not BinaryVoxels3DRequest:
        _reject("voxel_grid_request_type", "request must be a BinaryVoxels3DRequest")
    voxels = request.voxels
    if type(voxels) is not tuple or not 1 <= len(voxels) <= MAX_CUBICAL_VOXEL_GRID_SIDE:
        _reject(
            "voxel_grid_depth_bound", "voxel grid depth is outside its bound", "voxels"
        )
    height: int | None = None
    width: int | None = None
    volume = 0
    occupied_count = 0
    for plane in voxels:
        ledger.charge()
        if (
            type(plane) is not tuple
            or not 1 <= len(plane) <= MAX_CUBICAL_VOXEL_GRID_SIDE
        ):
            _reject(
                "voxel_grid_height_bound",
                "voxel grid height is outside its bound",
                "voxels",
            )
        if height is None:
            height = len(plane)
        elif len(plane) != height:
            _reject(
                "voxel_grid_not_rectangular",
                "voxel grid planes must have equal height",
                "voxels",
            )
        for row in plane:
            ledger.charge()
            if (
                type(row) is not tuple
                or not 1 <= len(row) <= MAX_CUBICAL_VOXEL_GRID_SIDE
            ):
                _reject(
                    "voxel_grid_width_bound",
                    "voxel grid width is outside its bound",
                    "voxels",
                )
            if width is None:
                width = len(row)
            elif len(row) != width:
                _reject(
                    "voxel_grid_not_rectangular",
                    "voxel grid rows must have equal width",
                    "voxels",
                )
            volume += len(row)
            if volume > MAX_CUBICAL_VOXEL_GRID_VOLUME:
                _reject(
                    "voxel_grid_volume_bound",
                    "voxel grid exceeds the admitted volume",
                    "voxels",
                )
            for voxel in row:
                ledger.charge()
                if type(voxel) is not bool:
                    _reject(
                        "voxel_grid_carrier_shape",
                        "voxels must be strict Boolean values in a rectangular 3D grid",
                        "voxels",
                    )
                occupied_count += int(voxel)
    assert width is not None and height is not None
    return request, width, height, len(voxels), occupied_count


def _face_keys(
    request: BinaryVoxels3DRequest,
    ledger: OperationWorkLedger,
) -> set[CellKey]:
    cells: set[CellKey] = set()
    for z, plane in enumerate(request.voxels):
        for y, row in enumerate(plane):
            for x, occupied in enumerate(row):
                ledger.charge()
                if not occupied:
                    continue
                coordinate_choices = tuple(
                    ((value, value), (value, value + 1), (value + 1, value + 1))
                    for value in (x, y, z)
                )
                for intervals in product(*coordinate_choices):
                    ledger.charge()
                    cells.add(cast(CellKey, intervals))
                    if len(cells) > MAX_FACE_CELLS:
                        raise OperationResourceAdmissionError(
                            location=("voxels",),
                            code="cubical_complex.voxel_grid_face_bound_exceeded",
                            message="voxel face closure exceeds the canonical complex cell limit",
                        )
    return cells


def binary_voxels_to_complex(request: BinaryVoxels3DRequest) -> CubicalComplex:
    """Return the closure of occupied unit voxels on the ordered (x,y,z) lattice."""

    request_checkpoint("before cubical voxel request validation")
    ledger = OperationWorkLedger(
        MAX_CUBICAL_VOXEL_GRID_VOLUME
        + MAX_CUBICAL_VOXEL_GRID_SIDE
        + MAX_CUBICAL_VOXEL_GRID_SIDE**2
        + MAX_CUBICAL_VOXEL_WORK
    )
    request, width, height, depth, occupied_count = _validate_raw_shape(
        request, ledger
    )
    voxel_count = width * height * depth
    request_checkpoint("after cubical voxel request validation")
    candidate_bound = 27 * occupied_count
    if candidate_bound > MAX_CUBICAL_VOXEL_FACE_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("voxels",),
            code="cubical_complex.voxel_grid_candidate_bound_exceeded",
            message=(
                "occupied voxel face candidates exceed the admitted construction "
                f"limit of {MAX_CUBICAL_VOXEL_FACE_CANDIDATES}"
            ),
        )
    output_cell_bound = min(candidate_bound, MAX_FACE_CELLS)
    sort_work_bound = output_cell_bound * max(1, ceil(log2(max(2, output_cell_bound))))
    construction_work_bound = (
        voxel_count + candidate_bound + sort_work_bound + 8 * output_cell_bound
    )
    if construction_work_bound > MAX_CUBICAL_VOXEL_WORK:
        raise OperationResourceAdmissionError(
            location=("voxels",),
            code="cubical_complex.voxel_grid_work_bound_exceeded",
            message="voxel face construction and canonicalization exceed the work limit",
        )
    # The complete generation, ordering, and value-check envelope is admitted
    # before expanding any occupied cube into its 27 possible faces.
    request_checkpoint("after cubical voxel construction admission")
    faces = _face_keys(request, ledger)
    ledger.charge(sort_work_bound + 8 * len(faces))
    ordered_faces = tuple(sorted(faces))
    request_checkpoint("before cubical voxel result construction")
    # Every key is generated from one occupied unit cube, and set/sort above
    # establish the canonical value invariants. Avoid replaying those checks for
    # each face while packaging the trusted kernel output.
    cells = tuple(
        CubicalCell.model_construct(intervals=intervals) for intervals in ordered_faces
    )
    return CubicalComplex(ambient_dimension=3, cells=cells)


__all__ = ["binary_voxels_to_complex"]
