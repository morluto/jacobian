"""Catalog declaration for finite 3D binary voxel construction."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.cubical_complexes._models import CubicalComplex
from jacobian.math.topology.cubical_complexes.voxel_grid._models import (
    MAX_CUBICAL_VOXEL_GRID_SIDE,
    BinaryVoxels3DRequest,
)
from jacobian.math.topology.cubical_complexes.voxel_grid.operations import (
    binary_voxels_to_complex,
)


def _run(request: BinaryVoxels3DRequest) -> CubicalComplex:
    return binary_voxels_to_complex(request)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="topology.cubical_complex.from_binary_voxels_3d.compute",
        title="Construct a cubical complex from binary voxels",
        description=(
            "Interpret `voxels[z][y][x]` as a rectangular 3D Boolean array, "
            "with each true entry the closed unit cube `[x,x+1]` times `[y,y+1]` times `"
            "[z,z+1]`. Return the canonical cubical complex containing every "
            "face of every occupied voxel on the ordered (x,y,z) axes. The "
            f"grid side lengths are at most {MAX_CUBICAL_VOXEL_GRID_SIDE}; "
            "occupied face candidates and the exact output cell count are "
            "bounded before result construction. An all-false grid returns "
            "the empty cubical complex while retaining ambient dimension 3."
        ),
        request_type=BinaryVoxels3DRequest,
        result_type=CubicalComplex,
        run=_run,
        tags=("topology", "cubical", "voxel-grid", "exact"),
        discovery_terms=(
            "3D binary voxel grid to cubical complex",
            "cubical complex of occupied unit voxels",
            "face closure of binary 3D voxels",
        ),
        examples=(
            OperationExample(
                name="one_occupied_voxel",
                description=(
                    "The sole occupied voxel contributes its cube and all 26 "
                    "proper cubical faces."
                ),
                input={"voxels": [[[True]]]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
