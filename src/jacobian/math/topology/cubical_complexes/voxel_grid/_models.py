"""Typed request for bounded three-dimensional binary voxel grids."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictBool, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_CUBICAL_VOXEL_GRID_SIDE = 64
MAX_CUBICAL_VOXEL_GRID_VOLUME = MAX_CUBICAL_VOXEL_GRID_SIDE**3

VoxelRow = Annotated[
    tuple[StrictBool, ...],
    Field(min_length=1, max_length=MAX_CUBICAL_VOXEL_GRID_SIDE),
]
VoxelPlane = Annotated[
    tuple[VoxelRow, ...],
    Field(min_length=1, max_length=MAX_CUBICAL_VOXEL_GRID_SIDE),
]
VoxelGrid = Annotated[
    tuple[VoxelPlane, ...],
    Field(min_length=1, max_length=MAX_CUBICAL_VOXEL_GRID_SIDE),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"cubical_complex.{reason}", message)


class BinaryVoxels3DRequest(StrictModel):
    """Rectangular ``voxels[z][y][x]`` binary array with one ordered (x,y,z) axis."""

    voxels: VoxelGrid

    @model_validator(mode="after")
    def require_rectangular_grid(self) -> Self:
        width = len(self.voxels[0][0])
        height = len(self.voxels[0])
        if any(
            len(plane) != height or any(len(row) != width for row in plane)
            for plane in self.voxels
        ):
            raise _validation_error(
                "voxel_grid_not_rectangular",
                "binary voxel input must be a rectangular nonempty 3D grid",
            )
        if len(self.voxels) * height * width > MAX_CUBICAL_VOXEL_GRID_VOLUME:
            raise _validation_error(
                "voxel_grid_volume_bound",
                f"voxel grid volume must not exceed {MAX_CUBICAL_VOXEL_GRID_VOLUME}",
            )
        return self


__all__ = [
    "MAX_CUBICAL_VOXEL_GRID_SIDE",
    "MAX_CUBICAL_VOXEL_GRID_VOLUME",
    "BinaryVoxels3DRequest",
]
