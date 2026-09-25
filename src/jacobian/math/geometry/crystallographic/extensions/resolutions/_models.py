"""Canonical rank-two Bieberbach free-resolution values."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.geometry.crystallographic.extensions._models import (
    BieberbachFaceOrbitComplex,
    BieberbachGroupRingBoundaryEntry,
    FiniteLatticeExtension,
)
from jacobian.math.topology.chain_complexes.values import ChainComplexValue

_CellIndex = Annotated[StrictInt, Field(ge=0, le=31)]


class BieberbachResolutionAugmentationEntry(StrictModel):
    """The augmentation sends one orbit representative to 1 in Z."""

    source_cell_index: _CellIndex
    coefficient: StrictInt = Field(ge=1, le=1)


class BieberbachPolygonFreeResolutionRequest(StrictModel):
    """A source-bound quotient cell structure to lift to the group cover."""

    source: BieberbachFaceOrbitComplex


class BieberbachPolygonFreeResolution(StrictModel):
    """The cellular free ZGamma resolution from a checked flat polygon.

    ``boundary_1_to_0`` and ``boundary_2_to_1`` are sparse matrices over the
    integral group ring. Each incidence coefficient multiplies the group
    element encoded by ``(lattice_translation, holonomy_element)``. The
    universal cover is the contractible Euclidean plane, so these cellular
    modules, followed by the displayed augmentation, form a free resolution.
    ``augmented_chain_complex`` is the result after tensoring with the trivial
    ZGamma-module Z.
    """

    source: BieberbachFaceOrbitComplex
    group: FiniteLatticeExtension
    vertex_orbit_count: StrictInt = Field(ge=1, le=32)
    edge_orbit_count: StrictInt = Field(ge=1, le=32)
    augmentation: tuple[BieberbachResolutionAugmentationEntry, ...] = Field(
        min_length=1, max_length=32
    )
    boundary_1_to_0: tuple[BieberbachGroupRingBoundaryEntry, ...] = Field(
        min_length=1, max_length=256
    )
    boundary_2_to_1: tuple[BieberbachGroupRingBoundaryEntry, ...] = Field(
        min_length=1, max_length=128
    )
    augmented_chain_complex: ChainComplexValue

    @model_validator(mode="after")
    def require_source_aligned_modules(self) -> Self:
        chain = self.source.quotient_chain_complex
        source_group = self.source.source.source.affine_realization.source
        if (
            self.group != source_group
            or len(self.source.vertex_orbits) != self.vertex_orbit_count
            or len(self.source.edge_orbit_representatives) != self.edge_orbit_count
            or len(self.augmentation) != self.vertex_orbit_count
            or tuple(entry.source_cell_index for entry in self.augmentation)
            != tuple(range(self.vertex_orbit_count))
            or self.boundary_1_to_0 != self.source.boundary_1_to_0
            or self.boundary_2_to_1 != self.source.boundary_2_to_1
            or self.augmented_chain_complex != chain
            or chain.basis_sizes != (self.vertex_orbit_count, self.edge_orbit_count, 1)
        ):
            raise PydanticCustomError(
                "crystallographic.resolution_source_alignment",
                "resolution modules, augmentation, or boundaries do not align with the checked polygon",
            )
        return self


__all__ = [
    "BieberbachPolygonFreeResolution",
    "BieberbachPolygonFreeResolutionRequest",
    "BieberbachResolutionAugmentationEntry",
]
