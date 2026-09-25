"""Finite degreewise images of simplicial maps."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.simplicial_sets._models import (
    FiniteTruncatedSimplicialSet,
)
from jacobian.math.topology.simplicial_sets.maps import (
    TruncatedSimplicialMap,
    _require_naturality,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables

_MAX_IMAGE_WORK = 100_000
_MAX_IMAGE_OUTPUT_CELLS = 131_072


class SimplicialMapImageRequest(StrictModel):
    """A caller-supplied simplicial map whose degreewise image is requested."""

    simplicial_map: TruncatedSimplicialMap


class SimplicialMapImageResult(StrictModel):
    """The image prefix and its canonical epi-mono factorization."""

    simplicial_map: TruncatedSimplicialMap
    image: FiniteTruncatedSimplicialSet
    surjection: TruncatedSimplicialMap
    inclusion: TruncatedSimplicialMap

    @model_validator(mode="after")
    def bind_factorization_axes(self) -> Self:
        if (
            self.simplicial_map.source != self.surjection.source
            or self.simplicial_map.target != self.inclusion.target
            or self.surjection.target != self.image
            or self.inclusion.source != self.image
        ):
            raise ValueError(
                "image factor maps must retain the original source and target"
            )
        return self


def _identity_work(value: FiniteTruncatedSimplicialSet) -> int:
    """Upper bound map entries touched when replaying all visible identities."""
    sizes = tuple(map(len, value.sets))
    degree = value.max_degree
    return 4 * sum(
        size
        * (
            ((n + 1) * n // 2 if n >= 2 else 0)
            + ((n + 1) * (n + 2) // 2 if n + 2 <= degree else 0)
            + ((n + 1) * (n + 2) if n < degree else 0)
        )
        for n, size in enumerate(sizes)
    )


def _preflight(value: TruncatedSimplicialMap) -> None:
    source, target = value.source, value.target
    source_cells = sum(len(label) for level in source.sets for label in level)
    target_cells = sum(len(label) for level in target.sets for label in level)
    map_cells = sum(len(row) for row in value.maps)
    output_cells = 2 * source_cells + 5 * target_cells + 3 * map_cells + 1024
    if output_cells > _MAX_IMAGE_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("simplicial_map",),
            code="simplicial_map.image_output_budget",
            message="the image factorization exceeds the admitted JSON output bound",
        )


def _require_checked(
    value: FiniteTruncatedSimplicialSet, *, location: str
) -> FiniteTruncatedSimplicialSet:
    checked = from_tables(
        value.max_degree, value.sets, value.face_maps, value.degeneracy_maps
    )
    if checked.status != "SIMPLICIAL_SET":
        raise OperationDomainValidationError(
            location=(location,),
            code="simplicial_map.image_carrier_invalid",
            message=f"{location} does not satisfy every visible simplicial identity",
        )
    if checked.simplicial_set is None:
        raise OperationDomainValidationError(
            location=(location,),
            code="simplicial_map.image_carrier_missing",
            message="validated carrier is missing",
        )
    return checked.simplicial_set


def simplicial_map_image(
    request: SimplicialMapImageRequest,
) -> SimplicialMapImageResult:
    """Return the finite-prefix image with exact simplicial epi-mono factors.

    In each degree the image is the subset of target simplices hit by the
    supplied map. Naturality makes this family closed under all visible faces
    and degeneracies, so restriction gives a simplicial set and a factorization
    through it.
    """
    value = request.simplicial_map
    _preflight(value)
    source = _require_checked(value.source, location="source")
    target = _require_checked(value.target, location="target")
    checked_map = TruncatedSimplicialMap(source=source, target=target, maps=value.maps)
    _require_naturality(checked_map, location="simplicial_map")

    image_levels: list[tuple[str, ...]] = []
    target_to_image: list[tuple[int, ...]] = []
    for degree, row in enumerate(checked_map.maps):
        selected = tuple(sorted(set(row)))
        target_to_image.append(selected)
        image_levels.append(tuple(target.sets[degree][index] for index in selected))

    image_positions = tuple(
        {target_index: image_index for image_index, target_index in enumerate(row)}
        for row in target_to_image
    )
    image_faces = tuple(
        tuple(
            tuple(
                image_positions[degree - 1][face[index]]
                for index in target_to_image[degree]
            )
            for face in target.face_maps[degree - 1]
        )
        for degree in range(1, target.max_degree + 1)
    )
    image_degeneracies = tuple(
        tuple(
            tuple(
                image_positions[degree + 1][degeneracy[index]]
                for index in target_to_image[degree]
            )
            for degeneracy in target.degeneracy_maps[degree]
        )
        for degree in range(target.max_degree)
    )

    # Naturality makes the selected target simplices closed under all visible
    # maps. Their restricted tables inherit the target's checked identities.
    image = FiniteTruncatedSimplicialSet._from_kernel(
        max_degree=target.max_degree,
        sets=tuple(image_levels),
        face_maps=image_faces,
        degeneracy_maps=image_degeneracies,
        total_simplices=sum(map(len, image_levels)),
        checked_identities=target.checked_identities,
    )
    surjection_rows = tuple(
        tuple(image_positions[degree][index] for index in row)
        for degree, row in enumerate(checked_map.maps)
    )
    surjection = TruncatedSimplicialMap(
        source=source, target=image, maps=surjection_rows
    )
    inclusion = TruncatedSimplicialMap(
        source=image, target=target, maps=tuple(target_to_image)
    )
    return SimplicialMapImageResult(
        simplicial_map=checked_map,
        image=image,
        surjection=surjection,
        inclusion=inclusion,
    )


__all__ = [
    "SimplicialMapImageRequest",
    "SimplicialMapImageResult",
    "simplicial_map_image",
]
