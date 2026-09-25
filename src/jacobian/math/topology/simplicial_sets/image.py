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
    MAX_SIMPLICES_PER_DEGREE,
    FiniteTruncatedSimplicialSet,
)
from jacobian.math.topology.simplicial_sets.maps import (
    TruncatedSimplicialMap,
    _require_naturality,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables

_MAX_IMAGE_WORK = 100_000
_MAX_IMAGE_OUTPUT_BYTES = 131_072


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
    sizes = tuple(map(len, source.sets))
    map_cells = sum(map(len, value.maps))
    source_naturality_work = sum(
        size * (degree + 1) * (int(degree > 0) + int(degree < source.max_degree))
        for degree, size in enumerate(sizes)
    )
    target_naturality_work = sum(
        len(level) * (degree + 1) * (int(degree > 0) + int(degree < target.max_degree))
        for degree, level in enumerate(target.sets)
    )
    identity_work = _identity_work(source) + _identity_work(target)
    # Each source simplex is visited once to collect its target image. The
    # image tables then scan at most every target face/degeneracy entry.
    construction_work = map_cells * (
        1 + MAX_SIMPLICES_PER_DEGREE.bit_length()
    ) + 2 * target.total_simplices * (source.max_degree + 1)
    if (
        identity_work
        + _identity_work(target)
        + 2 * source_naturality_work
        + target_naturality_work
        + construction_work
        > _MAX_IMAGE_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("simplicial_map",),
            code="simplicial_map.image_work_budget",
            message="the image factorization exceeds the admitted finite work bound",
        )

    source_bytes = len(source.model_dump_json())
    target_bytes = len(target.model_dump_json())
    map_bytes = len(value.model_dump_json()) - source_bytes - target_bytes
    # The result retains the original map, a source-to-image map, and an
    # image-to-target inclusion. The image carrier is bounded by the target;
    # both derived map tables have at most the input map's row widths.
    output_bytes = 2 * source_bytes + 5 * target_bytes + 3 * map_bytes + 1024
    if output_bytes > _MAX_IMAGE_OUTPUT_BYTES:
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
    assert checked.simplicial_set is not None
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
    source = _require_checked(value.source, location="source")
    target = _require_checked(value.target, location="target")
    value = TruncatedSimplicialMap(source=source, target=target, maps=value.maps)
    _preflight(value)
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
