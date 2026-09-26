"""Exact construction of rank-two Bieberbach free resolutions."""

from __future__ import annotations

from typing import NoReturn

from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.crystallographic.extensions._models import (
    BieberbachFaceOrbitComplex,
)
from jacobian.math.geometry.crystallographic.extensions.face_orbits import (
    MAX_FACE_ORBIT_POLYGON_FACETS,
    MAX_FACE_ORBIT_POLYGON_VERTICES,
    quotient_face_orbit_complex,
)
from jacobian.math.geometry.crystallographic.extensions.resolutions._models import (
    BieberbachPolygonFreeResolution,
    BieberbachResolutionAugmentationEntry,
)

MAX_POLYGON_RESOLUTION_RESULT_BYTES = 24_000_000


def _domain(reason: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("source",),
        code=f"crystallographic.polygon_resolution.{reason}",
        message=message,
    )


def _resource(reason: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("source",),
        code=f"crystallographic.polygon_resolution.{reason}",
        message=message,
    )


def polygon_free_resolution(
    source: BieberbachFaceOrbitComplex,
) -> BieberbachPolygonFreeResolution:
    """Return the cellular free ZGamma resolution associated to a flat polygon.

    The polygon producer establishes a complete side-pairing fundamental
    domain and torsion-freeness. Reconstructing the face-orbit complex checks
    those submitted relations before their group-ring incidence maps are used.
    """
    try:
        checked = BieberbachFaceOrbitComplex.model_validate(
            source.model_dump(mode="python", warnings=False), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("source",),
            code="crystallographic.polygon_resolution.source_shape",
            message="source does not satisfy the Bieberbach face-orbit value contract",
        ) from exc
    vertex_count = len(checked.source.source.facet_profile.vertices)
    facet_count = len(checked.source.source.facet_profile.facets)
    rank = len(checked.source.source.affine_realization.source.action_matrices[0])
    group = checked.source.source.affine_realization.source
    if rank != 2:
        _resource(
            "dimension_bound",
            "free-resolution construction currently supports dimension two",
        )
    if (
        vertex_count > MAX_FACE_ORBIT_POLYGON_VERTICES
        or facet_count > MAX_FACE_ORBIT_POLYGON_FACETS
    ):
        _resource(
            "polygon_bound",
            "resolution construction is bounded to 32 polygon vertices and facets",
        )
    input_bytes = len(checked.model_dump_json().encode("utf-8"))
    # The output retains the source and repeats the sparse incidence data and
    # augmented chain. Admit those exact serialized components before building
    # the result value. No geometric search is performed.
    predicted_result_bytes = (
        input_bytes
        + sum(
            len(entry.model_dump_json().encode("utf-8"))
            for entry in checked.boundary_1_to_0
        )
        + sum(
            len(entry.model_dump_json().encode("utf-8"))
            for entry in checked.boundary_2_to_1
        )
        + len(checked.quotient_chain_complex.model_dump_json().encode("utf-8"))
        + len(group.model_dump_json().encode("utf-8"))
        + 96 * len(checked.vertex_orbits)
        + 2048
    )
    if predicted_result_bytes > MAX_POLYGON_RESOLUTION_RESULT_BYTES:
        _resource("result_bound", "free-resolution result exceeds its byte envelope")

    try:
        reconstructed = quotient_face_orbit_complex(checked.source)
    except (OperationDomainValidationError, OperationResourceAdmissionError):
        raise
    if reconstructed != checked:
        _domain(
            "source_relation",
            "face-orbit boundaries do not match the checked polygon source",
        )

    result = BieberbachPolygonFreeResolution(
        source=checked,
        group=group,
        vertex_orbit_count=len(checked.vertex_orbits),
        edge_orbit_count=len(checked.edge_orbit_representatives),
        augmentation=tuple(
            BieberbachResolutionAugmentationEntry(
                source_cell_index=index, coefficient=1
            )
            for index in range(len(checked.vertex_orbits))
        ),
        boundary_1_to_0=checked.boundary_1_to_0,
        boundary_2_to_1=checked.boundary_2_to_1,
        augmented_chain_complex=checked.quotient_chain_complex,
    )
    return result


__all__ = ["polygon_free_resolution"]
