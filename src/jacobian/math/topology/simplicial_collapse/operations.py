"""Bounded exact elementary-collapse construction."""

from __future__ import annotations

from itertools import combinations

from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_DIMENSION,
    MAX_TOPOLOGY_FACES,
    MAX_TOPOLOGY_FACETS,
    MAX_TOPOLOGY_VERTICES,
    Simplex,
    SimplicialComplexRequest,
    _require_request_complex,
    canonical_complex,
)
from jacobian.math.topology.simplicial_collapse._models import (
    ElementaryCollapsePair,
    ElementaryCollapseRequest,
    ElementaryCollapseResult,
)

MAX_ELEMENTARY_COLLAPSE_WORK = 1_100_000
MAX_ELEMENTARY_COLLAPSE_OUTPUT_BYTES = 1_300_000
_MAX_SIMPLEX_JSON_BYTES = (MAX_TOPOLOGY_DIMENSION + 1) * 34 + MAX_TOPOLOGY_DIMENSION + 3


def _resource_error(code: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("complex",), code=code, message=message
    )


def _domain_error(code: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("pair",), code=code, message=message
    )


def _bounded_request(request: ElementaryCollapseRequest) -> ElementaryCollapseRequest:
    """Revalidate raw model_construct inputs after shallow cardinality checks."""

    if not isinstance(request, ElementaryCollapseRequest):
        raise _domain_error(
            "topology.elementary_collapse.request_type",
            "request must be an ElementaryCollapseRequest",
        )
    complex_value = request.complex
    if not isinstance(complex_value, SimplicialComplexRequest):
        raise _domain_error(
            "topology.elementary_collapse.complex_type",
            "complex must be a simplicial-complex request",
        )
    vertices = complex_value.vertices
    facets = complex_value.facets
    pair = request.pair
    if (
        not isinstance(vertices, tuple)
        or not 1 <= len(vertices) <= MAX_TOPOLOGY_VERTICES
    ):
        raise _resource_error(
            "topology.elementary_collapse.vertex_bound",
            f"the source must have between 1 and {MAX_TOPOLOGY_VERTICES} vertices",
        )
    if not isinstance(facets, tuple) or not 1 <= len(facets) <= MAX_TOPOLOGY_FACETS:
        raise _resource_error(
            "topology.elementary_collapse.facet_bound",
            f"the source must have between 1 and {MAX_TOPOLOGY_FACETS} facets",
        )
    if not isinstance(pair, ElementaryCollapsePair):
        raise _domain_error(
            "topology.elementary_collapse.pair_type",
            "pair must be an ElementaryCollapsePair",
        )
    if not isinstance(pair.face, tuple) or not isinstance(pair.coface, tuple):
        raise _domain_error(
            "topology.elementary_collapse.pair_shape", "pair faces must be tuples"
        )
    if not (
        1 <= len(pair.face) <= MAX_TOPOLOGY_DIMENSION
        and 2 <= len(pair.coface) <= MAX_TOPOLOGY_DIMENSION + 1
    ):
        raise _resource_error(
            "topology.elementary_collapse.pair_bound",
            "the face and coface dimensions exceed the admitted simplicial dimension",
        )
    for facet in facets:
        if (
            not isinstance(facet, tuple)
            or not 1 <= len(facet) <= MAX_TOPOLOGY_DIMENSION + 1
        ):
            raise _resource_error(
                "topology.elementary_collapse.facet_dimension",
                "each source facet must have between 1 and 8 vertices",
            )
    try:
        return ElementaryCollapseRequest.model_validate(
            {
                "complex": {"vertices": vertices, "facets": facets},
                "pair": {"face": pair.face, "coface": pair.coface},
            }
        )
    except ValidationError as exc:
        raise OperationDomainValidationError(
            location=("complex",),
            code="topology.elementary_collapse.invalid_input",
            message="the source complex or collapse pair is not canonical input",
        ) from exc


def _face_closure_bounded(
    facets: tuple[Simplex, ...],
) -> tuple[tuple[Simplex, ...], ...]:
    """Enumerate the complete source closure after its work was admitted."""

    by_dimension: list[set[Simplex]] = [
        set() for _ in range(MAX_TOPOLOGY_DIMENSION + 1)
    ]
    count = 0
    for facet in facets:
        for size in range(1, len(facet) + 1):
            dimension_faces = by_dimension[size - 1]
            for face in combinations(facet, size):
                if face not in dimension_faces:
                    dimension_faces.add(face)
                    count += 1
                    if count > MAX_TOPOLOGY_FACES:
                        raise _resource_error(
                            "topology.elementary_collapse.face_bound",
                            f"the source face closure exceeds {MAX_TOPOLOGY_FACES} faces",
                        )
    highest = max(index for index, faces in enumerate(by_dimension) if faces)
    return tuple(tuple(sorted(faces)) for faces in by_dimension[: highest + 1])


def _target_facets(
    facets: tuple[Simplex, ...],
    face: Simplex,
    coface: Simplex,
) -> tuple[Simplex, ...]:
    """Expose exactly the surviving codimension-one faces of the removed facet."""

    remaining = {facet for facet in facets if facet != coface}
    for ridge in combinations(coface, len(coface) - 1):
        if ridge == face:
            continue
        ridge_set = set(ridge)
        if not any(ridge_set.issubset(facet) for facet in remaining):
            remaining.add(ridge)
    return tuple(sorted(remaining))


def _output_bound() -> int:
    """Bound source, target, pair and JSON structure by the value limits."""

    complex_bytes = (
        (MAX_TOPOLOGY_FACES + MAX_TOPOLOGY_FACETS) * _MAX_SIMPLEX_JSON_BYTES
        + MAX_TOPOLOGY_VERTICES * 34
        + (MAX_TOPOLOGY_DIMENSION + 1) * 128
        + 4_096
    )
    return 2 * complex_bytes + 2 * _MAX_SIMPLEX_JSON_BYTES + 8_192


def elementary_collapse(
    request: ElementaryCollapseRequest,
) -> ElementaryCollapseResult:
    """Delete one free face and its unique codimension-one coface.

    A valid result carries the canonical source and target complexes. The
    target retains every source face except the selected pair.
    """

    request = _bounded_request(request)
    vertices = request.complex.vertices
    raw_facets = request.complex.facets
    face_candidates = sum((1 << len(facet)) - 1 for facet in raw_facets)
    facet_checks = len(raw_facets) * len(raw_facets) * (MAX_TOPOLOGY_DIMENSION + 1)
    target_checks = (
        (len(raw_facets) + len(request.pair.coface))
        * len(raw_facets)
        * (MAX_TOPOLOGY_DIMENSION + 1)
    )
    face_filter_work = MAX_TOPOLOGY_FACES * (MAX_TOPOLOGY_DIMENSION + 1)
    canonical_facet_checks = 2 * MAX_TOPOLOGY_FACETS**2 * (MAX_TOPOLOGY_DIMENSION + 1)
    canonical_face_order_checks = (
        2
        * MAX_TOPOLOGY_FACES
        * (MAX_TOPOLOGY_DIMENSION + 1)
        * (MAX_TOPOLOGY_FACES.bit_length())
    )
    work_bound = (
        face_candidates
        + facet_checks
        + target_checks
        + face_filter_work
        + canonical_facet_checks
        + canonical_face_order_checks
    )
    if work_bound > MAX_ELEMENTARY_COLLAPSE_WORK:
        raise _resource_error(
            "topology.elementary_collapse.work_bound",
            f"the elementary collapse has an admitted work bound of {work_bound}, above {MAX_ELEMENTARY_COLLAPSE_WORK}",
        )
    output_bound = _output_bound()
    if output_bound > MAX_ELEMENTARY_COLLAPSE_OUTPUT_BYTES:
        raise _resource_error(
            "topology.elementary_collapse.output_bound",
            f"the canonical source and target may require {output_bound} bytes, above {MAX_ELEMENTARY_COLLAPSE_OUTPUT_BYTES}",
        )

    facets = _require_request_complex(vertices, raw_facets, check_closure=False)
    face = tuple(sorted(request.pair.face))
    coface = tuple(sorted(request.pair.coface))
    face_set = set(face)
    coface_set = set(coface)
    if (
        len(set(face)) != len(face)
        or len(set(coface)) != len(coface)
        or len(coface) != len(face) + 1
        or not face_set.issubset(coface_set)
    ):
        raise _domain_error(
            "topology.elementary_collapse.not_codimension_one",
            "the selected face must be a codimension-one face of its coface",
        )
    if coface not in facets:
        raise _domain_error(
            "topology.elementary_collapse.coface_not_facet",
            "the selected coface must be a maximal simplex of the source",
        )
    containing = tuple(facet for facet in facets if face_set.issubset(facet))
    if containing != (coface,):
        raise _domain_error(
            "topology.elementary_collapse.not_free",
            "the selected face must be contained in exactly the selected coface",
        )

    target_facets = _target_facets(facets, face, coface)
    if len(target_facets) > MAX_TOPOLOGY_FACETS:
        raise _resource_error(
            "topology.elementary_collapse.target_facet_bound",
            f"the collapsed target has {len(target_facets)} facets, above {MAX_TOPOLOGY_FACETS}",
        )

    source_closure = _face_closure_bounded(facets)
    target_closure = tuple(
        tuple(cell for cell in cells if cell != face and cell != coface)
        for cells in source_closure
    )
    while target_closure and not target_closure[-1]:
        target_closure = target_closure[:-1]
    if not target_closure or not target_closure[0]:
        raise RuntimeError("a valid elementary collapse must retain a nonempty complex")

    source = canonical_complex(vertices, facets, closure=source_closure)
    target_vertices = tuple(
        sorted({vertex for cell in target_facets for vertex in cell})
    )
    target = canonical_complex(target_vertices, target_facets, closure=target_closure)
    return ElementaryCollapseResult(
        source=source,
        target=target,
        removed_pair=request.pair.model_copy(update={"face": face, "coface": coface}),
    )


__all__ = [
    "MAX_ELEMENTARY_COLLAPSE_OUTPUT_BYTES",
    "MAX_ELEMENTARY_COLLAPSE_WORK",
    "elementary_collapse",
]
