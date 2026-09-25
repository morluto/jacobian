"""Exact bounded associated-simplicial-set prefix of a finite complex."""

from __future__ import annotations

from itertools import combinations
from math import comb

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._request_admission import (
    require_canonical_complex_admission,
    run_topology_admission,
)
from jacobian.math.topology.simplicial_sets._models import (
    MAX_SIMPLICES_PER_DEGREE,
    MAX_TOTAL_SIMPLICES,
    FiniteTruncatedSimplicialSet,
)
from jacobian.math.topology.simplicial_sets.complex_conversion_models import (
    ComplexFaceSimplexIndex,
    SimplicialComplexPrefixRequest,
    SimplicialComplexPrefixResult,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables

MAX_COMPLEX_PREFIX_MAP_ENTRIES = 2_000
MAX_COMPLEX_PREFIX_IDENTITY_WORK = 100_000
MAX_COMPLEX_PREFIX_OUTPUT_BYTES = 1_000_000


def _admitted_faces(
    request: SimplicialComplexPrefixRequest,
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    source = request.complex
    run_topology_admission(
        lambda: require_canonical_complex_admission(source), location=("complex",)
    )
    actual_vertices = tuple(
        sorted({vertex for facet in source.maximal_simplices for vertex in facet})
    )
    if actual_vertices != source.vertices:
        raise OperationDomainValidationError(
            location=("complex",),
            code="simplicial_set.complex_vertex_axis_invalid",
            message="source vertex axis does not match its maximal simplices",
        )
    if (
        source.dimension != len(source.faces_by_dimension) - 1
        or source.f_vector
        != tuple(len(level.faces) for level in source.faces_by_dimension)
        or source.closure_size
        != sum(len(level.faces) for level in source.faces_by_dimension)
    ):
        raise OperationDomainValidationError(
            location=("complex",),
            code="simplicial_set.complex_source_summary_invalid",
            message="source dimension, f-vector, or closure size disagrees with its face table",
        )
    return tuple(
        tuple(face for face in level.faces)
        for level in source.faces_by_dimension[: request.max_degree + 1]
    )


def _level_labels(
    faces: tuple[tuple[tuple[str, ...], ...], ...],
    degree: int,
    vertex_index: dict[str, int],
) -> tuple[tuple[int, ...], ...]:
    """Generate each monotone tuple exactly once, grouped by its support."""
    tuples: set[tuple[int, ...]] = set()
    for level in faces[: degree + 1]:
        for face in level:
            support = tuple(vertex_index[vertex] for vertex in face)
            size = len(support)
            if size > degree + 1:
                continue
            for cuts in combinations(range(1, degree + 1), size - 1):
                endpoints = (0, *cuts, degree + 1)
                expanded = tuple(
                    vertex
                    for position, vertex in enumerate(support)
                    for _ in range(endpoints[position + 1] - endpoints[position])
                )
                tuples.add(expanded)
    return tuple(sorted(tuples))


def _label(simplex: tuple[int, ...]) -> str:
    return ",".join(str(vertex) for vertex in simplex)


def _build(request: SimplicialComplexPrefixRequest) -> SimplicialComplexPrefixResult:
    source = request.complex
    faces = _admitted_faces(request)
    vertex_index = {vertex: index for index, vertex in enumerate(source.vertices)}

    counts = tuple(
        sum(
            len(level) * comb(degree, dimension)
            for dimension, level in enumerate(faces)
            if dimension <= degree
        )
        for degree in range(request.max_degree + 1)
    )
    if any(count > MAX_SIMPLICES_PER_DEGREE for count in counts):
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="simplicial_set.complex_prefix_degree_budget",
            message=(
                "the associated prefix exceeds the per-degree simplex limit "
                f"{MAX_SIMPLICES_PER_DEGREE}"
            ),
        )
    total = sum(counts)
    if total > MAX_TOTAL_SIMPLICES:
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="simplicial_set.complex_prefix_total_budget",
            message=(
                f"the associated prefix has {total} simplices, above the "
                f"{MAX_TOTAL_SIMPLICES}-simplex limit"
            ),
        )

    face_map_entries = sum(
        counts[degree] * (degree + 1) for degree in range(1, len(counts))
    )
    degeneracy_map_entries = sum(
        counts[degree] * (degree + 1) for degree in range(len(counts) - 1)
    )
    map_entries = face_map_entries + degeneracy_map_entries
    transport_entries = sum(len(level) for level in faces)
    if map_entries + transport_entries > MAX_COMPLEX_PREFIX_MAP_ENTRIES:
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="simplicial_set.complex_prefix_map_budget",
            message="the associated prefix map and transport rows exceed their bound",
        )
    identity_work = sum(
        comb(degree + 1, 2) * counts[degree]
        for degree in range(2, request.max_degree + 1)
    )
    identity_work += sum(
        (degree + 1) * (degree + 2) // 2 * counts[degree]
        for degree in range(request.max_degree - 1)
    )
    identity_work += sum(
        (degree + 1) * (degree + 2) * counts[degree]
        for degree in range(request.max_degree)
    )
    if identity_work > MAX_COMPLEX_PREFIX_IDENTITY_WORK:
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="simplicial_set.complex_prefix_identity_work_budget",
            message="the associated prefix identity checks exceed their work bound",
        )
    # All generated labels have at most 14 ASCII characters (five vertex
    # indices in 0..63); mapping rows are bounded by the canonical carrier.
    source_bytes = len(source.model_dump_json().encode("utf-8"))
    estimated_bytes = (
        source_bytes + total * 24 + map_entries * 4 + transport_entries * 128 + 4096
    )
    if estimated_bytes > MAX_COMPLEX_PREFIX_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="simplicial_set.complex_prefix_output_budget",
            message="the associated prefix and face transport exceed the output bound",
        )

    values = tuple(
        _level_labels(faces, degree, vertex_index)
        for degree in range(request.max_degree + 1)
    )
    labels = tuple(tuple(_label(simplex) for simplex in level) for level in values)
    indices = tuple(
        {simplex: index for index, simplex in enumerate(level)} for level in values
    )
    face_maps = tuple(
        tuple(
            tuple(
                indices[degree - 1][simplex[:omitted] + simplex[omitted + 1 :]]
                for simplex in values[degree]
            )
            for omitted in range(degree + 1)
        )
        for degree in range(1, request.max_degree + 1)
    )
    degeneracy_maps = tuple(
        tuple(
            tuple(
                indices[degree + 1][
                    (*simplex[: omitted + 1], simplex[omitted], *simplex[omitted + 1 :])
                ]
                for simplex in values[degree]
            )
            for omitted in range(degree + 1)
        )
        for degree in range(request.max_degree)
    )
    checked = from_tables(request.max_degree, labels, face_maps, degeneracy_maps)
    if checked.status != "SIMPLICIAL_SET" or checked.simplicial_set is None:
        raise RuntimeError("complex conversion generated an invalid simplicial prefix")
    simplicial_set: FiniteTruncatedSimplicialSet = checked.simplicial_set

    transport = tuple(
        ComplexFaceSimplexIndex(
            dimension=dimension,
            face_index=face_index,
            simplex_index=indices[dimension][
                tuple(vertex_index[vertex] for vertex in face)
            ],
        )
        for dimension, level in enumerate(faces)
        for face_index, face in enumerate(level)
    )
    return SimplicialComplexPrefixResult(
        source_complex=source,
        simplicial_set=simplicial_set,
        face_simplex_indices=transport,
    )


def simplicial_set_from_complex(
    request: SimplicialComplexPrefixRequest,
) -> SimplicialComplexPrefixResult:
    """Return the complete bounded simplicial-set prefix associated to a complex."""
    return _build(request)


__all__ = ["simplicial_set_from_complex"]
