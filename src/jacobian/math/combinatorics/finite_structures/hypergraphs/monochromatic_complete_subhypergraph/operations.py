"""Exact complete monochromatic uniform-subhypergraph enumeration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations
from math import comb

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_LABEL_LENGTH,
    MAX_TOTAL_INCIDENCES,
    FiniteHypergraph,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.colorings import (
    IndexedHyperedgeColoring,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.monochromatic_complete_subhypergraph._models import (
    MonochromaticCompleteSubhypergraphProfile,
)

__all__ = ["construct"]

MAX_PROFILE_WORK = 2_000_000
MAX_PROFILE_ALLOCATION = 4_000_000
MAX_PROFILE_OUTPUT_BYTES = 33_554_432


@dataclass(frozen=True, slots=True)
class _Admission:
    source_uniformity: int
    target_uniformity: int
    target_vertices: tuple[str, ...]
    source_lookup: dict[frozenset[str], tuple[str, int]]
    shortcut: str | None = None


def _domain(location: tuple[str, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _resource(location: tuple[str, ...], code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=location,
        code=code,
        message=message,
    )


def _validate_uniformities(
    source_uniformity: int,
    target_uniformity: int,
    vertex_count: int,
) -> None:
    if type(source_uniformity) is not int or not 1 <= source_uniformity <= 256:
        _domain(
            ("source_uniformity",),
            "monochromatic_profile.source_uniformity",
            "source_uniformity must be an integer between 1 and 256",
        )
    if type(target_uniformity) is not int or not 1 <= target_uniformity <= 256:
        _domain(
            ("target_uniformity",),
            "monochromatic_profile.target_uniformity",
            "target_uniformity must be an integer between 1 and 256",
        )
    if target_uniformity < source_uniformity:
        _domain(
            ("target_uniformity",),
            "monochromatic_profile.target_before_source",
            "target_uniformity must be at least source_uniformity",
        )
    if target_uniformity > vertex_count:
        _domain(
            ("target_uniformity",),
            "monochromatic_profile.target_too_large",
            "target_uniformity cannot exceed the source vertex count",
        )


def _build_source_lookup(
    coloring: IndexedHyperedgeColoring, source_uniformity: int
) -> dict[frozenset[str], tuple[str, int]]:
    """Validate source uniformity and build the normalized edge index."""

    source = coloring.hypergraph
    source_lookup: dict[frozenset[str], tuple[str, int]] = {}
    source_edges = source.edges
    if source_edges:
        for index, (edge_id, members) in enumerate(source_edges):
            if len(members) != source_uniformity:
                _domain(
                    ("coloring", "hypergraph", "edges", str(index)),
                    "monochromatic_profile.nonuniform_source",
                    "every source edge must have source_uniformity members",
                )
            key = frozenset(members)
            if key in source_lookup:
                _domain(
                    ("coloring", "hypergraph", "edges", str(index)),
                    "monochromatic_profile.duplicate_source_edge",
                    "source vertex sets must occur once for an unambiguous lookup",
                )
            source_lookup[key] = (edge_id, coloring.assignments[index].color_index)
    return source_lookup


def _preflight_result(
    coloring: IndexedHyperedgeColoring,
    source_uniformity: int,
    target_uniformity: int,
    candidate_count: int,
    required_edges: int,
) -> None:
    """Admit complete work and a source-sensitive output envelope."""

    source = coloring.hypergraph
    source_edges = source.edges
    vertex_count = len(source.vertices)
    work = candidate_count * required_edges
    if work > MAX_PROFILE_WORK:
        _resource(
            ("target_uniformity",),
            "monochromatic_profile.work_bound",
            "complete target enumeration exceeds the 2000000 lookup bound",
        )

    # Each emitted candidate contributes q source-edge incidences.  A source
    # edge belongs to exactly C(n-r, s-r) target subsets, giving a sound,
    # source-sensitive upper bound before the candidate enumeration starts.
    candidate_upper_bound = min(
        candidate_count,
        len(source_edges)
        * comb(vertex_count - source_uniformity, target_uniformity - source_uniformity)
        // required_edges,
    )
    if candidate_upper_bound > MAX_EDGES:
        _resource(
            ("target_uniformity",),
            "monochromatic_profile.candidate_bound",
            "the complete candidate profile exceeds the 12000-edge bound",
        )
    if candidate_upper_bound * target_uniformity > MAX_TOTAL_INCIDENCES:
        _resource(
            ("target_uniformity",),
            "monochromatic_profile.incidence_bound",
            "the complete candidate profile exceeds the 36000-incidence bound",
        )

    # The retained profile allocates one target row, target incidence tuple,
    # and q source-ID witness entries per emitted candidate.  This cardinality
    # envelope is independent of JSON transport limits and is checked before
    # any target subset is enumerated.
    allocation = candidate_upper_bound * (target_uniformity + required_edges)
    if allocation > MAX_PROFILE_ALLOCATION:
        _resource(
            ("target_uniformity",),
            "monochromatic_profile.allocation_bound",
            "the complete candidate profile exceeds the 4000000-entry allocation bound",
        )

    max_vertex_bytes = max(
        (len(vertex.encode("utf-8")) for vertex in source.vertices),
        default=0,
    )
    max_edge_id_bytes = max(
        (len(edge_id.encode("utf-8")) for edge_id, _ in source_edges),
        default=0,
    )
    if (
        max_vertex_bytes > MAX_LABEL_LENGTH * 4
        or max_edge_id_bytes > MAX_LABEL_LENGTH * 4
    ):
        _resource(
            ("coloring",),
            "monochromatic_profile.label_width",
            "source labels exceed the hypergraph UTF-8 label envelope",
        )
    output_bytes = candidate_upper_bound * (
        target_uniformity * (max_vertex_bytes + 2)
        + required_edges * (max_edge_id_bytes + 2)
    )
    if output_bytes > MAX_PROFILE_OUTPUT_BYTES:
        _resource(
            ("target_uniformity",),
            "monochromatic_profile.output_bytes",
            "the complete candidate profile exceeds the 32-mebibyte JSON envelope",
        )


def _admit(
    coloring: IndexedHyperedgeColoring,
    source_uniformity: int,
    target_uniformity: int,
) -> _Admission:
    """Preflight exact enumeration and retained profile representation."""

    source = coloring.hypergraph
    _validate_uniformities(source_uniformity, target_uniformity, len(source.vertices))
    source_lookup = _build_source_lookup(coloring, source_uniformity)
    vertex_count = len(source.vertices)
    source_edges = source.edges
    candidate_count = comb(vertex_count, target_uniformity)
    required_edges = comb(target_uniformity, source_uniformity)
    if source_uniformity == target_uniformity:
        return _Admission(
            source_uniformity=source_uniformity,
            target_uniformity=target_uniformity,
            target_vertices=tuple(sorted(source.vertices)),
            source_lookup=source_lookup,
            shortcut="source_edges",
        )
    if len(source_edges) < required_edges:
        return _Admission(
            source_uniformity=source_uniformity,
            target_uniformity=target_uniformity,
            target_vertices=tuple(sorted(source.vertices)),
            source_lookup=source_lookup,
            shortcut="empty",
        )
    _preflight_result(
        coloring,
        source_uniformity,
        target_uniformity,
        candidate_count,
        required_edges,
    )

    return _Admission(
        source_uniformity=source_uniformity,
        target_uniformity=target_uniformity,
        target_vertices=tuple(sorted(source.vertices)),
        source_lookup=source_lookup,
    )


def construct(
    coloring: IndexedHyperedgeColoring,
    source_uniformity: int,
    target_uniformity: int,
) -> MonochromaticCompleteSubhypergraphProfile:
    """Return every complete monochromatic target subset with provenance."""

    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return construct(coloring, source_uniformity, target_uniformity)
    deadline = execution.started_at + 120.0
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    admission = _admit(coloring, source_uniformity, target_uniformity)
    target_edges: tuple[tuple[str, tuple[str, ...]], ...]
    candidate_colors: tuple[int, ...]
    source_witnesses: tuple[tuple[str, ...], ...]
    if admission.shortcut == "source_edges":
        source_edges = sorted(coloring.hypergraph.edges, key=lambda edge: edge[1])
        target_edges = tuple(
            (f"c{index}", members) for index, (_, members) in enumerate(source_edges)
        )
        candidate_colors = tuple(
            admission.source_lookup[frozenset(members)][1]
            for _, members in source_edges
        )
        source_witnesses = tuple((edge_id,) for edge_id, _ in source_edges)
    elif admission.shortcut == "empty":
        target_edges = ()
        candidate_colors = ()
        source_witnesses = ()
    else:
        target_edges_list: list[tuple[str, tuple[str, ...]]] = []
        candidate_colors_list: list[int] = []
        source_witnesses_list: list[tuple[str, ...]] = []
        for target in combinations(
            admission.target_vertices, admission.target_uniformity
        ):
            request_checkpoint("during monochromatic target enumeration")
            witness: list[str] = []
            candidate_color: int | None = None
            for required in combinations(target, admission.source_uniformity):
                source_entry = admission.source_lookup.get(frozenset(required))
                if source_entry is None:
                    break
                edge_id, color = source_entry
                if candidate_color is None:
                    candidate_color = color
                elif candidate_color != color:
                    break
                witness.append(edge_id)
            else:
                if candidate_color is None:
                    raise AssertionError(
                        "positive source uniformity requires a witness"
                    )
                target_edges_list.append((f"c{len(target_edges_list)}", target))
                candidate_colors_list.append(candidate_color)
                source_witnesses_list.append(tuple(witness))
        target_edges = tuple(target_edges_list)
        candidate_colors = tuple(candidate_colors_list)
        source_witnesses = tuple(source_witnesses_list)

    request_checkpoint("before monochromatic profile construction")
    return MonochromaticCompleteSubhypergraphProfile(
        coloring=coloring,
        source_uniformity=admission.source_uniformity,
        target_uniformity=admission.target_uniformity,
        hypergraph=FiniteHypergraph(
            vertices=coloring.hypergraph.vertices,
            edges=target_edges,
        ),
        candidate_colors=candidate_colors,
        source_edge_witnesses=source_witnesses,
    )
