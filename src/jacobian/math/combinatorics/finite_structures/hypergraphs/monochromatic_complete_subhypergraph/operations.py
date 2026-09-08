"""Exact complete monochromatic uniform-subhypergraph enumeration."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

from jacobian._execution import request_checkpoint
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
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
MAX_PROFILE_RESULT_BYTES = 10 * 1024 * 1024
MAX_PROFILE_LOOKUP_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _Admission:
    source_uniformity: int
    target_uniformity: int
    target_vertices: tuple[str, ...]
    source_lookup: dict[frozenset[str], tuple[str, int]]


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
    source_lookup_bytes = sum(
        32
        + len(edge_id.encode("utf-8"))
        + sum(len(vertex.encode("utf-8")) for vertex in members)
        for edge_id, members in source_edges
    )
    if source_lookup_bytes > MAX_PROFILE_LOOKUP_BYTES:
        _resource(
            ("coloring",),
            "monochromatic_profile.lookup_bound",
            "source edge lookup exceeds the 4 MiB admission bound",
        )
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

    try:
        source_bytes = len(
            encode_strict_json(coloring.model_dump(mode="json"), limits=None)
        )
        label_bytes = sorted(
            (len(vertex.encode("utf-8")) for vertex in source.vertices), reverse=True
        )
        target_label_bytes = sum(label_bytes[:target_uniformity])
        source_id_bytes = max(
            (len(edge_id.encode("utf-8")) for edge_id, _ in source_edges),
            default=0,
        )
    except (UnicodeError, ValueError) as exc:
        _domain(
            ("coloring",),
            "monochromatic_profile.source_encoding",
            "the source colouring cannot be represented in canonical JSON",
        )
        raise AssertionError("unreachable") from exc
    # This conservative row envelope includes target labels, one colour, q
    # source IDs, and JSON punctuation/field names.  It admits sparse sources
    # even when C is large, while rejecting a profile that cannot fit the wire.
    row_bytes = 256 + target_label_bytes + required_edges * (source_id_bytes + 8)
    if source_bytes + candidate_upper_bound * row_bytes > MAX_PROFILE_RESULT_BYTES:
        _resource(
            ("target_uniformity",),
            "monochromatic_profile.result_bytes",
            "the complete candidate profile exceeds the 10 MiB result bound",
        )
    if source_bytes > MAX_PROFILE_RESULT_BYTES:
        _resource(
            ("coloring",),
            "monochromatic_profile.source_bytes",
            "the retained source colouring exceeds the 10 MiB result bound",
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
    candidate_count = comb(vertex_count, target_uniformity)
    required_edges = comb(target_uniformity, source_uniformity)
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

    admission = _admit(coloring, source_uniformity, target_uniformity)
    target_edges: list[tuple[str, tuple[str, ...]]] = []
    candidate_colors: list[int] = []
    source_witnesses: list[tuple[str, ...]] = []
    for target in combinations(admission.target_vertices, admission.target_uniformity):
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
                raise AssertionError("positive source uniformity requires a witness")
            target_edges.append((f"c{len(target_edges)}", target))
            candidate_colors.append(candidate_color)
            source_witnesses.append(tuple(witness))

    request_checkpoint("before monochromatic profile construction")
    return MonochromaticCompleteSubhypergraphProfile(
        coloring=coloring,
        source_uniformity=admission.source_uniformity,
        target_uniformity=admission.target_uniformity,
        hypergraph=FiniteHypergraph(
            vertices=coloring.hypergraph.vertices,
            edges=tuple(target_edges),
        ),
        candidate_colors=tuple(candidate_colors),
        source_edge_witnesses=tuple(source_witnesses),
    )
