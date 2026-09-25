"""Deterministic greedy Morse matching and collapse transforms."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_DIMENSION,
    MAX_TOPOLOGY_FACES,
    FiniteSimplicialComplex,
    SimplicialComplexRequest,
    canonical_complex,
)
from jacobian.math.topology._request_admission import (
    require_canonical_complex_admission,
)
from jacobian.math.topology.discrete_morse._models import (
    DiscreteMorseMatchingResult,
    MatchingPair,
)
from jacobian.math.topology.discrete_morse.operations import construct_matching
from jacobian.math.topology.operations import canonicalize

MAX_COLLAPSE_SEQUENCE_STEPS = MAX_TOPOLOGY_FACES // 2
# Bound total face-set visits across all admitted sequence steps.
MAX_COLLAPSE_SEQUENCE_FACE_WORK = 128_000_000
MAX_GREEDY_COLLAPSE_WORK = 60_000_000


class GreedyMatchingRequest(StrictModel):
    complex: SimplicialComplexRequest


class GreedyCollapseRequest(StrictModel):
    """Request the lexicographically first maximal elementary collapse."""

    complex: SimplicialComplexRequest


class CollapseSequenceRequest(StrictModel):
    complex: SimplicialComplexRequest
    pairs: tuple[MatchingPair, ...] = Field(
        default=(), max_length=MAX_COLLAPSE_SEQUENCE_STEPS
    )


class CollapseSequenceResult(StrictModel):
    source: FiniteSimplicialComplex
    target: FiniteSimplicialComplex | None
    pairs: tuple[MatchingPair, ...]
    valid: bool
    collapsed_steps: int


def greedy_matching(request: GreedyMatchingRequest) -> DiscreteMorseMatchingResult:
    complex_ = canonicalize(request.complex.vertices, request.complex.facets).complex
    cells = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    pairs = []
    used = set()
    for coface in cells:
        if len(coface) < 2:
            continue
        for index in range(len(coface)):
            face = coface[:index] + coface[index + 1 :]
            if face in used or coface in used:
                continue
            pairs.append(MatchingPair(face=face, coface=coface))
            used |= {face, coface}
            break
    return construct_matching(complex_, tuple(pairs))


def _first_free_pair(
    facets: set[tuple[str, ...]],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Return the lexicographically first free face/facet pair."""
    candidates = sorted(
        face
        for coface in facets
        for position in range(len(coface))
        if (face := coface[:position] + coface[position + 1 :])
    )
    for face in candidates:
        containing = [facet for facet in facets if set(face).issubset(facet)]
        if len(containing) == 1 and len(containing[0]) == len(face) + 1:
            return face, containing[0]
    return None


def _remove_free_pair(
    faces: set[tuple[str, ...]],
    facets: set[tuple[str, ...]],
    face: tuple[str, ...],
    coface: tuple[str, ...],
) -> None:
    """Delete an elementary pair and expose only its surviving coface ridges."""
    facets.remove(coface)
    faces.remove(face)
    faces.remove(coface)
    for position in range(len(coface)):
        ridge = coface[:position] + coface[position + 1 :]
        if ridge == face:
            continue
        ridge_set = set(ridge)
        if not any(ridge_set.issubset(other) for other in facets):
            facets.add(ridge)


def greedy_collapse(complex_: FiniteSimplicialComplex) -> CollapseSequenceResult:
    """Return a canonical lexicographic sequence until no free pair remains.

    This is a deterministic maximal collapse, not a minimum-size result or a
    claim of noncollapsibility, contractibility, or any other homotopy theorem.
    """
    require_canonical_complex_admission(complex_)
    source = complex_
    max_steps = min(MAX_COLLAPSE_SEQUENCE_STEPS, source.closure_size // 2)
    # Per step: at most eight ridge-owner inserts, eight candidate reads, and
    # eight exposed-ridge containment scans over at most source.closure_size
    # facets. Add a linear allowance for the
    # final canonical value and matching-sized bookkeeping.
    work_bound = source.closure_size * (24 * max_steps + 256)
    if work_bound > MAX_GREEDY_COLLAPSE_WORK:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.greedy_collapse.admission.work",
            message=(
                f"the greedy collapse has a conservative work bound of "
                f"{work_bound}, above the {MAX_GREEDY_COLLAPSE_WORK}-unit envelope"
            ),
        )
    # Bound the result by its mathematical cardinalities: source/target face
    # families and matching pairs. Canonicalization also enforces the global
    # topology face and step limits.
    output_faces_bound = 3 * source.closure_size + 2 * len(source.vertices)
    output_pairs_bound = max_steps
    if (
        output_faces_bound > MAX_TOPOLOGY_FACES
        or output_pairs_bound > MAX_COLLAPSE_SEQUENCE_STEPS
    ):
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.greedy_collapse.admission.output_size",
            message="the greedy collapse result exceeds the admitted face or pair count",
        )

    faces = {face for degree in source.faces_by_dimension for face in degree.faces}
    facets = set(source.maximal_simplices)
    pairs: list[MatchingPair] = []
    while free_pair := _first_free_pair(facets):
        if len(pairs) == max_steps:
            raise RuntimeError(
                "elementary collapse exceeded its face-derived step bound"
            )
        face, coface = free_pair
        pairs.append(MatchingPair(face=face, coface=coface))
        _remove_free_pair(faces, facets, face, coface)

    closure = tuple(
        tuple(sorted(face for face in faces if len(face) == dimension + 1))
        for dimension in range(max(map(len, faces)))
    )
    vertices = tuple(sorted({vertex for face in faces for vertex in face}))
    target = canonical_complex(vertices, tuple(sorted(facets)), closure=closure)
    return CollapseSequenceResult(
        source=source,
        target=target,
        pairs=tuple(pairs),
        valid=True,
        collapsed_steps=len(pairs),
    )


def collapse_sequence(request: CollapseSequenceRequest) -> CollapseSequenceResult:
    source = canonicalize(request.complex.vertices, request.complex.facets).complex
    # Maintaining the face closure makes each step linear in the current face
    # count and source vertex count instead of rebuilding all subfaces of every
    # facet. Every intermediate is a subcomplex of the source, so its face
    # count cannot exceed source.closure_size.
    work_bound = source.closure_size * (
        (1 << (MAX_TOPOLOGY_DIMENSION + 1))
        + len(request.pairs) * (len(source.vertices) + MAX_TOPOLOGY_DIMENSION + 2)
    )
    if work_bound > MAX_COLLAPSE_SEQUENCE_FACE_WORK:
        raise OperationResourceAdmissionError(
            location=("pairs",),
            code="topology.collapse_sequence.admission.face_work",
            message=(
                f"the collapse sequence has a conservative face-expansion "
                f"bound of {work_bound}, above the "
                f"{MAX_COLLAPSE_SEQUENCE_FACE_WORK}-visit envelope"
            ),
        )
    faces = {face for degree in source.faces_by_dimension for face in degree.faces}
    facets = set(source.maximal_simplices)
    steps = 0
    for pair in request.pairs:
        face, coface = tuple(sorted(pair.face)), tuple(sorted(pair.coface))
        face_set, coface_set = set(face), set(coface)
        if (
            not face
            or not coface
            or len(coface) != len(face) + 1
            or len(face_set) != len(face)
            or len(coface_set) != len(coface)
            or not face_set.issubset(coface_set)
            or face not in faces
            or coface not in facets
        ):
            return CollapseSequenceResult(
                source=source,
                target=canonical_complex(
                    tuple(sorted({vertex for cell in faces for vertex in cell})),
                    tuple(sorted(facets)),
                    closure=tuple(
                        tuple(sorted(cell for cell in faces if len(cell) == dim + 1))
                        for dim in range(max(map(len, faces)))
                    ),
                ),
                pairs=request.pairs,
                valid=False,
                collapsed_steps=steps,
            )
        containing = tuple(facet for facet in facets if face_set.issubset(facet))
        if containing != (coface,):
            return CollapseSequenceResult(
                source=source,
                target=canonical_complex(
                    tuple(sorted({vertex for cell in faces for vertex in cell})),
                    tuple(sorted(facets)),
                    closure=tuple(
                        tuple(sorted(cell for cell in faces if len(cell) == dim + 1))
                        for dim in range(max(map(len, faces)))
                    ),
                ),
                pairs=request.pairs,
                valid=False,
                collapsed_steps=steps,
            )
        faces = {
            cell
            for cell in faces
            if not (face_set.issubset(cell) and set(cell).issubset(coface_set))
        }
        if not faces:
            return CollapseSequenceResult(
                source=source,
                target=None,
                pairs=request.pairs,
                valid=True,
                collapsed_steps=steps + 1,
            )
        vertices = {vertex for cell in faces for vertex in cell}
        facets = {
            cell
            for cell in faces
            if not any(
                tuple(sorted((*cell, vertex))) in faces
                for vertex in vertices.difference(cell)
            )
        }
        steps += 1
    remaining_vertices = tuple(sorted({vertex for face in faces for vertex in face}))
    return CollapseSequenceResult(
        source=source,
        target=canonical_complex(
            remaining_vertices,
            tuple(sorted(facets)),
            closure=tuple(
                tuple(sorted(cell for cell in faces if len(cell) == dim + 1))
                for dim in range(max(map(len, faces)))
            ),
        ),
        pairs=request.pairs,
        valid=True,
        collapsed_steps=steps,
    )


__all__ = [
    "CollapseSequenceRequest",
    "CollapseSequenceResult",
    "GreedyCollapseRequest",
    "GreedyMatchingRequest",
    "collapse_sequence",
    "greedy_collapse",
    "greedy_matching",
]
