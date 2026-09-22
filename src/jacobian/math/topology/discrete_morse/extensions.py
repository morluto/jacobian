"""Deterministic greedy Morse matching and collapse transforms."""

from __future__ import annotations

from pydantic import Field, ValidationError

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology._models import (
    FiniteSimplicialComplex,
    SimplicialComplexRequest,
)
from jacobian.math.topology._structural import (
    ElementaryCollapseRequest,
    compute_elementary_collapse,
)
from jacobian.math.topology.discrete_morse._models import (
    DiscreteMorseMatchingResult,
    MatchingPair,
)
from jacobian.math.topology.discrete_morse.operations import construct_matching
from jacobian.math.topology.operations import canonicalize


class GreedyMatchingRequest(StrictModel):
    complex: SimplicialComplexRequest


class CollapseSequenceRequest(StrictModel):
    complex: SimplicialComplexRequest
    pairs: tuple[MatchingPair, ...] = Field(default=())


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


def collapse_sequence(request: CollapseSequenceRequest) -> CollapseSequenceResult:
    current = request.complex
    source = canonicalize(current.vertices, current.facets).complex
    steps = 0
    for pair in request.pairs:
        try:
            elementary_request = ElementaryCollapseRequest(
                complex=current, free_face=pair.face, coface=pair.coface
            )
        except ValidationError as exc:
            raise OperationDomainValidationError(
                location=("pairs", steps),
                code="topology.collapse_sequence.pair_invalid",
                message=str(exc),
            ) from exc
        result = compute_elementary_collapse(elementary_request)
        if not result.is_free_face:
            return CollapseSequenceResult(
                source=source,
                target=canonicalize(current.vertices, current.facets).complex,
                pairs=request.pairs,
                valid=False,
                collapsed_steps=steps,
            )
        if result.remaining_complex is None:
            return CollapseSequenceResult(
                source=source,
                target=None,
                pairs=request.pairs,
                valid=True,
                collapsed_steps=steps + 1,
            )
        current = SimplicialComplexRequest(
            vertices=result.remaining_complex.vertices,
            facets=result.remaining_complex.maximal_simplices,
        )
        steps += 1
    return CollapseSequenceResult(
        source=source,
        target=canonicalize(current.vertices, current.facets).complex,
        pairs=request.pairs,
        valid=True,
        collapsed_steps=steps,
    )


__all__ = [
    "CollapseSequenceRequest",
    "CollapseSequenceResult",
    "GreedyMatchingRequest",
    "collapse_sequence",
    "greedy_matching",
]
