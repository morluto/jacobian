"""Typed contracts for hypergraph non-monochromatic colouring decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_VERTICES,
    FiniteHypergraph,
)

MAX_VERTEX_COUNT = MAX_VERTICES
MAX_EDGE_COUNT = MAX_EDGES
# Palette sizes are emitted as JSON integers, so use the interoperable integer
# range rather than tying a cheap injective presolve to the vertex count.
MAX_PALETTE_SIZE = (1 << 53) - 1
MAX_COLORING_WORK = 2_000_000

ColoringResult = Literal["COLORABLE", "NOT_COLORABLE"]


@dataclass(frozen=True)
class _ColoringAdmission:
    """Request-scoped facts established while admitting a coloring request."""

    has_forced_failure: bool
    has_injective_witness: bool


def _validate_coloring_envelope(
    hypergraph: FiniteHypergraph, palette_size: int
) -> _ColoringAdmission:
    """Validate the complete request and retained-result envelope."""
    vertex_count = len(hypergraph.vertices)
    edge_count = len(hypergraph.edges)
    if not isinstance(palette_size, int) or isinstance(palette_size, bool):
        raise PydanticCustomError(
            "hypergraph_coloring.palette_type", "palette_size must be an integer"
        )
    if not 1 <= palette_size <= MAX_PALETTE_SIZE:
        raise PydanticCustomError(
            "hypergraph_coloring.palette_out_of_range",
            f"palette_size must be between 1 and {MAX_PALETTE_SIZE}",
        )
    if vertex_count > MAX_VERTEX_COUNT:
        raise PydanticCustomError(
            "hypergraph_coloring.too_many_vertices",
            f"at most {MAX_VERTEX_COUNT} vertices are supported",
        )
    if edge_count > MAX_EDGE_COUNT:
        raise PydanticCustomError(
            "hypergraph_coloring.too_many_edges",
            f"at most {MAX_EDGE_COUNT} edges are supported",
        )
    has_forced_failure = any(len(members) <= 1 for _, members in hypergraph.edges)
    has_injective_witness = not has_forced_failure and palette_size >= vertex_count
    work = (
        0
        if has_forced_failure or has_injective_witness
        else palette_size**vertex_count * edge_count
    )
    if work > MAX_COLORING_WORK:
        raise PydanticCustomError(
            "hypergraph_coloring.work_too_large",
            f"coloring search requires at most {MAX_COLORING_WORK} edge checks",
        )

    # A COLORABLE result retains the source hypergraph and one assignment per
    # vertex. This is the largest result shape produced by the kernel.
    payload = {
        "hypergraph": hypergraph.model_dump(mode="json"),
        "palette_size": palette_size,
        "outcome": "NOT_COLORABLE" if has_forced_failure else "COLORABLE",
    }
    if not has_forced_failure:
        payload["witness"] = {
            "assignments": [
                [vertex, index] for index, vertex in enumerate(hypergraph.vertices)
            ]
        }
    else:
        payload["witness"] = None
    try:
        result_bytes = len(encode_strict_json(payload))
    except CanonicalizationError as exc:
        raise PydanticCustomError(
            "hypergraph_coloring.result_too_large",
            "the retained coloring result exceeds the canonical output limit",
        ) from exc
    if result_bytes > CanonicalLimits().max_output_bytes:
        raise PydanticCustomError(
            "hypergraph_coloring.result_too_large",
            "the retained coloring result exceeds the canonical output limit",
        )
    return _ColoringAdmission(
        has_forced_failure=has_forced_failure,
        has_injective_witness=has_injective_witness,
    )


class NonmonochromaticColoringRequest(StrictModel):
    """Decide whether a hypergraph has a q-colouring with no monochromatic edge."""

    hypergraph: FiniteHypergraph
    palette_size: StrictInt = Field(ge=1, le=MAX_PALETTE_SIZE)


class ColoringWitness(StrictModel):
    """A complete vertex-to-colour assignment."""

    assignments: tuple[tuple[str, int], ...]


class NonmonochromaticColoringResult(StrictModel):
    """Result of a non-monochromatic colouring decision."""

    hypergraph: FiniteHypergraph
    palette_size: int
    outcome: ColoringResult
    witness: ColoringWitness | None = None

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        if self.outcome == "COLORABLE" and self.witness is None:
            raise PydanticCustomError(
                "hypergraph_coloring.colorable_requires_witness",
                "COLORABLE results must carry a witness",
            )
        if self.outcome == "NOT_COLORABLE" and self.witness is not None:
            raise PydanticCustomError(
                "hypergraph_coloring.not_colorable_has_no_witness",
                "NOT_COLORABLE results must not carry a witness",
            )
        return self


__all__ = [
    "MAX_COLORING_WORK",
    "MAX_EDGE_COUNT",
    "MAX_PALETTE_SIZE",
    "MAX_VERTEX_COUNT",
    "ColoringWitness",
    "NonmonochromaticColoringRequest",
    "NonmonochromaticColoringResult",
    "_validate_coloring_envelope",
]
