"""Typed contracts for edge-deletion diameter profile."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, WithJsonSchema, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_EDGE_DELETION_DIAMETER_VERTICES = 64
MAX_EDGE_DELETION_DIAMETER_EDGES = 256


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"graph.{reason}", message)


def _graph_schema() -> JsonSchemaValue:
    schema = SimpleUndirectedGraph.model_json_schema()
    schema["description"] = (
        "A nonempty connected simple graph. The diameter profile requires "
        "a connected graph; admission bounds vertices, edges, and aggregate "
        "BFS work."
    )
    return schema


DiameterGraph = Annotated[
    SimpleUndirectedGraph,
    WithJsonSchema(_graph_schema()),
]


class EdgeDeletionDiameterProfileRequest(StrictModel):
    """Complete diameter response to each single-edge deletion."""

    graph: DiameterGraph

    @model_validator(mode="before")
    @classmethod
    def bound_raw(cls, value: object) -> object:
        return canonicalize_json_containers(value)


class EdgeDeletionDiameterEntry(StrictModel):
    """Diameter after deleting one source edge, or disconnected."""

    edge: tuple[str, str] = Field(description="Source edge in graph.edges order, sorted lexicographically.")
    edge_index: int = Field(ge=0, description="Index into graph.edges.")
    result: Literal["DIAMETER", "DISCONNECTED"] = Field(description="DIAMETER if G-e remains connected, else DISCONNECTED.")
    diameter: int | None = Field(default=None, ge=1, description="Exact diameter of G-e when connected.")

    @model_validator(mode="after")
    def require_discriminated(self) -> Self:
        if self.result == "DIAMETER":
            if self.diameter is None:
                raise _validation_error("diameter_missing", "DIAMETER must have diameter")
        else:
            if self.diameter is not None:
                raise _validation_error("disconnected_diameter", "DISCONNECTED must have no diameter")
        return self


class EdgeDeletionDiameterProfileResult(StrictModel):
    """Source diameter and per-edge deletion diameters."""

    graph: SimpleUndirectedGraph
    source_diameter: int = Field(ge=1, description="Diameter of the original connected graph.")
    entries: tuple[EdgeDeletionDiameterEntry, ...] = Field(
        description="One entry per source edge, in graph.edges order."
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw(cls, value: object) -> object:
        return canonicalize_json_containers(value)

    @model_validator(mode="after")
    def require_complete(self) -> Self:
        if len(self.entries) != len(self.graph.edges):
            raise _validation_error(
                "edge_count_mismatch", "entries must have one per source edge"
            )
        for idx, entry in enumerate(self.entries):
            if entry.edge_index != idx:
                raise _validation_error("edge_index_order", "entries must be in graph.edges order")
            if tuple(entry.edge) != tuple(sorted(entry.edge)):
                raise _validation_error("edge_sorted", "edge must be lexicographically sorted")
            # Check edge matches graph.edges
            expected = tuple(sorted(self.graph.edges[idx]))
            if tuple(entry.edge) != expected:
                raise _validation_error(
                    "edge_mismatch", f"entry edge {entry.edge} does not match graph.edges[{idx}] {expected}"
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        graph: SimpleUndirectedGraph,
        source_diameter: int,
        entries: tuple[EdgeDeletionDiameterEntry, ...],
    ) -> Self:
        return cls.model_construct(graph=graph, source_diameter=source_diameter, entries=entries)


__all__ = [
    "EdgeDeletionDiameterEntry",
    "EdgeDeletionDiameterProfileRequest",
    "EdgeDeletionDiameterProfileResult",
]
