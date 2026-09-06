"""Typed contracts for the equitable k-colourability decision."""

from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_EQUITABLE_COLORING_SEARCH_NODES = 1_000_000


def _is_complete(graph: SimpleUndirectedGraph) -> bool:
    n = len(graph.vertices)
    return len(graph.edges) == n * (n - 1) // 2


class EquitableColoringRequest(StrictModel):
    """Request to decide equitable k-colourability."""

    graph: SimpleUndirectedGraph
    k: int = Field(gt=0)



class EquitableColoringAssignment(StrictModel):
    """Graph-indexed structural carrier for an equitable-coloring witness."""

    graph: SimpleUndirectedGraph
    k: int = Field(gt=0)
    coloring: tuple[StrictInt, ...]

    @model_validator(mode="after")
    def require_structural_assignment(self) -> Self:
        if len(self.coloring) != len(self.graph.vertices):
            raise PydanticCustomError(
                "graph.equitable_coloring_assignment_length",
                "coloring must assign one color per graph vertex",
            )
        if any(color < 0 or color >= self.k for color in self.coloring):
            raise PydanticCustomError(
                "graph.equitable_coloring_assignment_palette",
                "coloring values must lie in 0..k-1",
            )
        return self

    def __len__(self) -> int:
        return len(self.coloring)

    def __getitem__(self, index: int) -> int:
        return self.coloring[index]

    def count(self, value: int) -> int:
        return self.coloring.count(value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tuple):
            return self.coloring == other
        return super().__eq__(other)


class EquitableColoringResult(StrictModel):
    """The equitable k-colouring decision."""

    graph: SimpleUndirectedGraph
    k: int
    colorable: bool
    coloring: EquitableColoringAssignment | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_sequence_witness(cls, value: object) -> object:
        if isinstance(value, dict) and isinstance(value.get("coloring"), tuple):
            payload = dict(value)
            payload["coloring"] = {
                "graph": payload.get("graph"),
                "k": payload.get("k"),
                "coloring": payload["coloring"],
            }
            return payload
        return value

    @model_validator(mode="after")
    def require_structural_binding(self) -> Self:
        if self.colorable:
            if self.coloring is None:
                raise PydanticCustomError(
                    "graph.equitable_colorable_result_requires_coloring",
                    "a colorable result requires a coloring witness",
                )
            if self.coloring.graph != self.graph or self.coloring.k != self.k:
                raise PydanticCustomError(
                    "graph.equitable_coloring_witness_source_binding",
                    "coloring witness must bind the result graph and palette",
                )
        elif self.coloring is not None:
            raise PydanticCustomError(
                "graph.equitable_non_colorable_result_has_no_coloring",
                "a non-colorable result must not carry a coloring witness",
            )
        return self


__all__ = [
    "MAX_EQUITABLE_COLORING_SEARCH_NODES",
    "EquitableColoringAssignment",
    "EquitableColoringRequest",
    "EquitableColoringResult",
]
