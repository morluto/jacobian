"""Source-bound edge-colour-preserving subgraph decisions."""

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import ColoredUndirectedGraph


def require_edge_colors(value: ColoredUndirectedGraph) -> None:
    if (
        value.vertex_colors
        or not value.edge_colors
        or len(value.edge_colors) != len(value.graph.edges)
    ):
        raise PydanticCustomError(
            "graph.edge_colored_pattern.color_domain",
            "sources require a nonempty total edge coloring and no vertex coloring",
        )


class EdgeColoredPatternRequest(StrictModel):
    """An edge-color-preserving embedding request.

    Both sources require a nonempty total edge coloring (one color on every
    edge, with colors allowed to repeat) and the empty vertex coloring.
    """

    pattern: ColoredUndirectedGraph = Field(
        description=(
            "Pattern graph. The total edge coloring assigns one color to every "
            "edge, and vertex colors must be empty."
        ),
    )
    host: ColoredUndirectedGraph = Field(
        description=(
            "Host graph. The total edge coloring assigns one color to every "
            "edge, and vertex colors must be empty."
        ),
    )

    @model_validator(mode="after")
    def require_color_domain(self) -> Self:
        require_edge_colors(self.pattern)
        require_edge_colors(self.host)
        return self


class EdgeColoredPatternResult(StrictModel):
    pattern: ColoredUndirectedGraph
    host: ColoredUndirectedGraph
    decision: Literal["EXISTS", "DOES_NOT_EXIST"]
    vertex_map: tuple[str, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def require_witness_shape(self) -> Self:
        require_edge_colors(self.pattern)
        require_edge_colors(self.host)
        if self.decision == "DOES_NOT_EXIST":
            if self.vertex_map:
                raise ValueError("a negative decision cannot carry a witness")
        elif (
            len(self.vertex_map) != len(self.pattern.graph.vertices)
            or len(set(self.vertex_map)) != len(self.vertex_map)
            or not set(self.vertex_map) <= set(self.host.graph.vertices)
        ):
            raise ValueError(
                "an embedding must inject the complete pattern axis into the host"
            )
        return self
