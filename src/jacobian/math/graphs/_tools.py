"""Typed declarations for graph-owned operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictStr, model_validator

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.graph6 import (
    Graph6DecodeValue,
    decode_graph6,
)


class Graph6DecodeRequest(StrictModel):
    graph6: StrictStr = Field(min_length=1, max_length=352)

    @model_validator(mode="after")
    def require_valid_graph6(self) -> Self:
        """Validate the graph6 payload at the request boundary.

        This ensures that every accepted request returns a Graph6DecodeValue
        without a parser exception.  The full parsing and canonicalization is
        delegated to the maintained decode_graph6 path.
        """
        decode_graph6(self.graph6)
        return self


def _decode(request: Graph6DecodeRequest) -> Graph6DecodeValue:
    return decode_graph6(request.graph6)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.encoding.graph6.decode.compute",
        title="Decode canonical small-order graph6",
        description=(
            "Decode a headerless or standard-header graph6 string of order at "
            "most 62 using the column-major upper-triangle bit convention, "
            "returning sorted edges, degrees, and a canonical graph digest."
        ),
        request_type=Graph6DecodeRequest,
        result_type=Graph6DecodeValue,
        run=_decode,
        tags=("graph", "encoding", "graph6", "deterministic", "exact"),
        examples=(
            example(
                "triangle_graph6",
                "Decode the graph6 representation of the triangle graph.",
                {"graph6": "Bw"},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
