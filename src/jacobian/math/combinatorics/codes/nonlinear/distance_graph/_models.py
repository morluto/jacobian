"""Typed contracts for the binary code distance graph operation."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, WithJsonSchema, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.codes.nonlinear.values import ExplicitBinaryCode
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph

MAX_CODE_SIZE = 256


def _distance_graph_code_schema() -> JsonSchemaValue:
    schema = ExplicitBinaryCode.model_json_schema()
    # ``WithJsonSchema`` installs this object as an inline field schema. Its
    # definitions are not merged into the enclosing model, so a reference
    # copied from the standalone value schema would make Pydantic's enclosing
    # schema walk fail with a missing ``BinaryWord`` definition.
    binary_word_schema = schema.pop("$defs")["BinaryWord"]
    schema["properties"]["codewords"]["items"] = binary_word_schema
    schema["description"] = (
        "An explicit binary code with at most "
        f"{MAX_CODE_SIZE} codewords for distance-graph construction."
    )
    schema["properties"]["codewords"].update(maxItems=MAX_CODE_SIZE)
    return schema


DistanceGraphCode = Annotated[
    ExplicitBinaryCode,
    WithJsonSchema(_distance_graph_code_schema()),
]


class BinaryCodeDistanceGraphRequest(StrictModel):
    """Request for the Hamming distance graph of a binary code."""

    source: DistanceGraphCode
    target_distance: int = Field(
        strict=True,
        ge=0,
        description=(
            "Nonnegative Hamming distance at most the source code length; "
            "the exact upper bound is validated against `source.length`."
        ),
    )

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.target_distance > self.source.length:
            raise PydanticCustomError(
                "code.distance_exceeds_length",
                "target_distance must not exceed code length",
            )
        if len(self.source.codewords) > MAX_CODE_SIZE:
            raise PydanticCustomError(
                "code.too_many_codewords",
                f"at most {MAX_CODE_SIZE} codewords are supported",
            )
        return self


class BinaryCodeDistanceGraphResult(StrictModel):
    """The Hamming distance graph of a binary code."""

    source: ExplicitBinaryCode
    target_distance: int
    graph: IndexedSimpleUndirectedGraph
    edge_count: int

    @model_validator(mode="after")
    def require_consistent_graph(self) -> Self:
        if self.target_distance < 0 or self.target_distance > self.source.length:
            raise PydanticCustomError(
                "code.distance_exceeds_length",
                "target_distance must be between zero and the code length",
            )
        if len(self.source.codewords) > MAX_CODE_SIZE:
            raise PydanticCustomError(
                "code.too_many_codewords",
                f"at most {MAX_CODE_SIZE} codewords are supported",
            )
        if self.graph.vertex_count != len(self.source.codewords):
            raise PydanticCustomError(
                "code.graph_vertex_count_matches_source",
                "graph vertex_count must equal the source codeword count",
            )
        if self.edge_count != len(self.graph.edges):
            raise PydanticCustomError(
                "code.edge_count_matches_graph",
                "edge_count must equal the graph edge count",
            )
        return self


__all__ = [
    "MAX_CODE_SIZE",
    "BinaryCodeDistanceGraphRequest",
    "BinaryCodeDistanceGraphResult",
    "DistanceGraphCode",
]
