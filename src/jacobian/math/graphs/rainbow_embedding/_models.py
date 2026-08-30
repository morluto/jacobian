"""Typed contracts for the rainbow embedding profile operation."""

from typing import Annotated

from pydantic import WithJsonSchema
from pydantic.json_schema import JsonSchemaValue

from jacobian._models import StrictModel
from jacobian.math.graphs.values import ColoredUndirectedGraph, SimpleUndirectedGraph

MAX_HOST_VERTICES = 16
MAX_PATTERN_VERTICES = 8


def _bounded_graph_schema(
    graph_type: type[SimpleUndirectedGraph] | type[ColoredUndirectedGraph],
    *,
    maximum: int,
    description: str,
) -> JsonSchemaValue:
    schema = graph_type.model_json_schema()
    schema["description"] = description
    definition = schema.get("$defs", {}).get("SimpleUndirectedGraph")
    if definition is None:
        definition = schema
    definition["properties"]["vertices"]["maxItems"] = maximum
    if "SimpleUndirectedGraph" in schema.get("$defs", {}):
        schema["properties"]["graph"] = definition
        del schema["$defs"]
    return schema


RainbowPatternGraph = Annotated[
    SimpleUndirectedGraph,
    WithJsonSchema(
        _bounded_graph_schema(
            SimpleUndirectedGraph,
            maximum=MAX_PATTERN_VERTICES,
            description=(
                f"A pattern graph with at most {MAX_PATTERN_VERTICES} vertices."
            ),
        )
    ),
]
RainbowHostGraph = Annotated[
    ColoredUndirectedGraph,
    WithJsonSchema(
        _bounded_graph_schema(
            ColoredUndirectedGraph,
            maximum=MAX_HOST_VERTICES,
            description=(
                "A coloured host graph with at most "
                f"{MAX_HOST_VERTICES} vertices and a total edge coloring."
            ),
        )
    ),
]


class RainbowEmbeddingRequest(StrictModel):
    """Request for the rainbow subgraph embedding profile."""

    pattern: RainbowPatternGraph
    host: RainbowHostGraph


class EmbeddingWitness(StrictModel):
    """One rainbow embedding."""

    pattern_to_host: tuple[tuple[str, str], ...]
    edge_color_labels: tuple[str, ...]


class RainbowEmbeddingResult(StrictModel):
    """The complete rainbow subgraph embedding profile."""

    pattern: SimpleUndirectedGraph
    host: ColoredUndirectedGraph
    embeddings: tuple[EmbeddingWitness, ...]
    total_embeddings: int
    rainbow_count: int


__all__ = [
    "MAX_HOST_VERTICES",
    "MAX_PATTERN_VERTICES",
    "EmbeddingWitness",
    "RainbowEmbeddingRequest",
    "RainbowEmbeddingResult",
    "RainbowHostGraph",
    "RainbowPatternGraph",
]
