"""Typed contracts for the rainbow embedding profile operation."""

from typing import Annotated

from pydantic import WithJsonSchema
from pydantic.json_schema import JsonSchemaValue

from jacobian._models import StrictModel
from jacobian.math.graphs.values import (
    MAX_INDEXED_SIMPLE_GRAPH_VERTICES,
    ColoredUndirectedGraph,
    SimpleUndirectedGraph,
)

MAX_HOST_VERTICES = MAX_INDEXED_SIMPLE_GRAPH_VERTICES
MAX_PATTERN_VERTICES = MAX_INDEXED_SIMPLE_GRAPH_VERTICES


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
                "A pattern graph within the canonical simple-graph vertex bound; "
                "admission uses exact work and retained-label bounds."
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
                "A coloured host graph within the canonical simple-graph vertex "
                "bound and with a total edge coloring; admission uses exact work "
                "and retained-label bounds."
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
