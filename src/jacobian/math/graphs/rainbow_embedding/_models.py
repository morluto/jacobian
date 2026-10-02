"""Typed contracts for the rainbow embedding profile operation."""

from typing import Annotated

from pydantic import WithJsonSchema
from pydantic.json_schema import JsonSchemaValue

from jacobian._models import StrictModel
from jacobian.math.graphs._input import (
    ColoredUndirectedGraphInput,
    SimpleUndirectedGraphInput,
    colored_graph_input_schema,
    simple_graph_input_schema,
)
from jacobian.math.graphs.values import (
    MAX_SIMPLE_GRAPH_VERTICES,
    ColoredUndirectedGraph,
    SimpleUndirectedGraph,
)

MAX_HOST_VERTICES = MAX_SIMPLE_GRAPH_VERTICES
MAX_PATTERN_VERTICES = MAX_SIMPLE_GRAPH_VERTICES


def _bounded_graph_schema(
    graph_type: type[SimpleUndirectedGraph] | type[ColoredUndirectedGraph],
    *,
    maximum: int,
    description: str,
) -> JsonSchemaValue:
    schema = (
        colored_graph_input_schema()
        if graph_type is ColoredUndirectedGraph
        else simple_graph_input_schema()
    )
    schema["description"] = description
    definition = schema.get("$defs", {}).get("_UnorientedSimpleGraph")
    if definition is None:
        definition = schema
    definition["properties"]["vertices"]["maxItems"] = maximum
    if "_UnorientedSimpleGraph" in schema.get("$defs", {}):
        schema["properties"]["graph"] = {
            **definition,
            **{
                key: value
                for key, value in schema["properties"]["graph"].items()
                if key != "$ref"
            },
        }
        del schema["$defs"]
    return schema


RainbowPatternGraph = Annotated[
    SimpleUndirectedGraphInput,
    WithJsonSchema(
        _bounded_graph_schema(
            SimpleUndirectedGraph,
            maximum=MAX_PATTERN_VERTICES,
            description=(
                "A pattern graph within the canonical simple-graph vertex bound; "
                "admission uses exact work and retained-label bounds."
            ),
        ),
        mode="validation",
    ),
]
RainbowHostGraph = Annotated[
    ColoredUndirectedGraphInput,
    WithJsonSchema(
        _bounded_graph_schema(
            ColoredUndirectedGraph,
            maximum=MAX_HOST_VERTICES,
            description=(
                "A coloured host graph within the canonical simple-graph vertex "
                "bound and with a total edge coloring; admission uses exact work "
                "and retained-label bounds."
            ),
        ),
        mode="validation",
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
