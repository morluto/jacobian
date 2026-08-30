"""Typed contracts for the monochromatic path hypergraph operation."""

from typing import Annotated

from pydantic import WithJsonSchema
from pydantic.json_schema import JsonSchemaValue

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph

MAX_VERTICES = 12


def _monochromatic_graph_schema() -> JsonSchemaValue:
    schema = ColoredUndirectedGraph.model_json_schema()
    schema["description"] = (
        "A coloured simple graph with at most "
        f"{MAX_VERTICES} vertices; the operation bounds its exact subset "
        "search and complete hypergraph result."
    )
    definition = schema.get("$defs", {}).get("SimpleUndirectedGraph")
    if definition is None:
        raise AssertionError("colored graph schema lost its simple-graph definition")
    definition["properties"]["vertices"]["maxItems"] = MAX_VERTICES
    definition["properties"]["edges"]["maxItems"] = (
        MAX_VERTICES * (MAX_VERTICES - 1) // 2
    )
    schema["properties"]["graph"] = definition
    del schema["$defs"]
    schema["properties"]["edge_colors"]["maxItems"] = (
        MAX_VERTICES * (MAX_VERTICES - 1) // 2
    )
    return schema


MonochromaticPathGraph = Annotated[
    ColoredUndirectedGraph,
    WithJsonSchema(_monochromatic_graph_schema()),
]


class MonochromaticPathRequest(StrictModel):
    """Request for the monochromatic path hypergraphs of a coloured graph."""

    graph: MonochromaticPathGraph


class MonochromaticPathResult(StrictModel):
    """The monochromatic path hypergraphs of a coloured graph."""

    graph: ColoredUndirectedGraph
    colour_to_hypergraph: dict[str, FiniteHypergraph]


__all__ = [
    "MAX_VERTICES",
    "MonochromaticPathGraph",
    "MonochromaticPathRequest",
    "MonochromaticPathResult",
]
