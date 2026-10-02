"""Opt-in JSON request spelling for canonical undirected graph carriers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import (
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    TypeAdapter,
    field_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from jacobian._models import StrictModel
from jacobian.math.graphs.values import (
    MAX_ENCODED_SIMPLE_GRAPH_EDGES,
    MAX_INDEXED_SIMPLE_GRAPH_EDGES,
    ColoredUndirectedGraph,
    IndexedSimpleUndirectedGraph,
    LoopedSimpleGraph,
    SimpleUndirectedGraph,
)


def _orient_pairs[Endpoint: (str, int)](
    edges: tuple[tuple[Endpoint, Endpoint], ...],
) -> tuple[tuple[Endpoint, Endpoint], ...]:
    return tuple(
        (right, left) if left > right else (left, right) for left, right in edges
    )


class _UnorientedSimpleGraph(SimpleUndirectedGraph):
    """A labelled simple graph with unordered endpoint pairs at JSON ingress."""

    model_config = ConfigDict(title="SimpleUndirectedGraphInput")

    edges: tuple[tuple[str, str], ...] = Field(
        max_length=MAX_ENCODED_SIMPLE_GRAPH_EDGES,
        description=(
            "Unique pairs of distinct declared vertices. JSON requests accept "
            "either endpoint order; each pair is oriented left < right by "
            "lexicographic label order (Unicode code points). Vertex and edge "
            "list order are preserved, including any aligned data. Loops and "
            "duplicates, including reversed duplicates, are rejected."
        ),
    )

    @field_validator("edges")
    @classmethod
    def orient_bounded_edges(
        cls, edges: tuple[tuple[str, str], ...]
    ) -> tuple[tuple[str, str], ...]:
        # Pydantic checks the raw count, tuple shape and string endpoints first.
        # The inherited carrier validator then checks labels, membership,
        # loops and duplicates against the oriented pairs. Never sort an axis.
        return _orient_pairs(edges)


class _UnorientedIndexedGraph(IndexedSimpleUndirectedGraph):
    """An indexed simple graph with unordered endpoint pairs at JSON ingress."""

    model_config = ConfigDict(title="IndexedSimpleUndirectedGraphInput")

    edges: tuple[tuple[int, int], ...] = Field(
        max_length=MAX_INDEXED_SIMPLE_GRAPH_EDGES,
        description=(
            "Unique pairs of distinct indices in 0..vertex_count-1. JSON requests "
            "accept either endpoint order; each pair is oriented left < right. "
            "Vertex and edge list order are preserved, including any aligned "
            "data. Loops and duplicates, including reversed duplicates, are rejected."
        ),
    )

    @field_validator("edges")
    @classmethod
    def orient_bounded_edges(
        cls, edges: tuple[tuple[int, int], ...]
    ) -> tuple[tuple[int, int], ...]:
        return _orient_pairs(edges)


class _UnorientedLoopedGraph(LoopedSimpleGraph):
    """A looped simple graph with unordered off-diagonal pairs at JSON ingress."""

    model_config = ConfigDict(title="LoopedSimpleGraphInput")

    edges: tuple[tuple[str, str], ...] = Field(
        max_length=MAX_ENCODED_SIMPLE_GRAPH_EDGES,
        description=(
            "Unique off-diagonal pairs of declared vertices. JSON requests accept "
            "either endpoint order; each pair is oriented left < right by label. "
            "Vertex and edge list order are preserved, including any aligned "
            "data. Duplicates, including reversed duplicates, are rejected. "
            "Loops use the unchanged separate loops list."
        ),
    )

    @field_validator("edges")
    @classmethod
    def orient_bounded_edges(
        cls, edges: tuple[tuple[str, str], ...]
    ) -> tuple[tuple[str, str], ...]:
        return _orient_pairs(edges)


def _serialize_graph(value: StrictModel) -> StrictModel:
    return value


@dataclass(frozen=True)
class GraphValueInputEncoding:
    """Explicit JSON decoder for a graph value, retaining its canonical native type.

    The decoder subclasses its carrier so inherited invariants still apply.
    Only declared graph fields opt into endpoint normalization; no arbitrary
    dictionaries or generated JSON schemas are traversed or reinterpreted.
    """

    canonical_type: type[StrictModel]
    input_type: type[StrictModel]

    def _canonical_graph_value(self, value: StrictModel) -> StrictModel:
        # The inherited carrier validator established every canonical invariant.
        # Retain its native type and nested values without repeating those checks.
        result = self.canonical_type.model_construct(
            _fields_set=value.model_fields_set,
            **{name: getattr(value, name) for name in self.canonical_type.model_fields},
        )
        if value.__pydantic_private__ is not None:
            object.__setattr__(
                result, "__pydantic_private__", value.__pydantic_private__.copy()
            )
        return result

    def __get_pydantic_core_schema__(
        self, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        if source is not self.canonical_type or not issubclass(
            self.input_type, self.canonical_type
        ):
            raise TypeError(
                "graph input encoding requires its canonical carrier and a decoder subclass"
            )
        canonical = handler(source)
        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(
                self._canonical_graph_value, handler.generate_schema(self.input_type)
            ),
            python_schema=canonical,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize_graph, return_schema=canonical
            ),
        )


SimpleUndirectedGraphInput = Annotated[
    SimpleUndirectedGraph,
    GraphValueInputEncoding(SimpleUndirectedGraph, _UnorientedSimpleGraph),
]

IndexedSimpleUndirectedGraphInput = Annotated[
    IndexedSimpleUndirectedGraph,
    GraphValueInputEncoding(IndexedSimpleUndirectedGraph, _UnorientedIndexedGraph),
]

LoopedSimpleGraphInput = Annotated[
    LoopedSimpleGraph,
    GraphValueInputEncoding(LoopedSimpleGraph, _UnorientedLoopedGraph),
]


class _UnorientedColoredGraph(ColoredUndirectedGraph):
    """A colored graph retaining its authoritative vertex and edge color axes."""

    model_config = ConfigDict(
        title="ColoredUndirectedGraphInput",
        json_schema_extra={
            "description": (
                "A materialized simple undirected graph with optional total vertex "
                "and edge colorings. Each color tuple is empty or aligned with "
                "the complete corresponding axis. JSON requests normalize endpoint pairs; "
                "the vertex axis, edge axis and their color assignments are preserved."
            )
        },
    )

    graph: SimpleUndirectedGraphInput = Field(
        description=ColoredUndirectedGraph.model_fields["graph"].description
    )


ColoredUndirectedGraphInput = Annotated[
    ColoredUndirectedGraph,
    GraphValueInputEncoding(ColoredUndirectedGraph, _UnorientedColoredGraph),
]


def simple_graph_input_schema() -> JsonSchemaValue:
    """The carrier-owned request schema for owner-specific admission overlays."""

    return TypeAdapter(SimpleUndirectedGraphInput).json_schema(mode="validation")


def colored_graph_input_schema() -> JsonSchemaValue:
    """The colored request schema for owner-specific admission overlays."""

    return TypeAdapter(ColoredUndirectedGraphInput).json_schema(mode="validation")


def indexed_graph_input_schema() -> JsonSchemaValue:
    """The indexed request schema for owner-specific admission overlays."""

    return TypeAdapter(IndexedSimpleUndirectedGraphInput).json_schema(mode="validation")
