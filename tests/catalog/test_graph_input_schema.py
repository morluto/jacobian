"""Permissive graph request descriptions remain separate from canonical output."""

from collections.abc import Iterator
from typing import Annotated, Any, Literal, cast, get_args, get_origin

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, TypeAdapter

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.math.graphs._input import (
    ColoredUndirectedGraphInput,
    GraphValueInputEncoding,
    SimpleUndirectedGraphInput,
)
from jacobian.math.graphs.values import (
    MAX_ENCODED_SIMPLE_GRAPH_EDGES,
    IndexedSimpleUndirectedGraph,
    LoopedSimpleGraph,
    SimpleUndirectedGraph,
)


def _graph_encodings(
    annotation: Any,
    metadata: tuple[Any, ...] = (),
    path: tuple[str, ...] = (),
    seen: frozenset[type[BaseModel]] = frozenset(),
) -> Iterator[tuple[tuple[str, ...], bool]]:
    if get_origin(annotation) is Annotated:
        annotation, *extras = get_args(annotation)
        metadata = (*extras, *metadata)
    codec = next(
        (item for item in metadata if isinstance(item, GraphValueInputEncoding)), None
    )
    if annotation in (
        SimpleUndirectedGraph,
        IndexedSimpleUndirectedGraph,
        LoopedSimpleGraph,
    ):
        yield path, codec is not None
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        model = codec.input_type if codec is not None else annotation
        if model in seen:
            return
        for name, field in model.model_fields.items():
            yield from _graph_encodings(
                field.annotation, tuple(field.metadata), (*path, name), seen | {model}
            )
    else:
        for argument in get_args(annotation):
            yield from _graph_encodings(argument, path=(*path, "[]"), seen=seen)


def _resolve(schema: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    if "$ref" in value:
        return cast(dict[str, Any], schema["$defs"][value["$ref"].rsplit("/", 1)[1]])
    return value


@pytest.mark.parametrize(
    "operation_id",
    (
        "graph.cut.maximum.compute",
        "graph.invariant.triangle_count.compute",
        "graph.invariant.maximum_matching.compute",
        "graph.k_core.compute",
        "graph.distance_matrix.compute",
        "graph.invariant.chromatic_number.compute",
        "graph.edge_coloring.k_decide",
        "graph.edge_coloring.list_capacity.assign",
        "graph.coloring.chromatic_number.check",
    ),
)
def test_request_schema_explains_pair_orientation_without_changing_output(
    operation_id: str,
) -> None:
    operation = Catalog.open().operation(operation_id)
    assert operation is not None
    modes: tuple[Literal["validation", "serialization"], ...] = (
        "validation",
        "serialization",
    )
    for mode in modes:
        schema = operation.request_type.model_json_schema(mode=mode)
        Draft202012Validator.check_schema(schema)
        graph = _resolve(schema, schema["properties"]["graph"])
        edges = graph["properties"]["edges"]
        assert edges["maxItems"] <= MAX_ENCODED_SIMPLE_GRAPH_EDGES
        description = edges["description"]
        if mode == "validation":
            assert "either endpoint order" in description
            assert "list order are preserved" in description
            assert "reversed duplicates" in description
        else:
            assert "Each pair must have" in description
            assert "either endpoint order" not in description

    output = operation.result_type.model_json_schema(mode="serialization")
    if "graph" in output["properties"]:
        graph = _resolve(output, output["properties"]["graph"])
        assert "Each pair must have" in graph["properties"]["edges"]["description"]
        assert (
            "either endpoint order" not in graph["properties"]["edges"]["description"]
        )


def test_owned_input_schema_matches_the_reversed_pair_alternative() -> None:
    graph = {"vertices": ["2", "10", "11"], "edges": [["2", "10"], ["11", "10"]]}
    adapter = TypeAdapter(SimpleUndirectedGraphInput)
    schema = adapter.json_schema(mode="validation")
    Draft202012Validator(schema).validate(graph)
    assert schema["properties"]["edges"]["items"]["minItems"] == 2
    assert schema["properties"]["edges"]["items"]["maxItems"] == 2
    assert adapter.json_schema(mode="serialization")["title"] == "SimpleUndirectedGraph"


def test_colored_request_retains_its_distinct_label_and_color_limits() -> None:
    schema = TypeAdapter(ColoredUndirectedGraphInput).json_schema()
    assert "empty or aligned" in schema["description"]
    for field in ("graph", "vertex_colors", "edge_colors"):
        description = schema["properties"][field]["description"]
        assert "NFC" in description
        assert "64 UTF-8 bytes" in description
    # The uncolored graph carrier intentionally admits longer and empty labels.
    plain = TypeAdapter(SimpleUndirectedGraphInput).json_schema()
    assert "maxLength" not in plain["properties"]["vertices"]["items"]


def test_maximum_cut_publishes_an_executable_orientation_example() -> None:
    operation = Catalog.open().operation("graph.cut.maximum.compute")
    assert operation is not None
    example = next(
        example
        for example in operation.examples
        if example.name == "numeric_labels_unoriented_path"
    )
    Draft202012Validator(operation.request_type.model_json_schema()).validate(
        example.input
    )
    assert example.input["graph"]["vertices"] == ["2", "10", "11"]
    assert example.input["graph"]["edges"][0] == ["2", "10"]


# Only these source/card/decomposition evidence paths retain strict graph decoding.
# Keep exact operation/path pairs: a new ordinary graph operand must opt in rather
# than being silently excluded by its namespace or a broad model-family rule.
_STRICT_REQUEST_GRAPH_PATHS = {
    "graph.deck.anonymous.equal.decide": (
        "left.classes.[].representative",
        "right.classes.[].representative",
    ),
    "graph.deck.anonymous_vertex.edge_count.compute": ("classes.[].representative",),
    "graph.deck.edge.isomorphism_classes.compute": (
        "deck.source",
        "deck.cards.[].card",
    ),
    "graph.deck.isomorphism_classes.compute": ("deck.source", "deck.cards.[].card"),
    "graph.deck.unlabelled.compute": ("deck.source", "deck.cards.[].card"),
    "graph.deck.vertex.anonymous.compute": ("family.source", "family.cards.[].card"),
    "graph.deck.vertex.degree_multiset.compute": (
        "deck.family.source",
        "deck.family.cards.[].card",
        "deck.classes.[].representative",
    ),
    "graph.deck.vertex.edge_count.compute": (
        "deck.family.source",
        "deck.family.cards.[].card",
        "deck.classes.[].representative",
    ),
    "graph.deck.vertex.induced_subgraph_count.compute": (
        "deck.family.source",
        "deck.family.cards.[].card",
        "deck.classes.[].representative",
    ),
    "graph.deck.vertex.subgraph_count.compute": (
        "deck.family.source",
        "deck.family.cards.[].card",
        "deck.classes.[].representative",
    ),
    "graph.deck.vertex.unlabelled.compute": ("deck.source", "deck.cards.[].card"),
    "graph.tree_decomposition.adhesions.compute": ("decomposition.graph",),
    "graph.tree_decomposition.bag_intersection_graph.compute": ("decomposition.graph",),
    "graph.tree_decomposition.reroot.compute": ("decomposition.graph",),
    "graph.tree_decomposition.restrict.compute": ("decomposition.graph",),
    "graph.tree_decomposition.vertex_occurrences.compute": ("decomposition.graph",),
    "graph.tree_decomposition.width.compute": ("decomposition.graph",),
}


def test_catalog_graph_ingress_policy_is_complete_and_results_stay_canonical() -> None:
    actual_strict = {}
    for operation in BUILTIN_TOOLS:
        strict_paths = tuple(
            ".".join(path)
            for path, encoded in _graph_encodings(operation.request_type)
            if not encoded
        )
        if strict_paths:
            actual_strict[operation.operation_id] = strict_paths
        assert not any(
            encoded for _, encoded in _graph_encodings(operation.result_type)
        ), operation.operation_id
    assert actual_strict == _STRICT_REQUEST_GRAPH_PATHS
