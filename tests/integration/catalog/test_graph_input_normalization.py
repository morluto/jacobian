"""Every advertised ordinary undirected graph operand shares request ingress.

The inventory follows actual typed values, never arbitrary dictionaries or a
generated JSON schema. Canonical evidence families deliberately stay strict.
"""

import copy
from collections.abc import Iterator
from typing import Any, get_args

import pytest
from pydantic import BaseModel, ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.dispatch import invoke_operation
from jacobian.math.graphs.decks._models import (
    AnonymousGraphCardMultiset,
    EdgeDeletionFamily,
    UnlabelledVertexDeck,
    VertexDeletionFamily,
)
from jacobian.math.graphs.decomposition.tree_decompositions.values import (
    TreeDecomposition,
)
from jacobian.math.graphs.values import (
    IndexedSimpleUndirectedGraph,
    LoopedSimpleGraph,
    SimpleUndirectedGraph,
)

_GRAPH_VALUES = (SimpleUndirectedGraph, IndexedSimpleUndirectedGraph, LoopedSimpleGraph)
# These operands encode source/card or decomposition evidence rather than raw
# graph arguments. Keep their canonical representatives and retained sources
# strict, including when an ordinary pattern operand is present beside them.
_STRICT_EVIDENCE = {
    AnonymousGraphCardMultiset: "canonical isomorphism representatives and multiplicities",
    EdgeDeletionFamily: "source-bound edge deletion cards",
    VertexDeletionFamily: "source-bound vertex deletion cards",
    UnlabelledVertexDeck: "canonical card classes bound to a source family",
    TreeDecomposition: "source-bound decomposition evidence",
}


def _graph_paths(
    value: Any, path: tuple[str | int, ...] = (), *, strict: bool = False
) -> Iterator[tuple[tuple[str | int, ...], bool]]:
    if isinstance(value, _GRAPH_VALUES):
        if value.edges:
            yield path, strict
    elif isinstance(value, BaseModel):
        strict = strict or type(value) in _STRICT_EVIDENCE
        for name in type(value).model_fields:
            yield from _graph_paths(getattr(value, name), (*path, name), strict=strict)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _graph_paths(item, (*path, index), strict=strict)


def _contains_graph(
    annotation: Any, seen: frozenset[type[BaseModel]] = frozenset()
) -> bool:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation in _GRAPH_VALUES:
            return True
        if annotation in seen:
            return False
        return any(
            _contains_graph(field.annotation, seen | {annotation})
            for field in annotation.model_fields.values()
        )
    return any(_contains_graph(argument, seen) for argument in get_args(annotation))


def _cases() -> list[Any]:
    cases = []
    for operation in BUILTIN_TOOLS:
        # Discover by declared carrier, including future cross-domain consumers.
        if not _contains_graph(operation.request_type):
            continue
        for example in operation.examples:
            request = operation.request_type.model_validate_json(
                encode_strict_json(example.input), strict=True
            )
            paths = tuple(_graph_paths(request))
            if paths:
                cases.append(
                    pytest.param(
                        operation,
                        example,
                        paths,
                        id=f"{operation.operation_id}:{example.name}",
                    )
                )
    return cases


@pytest.mark.parametrize(("operation", "example", "paths"), _cases())
def test_advertised_graph_operands_normalize_and_evidence_stays_strict(
    operation: MathTool[Any, Any],
    example: OperationExample,
    paths: tuple[tuple[tuple[str | int, ...], bool], ...],
) -> None:
    canonical = operation.request_type.model_validate_json(
        encode_strict_json(example.input), strict=True
    )
    payload = canonical.model_dump(mode="json")
    normalized = copy.deepcopy(payload)
    ordinary_paths = []
    for path, strict in paths:
        variant = copy.deepcopy(payload) if strict else normalized
        graph = variant
        for component in path:
            graph = graph[component]
        graph["edges"] = [list(reversed(edge)) for edge in graph["edges"]]
        if strict:
            with pytest.raises(ValidationError):
                operation.request_type.model_validate_json(
                    encode_strict_json(variant), strict=True
                )
        else:
            ordinary_paths.append(path)

    if ordinary_paths:
        parsed = operation.request_type.model_validate_json(
            encode_strict_json(normalized), strict=True
        )
        # This equality retains every non-graph field, axis, label, bound and
        # nested native value. Both requests reach exactly the same kernel input.
        assert parsed == canonical
        output = invoke_operation(
            operation.operation_id, normalized, Catalog.open()
        ).output
        decoded = operation.result_type.model_validate_json(
            encode_strict_json(output), strict=True
        )
        assert decoded.model_dump(mode="json") == output
