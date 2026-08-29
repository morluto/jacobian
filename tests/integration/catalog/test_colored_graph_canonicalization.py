"""Caller-visible integration for colored-graph canonicalization."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.graphs import ColoredUndirectedGraph
from jacobian.math.graphs.isomorphism import (
    ColoredGraphCanonicalizationResult,
    canonicalize_colored_graph,
)


def test_public_catalog_invocation_returns_replayable_value() -> None:
    result = invoke_operation(
        "graph.isomorphism.canonicalize.compute",
        {
            "colored_graph": {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"]],
                }
            }
        },
        Catalog.open(),
    )

    validated = ColoredGraphCanonicalizationResult.model_validate(result.output)
    assert validated.canonical_graph.graph.edges == (
        ("v00", "v01"),
        ("v00", "v02"),
    )


def test_public_catalog_invocation_matches_the_native_typed_value() -> None:
    payload = {
        "colored_graph": {
            "graph": {
                "vertices": ["a", "b", "c", "d"],
                "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
            },
            "vertex_colors": ["endpoint", "middle", "middle", "endpoint"],
            "edge_colors": ["outer", "middle", "outer"],
        }
    }
    dispatched = invoke_operation(
        "graph.isomorphism.canonicalize.compute", payload, Catalog.open()
    )
    native_graph = ColoredUndirectedGraph.model_validate(payload["colored_graph"])

    assert ColoredGraphCanonicalizationResult.model_validate(
        dispatched.output
    ) == canonicalize_colored_graph(native_graph)
