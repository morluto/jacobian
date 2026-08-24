"""Caller-visible integration for colored-graph canonicalization."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.graphs.isomorphism import ColoredGraphCanonicalizationResult


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
