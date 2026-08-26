"""Public schema guidance for directed bond reliability."""

from jacobian.catalog.catalog import Catalog
from jacobian.math.probability._models import (
    MAX_DIRECTED_BOND_RELIABILITY_ARCS,
    MAX_DIRECTED_BOND_RELIABILITY_DECLARED_VERTICES,
)

OPERATION_ID = "probability.digraph_bond_reliability.connection_probability.compute"


def test_directed_bond_reliability_schema_exposes_cross_field_contract() -> None:
    operation = Catalog.open().operation(OPERATION_ID)
    assert operation is not None

    schema = operation.request_type.model_json_schema()
    assert schema["description"] == (
        "Compute directed source-to-target bond connection probability.\n\n"
        f"The request admits at most {MAX_DIRECTED_BOND_RELIABILITY_ARCS} arcs, requires one probability for every\n"
        "arc exactly once (empty for an edgeless graph), and requires distinct\n"
        "declared source and target vertices.  Traversal work depends only on\n"
        "relevant vertices (terminals plus arc endpoints), so sparse graphs may\n"
        "declare very large vertex counts.  It normalizes arc rows\n"
        "lexicographically, so state indices and the source-bound result do not\n"
        "depend on input order."
    )
    properties = schema["properties"]
    assert (
        f"at most {MAX_DIRECTED_BOND_RELIABILITY_ARCS} arcs"
        in properties["graph"]["description"]
    )
    assert (
        properties["graph"]["properties"]["edges"]["maxItems"]
        == MAX_DIRECTED_BOND_RELIABILITY_ARCS
    )
    assert (
        properties["graph"]["properties"]["vertex_count"]["maximum"]
        == MAX_DIRECTED_BOND_RELIABILITY_DECLARED_VERTICES
    )
    assert (
        "every graph arc exactly once" in properties["arc_probabilities"]["description"]
    )
    assert (
        "normalized to lexicographic arc order"
        in properties["arc_probabilities"]["description"]
    )
    assert "distinct from target" in properties["source"]["description"]
    assert "distinct from source" in properties["target"]["description"]
