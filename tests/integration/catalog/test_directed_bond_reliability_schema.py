"""Public schema guidance for directed bond reliability."""

from jacobian.catalog.catalog import Catalog

OPERATION_ID = "probability.digraph_bond_reliability.connection_probability.compute"


def test_directed_bond_reliability_schema_exposes_cross_field_contract() -> None:
    operation = Catalog.open().operation(OPERATION_ID)
    assert operation is not None

    schema = operation.request_type.model_json_schema()
    assert schema["description"] == (
        "Compute directed source-to-target bond connection probability.\n\n"
        "The request admits at most 16 vertices and 12 arcs, requires one\n"
        "probability for every arc exactly once, and requires distinct declared\n"
        "source and target vertices.  It normalizes arc rows lexicographically, so\n"
        "state indices and the source-bound result do not depend on input order."
    )
    properties = schema["properties"]
    assert "at most 16 vertices and 12 arcs" in properties["graph"]["description"]
    assert (
        "every graph arc exactly once" in properties["arc_probabilities"]["description"]
    )
    assert (
        "normalized to lexicographic arc order"
        in properties["arc_probabilities"]["description"]
    )
    assert "distinct from target" in properties["source"]["description"]
    assert "distinct from source" in properties["target"]["description"]
