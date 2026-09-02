"""Public catalog contract for canonical graph-homomorphism checks."""

from jacobian.catalog.catalog import Catalog


def test_graph_homomorphism_check_exposes_only_the_canonical_map_contract() -> None:
    descriptor = Catalog.open().inspect("graph.homomorphism.check")

    assert descriptor is not None
    assert set(descriptor.input_schema["properties"]) == {"vertex_map"}
    assert "SimpleGraph" not in descriptor.input_schema.get("$defs", {})
    assert {
        "status",
        "homomorphism",
        "obstruction",
    } == set(descriptor.output_schema["properties"])
