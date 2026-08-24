"""Public catalog contract for canonical graph-homomorphism checks."""

from jacobian.catalog.catalog import Catalog


def test_graph_homomorphism_check_exposes_only_the_canonical_map_contract() -> None:
    descriptor = Catalog.open().inspect("graph.homomorphism.check")

    assert descriptor is not None
    assert descriptor.version == "2"
    assert set(descriptor.input_schema["properties"]) == {"vertex_map"}
    assert "SimpleGraph" not in descriptor.input_schema.get("$defs", {})
    assert {
        "status",
        "homomorphism",
        "obstruction",
    } == set(descriptor.output_schema["properties"])


def test_catalog_does_not_publish_retired_raw_graph_morphism_operations() -> None:
    catalog = Catalog.open()

    for operation_id in (
        "graph.homomorphism.find",
        "graph.core.check",
        "graph.retraction.check",
    ):
        assert catalog.operation(operation_id) is None
