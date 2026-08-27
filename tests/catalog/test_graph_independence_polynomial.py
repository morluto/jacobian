"""Catalog conformance for the tree independence-polynomial operation."""

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDiscoveryRequest
from jacobian.dispatch import invoke_operation


def test_catalog_example_executes_as_a_source_bound_polynomial_profile() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("graph.polynomial.independence.compute")
    assert operation is not None
    assert len(operation.examples) == 1

    example = operation.examples[0]
    result = invoke_operation(operation.operation_id, example.input, catalog)

    assert example.name == "path_tree_p4"
    assert result.output["graph"] == example.input["graph"]
    assert result.output["coefficients"] == ["1", "4", "3"]
    assert result.output["independence_number"] == 2
    assert result.output["independent_set_count"] == "8"
    assert result.output["polynomial"]["variables"] == ["x"]
    assert result.output["polynomial"]["polynomial"]["terms"] == [
        {
            "coefficient": {"num": "3", "den": "1"},
            "exponents": [2],
        },
        {
            "coefficient": {"num": "4", "den": "1"},
            "exponents": [1],
        },
        {
            "coefficient": {"num": "1", "den": "1"},
            "exponents": [0],
        },
    ]


def test_catalog_search_discovers_independent_set_cardinality_distribution() -> None:
    result = Catalog.open().search(
        OperationDiscoveryRequest(
            query="count independent vertex sets by cardinality in a tree",
            namespace="graph",
            limit=10,
        )
    )

    assert "graph.polynomial.independence.compute" in {
        match.operation_id for match in result.matches
    }
