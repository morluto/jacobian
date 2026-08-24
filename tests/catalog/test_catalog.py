from __future__ import annotations

import pytest

from jacobian.catalog import catalog as catalog_module
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDiscoveryRequest
from jacobian.catalog.search import browse_operations, discover_operations
from jacobian.dispatch import invoke_operation


def test_catalog_inspects_determinant_without_sqlite() -> None:
    catalog = Catalog.open()

    descriptor = catalog.inspect("matrix.determinant.compute")
    assert descriptor is not None
    assert descriptor.operation_id == "matrix.determinant.compute"


def test_every_served_operation_publishes_request_valid_examples() -> None:
    catalog = Catalog.open()

    for descriptor in catalog.snapshot().operations:
        operation = catalog.operation(descriptor.operation_id)
        assert operation is not None
        assert operation.examples, (
            f"{descriptor.operation_id} must publish an invocation example"
        )
        for invocation_example in operation.examples:
            operation.request_type.model_validate(invocation_example.input)


def test_invoke_operation_runs_determinant_without_state() -> None:
    catalog = Catalog.open()
    result = invoke_operation(
        "matrix.determinant.compute",
        {
            "matrix": {
                "matrix_schema_version": "1",
                "domain": "QQ",
                "entries": [
                    [{"num": "1", "den": "1"}, {"num": "2", "den": "1"}],
                    [{"num": "3", "den": "1"}, {"num": "4", "den": "1"}],
                ],
            }
        },
        catalog,
    )

    assert result.runtime_ms >= 0
    assert result.output is not None
    assert set(result.output) == {"determinant", "method"}
    assert result.output["determinant"] == {"num": "-2", "den": "1"}
    assert result.output["method"] == "FRACTION_FREE_BAREISS"


def test_invoke_operation_reports_unknown_removed_family_id() -> None:
    catalog = Catalog.open()
    with pytest.raises(ValueError, match="unknown operation"):
        invoke_operation(
            "graph.construct.explicit",
            {"vertices": ["a"], "edges": []},
            catalog,
        )


def test_compact_discovery_matches_full_descriptor_discovery() -> None:
    catalog = Catalog.open()
    descriptors = catalog.snapshot().operations
    operations = tuple(
        operation
        for descriptor in descriptors
        if (operation := catalog.operation(descriptor.operation_id)) is not None
    )
    request = OperationDiscoveryRequest(query="matrix determinant", limit=2)

    expected_search = discover_operations(descriptors, request)
    assert discover_operations(operations, request) == expected_search
    if expected_search.next_cursor is not None:
        next_request = request.model_copy(
            update={"cursor": expected_search.next_cursor}
        )
        assert discover_operations(operations, next_request) == discover_operations(
            descriptors, next_request
        )

    expected_browse = browse_operations(
        descriptors, domain="matrix", limit=2, cursor=None
    )
    assert (
        browse_operations(operations, domain="matrix", limit=2, cursor=None)
        == expected_browse
    )
    if expected_browse.next_cursor is not None:
        assert browse_operations(
            operations,
            domain="matrix",
            limit=2,
            cursor=expected_browse.next_cursor,
        ) == browse_operations(
            descriptors,
            domain="matrix",
            limit=2,
            cursor=expected_browse.next_cursor,
        )


def test_search_and_browse_do_not_materialize_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = Catalog.open()

    def fail_descriptor(*_args: object) -> None:
        raise AssertionError("compact discovery must not construct full descriptors")

    monkeypatch.setattr(catalog_module, "_descriptor", fail_descriptor)

    search = catalog.search(OperationDiscoveryRequest(query="matrix", limit=2))
    browse = catalog.browse(domain="matrix", limit=2, cursor=None)

    assert search.matches
    assert browse.operations


def test_natural_prime_power_query_ranks_factorization_before_prime_navigation() -> (
    None
):
    catalog = Catalog.open()

    result = catalog.search(
        OperationDiscoveryRequest(
            query="factor an integer into prime powers",
            limit=5,
        )
    )
    positions = {
        match.operation_id: index for index, match in enumerate(result.matches)
    }

    assert all(
        positions["integer.compute.prime_factorization"]
        < positions[prime_navigation_id]
        for prime_navigation_id in (
            "integer.compute.next_prime",
            "integer.compute.nth_prime",
            "integer.compute.previous_prime",
        )
    )


def test_natural_powerful_number_query_finds_bounded_decision() -> None:
    catalog = Catalog.open()

    result = catalog.search(
        OperationDiscoveryRequest(
            query="decide whether an integer is 2-full or powerful",
            limit=5,
        )
    )

    assert "integer.decide.powerful" in {match.operation_id for match in result.matches}


def test_catalog_runs_source_bound_powerful_decision() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("integer.decide.powerful")
    assert operation is not None
    assert operation.version == "3"

    result = invoke_operation(
        "integer.decide.powerful",
        {"value": "12168"},
        catalog,
    )

    assert result.output is not None
    assert result.output["is_powerful"] is True
    assert result.output["value"] == "12168"


def test_search_finds_lattice_hnf_in_matrix_domain() -> None:
    catalog = Catalog.open()

    result = catalog.search(
        OperationDiscoveryRequest(
            query="row Hermite normal form",
            domain="matrix",
            limit=10,
        )
    )

    assert "lattice.hermite_normal_form.compute" in {
        match.operation_id for match in result.matches
    }


def test_browse_includes_lattice_hnf_in_matrix_domain() -> None:
    catalog = Catalog.open()

    result = catalog.browse(domain="matrix", limit=100, cursor=None)

    assert "lattice.hermite_normal_form.compute" in {
        operation.operation_id for operation in result.operations
    }
