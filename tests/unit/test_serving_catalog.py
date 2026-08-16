from __future__ import annotations

import pytest

from jacobian import serving_catalog as serving_catalog_module
from jacobian.contracts.operations import OperationDiscoveryRequest
from jacobian.operation_discovery import browse_operations, discover_operations
from jacobian.operation_dispatcher import invoke_operation
from jacobian.serving_catalog import ServingCatalog


def test_serving_catalog_inspects_determinant_without_sqlite() -> None:
    catalog = ServingCatalog.open()

    descriptor = catalog.inspect("matrix.determinant.compute")
    assert descriptor is not None
    assert descriptor.operation_id == "matrix.determinant.compute"


def test_invoke_operation_runs_determinant_without_state() -> None:
    catalog = ServingCatalog.open()
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
    assert result.output["determinant"] == {"num": "-2", "den": "1"}


def test_invoke_operation_reports_unknown_removed_family_id() -> None:
    catalog = ServingCatalog.open()
    with pytest.raises(ValueError, match="unknown operation"):
        invoke_operation(
            "graph.construct.explicit",
            {"vertices": ["a"], "edges": []},
            catalog,
        )


def test_compact_discovery_matches_full_descriptor_discovery() -> None:
    catalog = ServingCatalog.open()
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
    catalog = ServingCatalog.open()

    def fail_descriptor(*_args: object) -> None:
        raise AssertionError("compact discovery must not construct full descriptors")

    monkeypatch.setattr(serving_catalog_module, "_descriptor", fail_descriptor)

    search = catalog.search(OperationDiscoveryRequest(query="matrix", limit=2))
    browse = catalog.browse(domain="matrix", limit=2, cursor=None)

    assert search.matches
    assert browse.operations


def test_search_finds_lattice_hnf_in_matrix_domain() -> None:
    catalog = ServingCatalog.open()

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
    catalog = ServingCatalog.open()

    result = catalog.browse(domain="matrix", limit=100, cursor=None)

    assert "lattice.hermite_normal_form.compute" in {
        operation.operation_id for operation in result.operations
    }
