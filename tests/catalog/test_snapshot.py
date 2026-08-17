"""Observable catalog inventory frozen across ownership-only refactors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDiscoveryRequest


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def test_operation_ids_and_request_result_schemas_match_snapshot() -> None:
    expected = json.loads(
        Path(__file__)
        .with_name("operation-schema-snapshot.json")
        .read_text(encoding="utf-8")
    )
    catalog = Catalog.open()
    snapshot = catalog.snapshot()
    actual = {
        descriptor.operation_id: {
            "input_schema": _digest(descriptor.input_schema),
            "output_schema": _digest(descriptor.output_schema),
        }
        for descriptor in snapshot.operations
    }

    assert snapshot.catalog_version == expected["catalog_version"]
    assert len(actual) == 336
    assert actual == expected["operations"]


def test_catalog_rejects_duplicate_tool_ids() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("integer.compute.gcd")
    assert operation is not None

    with pytest.raises(ValueError, match="duplicate built-in operation ID"):
        Catalog((operation, operation))


def test_each_tool_contract_and_function_have_one_math_owner() -> None:
    for operation in BUILTIN_TOOLS:
        modules = {
            operation.request_type.__module__,
            operation.result_type.__module__,
            operation.run.__module__,
        }
        non_math_modules = {
            module for module in modules if not module.startswith("jacobian.math.")
        }
        owners = {
            module.removeprefix("jacobian.math.").split(".", 1)[0]
            for module in modules
            if module.startswith("jacobian.math.")
        }

        assert not non_math_modules, (
            f"{operation.operation_id} has non-math owners: {sorted(non_math_modules)}"
        )
        assert len(owners) == 1, (
            f"{operation.operation_id} spans mathematical owners: {sorted(modules)}"
        )


def test_representative_search_browse_and_inspect_results_are_stable() -> None:
    catalog = Catalog.open()
    search = catalog.search(
        OperationDiscoveryRequest(query="finite field factorization", limit=5)
    )
    browse = catalog.browse(domain="graph", limit=5, cursor=None)
    inspected = catalog.inspect("integer.compute.gcd")

    assert [match.operation_id for match in search.matches] == [
        "finite_field.restrict_scalars.compute",
        "finite_abelian_group.exact_factorization.compute",
        "finite_field.linear_map.rank.compute",
        "finite_field.polynomial_map.table.compute",
        "finite_field.direction_rank_ledger.compute",
    ]
    assert [operation.operation_id for operation in browse.operations] == [
        "graph.coloring.k_colorability.decide",
        "graph.cut.minimum_st.compute",
        "graph.decomposition.biconnected_components.compute",
        "graph.decomposition.block_cut_tree.compute",
        "graph.decomposition.bridge_block_tree.compute",
    ]
    assert browse.total_operations == 48
    assert inspected is not None
    assert inspected.operation_id == "integer.compute.gcd"
    assert inspected.version == "2"
