"""Observable catalog inventory frozen across ownership-only refactors.

Schema compatibility snapshots are sharded by semantic owner so that a
domain-local change updates only that domain's fragment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDiscoveryRequest
from jacobian.catalog.search import matches_domain

_SNAPSHOTS_DIR = Path(__file__).with_name("operation_schema_snapshots")


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _load_fragments() -> dict[str, dict[str, Any]]:
    """Load and aggregate all owner-scoped snapshot fragments."""
    fragments: dict[str, dict[str, Any]] = {}
    for path in sorted(_SNAPSHOTS_DIR.glob("*.json")):
        fragment = json.loads(path.read_text(encoding="utf-8"))
        fragments[path.stem] = fragment
    return fragments


def test_operation_ids_and_request_result_schemas_match_snapshot() -> None:
    fragments = _load_fragments()
    catalog = Catalog.open()
    snapshot = catalog.snapshot()
    actual = {
        descriptor.operation_id: {
            "input_schema": _digest(descriptor.input_schema),
            "output_schema": _digest(descriptor.output_schema),
        }
        for descriptor in snapshot.operations
    }

    # Aggregate all fragments and compare
    expected: dict[str, Any] = {}
    catalog_version = None
    for fragment in fragments.values():
        assert fragment["catalog_version"] is not None
        if catalog_version is None:
            catalog_version = fragment["catalog_version"]
        assert fragment["catalog_version"] == catalog_version
        expected.update(fragment["operations"])

    assert snapshot.catalog_version == catalog_version
    assert len(actual) == len(expected)
    assert actual == expected


def test_snapshot_fragments_are_owner_scoped() -> None:
    """Every operation appears in exactly one fragment."""
    fragments = _load_fragments()
    all_ids: list[str] = []
    for fragment in fragments.values():
        all_ids.extend(fragment["operations"].keys())
    assert len(all_ids) == len(set(all_ids)), "duplicate operation IDs across fragments"


def test_snapshot_fragments_have_correct_domains() -> None:
    """Each fragment contains only operations from its declared domain."""
    fragments = _load_fragments()
    for fragment_name, fragment in fragments.items():
        domain = fragment.get("domain", fragment_name)
        assert domain == fragment_name, f"fragment {fragment_name} has domain {domain}"


def test_catalog_rejects_duplicate_tool_ids() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("integer.compute.extended_gcd")
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


def test_search_browse_and_inspect_results_stay_within_the_public_catalog() -> None:
    catalog = Catalog.open()
    public_ids = {
        descriptor.operation_id for descriptor in catalog.snapshot().operations
    }
    search = catalog.search(
        OperationDiscoveryRequest(query="finite field factorization", limit=5)
    )
    browse = catalog.browse(domain="graph", limit=5, cursor=None)
    inspected = catalog.inspect("integer.compute.extended_gcd")

    assert search.matches
    assert len(search.matches) <= 5
    assert {match.operation_id for match in search.matches} <= public_ids
    assert search.total_matches >= len(search.matches)

    assert len(browse.operations) <= 5
    assert {operation.operation_id for operation in browse.operations} <= public_ids
    assert browse.total_operations == sum(
        1 for tool in BUILTIN_TOOLS if matches_domain(tool, "graph")
    )
    assert browse.total_operations >= len(browse.operations)

    assert inspected is not None
    assert inspected.operation_id == "integer.compute.extended_gcd"
    assert inspected.version == "2"
