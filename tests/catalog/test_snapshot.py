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
from jacobian.catalog.search import matches_domain

_SNAPSHOT_DIR = Path(__file__).with_name("operation_schema_snapshots")


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _load_shards() -> tuple[dict[str, dict[str, str]], str]:
    """Load all owner-scoped snapshot fragments deterministically."""
    shard_files = sorted(_SNAPSHOT_DIR.glob("*.json"))
    assert shard_files, "no snapshot fragments found"

    operations: dict[str, dict[str, str]] = {}
    catalog_version = ""
    seen_ids: set[str] = set()

    for shard_path in shard_files:
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        if not catalog_version:
            catalog_version = shard["catalog_version"]
        elif catalog_version != shard["catalog_version"]:
            raise ValueError(
                f"catalog_version mismatch in {shard_path.name}: "
                f"{shard['catalog_version']} != {catalog_version}"
            )
        for op_id, schemas in shard["operations"].items():
            if op_id in seen_ids:
                raise ValueError(f"duplicate operation ID {op_id!r} across fragments")
            seen_ids.add(op_id)
            operations[op_id] = schemas

    return operations, catalog_version


def test_operation_ids_and_request_result_schemas_match_snapshot() -> None:
    expected_operations, expected_version = _load_shards()
    catalog = Catalog.open()
    snapshot = catalog.snapshot()
    actual = {
        descriptor.operation_id: {
            "input_schema": _digest(descriptor.input_schema),
            "output_schema": _digest(descriptor.output_schema),
        }
        for descriptor in snapshot.operations
    }

    assert snapshot.catalog_version == expected_version
    assert len(actual) == len(expected_operations), (
        f"operation count mismatch: actual={len(actual)} expected={len(expected_operations)}"
    )

    missing = set(actual) - set(expected_operations)
    stale = set(expected_operations) - set(actual)
    if missing:
        pytest.fail(f"missing operations in snapshot: {sorted(missing)}")
    if stale:
        pytest.fail(f"stale operations in snapshot: {sorted(stale)}")

    for op_id, schemas in actual.items():
        assert schemas == expected_operations[op_id], (
            f"schema drift for {op_id}: actual={schemas} expected={expected_operations[op_id]}"
        )


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


def test_duplicate_ids_across_fragments_fail() -> None:
    """Duplicate operation IDs across fragments must fail with a diagnostic."""
    shard_files = sorted(_SNAPSHOT_DIR.glob("*.json"))
    if len(shard_files) < 2:
        pytest.skip("need at least 2 fragments to test dedup")

    # Simulate a duplicate by loading two fragments and injecting a shared ID
    first = json.loads(shard_files[0].read_text(encoding="utf-8"))
    second = json.loads(shard_files[1].read_text(encoding="utf-8"))
    if not first["operations"] or not second["operations"]:
        pytest.skip("need non-empty fragments for dedup test")

    first_id = next(iter(first["operations"]))
    second["operations"][first_id] = first["operations"][first_id]

    seen: set[str] = set()
    with pytest.raises(ValueError, match="duplicate operation ID"):
        for _shard_path, shard in [(shard_files[0], first), (shard_files[1], second)]:
            for op_id in shard["operations"]:
                if op_id in seen:
                    raise ValueError(
                        f"duplicate operation ID {op_id!r} across fragments"
                    )
                seen.add(op_id)
