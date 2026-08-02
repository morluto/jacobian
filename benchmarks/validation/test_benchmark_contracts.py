from __future__ import annotations

import json
from dataclasses import replace

import pytest
from benchmarks.tooling import benchmark_contracts, benchmark_inventory
from benchmarks.tooling.harbor_suite import (
    HarborSuiteError,
    load_registry,
    validate_global_task_ids,
)


def test_every_committed_benchmark_contract_is_valid() -> None:
    assert benchmark_contracts.validate_all() == []


def test_registry_rejects_global_task_id_collisions() -> None:
    first, second, *_rest = load_registry()
    colliding = replace(second, tasks=(first.tasks[0],))

    with pytest.raises(HarborSuiteError, match="global task id"):
        validate_global_task_ids([first, colliding])


def test_inventory_covers_every_registered_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark_inventory, "task_digest", lambda _path: "a" * 64)
    inventory = benchmark_inventory.build_inventory()
    suites = load_registry()

    assert inventory["schema_version"] == "1"
    assert inventory["dataset_count"] == len(suites)
    assert inventory["task_count"] == sum(len(suite.tasks) for suite in suites)
    rendered = json.dumps(inventory)
    assert "verifier_digest" in rendered
    assert "submission_schema_digest" in rendered
