"""Split from test_benchmark_planner.py."""

from __future__ import annotations

import json

from benchmarks.validation._planner_helpers import (
    _assert_plan_valid,
    _host_matrix,
    _matrix,
    _raw_host_matrix,
    planner,
)


def test_executable_task_change_selects_exact_task_without_version_bump() -> None:
    result = planner.plan(
        [
            "benchmarks/datasets/mathematical-benchmarks-v1/"
            "parameterized-sharp-bound-audit/tests/verifier.py",
        ],
        event="pull_request",
    )

    assert result["run-benchmark-oracle"] == "true"
    assert result["benchmark-oracle-scope"] == "changed-tasks"
    assert result["benchmark-plan-mode"] == "changed"
    matrix = _matrix(result)
    assert len(matrix) == 1
    assert matrix[0]["dataset"] == "mathematical-benchmarks-v1"
    assert matrix[0]["tasks"] == ["parameterized-sharp-bound-audit"]
    host_matrix = _host_matrix(result)
    assert result["run-benchmark-host-validation"] == "true"
    assert {(entry["selector"], entry["keyword"]) for entry in host_matrix} == {
        (
            "benchmarks/validation/mathematical_benchmarks_v1/"
            "test_parameterized_sharp_bound_audit.py",
            "",
        ),
        (
            "benchmarks/validation/mathematical_benchmarks_v1/"
            "test_generic_verifier_contracts.py",
            "parameterized-sharp-bound-audit",
        ),
    }
    _assert_plan_valid(result)


def test_host_and_oracle_matrices_record_predictions() -> None:
    task = "parameterized-sharp-bound-audit"
    result = planner.plan(
        [f"benchmarks/datasets/mathematical-benchmarks-v1/{task}/tests/verifier.py"],
        event="pull_request",
        timings={
            f"mathematical-benchmarks-v1/{task}": 73.5,
            f"host-validation/{task}-specific": 8.25,
            f"host-validation/{task}-generic": 3.5,
        },
    )

    assert _matrix(result)[0]["predicted_seconds"] == 73.5
    predictions = {
        entry["name"]: entry["predicted_seconds"] for entry in _raw_host_matrix(result)
    }
    assert predictions == {
        f"{task}-generic": 3.5,
        f"{task}-specific": 8.25,
    }
    _assert_plan_valid(result)


def test_dataset_owned_task_selects_shared_host_regression() -> None:
    result = planner.plan(
        [
            "benchmarks/datasets/symbolic-coordination-v1/"
            "symbolic-coordination-valid-inverse-01/tests/verifier.py"
        ],
        event="pull_request",
    )

    assert _host_matrix(result) == [
        {
            "name": "symbolic-coordination-v1-1",
            "selector": (
                "benchmarks/validation/symbolic_coordination_v1/test_pilot_contract.py"
            ),
            "keyword": "",
            "splits": 0,
            "group": 0,
        }
    ]
    _assert_plan_valid(result)


def test_conjecture_probe_task_selects_owned_host_regression() -> None:
    result = planner.plan(
        [
            "benchmarks/datasets/conjecture-probes-v1/"
            "vizing-bounded-cartesian-products/tests/verifier.py"
        ],
        event="pull_request",
    )
    assert _host_matrix(result) == [
        {
            "name": "vizing-bounded-cartesian-products",
            "selector": (
                "benchmarks/validation/conjecture_probes_v1/"
                "test_vizing_bounded_cartesian_products.py"
            ),
            "keyword": "",
            "splits": 0,
            "group": 0,
        }
    ]
    _assert_plan_valid(result)


def test_changed_validation_test_selects_itself() -> None:
    path = "benchmarks/validation/test_benchmark_timings.py"
    result = planner.plan([path], event="pull_request")

    assert _host_matrix(result) == [
        {
            "name": "test_benchmark_timings",
            "selector": path,
            "keyword": "",
            "splits": 0,
            "group": 0,
        }
    ]
    _assert_plan_valid(result)


def test_shared_benchmark_support_falls_back_to_full_host_validation() -> None:
    result = planner.plan(["benchmarks/tooling/harbor_suite.py"], event="pull_request")

    assert _host_matrix(result) == [
        {
            "name": f"full-{group}-of-4",
            "selector": "benchmarks/validation",
            "keyword": "",
            "splits": 4,
            "group": group,
        }
        for group in range(1, 5)
    ]
    assert any(
        "shared benchmark tooling requires full host validation" in reason
        for reason in json.loads(result["benchmark-plan-reasons"])
    )
    _assert_plan_valid(result)


def test_shared_verifier_harness_states_full_suite_reason() -> None:
    path = "benchmarks/validation/mathematical_benchmarks_v1/support.py"
    result = planner.plan([path], event="pull_request")

    assert len(_host_matrix(result)) == 4
    assert json.loads(result["benchmark-plan-reasons"]) == [
        "benchmark validation or documentation change",
        f"shared verifier harness requires full host validation: {path}",
    ]
    _assert_plan_valid(result)
