"""Split from test_benchmark_planner.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.validation._planner_helpers import (
    _assert_plan_valid,
    _matrix,
    _matrix_tasks,
    planner,
)


def test_new_task_directory_and_member_resolve_directly_without_version_bump() -> None:
    # A new task directory plus its member fragment resolves directly from the
    # current tree: no dataset version bump or prior inventory is required to
    # select that task's Oracle on an ordinary pull request.
    result = planner.plan(
        [
            "benchmarks/datasets/mathematical-benchmarks-v1/"
            "parameterized-sharp-bound-audit/tests/verifier.py",
            "benchmarks/datasets/mathematical-benchmarks-v1/"
            "members/parameterized-sharp-bound-audit.toml",
        ],
        event="pull_request",
    )

    assert result["run-benchmark-oracle"] == "true"
    assert result["benchmark-oracle-scope"] == "changed-tasks"
    assert result["run-benchmark-record-schema"] == "true"
    assert result["run-benchmark-prospective-digest"] == "true"
    assert result["run-benchmark-inventory"] == "false"
    assert result["benchmark-plan-mode"] == "changed"
    matrix = _matrix(result)
    assert len(matrix) == 1
    assert matrix[0]["dataset"] == "mathematical-benchmarks-v1"
    assert matrix[0]["tasks"] == ["parameterized-sharp-bound-audit"]
    digests = matrix[0]["task_digests"]
    assert isinstance(digests, list)
    assert digests[0]["digest"].startswith("sha256:")
    _assert_plan_valid(result)


def test_large_task_set_is_deferred_from_pull_request_to_merge_queue() -> None:
    by_task, _suites = planner._membership()
    task_ids = sorted(by_task)[:9]
    paths = [
        f"benchmarks/datasets/{dataset}/{task_id}/tests/verifier.py"
        for task_id in task_ids
        for dataset, _path in by_task[task_id]
    ]

    pull_request = planner.plan(paths, event="pull_request")
    merge_group = planner.plan(paths, event="merge_group")

    assert pull_request["run-benchmark-oracle"] == "false"
    assert pull_request["benchmark-oracle-scope"] == "none"
    assert _matrix(pull_request) == []
    assert merge_group["run-benchmark-oracle"] == "true"
    assert merge_group["benchmark-oracle-scope"] == "changed-tasks"
    assert set(task_ids) <= _matrix_tasks(merge_group)


def test_large_task_set_deferred_on_pull_request_runs_on_main_push() -> None:
    by_task, _suites = planner._membership()
    task_ids = sorted(by_task)[:9]
    paths = [
        f"benchmarks/datasets/{dataset}/{task_id}/tests/verifier.py"
        for task_id in task_ids
        for dataset, _path in by_task[task_id]
    ]

    pull_request = planner.plan(paths, event="pull_request")
    push = planner.plan(paths, event="push")

    assert pull_request["run-benchmark-oracle"] == "false"
    assert push["run-benchmark-oracle"] == "true"
    assert push["benchmark-oracle-scope"] == "changed-tasks"
    assert push["benchmark-plan-mode"] == "integration"
    assert push["run-benchmark-inventory"] == "true"
    assert set(task_ids) <= _matrix_tasks(push)
    _assert_plan_valid(push)


def test_documentation_changes_do_not_consume_the_oracle_task_cap() -> None:
    by_task, _suites = planner._membership()
    task_ids = sorted(by_task)[:9]
    paths = [
        f"benchmarks/datasets/{dataset}/{task_id}/README.md"
        for task_id in task_ids
        for dataset, _path in by_task[task_id]
    ]
    paths.append(
        "benchmarks/datasets/"
        f"{by_task[task_ids[0]][0][0]}/{task_ids[0]}/tests/verifier.py"
    )

    result = planner.plan(paths, event="pull_request")

    assert result["run-benchmark-oracle"] == "true"
    assert result["benchmark-oracle-scope"] == "changed-tasks"
    assert _matrix_tasks(result) == {task_ids[0]}


def test_force_full_includes_each_dataset_task_pair() -> None:
    result = planner.plan([], event="workflow_dispatch", force_full=True)

    assert result["benchmark-plan-version"] == "2"
    assert result["benchmark-oracle-scope"] == "all"
    assert result["benchmark-plan-mode"] == "full"
    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-record-schema"] == "true"
    assert result["run-benchmark-prospective-digest"] == "true"
    assert result["run-benchmark-inventory"] == "true"
    assert result["run-benchmark-oracle"] == "true"
    assert _matrix(result)
    assert _matrix_tasks(result) == {
        ref.path.name
        for suite in planner._harbor_suite().load_registry()
        for ref in suite.tasks
    }
    _assert_plan_valid(result)


def test_schedule_event_runs_full_portfolio() -> None:
    result = planner.plan([], event="schedule")

    assert result["benchmark-plan-mode"] == "full"
    assert result["benchmark-oracle-scope"] == "all"
    assert result["run-benchmark-inventory"] == "true"
    assert result["run-benchmark-oracle"] == "true"
    _assert_plan_valid(result)


def test_timing_weights_balance_slow_tasks_deterministically() -> None:
    suites = planner._harbor_suite().load_registry()
    suite = next((item for item in suites if len(item.tasks) > 12), None)
    assert suite is not None, (
        "expected a suite with more than 12 tasks for timing weight validation; "
        "found: " + ", ".join(f"{item.id}={len(item.tasks)}" for item in suites)
    )
    timings = {
        f"{suite.id}/{ref.path.name}": float(index + 1)
        for index, ref in enumerate(suite.tasks)
    }

    first = planner._shard_entries([suite], timings=timings)
    second = planner._shard_entries([suite], timings=timings)

    assert first == second
    assert {task for shard in first for task in shard["tasks"]} == {
        ref.path.name for ref in suite.tasks
    }


def test_task_bundles_live_directly_in_their_harbor_dataset() -> None:
    suites = planner._harbor_suite().load_registry()
    assert suites
    for suite in suites:
        assert all(task.path.parent == suite.path for task in suite.tasks)


def test_two_unrelated_synthetic_additions_yield_independent_matrices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alpha_task = Path("alpha-task")
    beta_task = Path("beta-task")
    alpha = SimpleNamespace(id="alpha-v1", tasks=(SimpleNamespace(path=alpha_task),))
    beta = SimpleNamespace(id="beta-v1", tasks=(SimpleNamespace(path=beta_task),))
    suites = {"alpha-v1": alpha, "beta-v1": beta}
    by_task = {
        "alpha-task": [("alpha-v1", alpha_task)],
        "beta-task": [("beta-v1", beta_task)],
    }
    monkeypatch.setattr(planner, "_membership", lambda: (by_task, suites))

    addition_alpha = planner.plan(
        [
            "benchmarks/datasets/alpha-v1/alpha-task/tests/verifier.py",
            "benchmarks/datasets/alpha-v1/members/alpha-task.toml",
        ],
        event="pull_request",
    )
    addition_beta = planner.plan(
        [
            "benchmarks/datasets/beta-v1/beta-task/tests/verifier.py",
            "benchmarks/datasets/beta-v1/members/beta-task.toml",
        ],
        event="pull_request",
    )

    alpha_tasks = _matrix_tasks(addition_alpha)
    beta_tasks = _matrix_tasks(addition_beta)
    assert alpha_tasks == {"alpha-task"}
    assert beta_tasks == {"beta-task"}
    assert alpha_tasks.isdisjoint(beta_tasks)
    assert {item["dataset"] for item in _matrix(addition_alpha)} == {"alpha-v1"}
    assert {item["dataset"] for item in _matrix(addition_beta)} == {"beta-v1"}
    # The topology digest binds to the full inventory, not the changed subset:
    # both plans see the same tree, so they share one topology digest while
    # their Oracle matrices remain independent.
    assert addition_alpha["benchmark-topology-digest"].startswith("sha256:")
    assert (
        addition_alpha["benchmark-topology-digest"]
        == (addition_beta["benchmark-topology-digest"])
    )
    _assert_plan_valid(addition_alpha)
    _assert_plan_valid(addition_beta)
