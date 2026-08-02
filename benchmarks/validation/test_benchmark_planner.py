"""Contract tests for the independent Harbor benchmark planner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLANNER_PATH = ROOT / ".github" / "scripts" / "plan-benchmarks"
_SPEC = importlib.util.spec_from_loader(
    "benchmark_planner", SourceFileLoader("benchmark_planner", str(PLANNER_PATH))
)
assert _SPEC is not None and _SPEC.loader is not None
planner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(planner)


@pytest.fixture(autouse=True)
def stable_digests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep planner tests independent of Harbor's optional runtime package."""

    monkeypatch.setattr(
        planner,
        "_digest",
        lambda path: f"sha256:{hashlib.sha256(path.name.encode()).hexdigest()}",
    )


def _matrix(result: dict[str, str]) -> list[dict[str, object]]:
    return json.loads(result["benchmark-oracle-matrix"])


def _matrix_tasks(result: dict[str, str]) -> set[str]:
    return {str(task) for item in _matrix(result) for task in item["tasks"]}


def test_product_only_changes_skip_benchmark_work() -> None:
    result = planner.plan(["src/jacobian/math.py"], event="pull_request")

    assert result["run-benchmark-check"] == "false"
    assert result["run-benchmark-oracle"] == "false"
    assert result["benchmark-oracle-scope"] == "none"
    assert _matrix(result) == []


@pytest.mark.parametrize(
    "path",
    [
        ".github/scripts/plan-benchmarks",
        ".github/scripts/validate-benchmark-plan",
        ".github/workflows/benchmarks.yml",
        "Makefile",
    ],
)
def test_benchmark_control_plane_changes_run_contract_checks(path: str) -> None:
    result = planner.plan([path], event="pull_request")

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-oracle"] == "false"
    assert result["benchmark-oracle-scope"] == "none"
    assert _matrix(result) == []


def test_execution_configuration_change_defers_oracle_to_merge_queue() -> None:
    path = "benchmarks/config/jacobian.mcp.json"

    pull_request = planner.plan([path], event="pull_request")
    merge_group = planner.plan([path], event="merge_group")

    assert pull_request["run-benchmark-check"] == "true"
    assert pull_request["run-benchmark-oracle"] == "false"
    assert pull_request["benchmark-oracle-scope"] == "none"
    assert _matrix(pull_request) == []
    assert merge_group["run-benchmark-oracle"] == "true"
    assert merge_group["benchmark-oracle-scope"] == "all"
    assert _matrix(merge_group)


def test_executable_control_plane_change_defers_oracle_to_merge_queue() -> None:
    path = "benchmarks/tooling/harbor_suite.py"

    pull_request = planner.plan([path], event="pull_request")
    merge_group = planner.plan([path], event="merge_group")

    assert pull_request["run-benchmark-check"] == "true"
    assert pull_request["run-benchmark-oracle"] == "false"
    assert merge_group["run-benchmark-oracle"] == "true"
    assert merge_group["benchmark-oracle-scope"] == "all"
    assert _matrix(merge_group)


def test_task_readme_change_runs_contract_checks_without_oracle() -> None:
    result = planner.plan(
        [
            "benchmarks/datasets/agent-workflow-v1/"
            "parameterized-sharp-bound-audit/README.md"
        ],
        event="pull_request",
    )

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-oracle"] == "false"
    assert result["benchmark-oracle-scope"] == "none"
    assert _matrix(result) == []


def test_executable_task_change_selects_exact_task_and_all_memberships() -> None:
    result = planner.plan(
        [
            "benchmarks/datasets/agent-workflow-v1/"
            "parameterized-sharp-bound-audit/tests/verifier.py",
            "benchmarks/datasets/agent-workflow-v1/suite.toml",
        ],
        event="pull_request",
    )

    assert result["run-benchmark-oracle"] == "true"
    assert result["benchmark-oracle-scope"] == "changed-tasks"
    matrix = _matrix(result)
    assert len(matrix) == 1
    assert matrix[0]["dataset"] == "agent-workflow-v1"
    assert matrix[0]["tasks"] == ["parameterized-sharp-bound-audit"]
    digests = matrix[0]["task_digests"]
    assert isinstance(digests, list)
    assert len(digests[0]["digest"]) == 71
    assert digests[0]["digest"].startswith("sha256:")


def test_membership_change_defers_dataset_oracle_until_merge_queue() -> None:
    result = planner.plan(
        [
            "benchmarks/datasets/agent-workflow-v1/members/new-task.toml",
            "benchmarks/datasets/agent-workflow-v1/suite.toml",
        ],
        event="pull_request",
    )

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-oracle"] == "false"
    assert result["benchmark-oracle-scope"] == "none"
    assert _matrix(result) == []


def test_membership_change_runs_affected_dataset_in_merge_queue() -> None:
    result = planner.plan(
        [
            "benchmarks/datasets/agent-workflow-v1/members/new-task.toml",
            "benchmarks/datasets/agent-workflow-v1/suite.toml",
        ],
        event="merge_group",
    )

    assert result["run-benchmark-oracle"] == "true"
    assert result["benchmark-oracle-scope"] == "affected-datasets"
    assert _matrix(result)
    assert {item["dataset"] for item in _matrix(result)} == {"agent-workflow-v1"}


def test_shared_tooling_change_is_contract_only_on_pull_request() -> None:
    result = planner.plan(
        ["benchmarks/tooling/verifier_support.py"], event="pull_request"
    )

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-oracle"] == "false"
    assert result["benchmark-oracle-scope"] == "none"
    assert _matrix(result) == []


def test_shared_tooling_change_runs_full_portfolio_in_merge_queue() -> None:
    result = planner.plan(
        ["benchmarks/tooling/verifier_support.py"], event="merge_group"
    )

    assert result["run-benchmark-oracle"] == "true"
    assert result["benchmark-oracle-scope"] == "all"
    assert len(_matrix(result)) > 1


def test_large_task_set_is_deferred_from_pull_request_to_merge_queue() -> None:
    by_task, _suites = planner._membership()
    task_ids = sorted(by_task)[:9]
    paths = [
        f"benchmarks/datasets/{dataset}/{task_id}/tests/verifier.py"
        for task_id in task_ids
        for dataset, _path in by_task[task_id]
    ]
    paths.extend(
        f"benchmarks/datasets/{dataset}/suite.toml"
        for dataset in {
            dataset for task_id in task_ids for dataset, _ in by_task[task_id]
        }
    )

    pull_request = planner.plan(paths, event="pull_request")
    merge_group = planner.plan(paths, event="merge_group")

    assert pull_request["run-benchmark-oracle"] == "false"
    assert pull_request["benchmark-oracle-scope"] == "none"
    assert _matrix(pull_request) == []
    assert merge_group["run-benchmark-oracle"] == "true"
    assert merge_group["benchmark-oracle-scope"] == "affected-datasets"
    assert set(task_ids) <= _matrix_tasks(merge_group)


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
    paths.append(f"benchmarks/datasets/{by_task[task_ids[0]][0][0]}/suite.toml")

    result = planner.plan(paths, event="pull_request")

    assert result["run-benchmark-oracle"] == "true"
    assert result["benchmark-oracle-scope"] == "changed-tasks"
    assert _matrix_tasks(result) == {task_ids[0]}


def test_main_push_does_not_repeat_merge_queue_oracles() -> None:
    result = planner.plan(["benchmarks/tooling/verifier_support.py"], event="push")

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-oracle"] == "false"
    assert result["benchmark-oracle-scope"] == "none"
    assert _matrix(result) == []


def test_executable_task_change_requires_dataset_version_bump() -> None:
    with pytest.raises(Exception, match="dataset version bump"):
        planner.plan(
            [
                "benchmarks/datasets/agent-workflow-v1/"
                "parameterized-sharp-bound-audit/tests/verifier.py"
            ],
            event="pull_request",
        )


def test_dataset_version_file_must_change_its_version_value() -> None:
    suite = planner._harbor_suite().get_suite("public-reproductions-v1")
    task = suite.tasks[0].path.name

    with pytest.raises(Exception, match="version value did not change"):
        planner.plan(
            [
                f"benchmarks/datasets/{suite.id}/{task}/tests/verifier.py",
                f"benchmarks/datasets/{suite.id}/suite.toml",
            ],
            event="pull_request",
            base="main",
        )


def test_suite_only_change_cannot_bypass_version_value_check() -> None:
    with pytest.raises(Exception, match="version value did not change"):
        planner.plan(
            ["benchmarks/datasets/public-reproductions-v1/suite.toml"],
            event="pull_request",
            base="main",
        )


def test_adapter_documentation_change_never_runs_oracle() -> None:
    result = planner.plan(["benchmarks/adapters/README.md"], event="merge_group")

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-oracle"] == "false"
    assert result["benchmark-oracle-scope"] == "none"


def test_unknown_benchmark_path_fails_closed_to_full_portfolio() -> None:
    result = planner.plan(["benchmarks/new-control-plane.py"], event="pull_request")

    assert result["run-benchmark-oracle"] == "true"
    assert result["benchmark-oracle-scope"] == "all"
    assert _matrix(result)
    assert len({item["dataset"] for item in _matrix(result)}) > 1


def test_force_full_includes_each_dataset_task_pair() -> None:
    result = planner.plan([], event="workflow_dispatch", force_full=True)

    assert result["benchmark-oracle-scope"] == "all"
    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-oracle"] == "true"
    assert _matrix(result)
    assert len(_matrix(result)) < 20
    assert _matrix_tasks(result) == {
        ref.path.name
        for suite in planner._harbor_suite().load_registry()
        for ref in suite.tasks
    }


def test_force_full_still_requires_dataset_version_bumps() -> None:
    with pytest.raises(Exception, match="dataset version bump"):
        planner.plan(
            [
                "benchmarks/datasets/agent-workflow-v1/"
                "parameterized-sharp-bound-audit/tests/verifier.py"
            ],
            event="workflow_dispatch",
            force_full=True,
        )


def test_force_full_pr_with_base_still_enforces_version_value_bumps() -> None:
    """A ci:benchmark-full PR must not bypass the version-value check.

    The workflow preserves changed paths and passes --base even when a forced
    full plan is requested, so a task change whose suite.toml version value is
    unchanged against the base must still fail closed.
    """
    suite = planner._harbor_suite().get_suite("public-reproductions-v1")
    task = suite.tasks[0].path.name

    with pytest.raises(Exception, match="version value did not change"):
        planner.plan(
            [
                f"benchmarks/datasets/{suite.id}/{task}/tests/verifier.py",
                f"benchmarks/datasets/{suite.id}/suite.toml",
            ],
            event="pull_request",
            force_full=True,
            base="main",
        )


def test_benchmark_workflow_preserves_changed_paths_for_forced_full_prs() -> None:
    workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(encoding="utf-8")

    # The force-full branch must still compute changed_paths against BASE_SHA so
    # the planner receives the PR's benchmark changes instead of an empty list.
    pr_branch = workflow.split("pull_request|merge_group|push)", 1)[1].split(
        "*) exit 2", 1
    )[0]
    assert "--force-full" in pr_branch
    assert "--base" in pr_branch
    assert "git diff --name-only -z" in pr_branch


def test_timing_weights_balance_slow_tasks_deterministically() -> None:
    suites = planner._harbor_suite().load_registry()
    suite = next(item for item in suites if len(item.tasks) > 12)
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
