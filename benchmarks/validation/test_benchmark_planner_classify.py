"""Split from test_benchmark_planner.py."""

from __future__ import annotations

import pytest
from benchmarks.validation._planner_helpers import (
    _assert_plan_valid,
    _host_matrix,
    _matrix,
    planner,
)


def test_product_only_changes_skip_benchmark_work() -> None:
    result = planner.plan(["src/jacobian/math.py"], event="pull_request")

    assert result["benchmark-plan-version"] == "2"
    assert result["benchmark-plan-event"] == "pull_request"
    assert result["benchmark-plan-mode"] == "none"
    assert result["benchmark-topology-digest"] == ""
    assert result["run-benchmark-check"] == "false"
    assert result["run-benchmark-record-schema"] == "false"
    assert result["run-benchmark-prospective-digest"] == "false"
    assert result["run-benchmark-inventory"] == "false"
    assert result["run-benchmark-oracle"] == "false"
    assert result["benchmark-oracle-scope"] == "none"
    assert _matrix(result) == []
    _assert_plan_valid(result)


@pytest.mark.parametrize(
    "path",
    [
        ".github/scripts/plan-benchmarks",
        ".github/scripts/validate-benchmark-plan",
        ".github/workflows/benchmarks.yml",
        "Makefile",
        "tools/harbor_task_workflow.py",
    ],
)
def test_benchmark_control_plane_changes_run_contract_checks(path: str) -> None:
    result = planner.plan([path], event="pull_request")

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-record-schema"] == "true"
    # Planner/workflow/Makefile changes do not alter task content digests.
    assert result["run-benchmark-prospective-digest"] == "false"
    assert result["run-benchmark-inventory"] == "false"
    assert result["run-benchmark-oracle"] == "false"
    assert result["benchmark-oracle-scope"] == "none"
    assert result["benchmark-plan-mode"] == "changed"
    assert _matrix(result) == []
    _assert_plan_valid(result)


@pytest.mark.parametrize(
    "path",
    [".github/scripts/manage-test-timings", ".github/scripts/emit-plan-receipt"],
)
def test_non_host_control_utilities_do_not_select_full_verifier_corpus(
    path: str,
) -> None:
    result = planner.plan([path], event="pull_request")

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-host-validation"] == "false"
    assert _host_matrix(result) == []
    _assert_plan_valid(result)


def test_execution_configuration_change_defers_oracle_to_merge_queue() -> None:
    path = "benchmarks/config/jacobian.mcp.json"

    pull_request = planner.plan([path], event="pull_request")
    merge_group = planner.plan([path], event="merge_group")

    assert pull_request["run-benchmark-check"] == "true"
    assert pull_request["run-benchmark-record-schema"] == "true"
    assert pull_request["run-benchmark-prospective-digest"] == "true"
    assert pull_request["run-benchmark-inventory"] == "false"
    assert pull_request["run-benchmark-oracle"] == "false"
    assert pull_request["benchmark-oracle-scope"] == "none"
    assert pull_request["benchmark-plan-mode"] == "changed"
    assert _matrix(pull_request) == []
    assert merge_group["run-benchmark-oracle"] == "true"
    assert merge_group["benchmark-oracle-scope"] == "all"
    assert merge_group["benchmark-plan-mode"] == "integration"
    assert merge_group["run-benchmark-inventory"] == "true"
    assert _matrix(merge_group)
    _assert_plan_valid(pull_request)
    _assert_plan_valid(merge_group)


def test_shared_tooling_change_defers_oracle_and_runs_digests_on_pull_request() -> None:
    path = "benchmarks/tooling/harbor_suite.py"

    pull_request = planner.plan([path], event="pull_request")
    merge_group = planner.plan([path], event="merge_group")

    assert pull_request["run-benchmark-check"] == "true"
    assert pull_request["run-benchmark-record-schema"] == "true"
    assert pull_request["run-benchmark-prospective-digest"] == "true"
    assert pull_request["run-benchmark-oracle"] == "false"
    assert pull_request["benchmark-plan-mode"] == "changed"
    assert merge_group["run-benchmark-oracle"] == "true"
    assert merge_group["benchmark-oracle-scope"] == "all"
    assert merge_group["run-benchmark-inventory"] == "true"
    assert _matrix(merge_group)


def test_task_readme_change_runs_record_schema_without_oracle_or_digests() -> None:
    result = planner.plan(
        [
            "benchmarks/datasets/mathematical-benchmarks-v1/"
            "parameterized-sharp-bound-audit/README.md"
        ],
        event="pull_request",
    )

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-record-schema"] == "true"
    assert result["run-benchmark-prospective-digest"] == "false"
    assert result["run-benchmark-inventory"] == "false"
    assert result["run-benchmark-oracle"] == "false"
    assert result["benchmark-oracle-scope"] == "none"
    assert result["benchmark-plan-mode"] == "changed"
    assert _matrix(result) == []
    _assert_plan_valid(result)


def test_membership_change_defers_dataset_oracle_until_merge_queue() -> None:
    result = planner.plan(
        ["benchmarks/datasets/mathematical-benchmarks-v1/members/new-task.toml"],
        event="pull_request",
    )

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-record-schema"] == "true"
    assert result["run-benchmark-prospective-digest"] == "true"
    assert result["run-benchmark-oracle"] == "false"
    assert result["benchmark-oracle-scope"] == "none"
    assert result["benchmark-plan-mode"] == "changed"
    assert _matrix(result) == []


def test_membership_change_runs_affected_dataset_in_merge_queue() -> None:
    result = planner.plan(
        ["benchmarks/datasets/mathematical-benchmarks-v1/members/new-task.toml"],
        event="merge_group",
    )

    assert result["run-benchmark-oracle"] == "true"
    assert result["benchmark-oracle-scope"] == "affected-datasets"
    assert result["benchmark-plan-mode"] == "integration"
    assert result["run-benchmark-inventory"] == "true"
    assert _matrix(result)
    assert {item["dataset"] for item in _matrix(result)} == {
        "mathematical-benchmarks-v1"
    }
    _assert_plan_valid(result)


def test_merge_group_keeps_widest_oracle_scope_for_mixed_changes() -> None:
    result = planner.plan(
        [
            "benchmarks/config/jacobian.mcp.json",
            "benchmarks/datasets/mathematical-benchmarks-v1/members/new-task.toml",
        ],
        event="merge_group",
    )

    assert result["run-benchmark-oracle"] == "true"
    assert result["benchmark-oracle-scope"] == "all"
    assert result["benchmark-plan-mode"] == "integration"
    assert len(_matrix(result)) > 1
    _assert_plan_valid(result)


def test_existing_task_member_change_selects_changed_task_oracle_on_pull_request() -> (
    None
):
    # A member fragment change for an existing task can change its assurance
    # ceiling or required provider, which affects Oracle execution.  It
    # resolves to that exact task's Oracle on an ordinary pull request.
    result = planner.plan(
        [
            "benchmarks/datasets/mathematical-benchmarks-v1/"
            "members/parameterized-sharp-bound-audit.toml"
        ],
        event="pull_request",
    )

    assert result["run-benchmark-oracle"] == "true"
    assert result["benchmark-oracle-scope"] == "changed-tasks"
    assert result["run-benchmark-prospective-digest"] == "true"
    assert result["run-benchmark-inventory"] == "false"
    assert result["benchmark-plan-mode"] == "changed"
    matrix = _matrix(result)
    assert len(matrix) == 1
    assert matrix[0]["dataset"] == "mathematical-benchmarks-v1"
    assert matrix[0]["tasks"] == ["parameterized-sharp-bound-audit"]
    _assert_plan_valid(result)


def test_deleted_task_is_deferred_to_merge_queue_on_pull_request() -> None:
    # A deleted task no longer appears in the current inventory.  Its changed
    # paths (task directory and member fragment) are classified as dataset
    # integration changes, deferred to the merge queue on an ordinary PR.
    pull_request = planner.plan(
        [
            "benchmarks/datasets/mathematical-benchmarks-v1/"
            "deleted-former-task/tests/verifier.py",
            "benchmarks/datasets/mathematical-benchmarks-v1/members/deleted-former-task.toml",
        ],
        event="pull_request",
    )
    merge_group = planner.plan(
        [
            "benchmarks/datasets/mathematical-benchmarks-v1/"
            "deleted-former-task/tests/verifier.py",
            "benchmarks/datasets/mathematical-benchmarks-v1/members/deleted-former-task.toml",
        ],
        event="merge_group",
    )

    assert pull_request["run-benchmark-check"] == "true"
    assert pull_request["run-benchmark-record-schema"] == "true"
    assert pull_request["run-benchmark-prospective-digest"] == "true"
    assert pull_request["run-benchmark-oracle"] == "false"
    assert pull_request["benchmark-oracle-scope"] == "none"
    assert pull_request["benchmark-plan-mode"] == "changed"
    assert _matrix(pull_request) == []
    assert merge_group["run-benchmark-oracle"] == "true"
    assert merge_group["benchmark-oracle-scope"] == "affected-datasets"
    assert {item["dataset"] for item in _matrix(merge_group)} == {
        "mathematical-benchmarks-v1"
    }
    _assert_plan_valid(pull_request)
    _assert_plan_valid(merge_group)


def test_shared_tooling_change_is_contract_only_on_pull_request() -> None:
    result = planner.plan(
        ["benchmarks/tooling/verifier_support.py"], event="pull_request"
    )

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-record-schema"] == "true"
    assert result["run-benchmark-prospective-digest"] == "true"
    assert result["run-benchmark-oracle"] == "false"
    assert result["benchmark-oracle-scope"] == "none"
    assert _matrix(result) == []


def test_shared_tooling_change_runs_full_portfolio_in_merge_queue() -> None:
    result = planner.plan(
        ["benchmarks/tooling/verifier_support.py"], event="merge_group"
    )

    assert result["run-benchmark-oracle"] == "true"
    assert result["benchmark-oracle-scope"] == "all"
    assert result["benchmark-plan-mode"] == "integration"
    assert result["run-benchmark-inventory"] == "true"
    assert len(_matrix(result)) > 1


def test_main_push_owns_integration_oracle_and_inventory() -> None:
    result = planner.plan(["benchmarks/tooling/verifier_support.py"], event="push")

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-record-schema"] == "true"
    assert result["run-benchmark-prospective-digest"] == "true"
    assert result["run-benchmark-inventory"] == "true"
    assert result["run-benchmark-oracle"] == "true"
    assert result["benchmark-oracle-scope"] == "all"
    assert result["benchmark-plan-mode"] == "integration"
    assert _matrix(result)
    _assert_plan_valid(result)


def test_adapter_documentation_change_never_runs_oracle() -> None:
    result = planner.plan(["benchmarks/adapters/README.md"], event="merge_group")

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-record-schema"] == "true"
    assert result["run-benchmark-prospective-digest"] == "false"
    assert result["run-benchmark-oracle"] == "false"
    assert result["benchmark-oracle-scope"] == "none"
    assert result["benchmark-plan-mode"] == "integration"
    assert result["run-benchmark-inventory"] == "true"
    _assert_plan_valid(result)


def test_snapshot_change_runs_contracts_without_oracle() -> None:
    result = planner.plan(
        ["benchmarks/snapshots/mathematical-benchmarks-v1/digest.lock.json"],
        event="pull_request",
    )

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-record-schema"] == "true"
    assert result["run-benchmark-prospective-digest"] == "false"
    assert result["run-benchmark-oracle"] == "false"
    assert "immutable benchmark snapshot change" in result["benchmark-plan-reasons"]
    _assert_plan_valid(result)


def test_environment_profile_change_defers_full_oracle_to_merge_queue() -> None:
    pull_request = planner.plan(
        ["benchmarks/environment-profiles.toml"], event="pull_request"
    )
    merge_group = planner.plan(
        ["benchmarks/environment-profiles.toml"], event="merge_group"
    )

    assert pull_request["run-benchmark-prospective-digest"] == "true"
    assert pull_request["run-benchmark-oracle"] == "false"
    assert merge_group["run-benchmark-oracle"] == "true"
    assert merge_group["benchmark-oracle-scope"] == "all"
    _assert_plan_valid(pull_request)
    _assert_plan_valid(merge_group)


def test_unknown_benchmark_path_fails_closed_to_full_portfolio() -> None:
    result = planner.plan(["benchmarks/new-control-plane.py"], event="pull_request")

    assert result["run-benchmark-oracle"] == "true"
    assert result["benchmark-oracle-scope"] == "all"
    assert result["benchmark-plan-mode"] == "full"
    assert result["run-benchmark-inventory"] == "true"
    assert result["run-benchmark-record-schema"] == "true"
    assert result["run-benchmark-prospective-digest"] == "true"
    assert _matrix(result)
    assert len({item["dataset"] for item in _matrix(result)}) > 1
    _assert_plan_valid(result)
