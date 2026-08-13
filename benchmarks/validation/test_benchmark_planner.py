"""Contract tests for the independent Harbor benchmark planner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLANNER_PATH = ROOT / ".github" / "scripts" / "plan-benchmarks"
VALIDATOR_PATH = ROOT / ".github" / "scripts" / "validate-benchmark-plan"


def _load_script(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_loader(
        module_name, SourceFileLoader(module_name, str(path))
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as module_state:
        module_state.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
    return module


planner = _load_script("benchmark_planner", PLANNER_PATH)


@pytest.fixture(autouse=True)
def stable_digests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep planner tests independent of Harbor's optional runtime package."""

    monkeypatch.setattr(
        planner,
        "_digest",
        lambda path: f"sha256:{hashlib.sha256(path.name.encode()).hexdigest()}",
    )


@pytest.mark.parametrize("preserve_existing", [False, True])
def test_load_script_scopes_sys_modules_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    preserve_existing: bool,
) -> None:
    module_name = "_benchmark_planner_module_probe"
    source = tmp_path / "module_probe.py"
    source.write_text(
        "import sys\nregistered_while_loading = sys.modules[__name__]\n",
        encoding="utf-8",
    )
    sentinel = ModuleType("sentinel")
    if preserve_existing:
        monkeypatch.setitem(sys.modules, module_name, sentinel)
    else:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    loaded = _load_script(module_name, source)

    assert vars(loaded)["registered_while_loading"] is loaded
    if preserve_existing:
        assert sys.modules[module_name] is sentinel
    else:
        assert module_name not in sys.modules


def _matrix(result: dict[str, str]) -> list[dict[str, object]]:
    return json.loads(result["benchmark-oracle-matrix"])


def _matrix_tasks(result: dict[str, str]) -> set[str]:
    return {str(task) for item in _matrix(result) for task in item["tasks"]}


def _host_matrix(result: dict[str, str]) -> list[dict[str, object]]:
    matrix = json.loads(result["benchmark-host-validation-matrix"])
    return [
        {key: value for key, value in entry.items() if key != "predicted_seconds"}
        for entry in matrix
    ]


def _raw_host_matrix(result: dict[str, str]) -> list[dict[str, object]]:
    return json.loads(result["benchmark-host-validation-matrix"])


def _assert_plan_valid(result: dict[str, str]) -> None:
    payload = "\n".join(f"{key}={value}" for key, value in result.items()) + "\n"
    proc = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH)],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def _lane(result: dict[str, str], key: str) -> bool:
    return result[key] == "true"


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


def test_plan_is_versioned_and_bound_to_event_base_head_sha() -> None:
    base = "0" * 40
    head = "1" * 40
    result = planner.plan(
        [
            "benchmarks/datasets/mathematical-benchmarks-v1/"
            "parameterized-sharp-bound-audit/tests/verifier.py"
        ],
        event="pull_request",
        base=base,
        head=head,
    )

    assert result["benchmark-plan-version"] == "2"
    assert result["benchmark-plan-event"] == "pull_request"
    assert result["benchmark-plan-base-sha"] == base
    assert result["benchmark-plan-head-sha"] == head
    assert result["benchmark-planner-digest"].startswith("sha256:")
    assert len(result["benchmark-planner-digest"]) == 71
    assert result["benchmark-topology-digest"].startswith("sha256:")
    assert len(result["benchmark-topology-digest"]) == 71
    _assert_plan_valid(result)


def test_planner_digest_binds_to_planner_and_path_policy_sources() -> None:
    payload = "\n".join(
        f"{path.relative_to(ROOT).as_posix()}\t{path.read_bytes().hex()}"
        for path in planner.PLANNER_DIGEST_SOURCES
    ).encode()
    expected = "sha256:" + hashlib.sha256(payload).hexdigest()
    result = planner.plan(
        [
            "benchmarks/datasets/mathematical-benchmarks-v1/"
            "parameterized-sharp-bound-audit/tests/verifier.py"
        ],
        event="pull_request",
    )
    assert result["benchmark-planner-digest"] == expected


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


def test_root_makefile_runs_contracts_without_host_verifier_replay() -> None:
    result = planner.plan(["Makefile"], event="pull_request")

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-record-schema"] == "true"
    assert result["run-benchmark-host-validation"] == "false"
    assert _host_matrix(result) == []
    _assert_plan_valid(result)


def test_harbor_makefile_change_keeps_full_host_verifier_coverage() -> None:
    result = planner.plan(["make/harbor.mk"], event="pull_request")

    assert result["run-benchmark-host-validation"] == "true"
    assert len(_host_matrix(result)) == 4
    assert any(
        "shared verifier execution harness requires full host validation: "
        "make/harbor.mk" in reason
        for reason in json.loads(result["benchmark-plan-reasons"])
    )
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


def _build_temp_topology(tmp_path: Path) -> tuple[Path, SimpleNamespace]:
    """Build a minimal temp benchmark tree with one suite and one task.

    Returns the suite object whose ``path`` and ``suite_manifest`` point at
    real temp files so ``_topology_digest`` binds their content.
    """

    bench = tmp_path / "benchmarks"
    dataset_dir = bench / "datasets" / "alpha-v1"
    members_dir = dataset_dir / "members"
    members_dir.mkdir(parents=True)
    (bench / "registry.toml").write_text('schema_version = "1"\n', encoding="utf-8")
    (bench / "environment-profiles.toml").write_text(
        '[profiles.default]\nimage = "default"\n', encoding="utf-8"
    )
    suite_manifest = dataset_dir / "suite.toml"
    suite_manifest.write_text(
        'schema_version = "2"\n[dataset]\nid = "jacobian/alpha-v1"\n',
        encoding="utf-8",
    )
    member = members_dir / "alpha-task.toml"
    member.write_text('task_id = "alpha-task"\n', encoding="utf-8")
    task = SimpleNamespace(path=Path("alpha-task"))
    suite = SimpleNamespace(
        id="alpha-v1", path=dataset_dir, suite_manifest=suite_manifest, tasks=(task,)
    )
    return bench, suite


def test_member_change_alters_topology_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _bench, suite = _build_temp_topology(tmp_path)
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    suites = {"alpha-v1": suite}
    by_task = {"alpha-task": [("alpha-v1", Path("alpha-task"))]}
    monkeypatch.setattr(planner, "_membership", lambda: (by_task, suites))

    before = planner._topology_digest([suite])

    (suite.path / "members" / "alpha-task.toml").write_text(
        'task_id = "alpha-task"\nassurance_ceiling = "VERIFIED"\n',
        encoding="utf-8",
    )
    after = planner._topology_digest([suite])

    assert before.startswith("sha256:")
    assert after.startswith("sha256:")
    assert before != after


def test_environment_profile_change_alters_topology_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bench, suite = _build_temp_topology(tmp_path)
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    suites = {"alpha-v1": suite}
    by_task = {"alpha-task": [("alpha-v1", Path("alpha-task"))]}
    monkeypatch.setattr(planner, "_membership", lambda: (by_task, suites))

    before = planner._topology_digest([suite])

    (bench / "environment-profiles.toml").write_text(
        '[profiles.default]\nimage = "changed"\n', encoding="utf-8"
    )
    after = planner._topology_digest([suite])

    assert before != after


def test_suite_manifest_change_alters_topology_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, suite = _build_temp_topology(tmp_path)
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    suites = {"alpha-v1": suite}
    by_task = {"alpha-task": [("alpha-v1", Path("alpha-task"))]}
    monkeypatch.setattr(planner, "_membership", lambda: (by_task, suites))

    before = planner._topology_digest([suite])

    (suite.suite_manifest).write_text(
        'schema_version = "2"\n[dataset]\nid = "jacobian/alpha-v1"\ntitle = "Changed"\n',
        encoding="utf-8",
    )
    after = planner._topology_digest([suite])

    assert before != after


def test_registry_change_alters_topology_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bench, suite = _build_temp_topology(tmp_path)
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    suites = {"alpha-v1": suite}
    by_task = {"alpha-task": [("alpha-v1", Path("alpha-task"))]}
    monkeypatch.setattr(planner, "_membership", lambda: (by_task, suites))

    before = planner._topology_digest([suite])

    (bench / "registry.toml").write_text(
        'schema_version = "1"\n[[datasets]]\nid = "jacobian/alpha-v1"\n',
        encoding="utf-8",
    )
    after = planner._topology_digest([suite])

    assert before != after
