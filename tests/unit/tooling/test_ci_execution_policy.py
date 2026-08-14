from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from pre_commit.clientlib import load_config
from pre_commit.lang_base import hook_cmd
from pre_commit.parse_shebang import normalize_cmd

ROOT = Path(__file__).parents[3]


def test_pr_base_edits_trigger_ci() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    pull_request_trigger = workflow.split("  pull_request:", 1)[1].split(
        "  merge_group:", 1
    )[0]

    assert "edited" in pull_request_trigger


def test_wheel_job_covers_supported_pythons_and_313_compatibility_smoke() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    wheel = workflow.split("  wheel:", 1)[1].split("  lean:", 1)[0]

    assert 'python-version: ["3.12", "3.13"]' in wheel
    assert "--only-binary :all:" in wheel
    assert '"$environment/bin/jacobian" --state-dir "$state_dir" init' in wheel
    assert "make test-compatibility" in wheel
    assert "make test\n" not in wheel


def test_global_timeout_is_not_a_pytest_deadline() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "timeout = 60" not in pyproject
    assert 'timeout_method = "thread"' in pyproject


def test_local_hook_commands_have_parseable_entrypoints_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(str(ROOT / ".pre-commit-config.yaml"))
    hooks = [
        hook
        for repo in config["repos"]
        if repo["repo"] == "local"
        for hook in repo["hooks"]
    ]
    assert hooks

    class ArgumentsParsedError(Exception):
        pass

    original_parse_args = argparse.ArgumentParser.parse_args

    def stop_after_parse(
        parser: argparse.ArgumentParser,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> None:
        original_parse_args(parser, args, namespace)
        raise ArgumentsParsedError

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop_after_parse)

    for hook in hooks:
        command = hook_cmd(hook["entry"], hook["args"])
        normalize_cmd(command)
        if hook["id"] == "jacobian-pre-push":
            assert command == ("make", "lint", "typecheck")
            continue

        assert command[:4] == ("uv", "run", "--locked", "python"), hook["id"]
        script_index = 4
        assert script_index < len(command), hook["id"]
        script = (ROOT / command[script_index]).resolve()
        assert script.is_relative_to(ROOT) and script.is_file(), hook["id"]

        namespace = runpy.run_path(str(script))
        main = namespace.get("main")
        assert callable(main), hook["id"]
        monkeypatch.setattr(
            sys,
            "argv",
            [str(script), *command[script_index + 1 :]],
        )
        with pytest.raises(ArgumentsParsedError):
            main()


def test_process_lane_is_invoked_by_ci() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    action = (ROOT / ".github/actions/run-test-lane/action.yml").read_text(
        encoding="utf-8"
    )

    assert "lane: [storage, process, mcp]" in workflow
    assert "uses: ./.github/actions/run-test-lane" in workflow
    assert "run: make test-${{ inputs.lane }}" in action


def test_python_and_boundary_lanes_share_evidence_collection() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    action = (ROOT / ".github/actions/run-test-lane/action.yml").read_text(
        encoding="utf-8"
    )

    assert workflow.count("uses: ./.github/actions/run-test-lane") == 2
    assert "--junitxml=pytest.xml" in action
    assert "--cov --cov-report= --cov-fail-under=0" in action
    assert action.count("actions/upload-artifact@") == 2
    assert "uv cache prune --ci" in action


def test_lean_job_is_required_on_every_event_and_builds_semantic_targets() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    action = (ROOT / ".github/actions/setup-lean/action.yml").read_text(
        encoding="utf-8"
    )

    lean = workflow.split("  lean:", 1)[1].split("  coverage:", 1)[0]
    assert "if: >-" not in lean
    assert "JACOBIAN_LEAN_REQUIRED" in lean
    assert "use-github-cache: false" in action
    assert "use-mathlib-cache: true" in action
    assert "lake build JacobianLeanRuntime repl jacobian_lean_proof_state" in action
    assert "tools/preflight_lean_runtime.py --required" in action


def test_optional_boundary_and_deployment_jobs_have_explicit_workflow_gates() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "ci:full" not in workflow
    assert "ci:lean" not in workflow
    assert "run: make deploy-check" in workflow
    assert "name: required" in workflow
    assert "name: Deployment Tests" not in workflow


def test_python_jobs_use_fixed_local_semantic_targets() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "name: python (${{ matrix.lane }})" in workflow
    assert "lane: component" in workflow
    assert "lane: domain" in workflow
    assert "lane: composition" in workflow
    assert "lane: e2e" in workflow
    assert "lane: provider" in workflow
    assert "uses: ./.github/actions/run-test-lane" in workflow
    assert (
        "ORDINARY_TEST_LANES := unit component domain composition e2e provider"
        in makefile
    )
    assert "quick: lint test-unit" in makefile
    assert "check: lint typecheck test-unit" in makefile
    assert "check-all: lint typecheck test-ordinary" in makefile
    assert "check-external: test-lean ##" in makefile
    assert "check-external: test-lean test-provider" not in makefile


def test_exhaustive_local_reproduction_includes_exhaustive_marker_lane() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    all_ci = makefile.split(
        "test-all-ci: ## Every local semantic pytest/Lean lane; not hosted CI, coverage, or docs.",
        1,
    )[1].split("test-stress:", 1)[0]

    assert "$(MAKE) test-component" in all_ci
    assert "$(MAKE) test-exhaustive" in all_ci
    assert "$(WORKTREE_ADMISSION) run --target test-all-ci" in all_ci
    assert all_ci.index("$(MAKE) test-component") < all_ci.index(
        "$(MAKE) test-exhaustive"
    )


def test_focused_unit_lane_skips_worktree_admission() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    unit = makefile.split("test-unit:", 1)[1].split("test-component:", 1)[0]
    exhaustive = makefile.split("test-exhaustive:", 1)[1].split("test-ordering:", 1)[0]
    harbor = (ROOT / "make" / "harbor.mk").read_text(encoding="utf-8")

    assert "WORKTREE_ADMISSION" not in unit
    assert "$(WORKTREE_ADMISSION) run --target test-exhaustive" in exhaustive
    assert "$(WORKTREE_ADMISSION) run --target harbor-check-all" in harbor
    assert "$(WORKTREE_ADMISSION) run --target harbor-host-validation" in harbor
    assert "$(WORKTREE_ADMISSION) run --target harbor-oracle-all" in harbor


def test_component_lane_uses_module_fixture_affinity() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    component = makefile.split("test-component:", 1)[1].split("test-domain:", 1)[0]
    domain = makefile.split("test-domain:", 1)[1].split("test-composition:", 1)[0]

    assert "--dist loadscope" in component
    assert "--dist worksteal" in domain


def test_benchmark_workflow_has_distinct_pr_merge_and_full_portfolio_tiers() -> None:
    workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(encoding="utf-8")

    assert "schedule:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "EVENT_NAME: ${{ github.event_name }}" in workflow
    assert '--event "$EVENT_NAME"' in workflow
    assert "validate-benchmark-plan" in workflow
    assert "ci:benchmark-full" in workflow


def test_oracle_workers_do_not_repeat_benchmark_contract_suite() -> None:
    workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(encoding="utf-8")
    oracle = workflow.split("  oracle:", 1)[1].split("  validation:", 1)[0]

    assert "needs: [plan, static, contracts]" in oracle
    assert "make harbor-oracle-task" in oracle
    assert "make harbor-oracle DATASET" not in oracle


def test_oracle_artifact_preserves_augmented_task_digest_manifest() -> None:
    workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(encoding="utf-8")
    oracle = workflow.split("  oracle:", 1)[1].split("  validation:", 1)[0]

    assert "jacobian-augmented-task-digests.*.json" in oracle
    assert ".jacobian-augmented-task-digests.*.json" not in oracle


def test_benchmark_contracts_run_once_for_record_and_digest_evidence() -> None:
    workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(encoding="utf-8")

    assert "run: make harbor-validate" not in workflow
    assert workflow.count("run: make harbor-contracts harbor-adapter-checks") == 1
    assert "  contracts:" in workflow
    assert "  host_validation:" in workflow
    assert "benchmarks.tooling.host_validation run-entry" in workflow
    assert '--entry-json "$HOST_ENTRY"' in workflow
    assert "--total-workers 8 --max-parallel 4" in workflow
    assert '--execution-sha "${{ github.sha }}"' in workflow
    assert "  prospective-digest:" not in workflow
    assert "python .github/scripts/emit-plan-receipt" in workflow
    assert "benchmark-plan-receipt" in workflow


def test_benchmark_stable_gate_validates_provenance_receipts_in_python() -> None:
    workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(encoding="utf-8")
    validation = workflow.split("  validation:", 1)[1].split("  timings:", 1)[0]

    assert "benchmarks.tooling.benchmark_validation" in validation
    assert "benchmark-plan-receipt" in validation
    assert "benchmark-host-timing-*" in validation
    assert "benchmark-test-durations-input" in validation
    assert '--execution-sha "${{ github.sha }}"' in validation
    assert "check_lane()" not in validation


def test_product_ci_does_not_emit_a_plan_receipt() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    script = (ROOT / ".github/scripts/emit-plan-receipt").read_text(encoding="utf-8")

    assert "python .github/scripts/emit-plan-receipt" not in workflow
    assert "ci-plan-receipt" not in workflow
    assert "classify-ci-paths" not in workflow
    assert "kind must be 'benchmark'" in script
    assert "CI or benchmark plan" not in script


def test_paths_file_stays_on_harbor_planning() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    harbor = (ROOT / "make" / "harbor.mk").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "PATHS_FILE" not in makefile
    assert "PATHS_FILE :=" not in harbor
    assert "$(shell mktemp)" not in harbor
    assert "tr '\\n' ' '" not in harbor
    assert '--paths-file "$$tmp_dir/changed-paths.txt"' in harbor
    assert "--config make/harbor.mk" in harbor
    assert "PATHS_FILE" not in workflow


def test_static_job_runs_docs_linkcheck() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    static = workflow.split("  static:", 1)[1].split("  python:", 1)[0]
    action = (ROOT / ".github/actions/setup-python-tests/action.yml").read_text(
        encoding="utf-8"
    )

    assert "astral-sh/setup-uv@" in action
    assert 'version: "0.11.28"' in action
    assert "run: make docs-linkcheck" in static
    assert "run: make npm-test" in static
    assert 'node-version: "24"' in static


def test_plan_receipt_digests_are_rendered_as_markdown_code() -> None:
    workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(encoding="utf-8")

    assert "Plan receipt: \\`$(python" in workflow
    assert "Plan receipt: \\\\`$(python" not in workflow


def test_required_ci_gates_fail_closed_after_cancellation() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    required = workflow.split("  required:", 1)[1]

    assert "treating gate as non-failure" not in workflow
    assert required.count("if: ${{ always() }}") == 1
    assert "if: ${{ always() && !cancelled() }}" not in required
    assert "if: ${{ always() }}" in required
    assert "name: required" in workflow
    assert "name: Python Tests" not in workflow
    assert "name: Lean Tests" not in workflow
    assert "name: Deployment Tests" not in workflow
    assert "python-test:" not in workflow
    assert "lean-test:" not in workflow
    assert "deployment-test:" not in workflow
    assert ("needs: [static, python, boundaries, wheel, coverage, lean]") in required
    assert "success|skipped" not in required
    assert 'test "$LEAN_RESULT" = success' in required
    assert "needs.lean.result" in required
    lean_job = workflow.split("  lean:", 1)[1].split("  coverage:", 1)[0]
    assert "github.event_name != 'pull_request'" not in lean_job
    assert "JACOBIAN_LEAN_REQUIRED" in lean_job


def test_required_pr_workflows_cancel_stale_evidence() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    benchmarks = (ROOT / ".github/workflows/benchmarks.yml").read_text(encoding="utf-8")

    expected_concurrency = (
        "cancel-in-progress: ${{ github.event_name == 'pull_request' }}"
    )
    assert expected_concurrency in ci
    assert expected_concurrency in benchmarks
    validation = benchmarks.split("  validation:", 1)[1].split("  timings:", 1)[0]
    assert "if: ${{ always() }}" in validation
    assert "if: ${{ always() && !cancelled() }}" not in validation
    assert (
        '--lane "static:${PLAN_CHECK:-false}:${STATIC_RESULT:-cancelled}"' in validation
    )
    assert (
        '--lane "oracle:${ORACLE_FLAG:-false}:${ORACLE_RESULT:-cancelled}"'
        in validation
    )


def test_subprocess_coverage_is_owned_by_one_focused_worker_lane() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    subprocess_config = (ROOT / ".coveragerc-subprocess").read_text(encoding="utf-8")
    target = (
        (ROOT / "Makefile")
        .read_text(encoding="utf-8")
        .split("test-checker-subprocess-coverage:", 1)[1]
        .split("test-all-ci:", 1)[0]
    )

    assert 'patch = ["subprocess"]' not in pyproject
    assert "patch = subprocess" in subprocess_config
    assert "tests/unit/test_checker_worker_manifest.py" in target
    assert "--cov-config=.coveragerc-subprocess" in target
    assert "--include=src/jacobian/checker_worker.py --fail-under=1" in target
    assert workflow.count("make test-checker-subprocess-coverage") == 1
    assert "needs: [python, boundaries, subprocess_coverage]" in workflow


def test_local_oracle_targets_require_explicit_scope() -> None:
    harbor = (ROOT / "make" / "harbor.mk").read_text(encoding="utf-8")
    oracle = harbor.split("harbor-oracle:", 1)[1].split("harbor-oracle-task:", 1)[0]
    runner = harbor.split("harbor-oracle-run:", 1)[1].split("harbor-oracle-all:", 1)[0]

    assert '"$(TASKS)" -o "$(FULL)" = "1"' in oracle
    assert '"$(TASKS)" -o "$(FULL)" = "1"' in runner
    assert "DATASET=$$dataset FULL=1" in harbor
    assert "$(MAKE) harbor-check\n" in oracle
    assert "harbor-check-all" not in oracle


def test_local_oracle_attempts_are_serialized_on_a_shared_docker_host() -> None:
    harbor = (ROOT / "make" / "harbor.mk").read_text(encoding="utf-8")

    assert "HARBOR_ORACLE_LOCK ?= benchmarks/results/.harbor-oracle.lock" in harbor
    assert harbor.count('exec 9>"$(HARBOR_ORACLE_LOCK)"; flock 9;') == 2
    assert "HARBOR_ORACLE_DOCKER_BUILD_MODE ?= auto" in harbor
    assert "export DOCKER_BUILDKIT=0 COMPOSE_BAKE=false" in harbor


def test_product_ci_does_not_use_timing_shards() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "composition-shard-count:" not in workflow
    assert "--splits" not in workflow
    assert "pytest-split" not in workflow


def test_stress_lane_selects_only_property_tests_and_repeats_them() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    stress = makefile.split("test-stress:", 1)[1].split("test-ordering:", 1)[0]

    assert "-m property" in stress
    assert "--count=$(STRESS_COUNT)" in stress


def test_ordering_lane_dispatches_through_named_make_targets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    ordering = makefile.split("test-ordering:", 1)[1].split("duplicate-code:", 1)[0]
    workflow = (ROOT / ".github/workflows/scheduled-validation.yml").read_text(
        encoding="utf-8"
    )

    assert "ORDERING_LANE is required" in ordering
    assert "$(MAKE) test-$(ORDERING_LANE)" in ordering
    assert "uv run --locked pytest" not in ordering
    assert (
        "lane: [unit, component, domain, composition, storage, process, mcp, e2e]"
        in workflow
    )
    assert "ORDERING_LANE: ${{ matrix.lane }}" in workflow
    assert "run: make security-audit" in workflow
    assert "run: make duplicate-code" in workflow
