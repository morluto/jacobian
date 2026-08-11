from __future__ import annotations

import argparse
import json
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


def test_python_313_uses_the_narrow_compatibility_smoke() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    compatibility = workflow.split("  compatibility-test:", 1)[1].split(
        "  python-test:", 1
    )[0]

    assert 'python-version: "3.13"' in compatibility
    assert "make test-compatibility" in compatibility
    assert "make test\n" not in compatibility


def test_makefile_changes_do_not_route_to_unrelated_provider_lanes() -> None:
    manifest = json.loads((ROOT / ".github/ci-impact.json").read_text(encoding="utf-8"))
    rule = next(rule for rule in manifest["rules"] if rule["name"] == "makefile")

    # Command-index edits stay narrow; lane topology lives in tools/ and
    # tests/topology.toml under test-topology-runners.
    assert set(rule["suites"]) == {"static", "build"}
    assert not {"lean", "npm", "provider"}.intersection(rule["suites"])


def test_makefile_exposes_separate_local_and_hosted_plans() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    harbor = (ROOT / "make" / "harbor.mk").read_text(encoding="utf-8")

    assert "ci-plan:" in makefile
    assert "test-plan:" in makefile
    assert "include make/harbor.mk" in makefile
    assert "harbor-plan:" in harbor


def test_domain_mathematical_sources_skip_storage_mcp_and_e2e() -> None:
    manifest = json.loads((ROOT / ".github/ci-impact.json").read_text(encoding="utf-8"))
    rule = next(
        rule
        for rule in manifest["rules"]
        if rule["name"] == "domain-mathematical-sources"
    )

    assert set(rule["suites"]) == {
        "unit",
        "component",
        "domain",
        "static",
        "build",
    }
    assert not {"storage", "process", "mcp", "e2e", "lean", "npm"}.intersection(
        rule["suites"]
    )


def test_benchmark_ci_changes_do_not_trigger_product_semantic_lanes() -> None:
    manifest = json.loads((ROOT / ".github/ci-impact.json").read_text(encoding="utf-8"))
    rule = next(
        rule for rule in manifest["rules"] if rule["name"] == "benchmark-ci-automation"
    )

    assert set(rule["suites"]) == {"unit", "process", "static", "build"}
    assert not {"domain", "composition", "storage", "mcp", "e2e"}.intersection(
        rule["suites"]
    )


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

    assert "process-test:" in workflow
    assert "run: make test-process" in workflow


def test_provider_opt_in_and_deployment_have_explicit_workflow_gates() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "ci:provider" in workflow
    assert "--include-provider" in workflow
    assert "run-deploy: ${{ steps.classify.outputs.run-deploy }}" in workflow
    assert "deployment-check:" in workflow
    assert "run: make deploy-check" in workflow
    assert "deployment-test:" in workflow


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


def test_product_ci_publishes_a_provenance_bound_plan_receipt() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "python .github/scripts/emit-plan-receipt" in workflow
    assert "ci-plan-receipt" in workflow
    assert "plan-receipt-digest" in workflow


def test_documentation_job_installs_uv_before_make_docs_linkcheck() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    docs = workflow.split("  docs:", 1)[1].split("  security-audit:", 1)[0]

    assert "astral-sh/setup-uv@" in docs
    assert 'version: "0.11.28"' in docs
    assert "run: make docs-linkcheck" in docs


def test_plan_receipt_digests_are_rendered_as_markdown_code() -> None:
    for workflow_name in ("ci.yml", "benchmarks.yml"):
        workflow = (ROOT / ".github/workflows" / workflow_name).read_text(
            encoding="utf-8"
        )

        assert "Plan receipt: \\`$(python" in workflow
        assert "Plan receipt: \\\\`$(python" not in workflow


def test_required_ci_gates_fail_when_the_plan_is_cancelled() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "treating gate as non-failure" not in workflow
    assert workflow.count("if: ${{ always() }}") >= 8


def test_local_oracle_targets_require_explicit_scope() -> None:
    harbor = (ROOT / "make" / "harbor.mk").read_text(encoding="utf-8")
    oracle = harbor.split("harbor-oracle:", 1)[1].split("harbor-oracle-task:", 1)[0]
    runner = harbor.split("harbor-oracle-run:", 1)[1].split("harbor-oracle-all:", 1)[0]

    assert '"$(TASKS)" -o "$(FULL)" = "1"' in oracle
    assert '"$(TASKS)" -o "$(FULL)" = "1"' in runner
    assert "DATASET=$$dataset FULL=1" in harbor


def test_local_oracle_attempts_are_serialized_on_a_shared_docker_host() -> None:
    harbor = (ROOT / "make" / "harbor.mk").read_text(encoding="utf-8")

    assert "HARBOR_ORACLE_LOCK ?= benchmarks/results/.harbor-oracle.lock" in harbor
    assert harbor.count('exec 9>"$(HARBOR_ORACLE_LOCK)"; flock 9;') == 2
    assert "HARBOR_ORACLE_DOCKER_BUILD_MODE ?= auto" in harbor
    assert "export DOCKER_BUILDKIT=0 COMPOSE_BAKE=false" in harbor


def test_composition_lane_uses_timing_shards() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "composition-shard-count:" in workflow
    assert "composition-test:" in workflow
    assert "--splits ${{ needs.plan.outputs.composition-shard-count }}" in workflow
    assert "composition-test-durations-input" in workflow


def test_timing_shard_configuration_matches_topology() -> None:
    import tomllib

    topology = tomllib.loads((ROOT / "tests/topology.toml").read_text(encoding="utf-8"))
    config = json.loads((ROOT / ".github/ci-config.json").read_text(encoding="utf-8"))
    timed_lanes = {
        lane["name"] for lane in topology["lanes"] if lane["timing_sharding"]
    }

    assert timed_lanes == {"domain", "composition"}
    assert {
        key.removesuffix("_shard_count")
        for key in config
        if key.endswith("_shard_count")
    } == timed_lanes


def test_stress_lane_selects_only_property_tests_and_repeats_them() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    stress = makefile.split("test-stress:", 1)[1].split("test-ordering:", 1)[0]

    assert "-m property" in stress
    assert "--count=$(STRESS_COUNT)" in stress


def test_ordering_lane_dispatches_through_the_semantic_runner() -> None:
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


def test_static_validation_enforces_test_architecture() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    development = (ROOT / "make" / "development.mk").read_text(encoding="utf-8")

    assert "test-architecture:" in development
    assert "check-static: lint-full typecheck test-architecture" in makefile


def test_process_and_provider_lanes_have_explicit_resource_policies() -> None:
    import tomllib

    manifest = tomllib.loads((ROOT / "tests/topology.toml").read_text(encoding="utf-8"))
    lanes = {lane["name"]: lane for lane in manifest["lanes"]}

    assert lanes["process"]["workers"] == 2
    assert lanes["process"]["timeout_seconds"] == 120
    assert lanes["process"]["required_environment"] == ["process-group"]
    assert lanes["provider"]["workers"] == 1
    assert lanes["provider"]["required_provider"] == "optional"
    assert lanes["provider"]["ci"]["pull_request"] is False
