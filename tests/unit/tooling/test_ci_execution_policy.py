from __future__ import annotations

import json
from pathlib import Path

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

    assert set(rule["suites"]) <= {
        "unit",
        "component",
        "domain",
        "composition",
        "storage",
        "process",
        "mcp",
        "e2e",
        "static",
        "build",
    }
    assert not {"lean", "npm"}.intersection(rule["suites"])


def test_global_timeout_is_not_a_pytest_deadline() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "timeout = 60" not in pyproject
    assert 'timeout_method = "thread"' in pyproject


def test_pre_push_hook_stays_in_the_static_feedback_lane() -> None:
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    hook = config.split("id: jacobian-pre-push", 1)[1]

    assert "entry: make lint typecheck" in hook
    assert "entry: make check" not in hook


def test_process_lane_is_invoked_by_ci() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "process-test:" in workflow
    assert "run: make test-process" in workflow


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

    assert "needs: [plan, record-schema, prospective-digest]" in oracle
    assert "make harbor-oracle-task" in oracle
    assert "make harbor-oracle DATASET" not in oracle


def test_local_oracle_targets_require_explicit_scope() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    oracle = makefile.split("harbor-oracle:", 1)[1].split("harbor-oracle-task:", 1)[0]
    runner = makefile.split("harbor-oracle-run:", 1)[1].split("harbor-oracle-all:", 1)[
        0
    ]

    assert '"$(TASKS)" -o "$(FULL)" = "1"' in oracle
    assert '"$(TASKS)" -o "$(FULL)" = "1"' in runner
    assert "DATASET=$$dataset FULL=1" in makefile


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

    assert "test-architecture:" in makefile
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
