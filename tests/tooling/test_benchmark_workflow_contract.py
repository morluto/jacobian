"""Owner-local CI policy tests split from test_ci_execution_policy.py."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]


def _pull_request_trigger(workflow_path: str) -> str:
    workflow = (ROOT / workflow_path).read_text(encoding="utf-8")
    return workflow.split("  pull_request:", 1)[1].split("  merge_group:", 1)[0]


def test_benchmark_labels_replan_without_generic_pr_edit_restarts() -> None:
    pull_request_trigger = _pull_request_trigger(".github/workflows/benchmarks.yml")

    assert "edited" not in pull_request_trigger
    assert "labeled" in pull_request_trigger
    assert "unlabeled" in pull_request_trigger


def test_benchmark_workflow_has_distinct_pr_merge_and_full_portfolio_tiers() -> None:
    workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(encoding="utf-8")

    assert "schedule:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "EVENT_NAME: ${{ github.event_name }}" in workflow
    assert '--event "$EVENT_NAME"' in workflow
    assert '--output "$plan_dir/plan.json"' in workflow
    assert '--github-output "$GITHUB_OUTPUT"' in workflow
    assert "validate-benchmark-plan" not in workflow
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
    assert "python .github/scripts/emit-plan-receipt" not in workflow
    assert "benchmark-plan-receipt" not in workflow
    assert "name: benchmark-plan" in workflow


def test_benchmark_stable_gate_validates_provenance_receipts_in_python() -> None:
    workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(encoding="utf-8")
    validation = workflow.split("  validation:", 1)[1].split("  timings:", 1)[0]

    assert "benchmarks.tooling.benchmark_validation" in validation
    assert "--plan" in validation
    assert "plan.json" in validation
    assert "benchmark-plan-receipt" not in validation
    assert "benchmark-host-timing-*" in validation
    assert "benchmark-test-durations-input" in validation
    assert '--execution-sha "${{ github.sha }}"' in validation
    assert "check_lane()" not in validation


def test_benchmark_plan_is_shown_as_json_in_the_job_summary() -> None:
    workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(encoding="utf-8")

    assert "echo '```json'" in workflow
    assert 'cat "$plan_dir/plan.json"' in workflow
    assert "Plan receipt:" not in workflow


def test_benchmark_job_outputs_are_only_if_projections() -> None:
    workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(encoding="utf-8")
    outputs = workflow.split("    outputs:", 1)[1].split("    steps:", 1)[0]

    assert "run-benchmark-check:" in outputs
    assert "run-benchmark-record-schema:" in outputs
    assert "run-benchmark-inventory:" in outputs
    assert "run-benchmark-host-validation:" in outputs
    assert "benchmark-host-validation-matrix:" in outputs
    assert "run-benchmark-oracle:" in outputs
    assert "benchmark-oracle-matrix:" in outputs
    assert "benchmark-plan-version:" not in outputs
    assert "benchmark-planner-digest:" not in outputs
    assert "benchmark-plan-reasons:" not in outputs
    assert "run-benchmark-prospective-digest:" not in outputs
    assert "benchmark-oracle-scope:" not in outputs
    assert "PROSPECTIVE_DIGEST_FLAG" not in workflow


def test_ci_does_not_schedule_deleted_checker_worker_coverage() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "subprocess_coverage:" not in workflow
    assert "test-checker-subprocess-coverage" not in (ROOT / "Makefile").read_text(
        encoding="utf-8"
    )


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
