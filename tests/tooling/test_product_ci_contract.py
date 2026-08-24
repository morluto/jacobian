"""Owner-local CI policy tests split from test_ci_execution_policy.py."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]


def _pull_request_trigger(workflow_path: str) -> str:
    workflow = (ROOT / workflow_path).read_text(encoding="utf-8")
    return workflow.split("  pull_request:", 1)[1].split("  merge_group:", 1)[0]


def test_pr_metadata_edits_do_not_restart_product_ci() -> None:
    pull_request_trigger = _pull_request_trigger(".github/workflows/ci.yml")

    assert "edited" not in pull_request_trigger
    assert "labeled" not in pull_request_trigger
    assert "unlabeled" not in pull_request_trigger


def test_wheel_job_covers_supported_pythons_and_313_compatibility_smoke() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    wheel = workflow.split("  wheel:", 1)[1].split("  lean:", 1)[0]

    assert 'python-version: ["3.12", "3.13"]' in wheel
    assert "--only-binary :all:" in wheel
    assert '"$environment/bin/jacobian" run integer.compute.extended_gcd' in wheel
    assert "make test-compatibility" in wheel
    assert "make test\n" not in wheel


def test_global_timeout_is_not_a_pytest_deadline() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "timeout = 60" not in pyproject
    assert 'timeout_method = "thread"' in pyproject


def test_process_lane_is_invoked_by_ci() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    action = (ROOT / ".github/actions/run-test-lane/action.yml").read_text(
        encoding="utf-8"
    )

    assert "lane: [process, mcp]" in workflow
    assert "uses: ./.github/actions/run-test-lane" in workflow
    assert "run: make test-${{ inputs.lane }}" in action


def test_singular_backend_has_a_pinned_required_ci_lane() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    singular = workflow.split("  singular:", 1)[1].split("  wheel:", 1)[0]
    required = workflow.split("  required:", 1)[1]

    assert "SINGULAR_DEBIAN_VERSION: 1:4.4.1+ds-2" in singular
    assert '"singular=${SINGULAR_DEBIAN_VERSION}"' in singular
    assert 'system("version")' in singular
    assert "make test-singular" in singular
    assert (
        "needs: [static, python, boundaries, singular, wheel, coverage, lean]"
        in required
    )
    assert 'test "$SINGULAR_RESULT" = success' in required


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


def test_full_lean_runs_on_merge_group_and_main() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    action = (ROOT / ".github/actions/setup-lean/action.yml").read_text(
        encoding="utf-8"
    )

    lean = workflow.split("  lean:", 1)[1].split("  coverage:", 1)[0]
    # Pull requests and autofix dispatches that select pr_lanes skip Lean;
    # merge groups and main pushes always run it.
    assert "github.event_name != 'pull_request'" in lean
    assert "inputs.pr_lanes" in lean
    assert "inputs.pr_lanes" in workflow.split("EVENT_NAME", 1)[0] or (
        "PR_LANE_SELECTION" in workflow
    )
    assert "JACOBIAN_LEAN_REQUIRED" not in lean
    assert "python tools/setup_lean.py --repo ." in action
    assert "lake-manifest.json" not in action
    assert "leanprover/lean-action" not in action
    assert "JacobianLeanRuntime" not in action
    assert "jacobian_lean_proof_state" not in action
    assert "preflight_lean_runtime" not in action
    assert "Mathlib" not in action
    assert "uses: actions/cache" not in action


def test_optional_boundary_jobs_have_explicit_workflow_gates() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "ci:full" not in workflow
    assert "ci:lean" not in workflow
    assert "name: required" in workflow
    assert "name: Deployment Tests" not in workflow


def test_python_jobs_use_fixed_local_semantic_targets() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "name: python (${{ matrix.lane }})" in workflow
    for lane in ("math", "catalog", "dispatch", "cli", "tooling", "integration"):
        assert f"lane: {lane}" in workflow
    assert "lane: e2e" not in workflow
    assert "lane: provider" not in workflow
    assert "uses: ./.github/actions/run-test-lane" in workflow
    assert (
        "ORDINARY_TEST_LANES := math catalog dispatch cli tooling integration"
        in makefile
    )
    assert "quick: lint test-fast" in makefile
    assert "check: lint typecheck test-fast" in makefile
    assert "check-all: lint typecheck test-ordinary" in makefile
    assert "check-external: test-lean ##" in makefile
    assert "check-external: test-lean test-provider" not in makefile


def test_product_ci_does_not_emit_a_plan_receipt() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "python .github/scripts/emit-plan-receipt" not in workflow
    assert "ci-plan-receipt" not in workflow
    assert "classify-ci-paths" not in workflow
    assert not (ROOT / ".github/scripts/emit-plan-receipt").exists()
    assert not (ROOT / ".github/scripts/validate-benchmark-plan").exists()


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


def test_product_ci_does_not_use_timing_shards() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "composition-shard-count:" not in workflow
    assert "--splits" not in workflow
    assert "pytest-split" not in workflow


def test_coverage_report_lives_in_the_job_summary_not_a_pr_comment() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    coverage = workflow.split("  coverage:", 1)[1].split("  required:", 1)[0]

    assert "$GITHUB_STEP_SUMMARY" in coverage
    assert "jacobian-coverage-report" not in workflow
    assert "issues/$PR_NUMBER/comments" not in workflow
    assert "pull-requests: write" not in coverage
