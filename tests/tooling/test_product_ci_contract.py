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
    wheel = workflow.split("  wheel:", 1)[1].split("  coverage:", 1)[0]

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

    assert "fromJSON(needs.plan.outputs.boundary_lanes)" in workflow
    assert "needs.plan.outputs.run_boundaries == 'true'" in workflow
    assert "uses: ./.github/actions/run-test-lane" in workflow
    assert 'make test-${{ inputs.lane }} TESTS="$TESTS"' in action


def test_singular_backend_has_a_pinned_required_ci_lane() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    singular = workflow.split("  singular:", 1)[1].split("  wheel:", 1)[0]
    required = workflow.split("  required:", 1)[1]

    assert "SINGULAR_DEBIAN_VERSION: 1:4.4.1+ds-2" in singular
    assert '"singular=${SINGULAR_DEBIAN_VERSION}"' in singular
    assert 'system("version")' in singular
    assert "make test-singular" in singular
    assert "QEPCAD_DEBIAN_VERSION: 1.74+ds-5" in singular
    assert "run_qepcad" in workflow
    assert '"qepcad=${QEPCAD_DEBIAN_VERSION}"' in singular
    assert 'qepcad -v | grep -F "Version B 1.74,"' in singular
    assert "make test-qepcad" in singular
    assert "needs.plan.outputs.run_qepcad == 'true'" in singular
    assert singular.count('RUN_SINGULAR" = true ] || [ "$RUN_QEPCAD" = true') == 2
    assert (
        "needs: [plan, static, math, scale, catalog, catalog_examples, python, boundaries, singular, wheel, coverage]"
        in required
    )
    assert 'selected_result true "$SINGULAR_RESULT"' in required


def test_python_and_boundary_lanes_share_evidence_collection() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    action = (ROOT / ".github/actions/run-test-lane/action.yml").read_text(
        encoding="utf-8"
    )

    assert workflow.count("uses: ./.github/actions/run-test-lane") == 6
    assert "--junitxml=pytest.xml" in action
    assert "-p tools.pytest_timing" in action
    assert "--jacobian-timing-json=timing.json" in action
    assert "pytest_args+=(--cov --cov-report= --cov-fail-under=0)" in action
    assert "inputs.collect-coverage == 'true'" in action
    assert action.count("actions/upload-artifact@") == 4
    assert "${{ inputs.junit-artifact }}-timing" in action
    assert "tools/test_timing_report.py" in action
    assert "uv cache prune --ci" in action


def test_optional_boundary_jobs_have_explicit_workflow_gates() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "ci:full" not in workflow
    assert "name: required" in workflow
    assert "name: Deployment Tests" not in workflow


def test_python_jobs_select_math_and_public_contract_evidence_from_the_plan() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "name: python (${{ matrix.lane }})" in workflow
    assert "name: python (math ${{ matrix.group }}/${{ matrix.splits }})" in workflow
    assert "name: python (catalog)" in workflow
    assert "name: python (catalog examples)" in workflow
    assert "name: python (scale)" in workflow
    scale = workflow.split("  scale:", 1)[1].split("  catalog:", 1)[0]
    assert 'SCALE_WORKERS: "4"' in scale
    assert "tests: ${{ needs.plan.outputs.math_tests }}" in workflow
    assert "tests: tests/integration/catalog/" in workflow
    assert "fromJSON(needs.plan.outputs.python_lanes)" in workflow
    assert "needs.plan.outputs.run_python == 'true'" in workflow
    assert "lane: e2e" not in workflow
    assert "lane: provider" not in workflow
    assert "uses: ./.github/actions/run-test-lane" in workflow
    assert (
        "ORDINARY_TEST_LANES := math catalog dispatch cli tooling integration"
        in makefile
    )
    assert "handoff: lint typecheck test-focused" in makefile
    assert "quick: lint test-focused" in makefile
    assert "quick-scoped: lint-scoped test-focused" in makefile
    assert "check: ## Final broad gate:" in makefile
    assert "check-all: ## Escalation:" in makefile
    assert "$(VALIDATION_LOCK) run --target check -- $(MAKE) _check" in makefile
    assert "$(VALIDATION_LOCK) run --target check-all -- $(MAKE) _check-all" in makefile
    assert "_check: lint typecheck test-fast" in makefile
    assert "_check-all: lint typecheck test-ordinary" in makefile


def test_product_ci_uses_a_versioned_checked_in_test_plan() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "name: plan test evidence" in workflow
    assert "python tools/ci_test_plan.py" in workflow
    assert '--base "$BASE_SHA"' in workflow
    assert '--head "$HEAD_SHA"' in workflow
    assert 'event="$EVENT_NAME"' in workflow
    assert "run_scale: ${{ steps.plan.outputs.run_scale }}" in workflow
    assert "python_lanes: ${{ steps.plan.outputs.python_lanes }}" in workflow
    assert "math_shards: ${{ steps.plan.outputs.math_shards }}" in workflow
    assert (ROOT / "tools" / "ci_test_plan.py").exists()


def test_autofix_dispatch_binds_pr_plans_to_the_exact_commit_range() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    autofix = (ROOT / ".github/workflows/autofix.yml").read_text(encoding="utf-8")

    assert "base_sha:" in ci
    assert "head_sha:" in ci
    assert "DISPATCH_BASE_SHA: ${{ inputs.base_sha }}" in ci
    assert "DISPATCH_HEAD_SHA: ${{ inputs.head_sha }}" in ci
    assert 'BASE_SHA="$DISPATCH_BASE_SHA"' in ci
    assert 'HEAD_SHA="$DISPATCH_HEAD_SHA"' in ci
    assert 'test "$HEAD_SHA" = "$(git rev-parse HEAD)"' in ci
    assert 'git merge-base "$BASE_SHA" "$HEAD_SHA" >/dev/null' in ci
    assert "PR_BASE_SHA: ${{ github.event.pull_request.base.sha }}" in autofix
    assert '-f base_sha="$PR_BASE_SHA"' in autofix
    assert '-f head_sha="$(git rev-parse HEAD)"' in autofix


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


def test_product_ci_uses_bounded_timing_balanced_math_shards() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    action = (ROOT / ".github/actions/run-test-lane/action.yml").read_text(
        encoding="utf-8"
    )

    assert "fromJSON(needs.plan.outputs.math_shards)" in workflow
    assert "max-parallel: 4" in workflow
    assert "--splits" in action
    assert "--group" in action
    assert "--splitting-algorithm least_duration" in action
    assert "--store-durations" in action
    assert "name: math-test-durations" in workflow
    assert "needs.math.result == 'success'" in workflow


def test_scheduled_deferred_lanes_run_as_independent_jobs() -> None:
    workflow = (ROOT / ".github/workflows/scheduled-validation.yml").read_text(
        encoding="utf-8"
    )
    stress = workflow.split("  stress:", 1)[1].split("  exhaustive:", 1)[0]
    exhaustive = workflow.split("  exhaustive:", 1)[1].split("  scale:", 1)[0]
    scale = workflow.split("  scale:", 1)[1].split("  ordering:", 1)[0]

    assert "run: make test-stress" in stress
    assert "run: make test-exhaustive" not in stress
    assert "run: make test-scale" not in stress
    assert "run: make test-exhaustive" in exhaustive
    assert "run: make test-scale" in scale
    assert 'SCALE_WORKERS: "4"' in scale
    assert all(
        "uses: ./.github/actions/setup-python-tests" in job
        for job in (stress, exhaustive, scale)
    )


def test_coverage_report_lives_in_the_job_summary_not_a_pr_comment() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    coverage = workflow.split("  coverage:", 1)[1].split("  required:", 1)[0]

    assert "$GITHUB_STEP_SUMMARY" in coverage
    assert "jacobian-coverage-report" not in workflow
    assert "issues/$PR_NUMBER/comments" not in workflow
    assert "pull-requests: write" not in coverage


def test_required_accepts_only_explicitly_skipped_pr_evidence() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    required = workflow.split("  required:", 1)[1]

    assert "selected_result" in required
    assert "true:success|false:skipped" in required
    assert 'selected_result "$RUN_SCALE" "$SCALE_RESULT"' in required
    assert 'selected_result "$RUN_WHEEL" "$WHEEL_RESULT"' in required
    assert 'test "$COVERAGE_RESULT" = skipped' in required
    assert 'test "$COVERAGE_RESULT" = success' in required


def _scale_marked_tests(path: Path) -> set[str]:
    """Return test-function names carrying the ``scale`` tier marker."""

    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    marked: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            candidate = decorator
            if isinstance(candidate, ast.Call):
                candidate = candidate.func
            if not isinstance(candidate, ast.Attribute):
                continue
            if candidate.attr != "scale":
                continue
            owner = candidate.value
            if isinstance(owner, ast.Attribute) and owner.attr == "mark":
                marked.add(node.name)
    return marked


def _all_test_functions(path: Path) -> set[str]:
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def test_scale_evidence_keeps_an_ordinary_regression_in_its_module() -> None:
    """A module may move near-envelope proof to ``scale`` only with ordinary peers.

    The tier contract in ``docs/reference/testing-strategy.md`` requires a small
    ordinary regression for the same public behavior.  Every module that owns a
    ``scale`` test must also own at least one ordinary test, so a module cannot
    be wholly demoted out of the lane that ``main`` always runs.  A module whose
    filename explicitly declares the scale purpose (``*_scale.py`` or
    ``test_scale*.py``) is dedicated scale evidence and is exempt.
    """

    math_root = ROOT / "tests" / "math"
    offenders: list[str] = []
    scale_modules = 0
    for path in sorted(math_root.rglob("test_*.py")):
        marked = _scale_marked_tests(path)
        if not marked:
            continue
        scale_modules += 1
        if path.stem.endswith("scale") or path.stem.startswith("test_scale"):
            continue
        ordinary = _all_test_functions(path) - marked
        if not ordinary:
            offenders.append(str(path.relative_to(ROOT)))

    assert scale_modules > 0, "expected at least one scale-marked module"
    assert offenders == [], (
        "scale-only test modules must keep an ordinary regression: "
        + ", ".join(offenders)
    )


def test_ordinary_math_lane_excludes_every_heavy_tier() -> None:
    """The ordinary lane expression excludes all non-ordinary execution tiers."""

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    line = next(
        entry
        for entry in makefile.splitlines()
        if entry.startswith("ORDINARY_MARKER_EXPRESSION")
    )

    for tier in ("property", "exhaustive", "scale"):
        assert f"not {tier}" in line, f"{tier} must be excluded from the ordinary lane"
