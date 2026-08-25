"""Owner-local CI policy tests split from test_ci_execution_policy.py."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]


def _pull_request_trigger(workflow_path: str) -> str:
    workflow = (ROOT / workflow_path).read_text(encoding="utf-8")
    return workflow.split("  pull_request:", 1)[1].split("  merge_group:", 1)[0]


def test_exhaustive_local_reproduction_includes_exhaustive_marker_lane() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    all_ci = makefile.split(
        "test-full: ## Every local semantic pytest lane; not hosted CI, coverage, or docs.",
        1,
    )[1].split("test-stress:", 1)[0]

    assert "$(MAKE) test-math" in all_ci
    assert "$(MAKE) _test-exhaustive" in all_ci
    assert "$(VALIDATION_LOCK) run --target test-full" in all_ci
    assert all_ci.index("$(MAKE) test-math") < all_ci.index("$(MAKE) _test-exhaustive")


def test_focused_math_lane_skips_validation_lock() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    math = makefile.split("test-math:", 1)[1].split("test-catalog:", 1)[0]
    exhaustive = makefile.split("test-exhaustive:", 1)[1].split("test-ordering:", 1)[0]
    harbor = (ROOT / "make" / "harbor.mk").read_text(encoding="utf-8")

    assert "VALIDATION_LOCK" not in math
    assert "$(VALIDATION_LOCK) run --target test-exhaustive" in exhaustive
    assert "$(MAKE) _test-exhaustive" in exhaustive
    assert "$(VALIDATION_LOCK) run --target harbor-check-all" in harbor
    assert "$(VALIDATION_LOCK) run --target harbor-host-validation" in harbor
    assert "$(VALIDATION_LOCK) run --target harbor-oracle-all" in harbor
    assert "harbor-check-all -- $(MAKE) _harbor-check-all" in harbor
    assert "_harbor-check-all: harbor-check _harbor-host-validation" in harbor
    assert "_harbor-oracle-all: _harbor-check-all" in harbor


def test_lanes_use_their_declared_worker_and_fixture_affinity() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    math = makefile.split("test-math:", 1)[1].split("test-catalog:", 1)[0]
    catalog = makefile.split("test-catalog:", 1)[1].split("test-dispatch:", 1)[0]
    integration = makefile.split("test-integration:", 1)[1].split("test-fast:", 1)[0]

    assert "pytest -n 4 --dist worksteal" in math
    assert "pytest -n 2 --dist worksteal" in catalog
    assert "pytest -n 2 --dist worksteal" in integration


def test_paths_file_stays_on_harbor_planning() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    harbor = (ROOT / "make" / "harbor.mk").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "PATHS_FILE" not in makefile
    assert "PATHS_FILE :=" not in harbor
    assert "$(shell mktemp)" not in harbor
    assert "tr '\\n' ' '" not in harbor
    assert '--paths-file "$$tmp_dir/changed-paths.txt"' in harbor
    assert '--output "$$tmp_dir/plan.json"' in harbor
    assert "validate-benchmark-plan" not in harbor
    assert "emit-plan-receipt" not in harbor
    assert "PATHS_FILE" not in workflow
