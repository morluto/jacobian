from __future__ import annotations

import json
from pathlib import Path

from tests.boundary.process.tooling.ci import run_ci_script


def _write_inputs(tmp_path: Path, *, paths: str = "README.md\n") -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan = tmp_path / "plan.txt"
    changed = tmp_path / "paths.txt"
    plan.write_text("classification=docs\nrun-docs=true\n", encoding="utf-8")
    changed.write_text(paths, encoding="utf-8")
    return plan, changed


def _emit(tmp_path: Path, *, paths: str = "README.md\n") -> dict[str, object]:
    plan, changed = _write_inputs(tmp_path, paths=paths)
    output = tmp_path / "receipt.json"
    run_ci_script(
        "emit-plan-receipt",
        "--kind",
        "product-ci",
        "--event",
        "pull_request",
        "--base",
        "a" * 40,
        "--head",
        "b" * 40,
        "--planner",
        ".github/scripts/classify-ci-paths",
        "--config",
        ".github/ci-impact.json",
        "--config",
        "tests/topology.toml",
        "--config",
        ".github/scripts/validate-ci-plan",
        "--config",
        ".github/workflows/ci.yml",
        "--config",
        "Makefile",
        "--plan-file",
        plan,
        "--paths-file",
        changed,
        "--output",
        output,
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_plan_receipt_binds_plan_paths_source_and_configuration(tmp_path: Path) -> None:
    receipt = _emit(tmp_path)

    assert receipt["receipt_version"] == "1"
    assert receipt["plan_kind"] == "product-ci"
    assert receipt["changed_paths"] == ["README.md"]
    assert str(receipt["changed_paths_digest"]).startswith("sha256:")
    assert str(receipt["planner_digest"]).startswith("sha256:")
    assert set(receipt["config_digests"]) == {
        ".github/ci-impact.json",
        "tests/topology.toml",
        ".github/scripts/validate-ci-plan",
        ".github/workflows/ci.yml",
        "Makefile",
    }
    assert str(receipt["plan_digest"]).startswith("sha256:")
    assert str(receipt["receipt_digest"]).startswith("sha256:")


def test_plan_receipt_changes_when_changed_paths_change(tmp_path: Path) -> None:
    first = _emit(tmp_path / "first")
    second = _emit(tmp_path / "second", paths="CONTRIBUTING.md\n")

    assert first["changed_paths_digest"] != second["changed_paths_digest"]
    assert first["receipt_digest"] != second["receipt_digest"]


def test_plan_receipt_rejects_absolute_changed_paths(tmp_path: Path) -> None:
    plan, changed = _write_inputs(tmp_path, paths="/tmp/outside\n")
    result = run_ci_script(
        "emit-plan-receipt",
        "--kind",
        "product-ci",
        "--event",
        "pull_request",
        "--planner",
        ".github/scripts/classify-ci-paths",
        "--config",
        ".github/ci-impact.json",
        "--plan-file",
        plan,
        "--paths-file",
        changed,
    )

    assert result.returncode != 0
    assert "inside the repository" in result.stderr
