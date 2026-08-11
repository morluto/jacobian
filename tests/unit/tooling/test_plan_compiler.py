"""Unit tests for the authoritative test-plan compiler."""

from __future__ import annotations

import json
from pathlib import Path

from tools.test_plan.compile import compile_manifest, impact_json, load_manifest


def test_plan_manifest_loads_pytest_lanes() -> None:
    root = Path(__file__).resolve().parents[3]
    manifest = load_manifest(root / "tests" / "plan_manifest.toml")
    names = {lane.name for lane in manifest.pytest_lanes}
    assert {
        "unit",
        "component",
        "domain",
        "composition",
        "storage",
        "process",
        "mcp",
        "provider",
        "lean",
        "e2e",
    } <= names


def test_compile_topology_matches_checked_in_projection() -> None:
    root = Path(__file__).resolve().parents[3]
    result = compile_manifest(root / "tests" / "plan_manifest.toml")
    assert (root / "tests" / "topology.toml").read_text(
        encoding="utf-8"
    ) == result.topology
    assert 'execution_profile = "' in result.topology
    assert "process_supervision =" in result.topology


def test_compile_impact_matches_checked_in_projection() -> None:
    root = Path(__file__).resolve().parents[3]
    result = compile_manifest(root / "tests" / "plan_manifest.toml")
    checked_in = (root / ".github" / "ci-impact.json").read_text(encoding="utf-8")
    assert checked_in == impact_json(result.impact)
    assert result.impact["meta"]["compiler"] == "tools.test_plan.compile"
    assert result.impact["fallback"]["name"] == "unclassified-fail-closed"
    assert result.impact["catalog"]["component"]["matrix"] is True
    assert "matrix" not in result.impact["catalog"]["unit"]


def test_manifest_carries_suppression_rules() -> None:
    root = Path(__file__).resolve().parents[3]
    result = compile_manifest(root / "tests" / "plan_manifest.toml")
    assert "domain-mathematical-sources" in result.suppressions
    assert "python-source" in result.suppressions["domain-mathematical-sources"]


def test_derived_lane_tests_are_not_duplicated() -> None:
    root = Path(__file__).resolve().parents[3]
    result = compile_manifest(root / "tests" / "plan_manifest.toml")
    names = [rule["name"] for rule in result.impact["rules"]]
    assert len(names) == len(set(names))
    assert "unit-tests" in names
    assert "provider-tests" in names
    assert "lean-tests" in names
    assert "e2e-tests" in names
    assert "optional-provider-tests" not in names
    assert "lean-python-tests" not in names
    assert "end-to-end-tests" not in names
    provider_rule = next(
        rule for rule in result.impact["rules"] if rule["name"] == "provider-tests"
    )
    assert provider_rule["suites"] == ["provider", "static"]


def test_impact_rule_count_covers_manifest_plus_derived_lanes() -> None:
    root = Path(__file__).resolve().parents[3]
    manifest = load_manifest(root / "tests" / "plan_manifest.toml")
    result = compile_manifest(root / "tests" / "plan_manifest.toml")
    assert len(result.impact["rules"]) == len(manifest.impact_rules) + len(
        manifest.pytest_lanes
    )
    payload = json.loads(impact_json(result.impact))
    assert payload["version"] == 2
