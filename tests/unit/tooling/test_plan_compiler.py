"""Unit tests for the authoritative test-plan compiler."""

from __future__ import annotations

from pathlib import Path

from tools.test_plan.compile import compile_manifest, load_manifest


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


def test_manifest_carries_suppression_rules() -> None:
    root = Path(__file__).resolve().parents[3]
    result = compile_manifest(root / "tests" / "plan_manifest.toml")
    assert "domain-mathematical-sources" in result.suppressions
    assert "python-source" in result.suppressions["domain-mathematical-sources"]
