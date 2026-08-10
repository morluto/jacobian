"""Unit tests for complete-runtime inventory reporting."""

from __future__ import annotations

from pathlib import Path

from tools.inventory_test_runtime import inventory_modules


def test_inventory_flags_modules_and_owners() -> None:
    root = Path(__file__).resolve().parents[3]
    rows = inventory_modules(root)
    by_path = {row.path: row for row in rows}
    assert "tests/composition/runtime/test_structure_canonicalization.py" in by_path
    row = by_path["tests/composition/runtime/test_structure_canonicalization.py"]
    assert row.semantic_owner == "composition"
    assert "attached_complete_runtime" in row.complete_runtime_fixtures
    assert "authorized_complete_runtime" not in row.complete_runtime_fixtures


def test_inventory_accepts_authorized_modules_with_verify_signals() -> None:
    root = Path(__file__).resolve().parents[3]
    rows = inventory_modules(root)
    by_path = {row.path: row for row in rows}
    row = by_path["tests/composition/runtime/test_graph_neighborhood_independence.py"]
    assert "authorized_complete_runtime" in row.complete_runtime_fixtures
    assert row.has_verify_signal
    assert not row.unjustified_authorized


def test_inventory_graph_installation_stays_attached_only() -> None:
    root = Path(__file__).resolve().parents[3]
    rows = inventory_modules(root)
    by_path = {row.path: row for row in rows}
    row = by_path["tests/composition/runtime/test_graph_installation_contract.py"]
    assert row.complete_runtime_fixtures == ("attached_complete_runtime",)
    assert not row.unjustified_authorized
