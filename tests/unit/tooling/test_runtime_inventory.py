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
