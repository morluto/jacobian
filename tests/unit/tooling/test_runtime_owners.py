"""Unit tests for shared complete-runtime ownership rules."""

from __future__ import annotations

from tools.test_plan.runtime_owners import (
    allows_complete_runtime_fixture,
    allows_create_runtime,
)


def test_fixture_owners_cover_boundary_seams() -> None:
    assert allows_complete_runtime_fixture("tests/boundary/storage/recovery/test_x.py")
    assert allows_complete_runtime_fixture("tests/composition/runtime/test_x.py")
    assert not allows_complete_runtime_fixture("tests/domain/graph/test_x.py")
    assert not allows_complete_runtime_fixture("tests/unit/tooling/test_x.py")


def test_create_runtime_requires_named_boundary_seam() -> None:
    assert allows_create_runtime("tests/boundary/providers/sympy/runtime/test_x.py")
    assert allows_create_runtime("tests/support/runtime_templates.py")
    assert allows_create_runtime("tests/support/runtime_profiles.py")
    assert not allows_create_runtime("tests/support/runtime_instances.py")
    assert not allows_create_runtime("tests/boundary/mcp/test_x.py")
    assert allows_complete_runtime_fixture("tests/boundary/mcp/test_x.py")
