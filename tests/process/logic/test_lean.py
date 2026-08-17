"""Live boundary checks for the fixed one-shot Lean toolchain."""

from __future__ import annotations

import pytest

from jacobian.math.logic._operations import LeanCheckRequest, check_lean_source

pytestmark = pytest.mark.requires_lean


def test_lean_check_elaborates_a_bounded_source() -> None:
    result = check_lean_source(LeanCheckRequest(source="example : True := by trivial"))
    if result.outcome == "UNAVAILABLE":
        pytest.skip("the fixed Lean toolchain is not installed")

    assert result.outcome == "ELABORATED"
    assert result.diagnostics == ()
