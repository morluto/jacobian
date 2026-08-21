"""Live boundary checks for the fixed one-shot Lean and Mathlib environment."""

from __future__ import annotations

import pytest

from jacobian.math.logic._operations import (
    LeanCheckRequest,
    LeanDeclarationKind,
    LeanDeclarationSearchRequest,
    check_lean_source,
    search_mathlib_declarations,
)

pytestmark = pytest.mark.requires_lean


def test_lean_check_elaborates_a_bounded_source() -> None:
    result = check_lean_source(
        LeanCheckRequest(
            source=(
                "import Mathlib\n"
                "example {R : Type} [Ring R] [LinearOrder R] (a : R) : "
                "|a| ^ 2 = a ^ 2 := by\n"
                "  exact sq_abs a"
            )
        )
    )
    if result.outcome == "UNAVAILABLE":
        pytest.skip("the fixed Lean and Mathlib environment is not installed")

    assert result.outcome == "ELABORATED"
    assert result.diagnostics == ()


def test_mathlib_declaration_search_finds_an_exact_theorem_header() -> None:
    result = search_mathlib_declarations(
        LeanDeclarationSearchRequest(
            name_contains="sq_abs",
            kinds=(LeanDeclarationKind.THEOREM,),
            result_limit=5,
            timeout_seconds=30,
        )
    )
    if result.outcome == "UNAVAILABLE":
        pytest.skip("the fixed Mathlib environment is not installed")

    assert result.outcome == "COMPLETED"
    assert result.stop_reason == "EXHAUSTED"
    assert any(
        declaration.name == "sq_abs"
        and declaration.type.endswith("|a| ^ 2 = a ^ 2")
        and declaration.type_truncated is False
        and declaration.module == "Mathlib.Algebra.Order.Ring.Abs"
        for declaration in result.declarations
    )
