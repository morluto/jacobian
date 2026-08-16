"""Code theory operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools
__all__ = ["code_theory_operations"]


def code_theory_operations() -> MathTools:
    from jacobian.domains.code_theory.math_tools import CODE_THEORY_OPERATIONS

    return CODE_THEORY_OPERATIONS
