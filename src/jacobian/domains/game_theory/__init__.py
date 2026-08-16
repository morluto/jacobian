"""Finite game theory operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["game_theory_operations"]


def game_theory_operations() -> MathTools:
    from jacobian.domains.game_theory.math_tools import GAME_THEORY_OPERATIONS

    return GAME_THEORY_OPERATIONS
