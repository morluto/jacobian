"""Finite game theory operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["finite_game_theory_operations"]


def finite_game_theory_operations() -> MathTools:
    from jacobian.domains.finite_game_theory.math_tools import (
        FINITE_GAME_THEORY_OPERATIONS,
    )

    return FINITE_GAME_THEORY_OPERATIONS
