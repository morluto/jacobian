"""Extended coding theory operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["coding_theory_extended_operations"]


def coding_theory_extended_operations() -> MathTools:
    from jacobian.domains.coding_theory_extended.math_tools import (
        CODING_THEORY_EXTENDED_OPERATIONS,
    )

    return CODING_THEORY_EXTENDED_OPERATIONS
