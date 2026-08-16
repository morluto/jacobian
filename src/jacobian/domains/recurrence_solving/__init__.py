"""Recurrence solving operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools
__all__ = ["recurrence_solving_operations"]


def recurrence_solving_operations() -> MathTools:
    from jacobian.domains.recurrence_solving.math_tools import (
        RECURRENCE_SOLVING_OPERATIONS,
    )

    return RECURRENCE_SOLVING_OPERATIONS
