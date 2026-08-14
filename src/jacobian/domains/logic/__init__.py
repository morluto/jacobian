"""Exact logic operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["logic_operations"]


def logic_operations() -> MathTools:
    from jacobian.domains.logic.operations import LOGIC_OPERATIONS

    return LOGIC_OPERATIONS
