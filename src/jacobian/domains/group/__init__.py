"""Exact finite group operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["group_operations"]


def group_operations() -> MathTools:
    from jacobian.domains.group.math_tools import GROUP_OPERATIONS

    return GROUP_OPERATIONS
