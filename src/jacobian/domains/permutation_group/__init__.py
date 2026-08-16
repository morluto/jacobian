"""Permutation group operations."""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from jacobian.math_tools import MathTools
__all__ = ["permutation_group_operations"]
def permutation_group_operations() -> MathTools:
    from jacobian.domains.permutation_group.math_tools import PERMUTATION_GROUP_OPERATIONS
    return PERMUTATION_GROUP_OPERATIONS
