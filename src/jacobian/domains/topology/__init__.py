"""Finite simplicial topology domain."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["topology_operations"]


def topology_operations() -> MathTools:
    from jacobian.domains.topology.operations import TOPOLOGY_OPERATIONS

    return TOPOLOGY_OPERATIONS
