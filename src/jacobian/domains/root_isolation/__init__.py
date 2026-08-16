"""Root isolation operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools
__all__ = ["root_isolation_operations"]


def root_isolation_operations() -> MathTools:
    from jacobian.domains.root_isolation.math_tools import ROOT_ISOLATION_OPERATIONS

    return ROOT_ISOLATION_OPERATIONS
