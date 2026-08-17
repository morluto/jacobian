"""Polynomial map operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["polynomial_map_operations"]


def polynomial_map_operations() -> MathTools:
    from jacobian.domains.polynomial_maps.math_tools import (
        POLYNOMIAL_MAP_OPERATIONS,
    )

    return POLYNOMIAL_MAP_OPERATIONS
