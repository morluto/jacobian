"""Exact truncated formal power series operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["formal_power_series_operations"]


def formal_power_series_operations() -> MathTools:
    from jacobian.domains.formal_power_series.math_tools import (
        FORMAL_POWER_SERIES_OPERATIONS,
    )

    return FORMAL_POWER_SERIES_OPERATIONS
