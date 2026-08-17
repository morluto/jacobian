"""Exact electrical-network operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["electrical_network_operations"]


def electrical_network_operations() -> MathTools:
    from jacobian.domains.electrical_networks.math_tools import (
        ELECTRICAL_NETWORK_OPERATIONS,
    )

    return ELECTRICAL_NETWORK_OPERATIONS
