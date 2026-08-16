"""Exact graph spectral operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["graph_spectral_operations"]


def graph_spectral_operations() -> MathTools:
    from jacobian.domains.graph_spectral.math_tools import GRAPH_SPECTRAL_OPERATIONS

    return GRAPH_SPECTRAL_OPERATIONS
