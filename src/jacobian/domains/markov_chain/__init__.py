"""Markov chain operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools
__all__ = ["markov_chain_operations"]


def markov_chain_operations() -> MathTools:
    from jacobian.domains.markov_chain.math_tools import MARKOV_CHAIN_OPERATIONS

    return MARKOV_CHAIN_OPERATIONS
