"""Bounded lattice-reduction and integer normal-form operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["lattice_operations"]


def lattice_operations() -> MathTools:
    from jacobian.domains.lattices.hnf import HERMITE_NORMAL_FORM_OPERATION
    from jacobian.domains.lattices.lattice import LATTICE_OPERATIONS

    return (*LATTICE_OPERATIONS, HERMITE_NORMAL_FORM_OPERATION)
