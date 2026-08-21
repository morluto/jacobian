"""Bounded lattice-reduction and integer normal-form operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.lattices._hnf import HERMITE_NORMAL_FORM_OPERATION
from jacobian.math.lattices._lattice import LATTICE_OPERATIONS
from jacobian.math.lattices._lattice_tools import LATTICE_STRUCTURE_OPERATIONS

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    *LATTICE_OPERATIONS,
    HERMITE_NORMAL_FORM_OPERATION,
    *LATTICE_STRUCTURE_OPERATIONS,
)
