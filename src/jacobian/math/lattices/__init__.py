"""Supported exact lattice API.

Lattice mathematics is owned separately from matrix mathematics.  This module
exposes bounded LLL reduction over integer lattices backed by Python-FLINT,
the row Hermite normal form, and the structural integer-lattice operations
requested in issue #1739.
"""

from jacobian.math.lattices._lattice_operations import (
    compute_canonical_basis,
    compute_direct_sum,
    compute_discriminant_group,
    compute_dual,
    compute_orthogonal_complement,
    compute_orthogonal_sum,
    compute_rank_gram,
    compute_saturation,
    compute_sublattice_index,
)
from jacobian.math.lattices.operations import hermite_normal_form, reduce_basis

__all__ = [
    "compute_canonical_basis",
    "compute_direct_sum",
    "compute_discriminant_group",
    "compute_dual",
    "compute_orthogonal_complement",
    "compute_orthogonal_sum",
    "compute_rank_gram",
    "compute_saturation",
    "compute_sublattice_index",
    "hermite_normal_form",
    "reduce_basis",
]
