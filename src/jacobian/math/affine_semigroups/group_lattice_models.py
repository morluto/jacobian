"""Typed requests for affine group-lattice operations."""

from jacobian._models import StrictModel
from jacobian.math.affine_semigroups.group_lattice import AffineGroupLattice
from jacobian.math.affine_semigroups.semigroup import AffineConfiguration


class AffineGroupLatticeRequest(StrictModel):
    configuration: AffineConfiguration


__all__ = ["AffineGroupLattice", "AffineGroupLatticeRequest"]
