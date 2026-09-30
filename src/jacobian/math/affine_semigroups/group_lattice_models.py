"""Typed requests for affine group-lattice operations."""

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.affine_semigroups.group_lattice import (
    MAX_AFFINE_GROUP_LATTICE_LABEL_CHARS,
    AffineGroupLattice,
)
from jacobian.math.affine_semigroups.semigroup import AffineConfiguration

# Envelope the operation admits, stated here so a client generating a request
# from the JSON Schema sees the same numbers the native boundary enforces.
_MAX_ROWS = 8
_MAX_GENERATORS = 10
_MAX_ENTRY_MAGNITUDE = 10**8


class AffineGroupLatticeRequest(StrictModel):
    configuration: AffineConfiguration = Field(
        description=(
            "The affine configuration whose Hermite-normal-form group lattice is "
            f"computed. It must carry 1-{_MAX_ROWS} row labels, 1-{_MAX_GENERATORS} "
            "generator labels, one entry per (row, generator) pair, and entries "
            f"of magnitude below {_MAX_ENTRY_MAGNITUDE}. The row and generator "
            "labels together may use at most "
            f"{MAX_AFFINE_GROUP_LATTICE_LABEL_CHARS} characters. The shared "
            "scalar and nested configuration schemas are deliberately broader, "
            "so these operation-specific limits are stated here; a "
            "schema-shaped request that exceeds them is refused at admission."
        )
    )


__all__ = ["AffineGroupLattice", "AffineGroupLatticeRequest"]
