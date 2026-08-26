"""Owner-local geometry admission shared by lattice-polytope contracts.

The request model owns the public structural schema.  This module owns the
coupled exact geometry admission which is needed before a lattice-point scan;
keeping that boundary out of ``_models`` prevents contract construction from
depending directly on an operation module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math.lattice_polytopes._models import LatticePolytopeRequest
    from jacobian.math.lattice_polytopes._operations import AdmittedGeometry


def admitted_geometry(request: LatticePolytopeRequest) -> AdmittedGeometry:
    """Compute the one memoized exact geometry admission for ``request``."""
    # The exact kernel remains private to the owner.  This is the sole bridge
    # from request admission to it, rather than a request-model re-entry.
    from jacobian.math.lattice_polytopes._operations import _facets_and_box

    return _facets_and_box(request)


def enumeration_output_admission(request: LatticePolytopeRequest) -> None:
    """Check the enumerate-only materialization and transport envelope."""
    from jacobian.math.lattice_polytopes._operations import (
        enumeration_output_admission as _admit,
    )

    _admit(request)
