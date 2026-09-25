"""Finite lattice extensions underlying bounded Bieberbach checks."""

from jacobian.math.geometry.crystallographic.extensions._models import (
    CrystallographicAffineRealization,
    CrystallographicAffineSectionMap,
    CrystallographicExtensionTorsionResult,
    CrystallographicFundamentalDomainResult,
    CrystallographicPolytopePairingRequest,
    CrystallographicPolytopePairingResult,
    FiniteLatticeExtension,
    PolytopeFacetPairing,
)
from jacobian.math.geometry.crystallographic.extensions.operations import (
    affine_section_realization,
    check_crystallographic_fundamental_domain,
    decide_extension_torsion,
    pair_crystallographic_polytope_facets,
)

__all__ = [
    "CrystallographicAffineRealization",
    "CrystallographicAffineSectionMap",
    "CrystallographicExtensionTorsionResult",
    "CrystallographicFundamentalDomainResult",
    "CrystallographicPolytopePairingRequest",
    "CrystallographicPolytopePairingResult",
    "FiniteLatticeExtension",
    "PolytopeFacetPairing",
    "affine_section_realization",
    "check_crystallographic_fundamental_domain",
    "decide_extension_torsion",
    "pair_crystallographic_polytope_facets",
]
