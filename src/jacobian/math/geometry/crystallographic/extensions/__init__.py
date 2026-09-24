"""Finite lattice extensions underlying bounded Bieberbach checks."""

from jacobian.math.geometry.crystallographic.extensions._models import (
    BieberbachFaceOrbitComplex,
    CrystallographicAffineRealization,
    CrystallographicAffineSectionMap,
    CrystallographicExtensionTorsionResult,
    CrystallographicFundamentalDomainResult,
    CrystallographicPolytopePairingRequest,
    CrystallographicPolytopePairingResult,
    FiniteLatticeExtension,
    PolytopeFacetPairing,
)
from jacobian.math.geometry.crystallographic.extensions.face_orbits import (
    quotient_face_orbit_complex,
)
from jacobian.math.geometry.crystallographic.extensions.operations import (
    affine_section_realization,
    check_crystallographic_fundamental_domain,
    decide_extension_torsion,
    pair_crystallographic_polytope_facets,
)

__all__ = [
    "BieberbachFaceOrbitComplex",
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
    "quotient_face_orbit_complex",
]
