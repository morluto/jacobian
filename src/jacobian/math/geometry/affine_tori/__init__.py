"""Exact affine geometry on standard real tori."""

from jacobian.math.geometry.affine_tori._models import (
    AffineTorusFixedLocusOutcome,
    AffineTorusFixedLocusResult,
    EmptyAffineTorusFixedLocus,
    NonemptyAffineTorusFixedLocus,
)
from jacobian.math.geometry.affine_tori.operations import (
    affine_torus_fixed_locus,
    verify_integral_torus_character,
)
from jacobian.math.geometry.affine_tori.values import (
    ConnectedSubtorusParameterization,
    FiniteTorusComponentPresentation,
    IntegralTorusCharacter,
    RationalAffineTorusMap,
    RationalTorusCosetFamily,
    RationalTorusPoint,
    StandardRealTorus,
)

__all__ = [
    "AffineTorusFixedLocusOutcome",
    "AffineTorusFixedLocusResult",
    "ConnectedSubtorusParameterization",
    "EmptyAffineTorusFixedLocus",
    "FiniteTorusComponentPresentation",
    "IntegralTorusCharacter",
    "NonemptyAffineTorusFixedLocus",
    "RationalAffineTorusMap",
    "RationalTorusCosetFamily",
    "RationalTorusPoint",
    "StandardRealTorus",
    "affine_torus_fixed_locus",
    "verify_integral_torus_character",
]
