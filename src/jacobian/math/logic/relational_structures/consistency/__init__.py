"""Typed, bounded local-consistency operations for finite CSP instances."""

from jacobian.math.logic.relational_structures.consistency._models import (
    CspDomainConsistency,
)
from jacobian.math.logic.relational_structures.consistency.operations import (
    generalized_arc_consistency,
)

__all__ = ["CspDomainConsistency", "generalized_arc_consistency"]
