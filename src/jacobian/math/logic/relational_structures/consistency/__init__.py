"""Typed, bounded local-consistency operations for finite CSP instances."""

from jacobian.math.logic.relational_structures.consistency._models import (
    CspDomainConsistency,
    CspDomainRequest,
)
from jacobian.math.logic.relational_structures.consistency.operations import (
    generalized_arc_consistency,
)

__all__ = ["CspDomainConsistency", "CspDomainRequest", "generalized_arc_consistency"]
