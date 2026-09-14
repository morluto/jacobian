"""Request adapter for exact ring-of-integers basis construction."""

from __future__ import annotations

from jacobian.math.number_theory.number_fields._models import (
    NumberFieldRingOfIntegersRequest,
)
from jacobian.math.number_theory.number_fields._ring_of_integers import (
    NumberFieldRingOfIntegersResult,
    ring_of_integers,
)


def compute_nf_ring_of_integers(
    request: NumberFieldRingOfIntegersRequest,
) -> NumberFieldRingOfIntegersResult:
    """Run the shared native entry for one catalog request.

    The catalog and native paths deliberately converge here: the degree bound,
    the killable worker boundary, and result construction are owned by
    :func:`~jacobian.math.number_theory.number_fields.ring_of_integers`, so a
    direct Python call cannot take a cheaper or less bounded route than
    ``math.run``.
    """

    return ring_of_integers(request.field)


__all__ = ["compute_nf_ring_of_integers"]
