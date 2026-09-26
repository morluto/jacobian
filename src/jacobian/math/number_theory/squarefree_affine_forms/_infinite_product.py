"""Certified enclosures for simultaneous square-free local Euler products."""

from __future__ import annotations

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.math.analysis.intervals import ClosedRationalInterval
from jacobian.math.number_theory.squarefree_affine_forms._models import (
    MAX_INFINITE_PRODUCT_CUTOFF,
)
from jacobian.math.number_theory.squarefree_affine_forms.values import (
    SquarefreeAffineFamily,
)


class SquarefreeInfiniteProductRequest(StrictModel):
    """Bound the simultaneous square-free Euler product through prime P."""

    source: SquarefreeAffineFamily
    prime_cutoff: StrictInt = Field(
        ge=1,
        le=MAX_INFINITE_PRODUCT_CUTOFF,
        description=(
            "Include every prime p <= this value in the exact prefix. It must "
            "be at least the family-derived tail theorem cutoff."
        ),
    )


class SquarefreeInfiniteProductEnclosure(StrictModel):
    """Exact rational interval containing the simultaneous infinite product.

    For locally admissible families the product is the convergent product of
    their square-free local densities. A finite local obstruction gives the
    singleton zero interval. This value does not assert that the family has
    any simultaneous square-free integer values.
    """

    source: SquarefreeAffineFamily
    prime_cutoff: StrictInt = Field(ge=1, le=MAX_INFINITE_PRODUCT_CUTOFF)
    enclosure: ClosedRationalInterval


__all__ = [
    "SquarefreeInfiniteProductEnclosure",
    "SquarefreeInfiniteProductRequest",
]
