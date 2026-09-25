"""Typed result and request for a graded free-algebra projection."""

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.free_algebras._models import (
    FreeAlgebraPolynomial,
)

MAX_HOMOGENEOUS_COMPONENT_WORK = 600_000


class FreeAlgebraHomogeneousComponentRequest(StrictModel):
    """Select the degree-n graded component of a sparse QQ<X> polynomial."""

    polynomial: FreeAlgebraPolynomial
    degree: int = Field(ge=0)


class FreeAlgebraHomogeneousComponent(StrictModel):
    """A polynomial together with the degree of the component it represents.

    The alphabet is retained by ``polynomial`` even when the projection is
    zero. A decoded carrier's projection claim is source-operation data; a
    consumer that relies on it checks that claim against its supplied source.
    """

    degree: int = Field(ge=0)
    polynomial: FreeAlgebraPolynomial


__all__ = [
    "MAX_HOMOGENEOUS_COMPONENT_WORK",
    "FreeAlgebraHomogeneousComponent",
    "FreeAlgebraHomogeneousComponentRequest",
]
