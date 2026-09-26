"""Typed result and request for a graded free-algebra projection."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, WithJsonSchema

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS, ExactInteger
from jacobian._models import StrictModel
from jacobian.math.free_algebras._models import (
    FreeAlgebraPolynomial,
)

MAX_HOMOGENEOUS_COMPONENT_WORK = 600_000

# The grading degree is an arbitrary nonnegative integer. The source carrier
# can hold words only up to MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH letters, so a
# larger degree is valid and has the canonical zero component. Preserve the
# requested degree exactly across JSON with the shared exact-integer codec: the
# wire value is a canonical decimal string, so degrees beyond JavaScript's
# safe-integer range survive the boundary. The shared 32,768-digit envelope is
# the representable structural limit, not an arithmetic ceiling.
_DEGREE_WIRE_SCHEMA = {
    "type": "string",
    "pattern": rf"^(?:0|[1-9][0-9]{{0,{MAX_CANONICAL_INTEGER_DIGITS - 1}}})(?![\s\S])",
    "maxLength": MAX_CANONICAL_INTEGER_DIGITS,
}
HomogeneousComponentDegree = Annotated[
    ExactInteger,
    Field(ge=0),
    WithJsonSchema(_DEGREE_WIRE_SCHEMA),
]


class FreeAlgebraHomogeneousComponentRequest(StrictModel):
    """Select the degree-n graded component of a sparse QQ<X> polynomial."""

    polynomial: FreeAlgebraPolynomial
    degree: HomogeneousComponentDegree


class FreeAlgebraHomogeneousComponent(StrictModel):
    """A polynomial together with the degree of the component it represents.

    The alphabet is retained by ``polynomial`` even when the projection is
    zero. A decoded carrier's projection claim is source-operation data; a
    consumer that relies on it checks that claim against its supplied source.
    """

    degree: HomogeneousComponentDegree
    polynomial: FreeAlgebraPolynomial


__all__ = [
    "MAX_HOMOGENEOUS_COMPONENT_WORK",
    "FreeAlgebraHomogeneousComponent",
    "FreeAlgebraHomogeneousComponentRequest",
    "HomogeneousComponentDegree",
]
