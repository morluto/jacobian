"""Wire contract for bounded crystallographic mapping-torus construction."""

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.math.matrices.values import IntegerMatrix

MAX_MAPPING_TORUS_RANK = 6
MAX_FINITE_ORDER_EXPONENT = 64
MAX_MAPPING_TORUS_ENTRY_DIGITS = 32


class CrystallographicMappingTorusRequest(StrictModel):
    """A finite-order integral automorphism defining one flat mapping torus.

    ``linear_part`` must be a square integral matrix of rank at most six.
    ``finite_order_exponent`` is an authored positive exponent ``m`` for which
    the operation checks ``linear_part**m = I``; it need not be the minimal
    order. Operation admission additionally bounds coefficient height, exterior
    power materialization, exact matrix-power intermediates, and result bytes.
    """

    linear_part: IntegerMatrix = Field(
        description=(
            "Square integral linear part A. The admitted operation checks A^m = I "
            "for the supplied finite-order exponent before constructing the flat "
            "mapping torus."
        )
    )
    finite_order_exponent: StrictInt = Field(
        ge=1,
        le=MAX_FINITE_ORDER_EXPONENT,
        description=(
            "A positive exponent m with A^m = I; the operation verifies this "
            "relation and does not require m to be minimal."
        ),
    )


__all__ = [
    "MAX_FINITE_ORDER_EXPONENT",
    "MAX_MAPPING_TORUS_ENTRY_DIGITS",
    "MAX_MAPPING_TORUS_RANK",
    "CrystallographicMappingTorusRequest",
]
