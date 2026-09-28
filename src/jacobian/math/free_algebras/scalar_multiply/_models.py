"""Request shape and operation-specific bounds for polynomial scalar action."""

from pydantic import Field

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS,
    FreeAlgebraPolynomial,
)

MAX_SCALAR_MULTIPLY_WORK = 170_000_000
MAX_SCALAR_MULTIPLY_OUTPUT_CELLS = 1_100_000
MAX_SCALAR_MULTIPLY_INTERMEDIATE_CELLS = 2_200_000
MAX_SCALAR_MULTIPLY_COEFFICIENT_DIGITS = MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS
MAX_SCALAR_MULTIPLY_SCALAR_DIGITS = 2 * MAX_SCALAR_MULTIPLY_COEFFICIENT_DIGITS


class FreeAlgebraPolynomialScalarMultiplyRequest(StrictModel):
    """Scale one exact QQ-polynomial while retaining its ordered alphabet."""

    polynomial: FreeAlgebraPolynomial = Field(
        description="The canonical sparse source polynomial over its ordered alphabet."
    )
    scalar: CanonicalRational = Field(
        description="An exact rational scalar acting on every polynomial coefficient."
    )
