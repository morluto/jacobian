"""Free associative algebra word and noncommutative polynomial ownership."""

from jacobian.math.free_algebras._models import (
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
    FreeAlgebraWord,
)
from jacobian.math.free_algebras.operations import multiply

__all__ = [
    "FreeAlgebraPolynomial",
    "FreeAlgebraTerm",
    "FreeAlgebraWord",
    "multiply",
]
