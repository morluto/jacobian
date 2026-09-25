"""Free associative algebra word and noncommutative polynomial ownership."""

from jacobian.math.free_algebras._models import (
    FreeAlgebraIdeal,
    FreeAlgebraIdealPrefixResult,
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
    FreeAlgebraWord,
    GroebnerShirshovResult,
)
from jacobian.math.free_algebras.homogeneous_component import homogeneous_component
from jacobian.math.free_algebras.operations import (
    groebner_shirshov_through_degree,
    ideal_generated_prefix,
    multiply,
)

__all__ = [
    "FreeAlgebraIdeal",
    "FreeAlgebraIdealPrefixResult",
    "FreeAlgebraPolynomial",
    "FreeAlgebraTerm",
    "FreeAlgebraWord",
    "GroebnerShirshovResult",
    "groebner_shirshov_through_degree",
    "homogeneous_component",
    "ideal_generated_prefix",
    "multiply",
]
