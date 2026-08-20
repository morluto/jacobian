"""Supported native universal-algebra API."""

from jacobian.math.universal_algebra.operations import (
    congruence_check,
    equation_profile,
    evaluate_term,
    generated_subalgebra,
    quotient,
)
from jacobian.math.universal_algebra.values import (
    FiniteAlgebra,
    FlatTerm,
    OperationSymbol,
    Term,
)

__all__ = [
    "FiniteAlgebra",
    "FlatTerm",
    "OperationSymbol",
    "Term",
    "congruence_check",
    "equation_profile",
    "evaluate_term",
    "generated_subalgebra",
    "quotient",
]
