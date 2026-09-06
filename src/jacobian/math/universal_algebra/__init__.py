"""Supported native universal-algebra API."""

from jacobian.math.universal_algebra.operations import (
    congruence_check,
    equation_profile,
    evaluate_term,
    generated_subalgebra,
    homomorphism_profile,
    quotient,
    verify_congruence,
    verify_equation_profile,
    verify_evaluate,
    verify_generated_subalgebra,
)
from jacobian.math.universal_algebra.values import (
    ApplicationTerm,
    FiniteAlgebra,
    FiniteAlgebraCarrierMap,
    FiniteAlgebraHomomorphism,
    FlatTerm,
    OperationSymbol,
    Term,
    VariableTerm,
)

__all__ = [
    "ApplicationTerm",
    "FiniteAlgebra",
    "FiniteAlgebraCarrierMap",
    "FiniteAlgebraHomomorphism",
    "FlatTerm",
    "OperationSymbol",
    "Term",
    "VariableTerm",
    "congruence_check",
    "equation_profile",
    "evaluate_term",
    "generated_subalgebra",
    "homomorphism_profile",
    "quotient",
    "verify_congruence",
    "verify_equation_profile",
    "verify_evaluate",
    "verify_generated_subalgebra",
]
