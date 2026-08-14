"""Exact SymPy-backed integer number-theory operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["number_theory_operations"]


def number_theory_operations() -> MathTools:
    from jacobian.domains.number_theory.derived import DERIVED_NUMBER_THEORY_OPERATIONS
    from jacobian.domains.number_theory.divisibility import DIVISIBILITY_OPERATIONS
    from jacobian.domains.number_theory.finite_abelian_groups import (
        FINITE_ABELIAN_GROUP_FACTORIZATION_OPERATION,
    )
    from jacobian.domains.number_theory.modular import MODULAR_OPERATIONS
    from jacobian.domains.number_theory.modular_identity import (
        MODULAR_IDENTITY_OPERATIONS,
    )
    from jacobian.domains.number_theory.primes import PRIME_OPERATIONS

    return (
        *DIVISIBILITY_OPERATIONS,
        *PRIME_OPERATIONS,
        *MODULAR_OPERATIONS,
        *MODULAR_IDENTITY_OPERATIONS,
        *DERIVED_NUMBER_THEORY_OPERATIONS,
        FINITE_ABELIAN_GROUP_FACTORIZATION_OPERATION,
    )
