"""Thin operation binding for finite abelian group factorization."""

from __future__ import annotations

from jacobian.catalog._examples import example
from jacobian.math.finite_abelian_groups import (
    FiniteAbelianGroupFactorizationRequest,
    FiniteAbelianGroupFactorizationResult,
    finite_abelian_group_factorization,
)
from jacobian.math.number_theory._support import number_theory_operation

FINITE_ABELIAN_GROUP_FACTORIZATION_OPERATION = number_theory_operation(
    "finite_abelian_group.exact_factorization.compute",
    "Exact finite abelian group factorization",
    (
        "Normalize two bounded integer-vector factors in a declared product "
        "of cyclic groups, exhaustively count every sum representation, and "
        "decide whether every group element has exactly one representation."
    ),
    FiniteAbelianGroupFactorizationRequest,
    FiniteAbelianGroupFactorizationResult,
    finite_abelian_group_factorization,
    "number-theory",
    "finite-abelian-group",
    "cyclic-product",
    "factorization",
    "unique-representation",
    "coset-transversal",
    "exact",
    version="1",
    examples=(
        example(
            "z2_times_z4_transversal",
            "Verify eight representatives form a complete transversal.",
            {
                "moduli": [2, 4],
                "left": [
                    [0, 0],
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [1, 0],
                    [1, 1],
                    [1, 2],
                    [1, 3],
                ],
                "right": [[0, 0]],
            },
        ),
    ),
)

__all__ = ["FINITE_ABELIAN_GROUP_FACTORIZATION_OPERATION"]
