"""Thin operation binding for finite abelian group factorization."""

from __future__ import annotations

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domains._examples import example
from jacobian.math.finite_abelian_groups import (
    FiniteAbelianGroupFactorizationRequest,
    FiniteAbelianGroupFactorizationResult,
    finite_abelian_group_factorization,
)
from jacobian.operation_bindings import inline_operation
from jacobian.operations import OperationSpec

FINITE_ABELIAN_GROUP_FACTORIZATION_CAPABILITY = inline_operation(
    OperationSpec(
        operation_id="finite_abelian_group.exact_factorization.compute",
        version="1",
        title="Exact finite abelian group factorization",
        description=(
            "Normalize two bounded integer-vector factors in a declared product "
            "of cyclic groups, exhaustively count every sum representation, and "
            "decide whether every group element has exactly one representation."
        ),
        request_type=FiniteAbelianGroupFactorizationRequest,
        result_type=FiniteAbelianGroupFactorizationResult,
        execute=finite_abelian_group_factorization,
        tags=(
            "number-theory",
            "finite-abelian-group",
            "cyclic-product",
            "factorization",
            "unique-representation",
            "coset-transversal",
            "exact",
        ),
        invalid_request=CapabilityDiagnostic(
            code="INVALID_FINITE_ABELIAN_FACTORIZATION_REQUEST",
            stage="finite_abelian_group_input_validation",
            message=(
                "Input does not satisfy the bounded product-of-cyclic-groups "
                "factorization contract."
            ),
            hint=(
                "Supply rank at most 6, group order and factor product at most "
                "4,096, and distinct bounded factor elements after normalization."
            ),
        ),
        invocation_examples=(
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
)

__all__ = ["FINITE_ABELIAN_GROUP_FACTORIZATION_CAPABILITY"]
