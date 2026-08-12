"""Independent checker declarations for finite-field operations."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.domains.finite_fields.contracts import (
    CollisionCertificateRequest,
    FiberPartitionRequest,
    FiniteMapTableRequest,
    LinearMapRankRequest,
    PermutationCertificateRequest,
    RestrictScalarsRequest,
)

FINITE_FIELD_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "finite_field.restrict_scalars.compute",
        RestrictScalarsRequest,
        "check_finite_field_restriction",
        "finite-field.restriction.sympy-replay",
        entrypoint_module="jacobian_checkers.finite_field_rank",
        replay_method="SymPy polynomial-quotient replay",
        reason=(
            "operator-authorized SymPy replay independent of the Python-FLINT producer"
        ),
    ),
    ExactReplayCheckerDeclaration(
        "finite_field.linear_map.rank.compute",
        LinearMapRankRequest,
        "check_finite_field_linear_map_rank",
        "finite-field.linear-map-rank.sympy-replay",
        entrypoint_module="jacobian_checkers.finite_field_rank",
        replay_method="SymPy prime-field rank replay",
        reason=(
            "operator-authorized SymPy replay independent of the Python-FLINT producer"
        ),
    ),
    ExactReplayCheckerDeclaration(
        "finite_field.polynomial_map.table.compute",
        FiniteMapTableRequest,
        "check_finite_map_table",
        "finite-field.polynomial-map-table.sympy-replay",
        entrypoint_module="jacobian_checkers.finite_field_polynomial",
        replay_method="SymPy finite-field polynomial replay",
        reason="operator-authorized SymPy replay independent of the FLINT producer",
    ),
    ExactReplayCheckerDeclaration(
        "finite_field.polynomial_map.fibers.compute",
        FiberPartitionRequest,
        "check_finite_map_fibers",
        "finite-field.polynomial-map-fibers.sympy-replay",
        entrypoint_module="jacobian_checkers.finite_field_polynomial",
        replay_method="SymPy finite-map fiber replay",
        reason="operator-authorized SymPy replay of the exact map and its fibers",
    ),
    ExactReplayCheckerDeclaration(
        "finite_field.polynomial_map.collision.compute",
        CollisionCertificateRequest,
        "check_finite_map_collision",
        "finite-field.polynomial-map-collision.sympy-replay",
        entrypoint_module="jacobian_checkers.finite_field_polynomial",
        replay_method="SymPy finite-map collision replay",
        reason="operator-authorized SymPy replay of the exact map and collision",
    ),
    ExactReplayCheckerDeclaration(
        "finite_field.polynomial_map.permutation.compute",
        PermutationCertificateRequest,
        "check_finite_map_permutation",
        "finite-field.polynomial-map-permutation.sympy-replay",
        entrypoint_module="jacobian_checkers.finite_field_polynomial",
        replay_method="SymPy finite-map permutation replay",
        reason="operator-authorized SymPy replay of the exact map and inverse",
    ),
)

__all__ = ["FINITE_FIELD_EXACT_REPLAY_CHECKERS"]
