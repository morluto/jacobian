"""Independent checker declarations for finite-field operations."""

from jacobian.checker_operations import AuthorizedChecker
from jacobian.contracts.operations import (
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.domains.finite_fields.contracts import (
    CollisionCertificateRequest,
    FiberPartitionRequest,
    FiniteMapTableRequest,
    LinearMapRankRequest,
    PermutationCertificateRequest,
    RestrictScalarsRequest,
)
from jacobian.provider_runtime import (
    composite_provider_runtime,
    known_provider_runtime,
    source_provider_runtime,
)


def _finite_field_rank_runtime(
    *, checker_ids: tuple[str, ...] = ()
) -> ProviderObservation:
    source = source_provider_runtime(
        "jacobian.finite-field-rank-checker-source",
        version="1",
        entrypoint=(
            "jacobian_checkers.finite_field_rank:check_finite_field_linear_map_rank"
        ),
        install_tier=ProviderInstallTier.T0,
        license_id="MIT",
        features=("clean-process-checker",),
    )
    sympy = known_provider_runtime(
        "jacobian.sympy",
        features=("prime-field-rank-replay",),
    )
    return composite_provider_runtime(
        "jacobian.finite-field-rank-checker",
        components=(source, sympy),
        features=("independent-prime-field-rank-replay",),
        checker_ids=checker_ids,
    )


def _finite_field_polynomial_runtime(
    *, checker_ids: tuple[str, ...] = ()
) -> ProviderObservation:
    source = source_provider_runtime(
        "jacobian.finite-field-polynomial-checker-source",
        version="1",
        entrypoint="jacobian_checkers.finite_field_polynomial:check_finite_map_table",
        install_tier=ProviderInstallTier.T0,
        license_id="MIT",
        features=("clean-process-checker",),
    )
    sympy = known_provider_runtime(
        "jacobian.sympy",
        features=("finite-field-polynomial-replay",),
    )
    return composite_provider_runtime(
        "jacobian.finite-field-polynomial-checker",
        components=(source, sympy),
        features=("independent-finite-field-polynomial-replay",),
        checker_ids=checker_ids,
    )


FINITE_FIELD_AUTHORIZED_CHECKERS = (
    AuthorizedChecker(
        "finite_field.restrict_scalars.compute",
        RestrictScalarsRequest,
        "check_finite_field_restriction",
        "finite-field.restriction.sympy-replay",
        entrypoint_module="jacobian_checkers.finite_field_rank",
        observation_loader=_finite_field_rank_runtime,
        replay_method="SymPy polynomial-quotient replay",
        reason=(
            "operator-authorized SymPy replay independent of the Python-FLINT producer"
        ),
    ),
    AuthorizedChecker(
        "finite_field.linear_map.rank.compute",
        LinearMapRankRequest,
        "check_finite_field_linear_map_rank",
        "finite-field.linear-map-rank.sympy-replay",
        entrypoint_module="jacobian_checkers.finite_field_rank",
        observation_loader=_finite_field_rank_runtime,
        replay_method="SymPy prime-field rank replay",
        reason=(
            "operator-authorized SymPy replay independent of the Python-FLINT producer"
        ),
    ),
    AuthorizedChecker(
        "finite_field.polynomial_map.table.compute",
        FiniteMapTableRequest,
        "check_finite_map_table",
        "finite-field.polynomial-map-table.sympy-replay",
        entrypoint_module="jacobian_checkers.finite_field_polynomial",
        observation_loader=_finite_field_polynomial_runtime,
        replay_method="SymPy finite-field polynomial replay",
        reason="operator-authorized SymPy replay independent of the FLINT producer",
    ),
    AuthorizedChecker(
        "finite_field.polynomial_map.fibers.compute",
        FiberPartitionRequest,
        "check_finite_map_fibers",
        "finite-field.polynomial-map-fibers.sympy-replay",
        entrypoint_module="jacobian_checkers.finite_field_polynomial",
        observation_loader=_finite_field_polynomial_runtime,
        replay_method="SymPy finite-map fiber replay",
        reason="operator-authorized SymPy replay of the exact map and its fibers",
    ),
    AuthorizedChecker(
        "finite_field.polynomial_map.collision.compute",
        CollisionCertificateRequest,
        "check_finite_map_collision",
        "finite-field.polynomial-map-collision.sympy-replay",
        entrypoint_module="jacobian_checkers.finite_field_polynomial",
        observation_loader=_finite_field_polynomial_runtime,
        replay_method="SymPy finite-map collision replay",
        reason="operator-authorized SymPy replay of the exact map and collision",
    ),
    AuthorizedChecker(
        "finite_field.polynomial_map.permutation.compute",
        PermutationCertificateRequest,
        "check_finite_map_permutation",
        "finite-field.polynomial-map-permutation.sympy-replay",
        entrypoint_module="jacobian_checkers.finite_field_polynomial",
        observation_loader=_finite_field_polynomial_runtime,
        replay_method="SymPy finite-map permutation replay",
        reason="operator-authorized SymPy replay of the exact map and inverse",
    ),
)

__all__ = ["FINITE_FIELD_AUTHORIZED_CHECKERS"]
