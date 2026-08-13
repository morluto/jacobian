"""Independent checker declarations owned by rational optimization."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderRuntime,
)
from jacobian.contracts.validated_analysis import RationalLinearProgramRequest
from jacobian.provider_runtime import source_provider_runtime


def _rational_lp_runtime(
    *, checker_ids: tuple[str, ...] = ()
) -> CapabilityProviderRuntime:
    return source_provider_runtime(
        "jacobian.rational-lp-checker",
        version="1",
        entrypoint="jacobian_checkers.rational_lp:check_rational_linear_optimum",
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT",
        features=("standard-library-rational-replay", "clean-process-checker"),
        checker_ids=checker_ids,
    )


RATIONAL_OPTIMIZATION_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "optimization.linear.rational_optimum.compute",
        RationalLinearProgramRequest,
        "check_rational_linear_optimum",
        "optimization.linear.rational-optimum.fraction-replay",
        entrypoint_module="jacobian_checkers.rational_lp",
        provider_runtime_factory=_rational_lp_runtime,
        replay_method="standard-library Fraction primal/dual replay",
        reason=(
            "operator-authorized standard-library checker independently replays "
            "primal feasibility, unrestricted-dual feasibility, and strong-duality "
            "equality without importing the SymPy producer"
        ),
    ),
)

__all__ = ["RATIONAL_OPTIMIZATION_EXACT_REPLAY_CHECKERS"]
