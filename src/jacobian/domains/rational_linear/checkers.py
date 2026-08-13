"""Independent checker declarations for inline rational-linear candidates."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderRuntime,
)
from jacobian.contracts.linear import (
    LinearRationalInconsistencyFindRequest,
    LinearRationalSolutionFindRequest,
)
from jacobian.provider_runtime import source_provider_runtime

_ENTRYPOINT = "jacobian_checkers.linear"


def _linear_solution_runtime(
    *, checker_ids: tuple[str, ...] = ()
) -> CapabilityProviderRuntime:
    return source_provider_runtime(
        "jacobian.rational-linear-checker",
        version="1",
        entrypoint="jacobian_checkers.linear:check_rational_solution",
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT",
        features=("standard-library-rational-replay", "clean-process-checker"),
        checker_ids=checker_ids,
    )


def _linear_inconsistency_runtime(
    *, checker_ids: tuple[str, ...] = ()
) -> CapabilityProviderRuntime:
    return source_provider_runtime(
        "jacobian.rational-linear-inconsistency-checker",
        version="1",
        entrypoint="jacobian_checkers.linear:check_rational_inconsistency",
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT",
        features=("standard-library-rational-replay", "clean-process-checker"),
        checker_ids=checker_ids,
    )


RATIONAL_LINEAR_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "linear.rational_solution.compute",
        LinearRationalSolutionFindRequest,
        "check_rational_solution",
        "linear.rational_solution",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_linear_solution_runtime,
        replay_method="independent exact rational equation replay",
        reason="standard-library checker independently replays every equation",
    ),
    ExactReplayCheckerDeclaration(
        "linear.rational_inconsistency.compute",
        LinearRationalInconsistencyFindRequest,
        "check_rational_inconsistency",
        "linear.rational_inconsistency",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_linear_inconsistency_runtime,
        replay_method="independent exact rational left-nullspace replay",
        reason="standard-library checker independently replays the left witness",
    ),
)

__all__ = ["RATIONAL_LINEAR_EXACT_REPLAY_CHECKERS"]
