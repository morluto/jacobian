"""Independent checker declarations for inline rational-linear candidates."""

from jacobian.checker_operations import AuthorizedChecker
from jacobian.contracts.linear import (
    LinearRationalInconsistencyFindRequest,
    LinearRationalSolutionFindRequest,
)
from jacobian.contracts.operations import (
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.provider_runtime import source_provider_runtime

_ENTRYPOINT = "jacobian_checkers.linear"


def _linear_solution_runtime(
    *, checker_ids: tuple[str, ...] = ()
) -> ProviderObservation:
    return source_provider_runtime(
        "jacobian.rational-linear-checker",
        version="1",
        entrypoint="jacobian_checkers.linear:check_rational_solution",
        install_tier=ProviderInstallTier.T1,
        license_id="MIT",
        features=("standard-library-rational-replay", "clean-process-checker"),
        checker_ids=checker_ids,
    )


def _linear_inconsistency_runtime(
    *, checker_ids: tuple[str, ...] = ()
) -> ProviderObservation:
    return source_provider_runtime(
        "jacobian.rational-linear-inconsistency-checker",
        version="1",
        entrypoint="jacobian_checkers.linear:check_rational_inconsistency",
        install_tier=ProviderInstallTier.T1,
        license_id="MIT",
        features=("standard-library-rational-replay", "clean-process-checker"),
        checker_ids=checker_ids,
    )


RATIONAL_LINEAR_AUTHORIZED_CHECKERS = (
    AuthorizedChecker(
        "linear.rational_solution.compute",
        LinearRationalSolutionFindRequest,
        "check_rational_solution",
        "linear.rational_solution",
        entrypoint_module=_ENTRYPOINT,
        observation_loader=_linear_solution_runtime,
        replay_method="independent exact rational equation replay",
        reason="standard-library checker independently replays every equation",
    ),
    AuthorizedChecker(
        "linear.rational_inconsistency.compute",
        LinearRationalInconsistencyFindRequest,
        "check_rational_inconsistency",
        "linear.rational_inconsistency",
        entrypoint_module=_ENTRYPOINT,
        observation_loader=_linear_inconsistency_runtime,
        replay_method="independent exact rational left-nullspace replay",
        reason="standard-library checker independently replays the left witness",
    ),
)

__all__ = ["RATIONAL_LINEAR_AUTHORIZED_CHECKERS"]
