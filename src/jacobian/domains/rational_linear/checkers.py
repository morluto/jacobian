"""Independent checker declarations for inline rational-linear candidates."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.linear import (
    LinearRationalInconsistencyFindRequest,
    LinearRationalSolutionFindRequest,
)

_ENTRYPOINT = "jacobian_checkers.linear"

RATIONAL_LINEAR_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "linear.rational_solution.compute",
        LinearRationalSolutionFindRequest,
        "check_rational_solution",
        "linear.rational_solution",
        entrypoint_module=_ENTRYPOINT,
        replay_method="independent exact rational equation replay",
        reason="standard-library checker independently replays every equation",
    ),
    ExactReplayCheckerDeclaration(
        "linear.rational_inconsistency.compute",
        LinearRationalInconsistencyFindRequest,
        "check_rational_inconsistency",
        "linear.rational_inconsistency",
        entrypoint_module=_ENTRYPOINT,
        replay_method="independent exact rational left-nullspace replay",
        reason="standard-library checker independently replays the left witness",
    ),
)

__all__ = ["RATIONAL_LINEAR_EXACT_REPLAY_CHECKERS"]
