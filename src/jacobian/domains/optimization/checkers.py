"""Independent checker declarations owned by rational optimization."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.validated_analysis import RationalLinearProgramRequest

RATIONAL_OPTIMIZATION_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "optimization.linear.rational_optimum.compute",
        RationalLinearProgramRequest,
        "check_rational_linear_optimum",
        "optimization.linear.rational-optimum.fraction-replay",
        entrypoint_module="jacobian_checkers.rational_lp",
        replay_method="standard-library Fraction primal/dual replay",
        reason=(
            "operator-authorized standard-library checker independently replays "
            "primal feasibility, unrestricted-dual feasibility, and strong-duality "
            "equality without importing the SymPy producer"
        ),
    ),
)

__all__ = ["RATIONAL_OPTIMIZATION_EXACT_REPLAY_CHECKERS"]
