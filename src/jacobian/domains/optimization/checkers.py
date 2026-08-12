"""Independent checker declarations owned by rational optimization."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.validated_analysis import RationalLinearProgramRequest

RATIONAL_OPTIMIZATION_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "optimization.linear.rational_optimum.compute",
        RationalLinearProgramRequest,
        "check_rational_linear_optimum",
        "optimization.linear.rational-optimum.fraction-replay",
        replay_method="Python-FLINT exact rational primal/dual replay",
        reason=(
            "operator-authorized Python-FLINT checker independently replays "
            "primal feasibility, dual feasibility, and strong-duality equality "
            "without importing the SymPy producer"
        ),
    ),
)

__all__ = ["RATIONAL_OPTIMIZATION_EXACT_REPLAY_CHECKERS"]
