"""Independent checker declarations owned by exact combinatorics."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.combinatorics import (
    LinearRecurrenceEvaluationRequest,
    RationalGeneratingFunctionCoefficientsRequest,
)

_ENTRYPOINT = "jacobian_checkers.recurrence_series"
_REASON = (
    "operator-authorized standard-library Fraction replay independent of the "
    "SymPy recurrence and rational-series producer"
)

COMBINATORICS_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "combinatorics.recurrence.linear.evaluate",
        LinearRecurrenceEvaluationRequest,
        "check_linear_recurrence_evaluation",
        "combinatorics.linear-recurrence.fraction-replay",
        entrypoint_module=_ENTRYPOINT,
        replay_method="standard-library Fraction recurrence replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "combinatorics.generating_function.coefficients.compute",
        RationalGeneratingFunctionCoefficientsRequest,
        "check_rational_generating_function_coefficients",
        "combinatorics.rational-series.fraction-residual-replay",
        entrypoint_module=_ENTRYPOINT,
        replay_method="standard-library Fraction residual replay",
        reason=_REASON,
    ),
)

__all__ = ["COMBINATORICS_EXACT_REPLAY_CHECKERS"]
