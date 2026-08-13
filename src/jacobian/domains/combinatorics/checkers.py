"""Independent checker declarations owned by exact combinatorics."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.contracts.combinatorics import (
    BinomialRequest,
    CyclicDifferenceSetExtensionRequest,
    CyclicPerfectDifferenceSetRequest,
    IntegerSidonRequest,
    LinearRecurrenceEvaluationRequest,
    PolynomialCoefficientRecurrenceEvaluationRequest,
    RationalGeneratingFunctionCoefficientsRequest,
)
from jacobian.providers import flint_runtime

_ENTRYPOINT = "jacobian_checkers.recurrence_series"
_REASON = (
    "operator-authorized standard-library Fraction replay independent of the "
    "SymPy recurrence and rational-series producer"
)


def _combinatorics_runtime(
    *, checker_ids: tuple[str, ...] = ()
) -> CapabilityProviderRuntime:
    return flint_runtime.combinatorics_exact_checker_provider_runtime(
        checker_ids=checker_ids
    )


COMBINATORICS_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "combinatorics.compute.binomial",
        BinomialRequest,
        "check_binomial",
        "combinatorics.binomial.multiplicative-recurrence-replay",
        entrypoint_module="jacobian_checkers.additive_combinatorics",
        replay_method="standard-library multiplicative recurrence replay",
        reason=(
            "operator-authorized checker independently evaluates the bounded "
            "binomial coefficient by an exact multiplicative recurrence without "
            "calling math.comb or importing producer code"
        ),
        provider_runtime_factory=_combinatorics_runtime,
    ),
    ExactReplayCheckerDeclaration(
        "combinatorics.integer_set.sidon.decide",
        IntegerSidonRequest,
        "check_integer_sidon",
        "combinatorics.integer-sidon.ordered-difference-replay",
        entrypoint_module="jacobian_checkers.additive_combinatorics",
        provider_runtime_factory=_combinatorics_runtime,
        replay_method="standard-library ordered-difference replay",
        reason=(
            "operator-authorized standard-library checker independently enumerates "
            "every ordered integer difference without importing producer code"
        ),
    ),
    ExactReplayCheckerDeclaration(
        "combinatorics.cyclic_difference_set.perfect.decide",
        CyclicPerfectDifferenceSetRequest,
        "check_cyclic_perfect_difference_set",
        "combinatorics.cyclic-pds.residue-profile-replay",
        entrypoint_module="jacobian_checkers.additive_combinatorics",
        provider_runtime_factory=_combinatorics_runtime,
        replay_method="standard-library cyclic residue-profile replay",
        reason=(
            "operator-authorized standard-library checker independently rebuilds "
            "the complete nonzero cyclic difference multiplicities"
        ),
    ),
    ExactReplayCheckerDeclaration(
        "combinatorics.cyclic_difference_set.extension.decide",
        CyclicDifferenceSetExtensionRequest,
        "check_cyclic_difference_set_extension",
        "combinatorics.cyclic-pds-extension.exhaustive-replay",
        entrypoint_module="jacobian_checkers.additive_combinatorics",
        provider_runtime_factory=_combinatorics_runtime,
        replay_method="standard-library fixed-order exhaustive extension replay",
        reason=(
            "operator-authorized checker independently enumerates every bounded "
            "completion using itertools rather than the producer's pruning search"
        ),
    ),
    ExactReplayCheckerDeclaration(
        "combinatorics.recurrence.linear.evaluate",
        LinearRecurrenceEvaluationRequest,
        "check_linear_recurrence_evaluation",
        "combinatorics.linear-recurrence.fraction-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_combinatorics_runtime,
        replay_method="standard-library Fraction recurrence replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "combinatorics.recurrence.p_recursive.evaluate",
        PolynomialCoefficientRecurrenceEvaluationRequest,
        "check_polynomial_coefficient_recurrence_evaluation",
        "combinatorics.p-recursive.fraction-residual-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_combinatorics_runtime,
        replay_method="standard-library Fraction polynomial recurrence replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "combinatorics.generating_function.coefficients.compute",
        RationalGeneratingFunctionCoefficientsRequest,
        "check_rational_generating_function_coefficients",
        "combinatorics.rational-series.fraction-residual-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_combinatorics_runtime,
        replay_method="standard-library Fraction residual replay",
        reason=_REASON,
    ),
)

__all__ = ["COMBINATORICS_EXACT_REPLAY_CHECKERS"]
