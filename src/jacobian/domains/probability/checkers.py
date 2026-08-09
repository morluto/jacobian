"""Independent checker declarations owned by the probability domain."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.probability import (
    FiniteConvolutionRequest,
    FiniteEventRequest,
    FinitePushforwardRequest,
    GaussianPolynomialMomentRequest,
    GraphConnectionProbabilityRequest,
)
from jacobian.contracts.validated_analysis import FiniteRawMomentRequest

_ENTRYPOINT = "jacobian_checkers.exact_probability_operations"
_REASON = (
    "operator-authorized standard-library Fraction replay independent of the "
    "Python-FLINT producer"
)

PROBABILITY_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "probability.finite_distribution.raw_moment.compute",
        FiniteRawMomentRequest,
        "check_finite_raw_moment",
        "probability.finite-raw-moment.fraction-replay",
        entrypoint_module=_ENTRYPOINT,
        replay_method="standard-library Fraction replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "probability.finite_distribution.event_probability.compute",
        FiniteEventRequest,
        "check_finite_event_probability",
        "probability.finite-event.fraction-replay",
        entrypoint_module=_ENTRYPOINT,
        replay_method="standard-library Fraction replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "probability.finite_distribution.condition.compute",
        FiniteEventRequest,
        "check_finite_condition",
        "probability.finite-condition.fraction-replay",
        entrypoint_module=_ENTRYPOINT,
        replay_method="standard-library Fraction replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "probability.finite_distribution.pushforward.compute",
        FinitePushforwardRequest,
        "check_finite_pushforward",
        "probability.finite-pushforward.fraction-replay",
        entrypoint_module=_ENTRYPOINT,
        replay_method="standard-library Fraction replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "probability.finite_distribution.convolution.compute",
        FiniteConvolutionRequest,
        "check_finite_convolution",
        "probability.finite-convolution.fraction-replay",
        entrypoint_module=_ENTRYPOINT,
        replay_method="standard-library Fraction replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "probability.gaussian_polynomial.moment.compute",
        GaussianPolynomialMomentRequest,
        "check_gaussian_polynomial_moment",
        "probability.gaussian-polynomial-moment.fraction-replay",
        entrypoint_module=_ENTRYPOINT,
        replay_method="independent standard-library coefficient contraction",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "probability.graph_reliability.connection_probability.compute",
        GraphConnectionProbabilityRequest,
        "check_graph_connection_probability",
        "probability.graph-reliability-connection.fraction-replay",
        entrypoint_module=_ENTRYPOINT,
        replay_method="independent exhaustive edge-subset replay",
        reason=_REASON,
        verification_capability_id=(
            "probability.graph_reliability.connection_probability.verify"
        ),
        verification_title="Verify an exact terminal connection probability",
        verification_description=(
            "Independently enumerate every edge subset and replay exact terminal "
            "connectivity for one bounded graph-reliability result."
        ),
        verification_tags=(
            "verification",
            "exact",
            "probability",
            "graph",
            "reliability",
            "percolation",
            "connection",
            "terminals",
        ),
    ),
)

__all__ = ["PROBABILITY_EXACT_REPLAY_CHECKERS"]
