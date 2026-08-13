"""Independent checker declarations owned by the probability domain."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderRuntime,
)
from jacobian.contracts.probability import (
    FiniteConvolutionRequest,
    FiniteEventRequest,
    FinitePushforwardRequest,
    GraphConnectionProbabilityRequest,
)
from jacobian.contracts.validated_analysis import FiniteRawMomentRequest
from jacobian.domains.probability.gaussian_inputs import (
    CanonicalGaussianPolynomialMomentRequest,
)
from jacobian.domains.probability.mutual_information import (
    FiniteJointTableMutualInformationRequest,
)
from jacobian.provider_runtime import source_provider_runtime
from jacobian.providers import flint_runtime

_ENTRYPOINT = "jacobian_checkers.exact_probability_operations"
_REASON = (
    "operator-authorized standard-library Fraction replay independent of the "
    "Python-FLINT producer"
)


def _mutual_information_checker_runtime(
    *, checker_ids: tuple[str, ...] = ()
) -> CapabilityProviderRuntime:
    """Measure the checker source only when authorization installs it."""

    return source_provider_runtime(
        "jacobian.mutual-information-checker",
        version="1",
        entrypoint=(
            "jacobian_checkers.mutual_information:check_finite_joint_mutual_information"
        ),
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT",
        features=("standard-library-fraction-replay", "clean-process-checker"),
        checker_ids=checker_ids,
    )


def _probability_runtime(
    *, checker_ids: tuple[str, ...] = ()
) -> CapabilityProviderRuntime:
    return flint_runtime.probability_exact_checker_provider_runtime(
        checker_ids=checker_ids
    )


PROBABILITY_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "probability.joint.mutual_information.compute",
        FiniteJointTableMutualInformationRequest,
        "check_finite_joint_mutual_information",
        "probability.finite-joint-mutual-information.fraction-replay",
        entrypoint_module="jacobian_checkers.mutual_information",
        replay_method="standard-library Fraction logarithmic-product replay",
        reason=(
            "operator-authorized standard-library Fraction checker independently "
            "reconstructs marginals, likelihood ratios, and the scaled log product"
        ),
        provider_runtime_factory=_mutual_information_checker_runtime,
        verification_capability_id="probability.joint.mutual_information.verify",
        verification_title="Verify a finite-table mutual-information certificate",
        verification_description=(
            "Independently reconstruct ordered marginals, positive-support "
            "likelihood ratios, and the exact scaled logarithmic product for one "
            "bounded normalized rational joint table."
        ),
        verification_tags=(
            "verification",
            "exact",
            "probability",
            "information-theory",
            "mutual-information",
        ),
    ),
    ExactReplayCheckerDeclaration(
        "probability.finite_distribution.raw_moment.compute",
        FiniteRawMomentRequest,
        "check_finite_raw_moment",
        "probability.finite-raw-moment.fraction-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_probability_runtime,
        replay_method="standard-library Fraction replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "probability.finite_distribution.event_probability.compute",
        FiniteEventRequest,
        "check_finite_event_probability",
        "probability.finite-event.fraction-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_probability_runtime,
        replay_method="standard-library Fraction replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "probability.finite_distribution.condition.compute",
        FiniteEventRequest,
        "check_finite_condition",
        "probability.finite-condition.fraction-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_probability_runtime,
        replay_method="standard-library Fraction replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "probability.finite_distribution.pushforward.compute",
        FinitePushforwardRequest,
        "check_finite_pushforward",
        "probability.finite-pushforward.fraction-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_probability_runtime,
        replay_method="standard-library Fraction replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "probability.finite_distribution.convolution.compute",
        FiniteConvolutionRequest,
        "check_finite_convolution",
        "probability.finite-convolution.fraction-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_probability_runtime,
        replay_method="standard-library Fraction replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "probability.gaussian_polynomial.moment.compute",
        CanonicalGaussianPolynomialMomentRequest,
        "check_gaussian_polynomial_moment",
        "probability.gaussian-polynomial-moment.fraction-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_probability_runtime,
        replay_method="independent standard-library coefficient contraction",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "probability.graph_reliability.connection_probability.compute",
        GraphConnectionProbabilityRequest,
        "check_graph_connection_probability",
        "probability.graph-reliability-connection.fraction-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_probability_runtime,
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
