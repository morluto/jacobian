"""Independent checker declarations owned by the finite-poset domain."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.contracts.posets import (
    FinitePosetRequest,
    LinearExtensionRequest,
    MobiusFunctionRequest,
    PosetRequest,
)
from jacobian.providers import flint_runtime

_ENTRYPOINT = "jacobian_checkers.finite_posets"
_REASON = (
    "operator-authorized clean-process standard-library replay that imports "
    "neither NetworkX nor the poset producer or contracts"
)


def _poset_runtime(*, checker_ids: tuple[str, ...] = ()) -> CapabilityProviderRuntime:
    return flint_runtime.poset_exact_checker_provider_runtime(checker_ids=checker_ids)


FINITE_POSET_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "poset.finite.compute",
        FinitePosetRequest,
        "check_finite_poset_materialization",
        "poset.finite.closure-reduction-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_poset_runtime,
        replay_method="independent closure and transitive-reduction replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "poset.width.compute",
        PosetRequest,
        "check_poset_width",
        "poset.width.dilworth-dual-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_poset_runtime,
        replay_method="independent antichain and chain-partition replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "poset.linear_extensions.count",
        LinearExtensionRequest,
        "check_linear_extension_count",
        "poset.linear-extensions.complete-ideal-dp-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_poset_runtime,
        replay_method="independent complete order-ideal recurrence replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "poset.mobius_function.compute",
        MobiusFunctionRequest,
        "check_poset_mobius_function",
        "poset.mobius.interval-convolution-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_poset_runtime,
        replay_method="independent interval-convolution recurrence replay",
        reason=_REASON,
    ),
)

__all__ = ["FINITE_POSET_EXACT_REPLAY_CHECKERS"]
