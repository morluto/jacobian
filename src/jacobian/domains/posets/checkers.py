"""Independent checker declarations owned by the finite-poset domain."""

from jacobian.checker_operations import AuthorizedChecker
from jacobian.contracts.operations import ProviderObservation
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


def _poset_runtime(*, checker_ids: tuple[str, ...] = ()) -> ProviderObservation:
    return flint_runtime.poset_exact_checker_provider_runtime(checker_ids=checker_ids)


FINITE_POSET_AUTHORIZED_CHECKERS = (
    AuthorizedChecker(
        "poset.finite.compute",
        FinitePosetRequest,
        "check_finite_poset_materialization",
        "poset.finite.closure-reduction-replay",
        entrypoint_module=_ENTRYPOINT,
        observation_loader=_poset_runtime,
        replay_method="independent closure and transitive-reduction replay",
        reason=_REASON,
    ),
    AuthorizedChecker(
        "poset.width.compute",
        PosetRequest,
        "check_poset_width",
        "poset.width.dilworth-dual-replay",
        entrypoint_module=_ENTRYPOINT,
        observation_loader=_poset_runtime,
        replay_method="independent antichain and chain-partition replay",
        reason=_REASON,
    ),
    AuthorizedChecker(
        "poset.linear_extensions.count",
        LinearExtensionRequest,
        "check_linear_extension_count",
        "poset.linear-extensions.complete-ideal-dp-replay",
        entrypoint_module=_ENTRYPOINT,
        observation_loader=_poset_runtime,
        replay_method="independent complete order-ideal recurrence replay",
        reason=_REASON,
    ),
    AuthorizedChecker(
        "poset.mobius_function.compute",
        MobiusFunctionRequest,
        "check_poset_mobius_function",
        "poset.mobius.interval-convolution-replay",
        entrypoint_module=_ENTRYPOINT,
        observation_loader=_poset_runtime,
        replay_method="independent interval-convolution recurrence replay",
        reason=_REASON,
    ),
)

__all__ = ["FINITE_POSET_AUTHORIZED_CHECKERS"]
