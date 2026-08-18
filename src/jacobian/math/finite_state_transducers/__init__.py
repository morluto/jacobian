"""Exact bounded native APIs for finite-state transducers."""

from jacobian.math.finite_state_transducers.operations import (
    coaccessible_states,
    compose_subsequential,
    identity_transducer,
    invert_rational,
    reachable_states,
    replay_rational_path,
    run_subsequential,
    trim_subsequential,
)
from jacobian.math.finite_state_transducers.values import (
    RationalEdge,
    RationalTransducer,
    SubseqFinalOutput,
    SubseqTransition,
    SubsequentialTransducer,
)

__all__ = [
    "RationalEdge",
    "RationalTransducer",
    "SubseqFinalOutput",
    "SubseqTransition",
    "SubsequentialTransducer",
    "coaccessible_states",
    "compose_subsequential",
    "identity_transducer",
    "invert_rational",
    "reachable_states",
    "replay_rational_path",
    "run_subsequential",
    "trim_subsequential",
]
