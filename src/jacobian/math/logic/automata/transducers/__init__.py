"""Exact bounded native APIs for finite-state transducers."""

from jacobian.math.logic.automata.transducers.operations import (
    coaccessible_states,
    compose_subsequential,
    identity_transducer,
    invert_rational,
    minimize_subsequential,
    reachable_states,
    replay_rational_path,
    run_subsequential,
    trim_subsequential,
    verify_composition,
    verify_minimization,
    verify_subsequential_run,
)
from jacobian.math.logic.automata.transducers.values import (
    FiniteAlphabet,
    RationalEdge,
    RationalTransducer,
    SubseqFinalOutput,
    SubseqTransition,
    SubsequentialTransducer,
)

__all__ = [
    "FiniteAlphabet",
    "RationalEdge",
    "RationalTransducer",
    "SubseqFinalOutput",
    "SubseqTransition",
    "SubsequentialTransducer",
    "coaccessible_states",
    "compose_subsequential",
    "identity_transducer",
    "invert_rational",
    "minimize_subsequential",
    "reachable_states",
    "replay_rational_path",
    "run_subsequential",
    "trim_subsequential",
    "verify_composition",
    "verify_minimization",
    "verify_subsequential_run",
]
