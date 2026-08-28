"""Exact public API contract for jacobian.math.logic.automata.transducers."""

from __future__ import annotations

from jacobian.math.logic.automata import transducers as finite_state_transducers


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the finite_state_transducers public API."""
    expected = (
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
    )
    assert tuple(finite_state_transducers.__all__) == expected
    assert len(finite_state_transducers.__all__) == len(
        set(finite_state_transducers.__all__)
    )
    assert all(not name.startswith("_") for name in finite_state_transducers.__all__)
    assert all(
        hasattr(finite_state_transducers, name)
        for name in finite_state_transducers.__all__
    )
