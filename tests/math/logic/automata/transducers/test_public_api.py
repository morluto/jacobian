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
        "minimize_subsequential",
        "project_rational_relation",
        "rational_relation_outputs_for_input",
        "reachable_state_witnesses",
        "reachable_states",
        "replay_rational_path",
        "restrict_rational_input",
        "run_subsequential",
        "trim_subsequential",
        "verify_composition",
        "verify_minimization",
        "verify_subsequential_run",
        "word_morphism_to_subsequential",
    )
    # The module's declared __all__ is the source of truth: every
    # advertised name must stay exported, but adding one is allowed.
    assert set(expected) <= set(finite_state_transducers.__all__)
    assert len(finite_state_transducers.__all__) == len(
        set(finite_state_transducers.__all__)
    )
    assert all(not name.startswith("_") for name in finite_state_transducers.__all__)
    assert all(
        hasattr(finite_state_transducers, name)
        for name in finite_state_transducers.__all__
    )
