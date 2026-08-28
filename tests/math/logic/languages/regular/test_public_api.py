"""Exact public API contract for jacobian.math.logic.languages.regular."""

from __future__ import annotations

from jacobian.math.logic.languages import regular as regular_languages


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the regular_languages public API."""
    expected = (
        "DFA",
        "AutomatonTransition",
        "DFATransition",
        "FiniteLabeledAutomaton",
        "TransitionParikhCell",
        "TransitionParikhProfile",
        "count_accepted_words",
        "dfa_complement",
        "dfa_run",
        "dfa_transition_carrier",
        "transition_parikh_profile",
    )
    assert tuple(regular_languages.__all__) == expected
    assert len(regular_languages.__all__) == len(set(regular_languages.__all__))
    assert all(not name.startswith("_") for name in regular_languages.__all__)
    assert all(hasattr(regular_languages, name) for name in regular_languages.__all__)
