"""Exact public API contract for jacobian.math.probability."""

from __future__ import annotations

from jacobian.math import probability


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the probability public API."""
    expected = (
        "AllTerminalReliabilityResult",
        "AsymmetricLocalLemmaInequality",
        "AsymmetricLocalLemmaWitness",
        "AsymmetricLocalLemmaWitnessCheckResult",
        "FiniteJointTable",
        "MutualInformationLogRepresentation",
        "MutualInformationResult",
        "MutualInformationTerm",
        "all_terminal_reliability",
        "check_asymmetric_local_lemma_witness",
        "mutual_information",
    )
    assert tuple(probability.__all__) == expected
    assert len(probability.__all__) == len(set(probability.__all__))
    assert all(not name.startswith("_") for name in probability.__all__)
    assert all(hasattr(probability, name) for name in probability.__all__)
