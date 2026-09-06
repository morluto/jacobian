"""Exact public API contract for jacobian.math.probability."""

from __future__ import annotations

from jacobian.math import probability
from jacobian.math.probability import operations as probability_operations


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the probability public API."""
    expected = (
        "AllTerminalReliabilityResult",
        "AsymmetricLocalLemmaInequality",
        "AsymmetricLocalLemmaWitness",
        "AsymmetricLocalLemmaWitnessCheckResult",
        "ExactComplexRational",
        "FiniteJointTable",
        "GraphConnectionProbabilityResult",
        "GraphReliabilityEdgeProbability",
        "GraphReliabilitySource",
        "GraphReliabilityState",
        "MutualInformationLogRepresentation",
        "MutualInformationResult",
        "MutualInformationTerm",
        "all_terminal_reliability",
        "check_asymmetric_local_lemma_witness",
        "compute_graph_connection_probability",
        "condition",
        "convolution",
        "convolution_peak",
        "convolution_power",
        "event_probability",
        "gaussian_polynomial_moment",
        "mutual_information",
        "pushforward",
        "raw_moment",
        "verify_graph_connection_probability",
    )
    assert tuple(probability.__all__) == expected
    assert len(probability.__all__) == len(set(probability.__all__))
    assert all(not name.startswith("_") for name in probability.__all__)
    assert all(hasattr(probability, name) for name in probability.__all__)


def test_exact_operations_module_exports() -> None:
    """Native operations.py __all__ publishes the finite-distribution kernels."""
    expected = (
        "condition",
        "convolution",
        "convolution_peak",
        "convolution_power",
        "event_probability",
        "gaussian_polynomial_moment",
        "pushforward",
        "raw_moment",
    )
    assert tuple(probability_operations.__all__) == expected
    assert len(probability_operations.__all__) == len(
        set(probability_operations.__all__)
    )
    assert all(not name.startswith("_") for name in probability_operations.__all__)
    assert all(
        hasattr(probability_operations, name) for name in probability_operations.__all__
    )
    assert probability_operations.convolution_peak is probability.convolution_peak
    assert probability_operations.convolution_power is probability.convolution_power
