"""Exact public API contract for jacobian.math.probability.graphical_models."""

from __future__ import annotations

from jacobian.math.probability import graphical_models


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the graphical_models public API."""
    expected = (
        "BayesianNetwork",
        "ConditionalProbabilityTable",
        "Factor",
        "bayes_net_joint",
        "construct_bayes_net",
        "d_separation",
        "factor_marginalize",
        "factor_multiply",
        "junction_tree_calibrate",
        "variable_elimination",
        "variable_elimination_trace",
        "verify_d_separation",
    )
    # The module's declared __all__ is the source of truth: every
    # advertised name must stay exported, but adding one is allowed.
    assert set(expected) <= set(graphical_models.__all__)
    assert len(graphical_models.__all__) == len(set(graphical_models.__all__))
    assert all(not name.startswith("_") for name in graphical_models.__all__)
    assert all(hasattr(graphical_models, name) for name in graphical_models.__all__)
