"""Exact public API contract for jacobian.math.finite_stochastic_processes."""

from __future__ import annotations

from jacobian.math import finite_stochastic_processes


def test_exact_public_api_symbols() -> None:
    expected = (
        "FiniteProbabilitySpace",
        "FiniteRandomVariable",
        "FiniteSigmaAlgebra",
        "conditional_expectation",
        "doob_martingale",
        "filtration_natural",
        "poisson_binomial",
        "sigma_algebra_from_observation",
        "sigma_algebra_join",
    )
    assert tuple(finite_stochastic_processes.__all__) == expected
    assert len(finite_stochastic_processes.__all__) == len(
        set(finite_stochastic_processes.__all__)
    )
    assert all(not name.startswith("_") for name in finite_stochastic_processes.__all__)
    assert all(
        hasattr(finite_stochastic_processes, name)
        for name in finite_stochastic_processes.__all__
    )
