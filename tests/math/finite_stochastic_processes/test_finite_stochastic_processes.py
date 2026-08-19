"""Tests for finite stochastic process operations."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.math.finite_stochastic_processes import (
    FiniteProbabilitySpace,
    FiniteRandomVariable,
    FiniteSigmaAlgebra,
)
from jacobian.math.finite_stochastic_processes._models import (
    ConditionalExpectationRequest,
    DoobMartingaleRequest,
    FiltrationRequest,
    FromObservationRequest,
)
from jacobian.math.finite_stochastic_processes._operations import (
    compute_conditional_expectation,
    compute_doob_martingale,
    compute_filtration,
    compute_sigma_from_observation,
)
from jacobian.math.finite_stochastic_processes._tools import TOOLS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coin_space() -> FiniteProbabilitySpace:
    return FiniteProbabilitySpace(samples=("H", "T"), masses=("1/2", "1/2"))


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_catalog_contains_only_audited_agent_outcomes() -> None:
    expected_ids = {
        "probability.finite_sigma_algebra.from_observation.compute",
        "probability.finite_sigma_algebra.join.compute",
        "probability.conditional_expectation.finite.compute",
        "probability.filtration.natural.compute",
        "probability.process.doob_martingale.compute",
    }
    assert {
        t.operation_id
        for t in TOOLS
        if t.operation_id.startswith("probability.finite_sigma_algebra.")
        or t.operation_id.startswith("probability.conditional_expectation.")
        or t.operation_id.startswith("probability.filtration.")
        or t.operation_id.startswith("probability.process.doob_martingale.")
    } == expected_ids


# ---------------------------------------------------------------------------
# Sigma algebra from observation
# ---------------------------------------------------------------------------


class TestSigmaFromObservation:
    def test_coin_observation(self) -> None:
        result = compute_sigma_from_observation(
            FromObservationRequest(space=_coin_space(), observation=("heads", "tails"))
        )
        assert len(result.blocks) == 2

    def test_constant_observation(self) -> None:
        result = compute_sigma_from_observation(
            FromObservationRequest(space=_coin_space(), observation=("same", "same"))
        )
        assert len(result.blocks) == 1


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------


class TestJoin:
    def test_trivial_join(self) -> None:
        sigma = FiniteSigmaAlgebra(space=_coin_space(), blocks=(("H", "T"),))
        from jacobian.math.finite_stochastic_processes.operations import (
            sigma_algebra_join,
        )

        result = sigma_algebra_join(sigma, sigma)
        assert len(result.blocks) == 1


# ---------------------------------------------------------------------------
# Conditional expectation
# ---------------------------------------------------------------------------


class TestConditionalExpectation:
    def test_trivial_sigma(self) -> None:
        rv = FiniteRandomVariable(space=_coin_space(), values=("1", "0"))
        sigma = FiniteSigmaAlgebra(space=_coin_space(), blocks=(("H", "T"),))
        result = compute_conditional_expectation(
            ConditionalExpectationRequest(rv=rv, sigma=sigma)
        )
        # E[X | trivial sigma] = E[X] = 1/2
        assert Fraction(result.values[0]) == Fraction(1, 2)
        assert Fraction(result.values[1]) == Fraction(1, 2)

    def test_discrete_sigma(self) -> None:
        rv = FiniteRandomVariable(space=_coin_space(), values=("1", "0"))
        sigma = FiniteSigmaAlgebra(space=_coin_space(), blocks=(("H",), ("T",)))
        result = compute_conditional_expectation(
            ConditionalExpectationRequest(rv=rv, sigma=sigma)
        )
        # E[X | {H}] = 1, E[X | {T}] = 0
        assert Fraction(result.values[0]) == 1
        assert Fraction(result.values[1]) == 0


# ---------------------------------------------------------------------------
# Filtration
# ---------------------------------------------------------------------------


class TestFiltration:
    def test_single_step(self) -> None:
        result = compute_filtration(
            FiltrationRequest(
                space=_coin_space(),
                observations=(("heads", "tails"),),
            )
        )
        assert len(result.sigmas) == 1


# ---------------------------------------------------------------------------
# Doob martingale
# ---------------------------------------------------------------------------


class TestDoobMartingale:
    def test_coin_doob(self) -> None:
        result = compute_doob_martingale(
            DoobMartingaleRequest(
                space=_coin_space(),
                observations=(("heads", "tails"),),
                payoff=("1", "0"),
            )
        )
        assert len(result.martingale) == 1
        assert Fraction(result.martingale[0][0]) == 1
        assert Fraction(result.martingale[0][1]) == 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_nonpositive_mass_rejected(self) -> None:
        with pytest.raises(ValidationError, match="positive"):
            FiniteProbabilitySpace(samples=("a",), masses=("0",))

    def test_masses_not_summing_to_one_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sum to exactly 1"):
            FiniteProbabilitySpace(samples=("a", "b"), masses=("1/3", "1/3"))

    def test_duplicate_samples_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            FiniteProbabilitySpace(samples=("a", "a"), masses=("1/2", "1/2"))
