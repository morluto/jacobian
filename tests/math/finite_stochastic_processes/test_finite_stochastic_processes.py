"""Tests for finite stochastic process operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
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
    return FiniteProbabilitySpace(
        samples=("H", "T"),
        masses=(_q(1, 2), _q(1, 2)),
    )


def _q(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(numerator, denominator)


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
        rv = FiniteRandomVariable(space=_coin_space(), values=(_q(1), _q(0)))
        sigma = FiniteSigmaAlgebra(space=_coin_space(), blocks=(("H", "T"),))
        result = compute_conditional_expectation(
            ConditionalExpectationRequest(rv=rv, sigma=sigma)
        )
        # E[X | trivial sigma] = E[X] = 1/2
        assert result.values == (_q(1, 2), _q(1, 2))

    def test_discrete_sigma(self) -> None:
        rv = FiniteRandomVariable(space=_coin_space(), values=(_q(1), _q(0)))
        sigma = FiniteSigmaAlgebra(space=_coin_space(), blocks=(("H",), ("T",)))
        result = compute_conditional_expectation(
            ConditionalExpectationRequest(rv=rv, sigma=sigma)
        )
        # E[X | {H}] = 1, E[X | {T}] = 0
        assert result.values == (_q(1), _q(0))

    def test_exact_result_may_grow_beyond_input_digit_bound(self) -> None:
        left = 10**255 + 19
        right = 10**255 + 21
        rv = FiniteRandomVariable(
            space=_coin_space(), values=(_q(1, left), _q(1, right))
        )
        sigma = FiniteSigmaAlgebra(space=_coin_space(), blocks=(("H", "T"),))

        result = compute_conditional_expectation(
            ConditionalExpectationRequest(rv=rv, sigma=sigma)
        )

        assert len(result.values[0].den) > 256


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
                payoff=(_q(1), _q(0)),
            )
        )
        assert len(result.martingale) == 1
        assert result.martingale[0] == (_q(1), _q(0))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_nonpositive_mass_rejected(self) -> None:
        with pytest.raises(ValidationError) as error:
            FiniteProbabilitySpace(samples=("a",), masses=(_q(0),))
        assert (
            error.value.errors()[0]["type"]
            == "finite_stochastic_process.mass_nonpositive"
        )

    def test_masses_not_summing_to_one_rejected(self) -> None:
        with pytest.raises(ValidationError) as error:
            FiniteProbabilitySpace(samples=("a", "b"), masses=(_q(1, 3), _q(1, 3)))
        assert (
            error.value.errors()[0]["type"]
            == "finite_stochastic_process.mass_sum_invalid"
        )

    def test_duplicate_samples_rejected(self) -> None:
        with pytest.raises(ValidationError) as error:
            FiniteProbabilitySpace(samples=("a", "a"), masses=(_q(1, 2), _q(1, 2)))
        assert (
            error.value.errors()[0]["type"]
            == "finite_stochastic_process.sample_duplicate"
        )

    @pytest.mark.parametrize(
        "payoff",
        ["0/0", "1/0", {"num": "0", "den": "0"}, {"num": "1", "den": "0"}],
    )
    def test_doob_rejects_invalid_payoff_rationals(self, payoff: object) -> None:
        with pytest.raises(ValidationError):
            DoobMartingaleRequest.model_validate(
                {
                    "space": _coin_space().model_dump(mode="json"),
                    "observations": [["heads", "tails"]],
                    "payoff": [payoff, {"num": "0", "den": "1"}],
                }
            )
