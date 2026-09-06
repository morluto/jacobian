"""Tests for finite stochastic process operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.probability.stochastic_processes import (
    FiniteProbabilitySpace,
    FiniteRandomVariable,
    FiniteSigmaAlgebra,
)
from jacobian.math.probability.stochastic_processes._models import (
    ConditionalExpectationRequest,
    DoobMartingaleRequest,
    FiltrationRequest,
    FromObservationRequest,
)
from jacobian.math.probability.stochastic_processes._tools import (
    TOOLS,
    compute_conditional_expectation,
    compute_doob_martingale,
    compute_filtration,
    compute_sigma_from_observation,
)
from jacobian.math.probability.stochastic_processes.operations import (
    conditional_expectation,
    sigma_algebra_from_observation,
    sigma_algebra_join,
    verify_doob_martingale,
    verify_filtration,
)

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
        from jacobian.math.probability.stochastic_processes.operations import (
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
        decoded = type(result).model_validate_json(result.model_dump_json())
        assert verify_filtration(decoded)
        assert not verify_filtration(decoded.model_copy(update={"sigmas": ()}))


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
        assert result.martingale[0].values == (_q(1), _q(0))
        decoded = type(result).model_validate_json(result.model_dump_json())
        assert verify_doob_martingale(decoded)
        assert not verify_doob_martingale(
            decoded.model_copy(
                update={
                    "martingale": (
                        decoded.martingale[0].model_copy(
                            update={"values": (_q(0), _q(0))}
                        ),
                    )
                }
            )
        )
        other_space = FiniteProbabilitySpace(samples=("H",), masses=(_q(1),))
        assert not verify_doob_martingale(
            decoded.model_copy(
                update={
                    "martingale": (
                        decoded.martingale[0].model_copy(update={"space": other_space}),
                    )
                }
            )
        )

    def test_empty_observation_history_round_trips(self) -> None:
        result = compute_doob_martingale(
            DoobMartingaleRequest(
                space=_coin_space(), observations=(), payoff=(_q(1), _q(0))
            )
        )
        assert result.martingale == ()
        assert verify_doob_martingale(
            type(result).model_validate_json(result.model_dump_json())
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_nonpositive_mass_rejected(self) -> None:
        space = FiniteProbabilitySpace(samples=("a",), masses=(_q(0),))
        with pytest.raises(OperationDomainValidationError) as error:
            sigma_algebra_from_observation(
                FiniteProbabilitySpace.model_validate_json(space.model_dump_json()),
                ("x",),
            )
        assert (
            error.value.errors()[0]["type"]
            == "finite_stochastic_process.mass_nonpositive"
        )

    def test_masses_not_summing_to_one_rejected(self) -> None:
        space = FiniteProbabilitySpace(samples=("a", "b"), masses=(_q(1, 3), _q(1, 3)))
        with pytest.raises(OperationDomainValidationError) as error:
            sigma_algebra_from_observation(
                FiniteProbabilitySpace.model_validate_json(space.model_dump_json()),
                ("x", "y"),
            )
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

    @pytest.mark.parametrize("blocks", [(("H",),), (("H", "T"), ("T",))])
    def test_forged_partition_rejected_by_consumers(
        self, blocks: tuple[tuple[str, ...], ...]
    ) -> None:
        space = _coin_space()
        sigma = FiniteSigmaAlgebra.model_validate_json(
            FiniteSigmaAlgebra(space=space, blocks=blocks).model_dump_json()
        )
        valid = sigma_algebra_from_observation(space, ("x", "y"))
        with pytest.raises(OperationDomainValidationError):
            sigma_algebra_join(sigma, valid)
        with pytest.raises(OperationDomainValidationError):
            conditional_expectation(
                FiniteRandomVariable(space=space, values=(_q(0), _q(1))), sigma
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
