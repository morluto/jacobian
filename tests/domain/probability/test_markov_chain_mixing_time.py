"""Tests for the bounded exact Markov chain mixing time operation (#1674)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.markov_chain import MixingTimeRequest, MixingTimeResult
from jacobian.domains.markov_chain.operations import compute_mixing_time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mat(*rows: tuple[str, str]) -> tuple[dict, ...]:
    """Build a row-major matrix of canonical rationals from (num, den) pairs."""
    return tuple(
        tuple({"num": num, "den": den} for num, den in row) for row in rows
    )


TWO_STATE = _mat(
    (("1", "2"), ("1", "2")),
    (("1", "4"), ("3", "4")),
)


SWAP = _mat(
    (("0", "1"), ("1", "1")),
    (("1", "1"), ("0", "1")),
)


IDENTITY = _mat(
    (("1", "1"), ("0", "1")),
    (("0", "1"), ("1", "1")),
)


UNIFORM_2 = _mat(
    (("1", "2"), ("1", "2")),
    (("1", "2"), ("1", "2")),
)


# ---------------------------------------------------------------------------
# Mixing time correctness
# ---------------------------------------------------------------------------

class TestMixingTimeCorrectness:
    """Verify exact mixing time values."""

    def test_uniform_matrix_mixes_in_one_step(self) -> None:
        """The uniform 2x2 matrix is already stationary after one step."""
        result = compute_mixing_time(
            MixingTimeRequest.model_validate(
                {"matrix": UNIFORM_2, "epsilon": {"num": "1", "den": "100"},
                 "max_steps": 100}
            )
        )
        assert result.mixing_time == 1

    def test_two_state_chain_small_epsilon(self) -> None:
        """Two-state chain P=[[1/2,1/2],[1/4,3/4]] has eigenvalue 1/4.

        Stationary distribution is [1/3, 2/3].  At t=0 the TV distance is
        1/2; at t=1 it is 1/6; at t=2 it is 1/12; at t=3 it is 1/24;
        at t=4 it is 1/48 < 1/100 = 0.01.

        So the mixing time for epsilon=1/100 should be 4.
        """
        result = compute_mixing_time(
            MixingTimeRequest.model_validate(
                {"matrix": TWO_STATE, "epsilon": {"num": "1", "den": "100"},
                 "max_steps": 100}
            )
        )
        assert result.mixing_time == 4

    def test_two_state_chain_larger_epsilon(self) -> None:
        """With epsilon=1/6 the mixing time should be 1.

        At t=0: TV = 1/2 > 1/6.
        At t=1: TV = 1/6 <= 1/6.  So mixing time = 1.
        """
        result = compute_mixing_time(
            MixingTimeRequest.model_validate(
                {"matrix": TWO_STATE, "epsilon": {"num": "1", "den": "6"},
                 "max_steps": 100}
            )
        )
        assert result.mixing_time == 1

    def test_two_state_chain_loose_epsilon(self) -> None:
        """With a large enough epsilon the mixing time is 0.

        For P=[[1/2,1/2],[1/4,3/4]] the stationary distribution is [1/3, 2/3].
        At t=0 (identity), the max TV distance is 2/3 (from row [1,0] to [1/3,2/3]).
        So with epsilon = 1 (larger than any possible TV distance) mixing time = 0.
        """
        result = compute_mixing_time(
            MixingTimeRequest.model_validate(
                {"matrix": TWO_STATE, "epsilon": {"num": "1", "den": "1"},
                 "max_steps": 100}
            )
        )
        assert result.mixing_time == 0

    def test_method_is_sympy_exact(self) -> None:
        result = compute_mixing_time(
            MixingTimeRequest.model_validate(
                {"matrix": TWO_STATE, "epsilon": {"num": "1", "den": "100"},
                 "max_steps": 100}
            )
        )
        assert result.method == "SYMPY_EXACT"

    def test_result_is_mixing_time_result_type(self) -> None:
        result = compute_mixing_time(
            MixingTimeRequest.model_validate(
                {"matrix": TWO_STATE, "epsilon": {"num": "1", "den": "100"},
                 "max_steps": 100}
            )
        )
        assert isinstance(result, MixingTimeResult)
        assert isinstance(result.mixing_time, int)


# ---------------------------------------------------------------------------
# Fail-closed: budget exhaustion
# ---------------------------------------------------------------------------

class TestBudgetExhaustion:
    """The operation must fail closed when mixing time exceeds the budget."""

    def test_periodic_chain_exceeds_budget(self) -> None:
        """The swap chain is periodic and never mixes, so any budget is exceeded."""
        with pytest.raises(ValueError, match="max_steps budget"):
            compute_mixing_time(
                MixingTimeRequest.model_validate(
                    {"matrix": SWAP, "epsilon": {"num": "1", "den": "100"},
                     "max_steps": 5}
                )
            )

    def test_two_state_chain_tight_budget(self) -> None:
        """If max_steps is too small, the operation fails even for a valid chain."""
        with pytest.raises(ValueError, match="max_steps budget"):
            compute_mixing_time(
                MixingTimeRequest.model_validate(
                    {"matrix": TWO_STATE, "epsilon": {"num": "1", "den": "100"},
                     "max_steps": 3}
                )
            )

    def test_identity_chain_exceeds_budget(self) -> None:
        """The identity matrix is reducible and never mixes."""
        with pytest.raises(ValueError, match="max_steps budget"):
            compute_mixing_time(
                MixingTimeRequest.model_validate(
                    {"matrix": IDENTITY, "epsilon": {"num": "1", "den": "100"},
                     "max_steps": 10}
                )
            )


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------

class TestContractValidation:
    """The MixingTimeRequest must reject invalid inputs."""

    def test_non_square_matrix_rejected(self) -> None:
        with pytest.raises(ValidationError, match="square"):
            MixingTimeRequest.model_validate(
                {
                    "matrix": [
                        [{"num": "1", "den": "2"}, {"num": "1", "den": "2"}],
                    ],
                    "epsilon": {"num": "1", "den": "100"},
                    "max_steps": 100,
                }
            )

    def test_row_not_sum_to_one_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sum to one"):
            MixingTimeRequest.model_validate(
                {
                    "matrix": [
                        [{"num": "1", "den": "2"}, {"num": "1", "den": "3"}],
                        [{"num": "1", "den": "4"}, {"num": "3", "den": "4"}],
                    ],
                    "epsilon": {"num": "1", "den": "100"},
                    "max_steps": 100,
                }
            )

    def test_negative_entry_rejected(self) -> None:
        with pytest.raises(ValidationError, match="nonnegative"):
            MixingTimeRequest.model_validate(
                {
                    "matrix": [
                        [{"num": "-1", "den": "1"}, {"num": "2", "den": "1"}],
                        [{"num": "1", "den": "4"}, {"num": "3", "den": "4"}],
                    ],
                    "epsilon": {"num": "1", "den": "100"},
                    "max_steps": 100,
                }
            )

    def test_zero_epsilon_rejected(self) -> None:
        with pytest.raises(ValidationError, match="epsilon must be a positive"):
            MixingTimeRequest.model_validate(
                {
                    "matrix": TWO_STATE,
                    "epsilon": {"num": "0", "den": "1"},
                    "max_steps": 100,
                }
            )

    def test_negative_epsilon_rejected(self) -> None:
        with pytest.raises(ValidationError, match="epsilon must be a positive"):
            MixingTimeRequest.model_validate(
                {
                    "matrix": TWO_STATE,
                    "epsilon": {"num": "-1", "den": "100"},
                    "max_steps": 100,
                }
            )

    def test_max_steps_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MixingTimeRequest.model_validate(
                {
                    "matrix": TWO_STATE,
                    "epsilon": {"num": "1", "den": "100"},
                    "max_steps": 0,
                }
            )

    @pytest.mark.parametrize("max_steps", [0, -1, 100001])
    def test_invalid_max_steps_rejected(self, max_steps: int) -> None:
        with pytest.raises(ValidationError):
            MixingTimeRequest.model_validate(
                {
                    "matrix": TWO_STATE,
                    "epsilon": {"num": "1", "den": "100"},
                    "max_steps": max_steps,
                }
            )


# ---------------------------------------------------------------------------
# Three-state chains
# ---------------------------------------------------------------------------

class TestThreeStateChains:
    """Exercise 3x3 transition matrices."""

    def test_uniform_three_state_mixes_in_one(self) -> None:
        uniform = tuple(
            tuple({"num": "1", "den": "3"} for _ in range(3)) for _ in range(3)
        )
        result = compute_mixing_time(
            MixingTimeRequest.model_validate(
                {"matrix": uniform, "epsilon": {"num": "1", "den": "1000"},
                 "max_steps": 100}
            )
        )
        assert result.mixing_time == 1

    def test_deterministic_cycle_three_states(self) -> None:
        """Deterministic 3-cycle: 0->1->2->0.  This is periodic (period 3).

        It never mixes, so the operation must fail closed on any budget.
        """
        cycle = (
            (
                {"num": "0", "den": "1"},
                {"num": "1", "den": "1"},
                {"num": "0", "den": "1"},
            ),
            (
                {"num": "0", "den": "1"},
                {"num": "0", "den": "1"},
                {"num": "1", "den": "1"},
            ),
            (
                {"num": "1", "den": "1"},
                {"num": "0", "den": "1"},
                {"num": "0", "den": "1"},
            ),
        )
        with pytest.raises(ValueError, match="max_steps budget"):
            compute_mixing_time(
                MixingTimeRequest.model_validate(
                    {"matrix": cycle, "epsilon": {"num": "1", "den": "1000"},
                     "max_steps": 50}
                )
            )

    def test_lazy_three_state_chain(self) -> None:
        """A lazy chain on 3 states with self-loops that does mix."""
        chain = (
            (
                {"num": "1", "den": "2"},
                {"num": "1", "den": "4"},
                {"num": "1", "den": "4"},
            ),
            (
                {"num": "1", "den": "4"},
                {"num": "1", "den": "2"},
                {"num": "1", "den": "4"},
            ),
            (
                {"num": "1", "den": "4"},
                {"num": "1", "den": "4"},
                {"num": "1", "den": "2"},
            ),
        )
        result = compute_mixing_time(
            MixingTimeRequest.model_validate(
                {"matrix": chain, "epsilon": {"num": "1", "den": "1000"},
                 "max_steps": 500}
            )
        )
        assert result.mixing_time > 0


# ---------------------------------------------------------------------------
# Consistency with stationary distribution
# ---------------------------------------------------------------------------

class TestConsistencyWithStationary:
    """The mixing time computation must be consistent with the stationary distribution."""

    def test_uniform_matrix_zero_distance_at_step_one(self) -> None:
        """For the uniform 2x2 matrix, after one step every row equals the
        stationary distribution, so the TV distance is exactly zero."""
        result = compute_mixing_time(
            MixingTimeRequest.model_validate(
                {"matrix": UNIFORM_2, "epsilon": {"num": "1", "den": "10000000"},
                 "max_steps": 100}
            )
        )
        assert result.mixing_time == 1
