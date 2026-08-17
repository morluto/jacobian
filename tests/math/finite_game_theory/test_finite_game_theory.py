"""Tests for finite game theory operations."""

from jacobian.math.finite_game_theory._models import (
    PayoffMatrix,
    ZeroSumGameRequest,
)
from jacobian.math.finite_game_theory._operations import (
    compute_best_response,
    compute_nash_equilibrium,
)


class TestBestResponse:
    def test_simple(self):
        req = ZeroSumGameRequest(
            payoff_matrix=PayoffMatrix(
                n_rows=2,
                n_cols=2,
                entries=(
                    {"num": "3", "den": "1"},
                    {"num": "0", "den": "1"},
                    {"num": "0", "den": "1"},
                    {"num": "2", "den": "1"},
                ),
            ),
        )
        result = compute_best_response(req)
        assert result.best_row == 0  # Row 0 has minimum 0, Row 1 has minimum 0


class TestNashEquilibrium:
    def test_pure_strategy(self):
        req = ZeroSumGameRequest(
            payoff_matrix=PayoffMatrix(
                n_rows=2,
                n_cols=2,
                entries=(
                    {"num": "1", "den": "1"},
                    {"num": "1", "den": "1"},
                    {"num": "0", "den": "1"},
                    {"num": "0", "den": "1"},
                ),
            ),
        )
        result = compute_nash_equilibrium(req)
        assert result.value == "1"  # (0,0) is the pure Nash equilibrium

    def test_mixed_equilibrium(self):
        req = ZeroSumGameRequest(
            payoff_matrix=PayoffMatrix(
                n_rows=2,
                n_cols=2,
                entries=(
                    {"num": "2", "den": "1"},
                    {"num": "0", "den": "1"},
                    {"num": "-1", "den": "1"},
                    {"num": "3", "den": "1"},
                ),
            ),
        )
        result = compute_nash_equilibrium(req)
        assert result.row_strategy == ("2/3", "1/3")
        assert result.col_strategy == ("1/2", "1/2")
        assert result.value == "1"

    def test_maximin_uses_actual_payoff(self):
        req = ZeroSumGameRequest(
            payoff_matrix=PayoffMatrix(
                n_rows=1,
                n_cols=1,
                entries=({"num": "-100000000000000000000", "den": "1"},),
            ),
        )
        result = compute_best_response(req)
        assert result.value == "-100000000000000000000"
