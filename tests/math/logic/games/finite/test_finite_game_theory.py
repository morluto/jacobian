"""Tests for finite game theory operations."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.logic.games.finite._models import (
    NashEquilibriumRequest,
    PayoffMatrix,
    ZeroSumGameRequest,
)
from jacobian.math.logic.games.finite.operations import (
    best_response,
    nash_equilibrium,
)


def _r(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


class TestBestResponse:
    def test_simple(self) -> None:
        req = ZeroSumGameRequest(
            payoff_matrix=PayoffMatrix(
                n_rows=2,
                n_cols=2,
                entries=(
                    _r(3),
                    _r(0),
                    _r(0),
                    _r(2),
                ),
            ),
        )
        result = best_response(req.payoff_matrix)
        assert result.best_row == 0  # Row 0 has minimum 0, Row 1 has minimum 0

    def test_payoffs_beyond_the_equilibrium_bound_still_admit_best_response(
        self,
    ) -> None:
        large = 10**8_200
        req = ZeroSumGameRequest(
            payoff_matrix=PayoffMatrix(
                n_rows=2,
                n_cols=2,
                entries=(
                    _r(Fraction(1, large - 1)),
                    _r(Fraction(1, large - 2)),
                    _r(Fraction(1, large - 3)),
                    _r(Fraction(1, large - 4)),
                ),
            ),
        )
        result = best_response(req.payoff_matrix)
        assert result.best_row == 1
        assert result.value.as_fraction() == Fraction(1, large - 3)


class TestNashEquilibrium:
    def test_payoffs_must_bound_exact_strategy_growth(self) -> None:
        large = 10**32_767
        request = NashEquilibriumRequest(
            payoff_matrix=PayoffMatrix(
                n_rows=2,
                n_cols=2,
                entries=(
                    _r(Fraction(1, large - 1)),
                    _r(0),
                    _r(0),
                    _r(Fraction(1, large - 2)),
                ),
            )
        )
        with pytest.raises(ValueError, match="exact-equilibrium work bound"):
            nash_equilibrium(request.payoff_matrix)

    def test_pure_strategy(self) -> None:
        req = NashEquilibriumRequest(
            payoff_matrix=PayoffMatrix(
                n_rows=2,
                n_cols=2,
                entries=(
                    _r(1),
                    _r(1),
                    _r(0),
                    _r(0),
                ),
            ),
        )
        result = nash_equilibrium(req.payoff_matrix)
        assert result.value.as_fraction() == 1  # (0,0) is the pure equilibrium

    def test_mixed_equilibrium(self) -> None:
        req = NashEquilibriumRequest(
            payoff_matrix=PayoffMatrix(
                n_rows=2,
                n_cols=2,
                entries=(
                    _r(2),
                    _r(0),
                    _r(-1),
                    _r(3),
                ),
            ),
        )
        result = nash_equilibrium(req.payoff_matrix)
        assert tuple(value.as_fraction() for value in result.row_strategy) == (
            Fraction(2, 3),
            Fraction(1, 3),
        )
        assert tuple(value.as_fraction() for value in result.col_strategy) == (
            Fraction(1, 2),
            Fraction(1, 2),
        )
        assert result.value.as_fraction() == 1

    def test_degenerate_game_uses_exact_linear_programming(self) -> None:
        req = NashEquilibriumRequest(
            payoff_matrix=PayoffMatrix(
                n_rows=3,
                n_cols=3,
                entries=tuple(_r(value) for value in (1, -1, 0, -1, 1, 0, 0, 0, 0)),
            ),
        )

        result = nash_equilibrium(req.payoff_matrix)

        assert result.value.as_fraction() == 0
        assert sum(value.as_fraction() for value in result.row_strategy) == 1
        assert sum(value.as_fraction() for value in result.col_strategy) == 1

    def test_negative_game_value_is_not_clamped_by_simplex_nonnegativity(self) -> None:
        req = NashEquilibriumRequest(
            payoff_matrix=PayoffMatrix(
                n_rows=2,
                n_cols=2,
                entries=tuple(_r(value) for value in (-2, -3, -4, -5)),
            ),
        )

        result = nash_equilibrium(req.payoff_matrix)

        assert tuple(value.as_fraction() for value in result.row_strategy) == (
            Fraction(1),
            Fraction(0),
        )
        assert tuple(value.as_fraction() for value in result.col_strategy) == (
            Fraction(0),
            Fraction(1),
        )
        assert result.value.as_fraction() == -3

    def test_fractional_payoffs_are_scaled_before_exact_linear_programming(
        self,
    ) -> None:
        values = (Fraction(1, 3), Fraction(-2, 5), Fraction(7, 4), Fraction(1, 2))
        req = NashEquilibriumRequest(
            payoff_matrix=PayoffMatrix(
                n_rows=2,
                n_cols=2,
                entries=tuple(_r(value) for value in values),
            ),
        )

        result = nash_equilibrium(req.payoff_matrix)

        assert tuple(value.as_fraction() for value in result.row_strategy) == (
            Fraction(0),
            Fraction(1),
        )
        assert tuple(value.as_fraction() for value in result.col_strategy) == (
            Fraction(0),
            Fraction(1),
        )
        assert result.value.as_fraction() == Fraction(1, 2)

    def test_maximin_uses_actual_payoff(self) -> None:
        req = ZeroSumGameRequest(
            payoff_matrix=PayoffMatrix(
                n_rows=1,
                n_cols=1,
                entries=(_r(-100000000000000000000),),
            ),
        )
        result = best_response(req.payoff_matrix)
        assert result.value.as_fraction() == -100000000000000000000
