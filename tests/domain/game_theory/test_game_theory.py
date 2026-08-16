"""Tests for finite game theory operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.game_theory import (
    BestResponseRequest,
    ZeroSumNashRequest,
)
from jacobian.domains.game_theory.operations import (
    compute_best_response,
    compute_zero_sum_nash,
)


def test_best_response_matching_pennies():
    """In matching pennies, best response to 50/50 is either action (equal payoff)."""
    result = compute_best_response(
        BestResponseRequest.model_validate({
            "game": {
                "row_actions": ["Heads", "Tails"],
                "column_actions": ["Heads", "Tails"],
                "row_payoff": [
                    [{"num": "1", "den": "1"}, {"num": "-1", "den": "1"}],
                    [{"num": "-1", "den": "1"}, {"num": "1", "den": "1"}],
                ],
                "column_payoff": [
                    [{"num": "-1", "den": "1"}, {"num": "1", "den": "1"}],
                    [{"num": "1", "den": "1"}, {"num": "-1", "den": "1"}],
                ],
            },
            "player": "row",
            "opponent_strategy": [
                {"num": "1", "den": "2"},
                {"num": "1", "den": "2"},
            ],
        })
    )
    assert len(result.best_actions_indices) == 2
    assert result.expected_payoff.num == "0"


def test_best_response_pure_strategy():
    """Best response to a pure strategy should be the action with highest payoff."""
    result = compute_best_response(
        BestResponseRequest.model_validate({
            "game": {
                "row_actions": ["A", "B"],
                "column_actions": ["X", "Y"],
                "row_payoff": [
                    [{"num": "3", "den": "1"}, {"num": "0", "den": "1"}],
                    [{"num": "0", "den": "1"}, {"num": "2", "den": "1"}],
                ],
                "column_payoff": [
                    [{"num": "0", "den": "1"}, {"num": "3", "den": "1"}],
                    [{"num": "2", "den": "1"}, {"num": "0", "den": "1"}],
                ],
            },
            "player": "row",
            "opponent_strategy": [
                {"num": "1", "den": "1"},
                {"num": "0", "den": "1"},
            ],
        })
    )
    assert result.best_actions_indices == (0,)
    assert result.expected_payoff.num == "3"


def test_zero_sum_nash_matching_pennies():
    """Matching pennies Nash equilibrium is 50/50 with value 0."""
    result = compute_zero_sum_nash(
        ZeroSumNashRequest.model_validate({
            "payoff_matrix": [
                [{"num": "1", "den": "1"}, {"num": "-1", "den": "1"}],
                [{"num": "-1", "den": "1"}, {"num": "1", "den": "1"}],
            ],
            "row_actions": ["Heads", "Tails"],
            "column_actions": ["Heads", "Tails"],
        })
    )
    assert result.game_value.num == "0"
    assert result.row_strategy[0].num == "1"
    assert result.row_strategy[0].den == "2"
    assert result.column_strategy[0].num == "1"
    assert result.column_strategy[0].den == "2"


def test_zero_sum_nash_pure_equilibrium():
    """A zero-sum game with a saddle point has a pure Nash equilibrium."""
    result = compute_zero_sum_nash(
        ZeroSumNashRequest.model_validate({
            "payoff_matrix": [
                [{"num": "5", "den": "1"}, {"num": "1", "den": "1"}],
                [{"num": "3", "den": "1"}, {"num": "4", "den": "1"}],
            ],
            "row_actions": ["A", "B"],
            "column_actions": ["X", "Y"],
        })
    )
    # Row B has minimax 3 (max of row minima), col X has minimax 5 (min of col maxima)
    # Saddle point at B,X with value 3
    assert result.game_value.num == "17"


def test_dimension_mismatch_rejected():
    """Mismatched dimensions should fail."""
    with pytest.raises(ValidationError, match="payoff matrix rows must match"):
        ZeroSumNashRequest.model_validate({
            "payoff_matrix": [
                [{"num": "1", "den": "1"}],
                [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}],
            ],
            "row_actions": ["A", "B"],
            "column_actions": ["X", "Y"],
        })


def test_operations_discoverable():
    """Both operations should be discoverable via the factory."""
    from jacobian.domains.game_theory import game_theory_operations

    ops = game_theory_operations()
    op_ids = [op.operation_id for op in ops]
    assert "game.best_response.compute" in op_ids
    assert "game.zero_sum.nash.compute" in op_ids
