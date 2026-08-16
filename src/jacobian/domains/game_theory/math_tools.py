"""MathTool declarations for finite game theory operations."""

from __future__ import annotations

from jacobian.contracts.game_theory import (
    BestResponseRequest,
    BestResponseResult,
    ZeroSumNashRequest,
    ZeroSumNashResult,
)
from jacobian.domains._examples import example
from jacobian.domains.game_theory.operations import (
    compute_best_response,
    compute_zero_sum_nash,
)
from jacobian.math_tools import MathTool


GAME_THEORY_OPERATIONS: tuple[MathTool, ...] = (
    MathTool(
        operation_id="game.best_response.compute",
        version="1",
        title="Compute the best response in a normal-form game",
        description=(
            "Compute the best response for one player against an "
            "opponent's mixed strategy in a two-player normal-form "
            "game with rational payoffs."
        ),
        request_type=BestResponseRequest,
        result_type=BestResponseResult,
        run=compute_best_response,
        tags=(
            "game-theory",
            "best-response",
            "nash",
            "rational",
            "exact",
        ),
        examples=(
            example(
                "matching_pennies_br",
                "Best response in a matching pennies game.",
                {
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
                },
            ),
        ),
    ),
    MathTool(
        operation_id="game.zero_sum.nash.compute",
        version="1",
        title="Compute a Nash equilibrium for a zero-sum game",
        description=(
            "Compute an exact rational Nash equilibrium for a two-player "
            "zero-sum game using exact rational arithmetic."
        ),
        request_type=ZeroSumNashRequest,
        result_type=ZeroSumNashResult,
        run=compute_zero_sum_nash,
        tags=(
            "game-theory",
            "zero-sum",
            "nash",
            "equilibrium",
            "rational",
            "exact",
        ),
        examples=(
            example(
                "matching_pennies_nash",
                "Nash equilibrium of matching pennies.",
                {
                    "payoff_matrix": [
                        [{"num": "1", "den": "1"}, {"num": "-1", "den": "1"}],
                        [{"num": "-1", "den": "1"}, {"num": "1", "den": "1"}],
                    ],
                    "row_actions": ["Heads", "Tails"],
                    "column_actions": ["Heads", "Tails"],
                },
            ),
        ),
    ),
)

__all__ = ["GAME_THEORY_OPERATIONS"]
