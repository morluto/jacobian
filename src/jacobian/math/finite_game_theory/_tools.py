"""Finite game theory operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.finite_game_theory._models import (
    BestResponseResult,
    DeterministicTerminalGameRequest,
    NashEquilibriumResult,
    ZeroSumGameRequest,
)
from jacobian.math.finite_game_theory._operations import (
    compute_best_response,
    compute_deterministic_terminal_game,
    compute_nash_equilibrium,
)
from jacobian.math.finite_game_theory.values import DeterministicTerminalGameSolution


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


GAME_EXAMPLE = {
    "payoff_matrix": {
        "n_rows": 2,
        "n_cols": 2,
        "entries": [
            {"num": "3", "den": "1"},
            {"num": "0", "den": "1"},
            {"num": "0", "den": "1"},
            {"num": "2", "den": "1"},
        ],
    },
}

DETERMINISTIC_TERMINAL_GAME_EXAMPLE = {
    "game": {
        "positions": [
            {"label": "s", "owner": "MIN"},
            {"label": "u", "owner": "MAX"},
            {"label": "v", "owner": "MIN"},
            {
                "label": "t1",
                "owner": "TERMINAL",
                "payoff": {"num": "1", "den": "1"},
            },
            {
                "label": "t2",
                "owner": "TERMINAL",
                "payoff": {"num": "2", "den": "1"},
            },
        ],
        "moves": [
            {"source": "s", "target": "u"},
            {"source": "s", "target": "v"},
            {"source": "u", "target": "u"},
            {"source": "u", "target": "t1"},
            {"source": "v", "target": "t2"},
        ],
        "draw_payoff": {"num": "0", "den": "1"},
    }
}


FINITE_GAME_THEORY_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "game_theory.best_response.compute",
        "Compute best response in a zero-sum game",
        "Compute the maximin value and best row for the row player in a "
        "2-player zero-sum game using exact rational arithmetic.",
        ZeroSumGameRequest,
        BestResponseResult,
        compute_best_response,
        "game-theory",
        "best-response",
        "zero-sum",
        "exact",
        examples=(
            example(
                "simple_2x2",
                "Best response in a 2x2 zero-sum game.",
                GAME_EXAMPLE,
            ),
        ),
    ),
    _op(
        "game_theory.nash_equilibrium.compute",
        "Compute Nash equilibrium of a zero-sum game",
        "Find the Nash equilibrium of a 2-player zero-sum game using "
        "exact rational primal and dual linear programs.",
        ZeroSumGameRequest,
        NashEquilibriumResult,
        compute_nash_equilibrium,
        "game-theory",
        "nash-equilibrium",
        "zero-sum",
        "exact",
        examples=(
            example(
                "simple_2x2_nash",
                "Nash equilibrium of a 2x2 zero-sum game.",
                GAME_EXAMPLE,
            ),
        ),
    ),
    _op(
        "game.deterministic_terminal.solve",
        "Solve a finite deterministic terminal-payoff game",
        "Compute every position's exact minimax payoff and one canonical "
        "optimal stationary strategy for each player in a materialized finite "
        "turn-based arena. Terminal positions carry exact rational payoffs to "
        "MAX, and every infinite play has the declared draw payoff.",
        DeterministicTerminalGameRequest,
        DeterministicTerminalGameSolution,
        compute_deterministic_terminal_game,
        "game-theory",
        "deterministic-game",
        "terminal-payoff",
        "stationary-strategy",
        "exact",
        examples=(
            example(
                "owned_cycle_and_two_terminals",
                "Solve every position of an owned cyclic arena; positions must "
                "partition into MAX, MIN, and terminal owners, every nonterminal "
                "must have a move, and moves must use declared-position order.",
                DETERMINISTIC_TERMINAL_GAME_EXAMPLE,
            ),
        ),
    ),
)


TOOLS = FINITE_GAME_THEORY_OPERATIONS

__all__ = ["TOOLS"]
