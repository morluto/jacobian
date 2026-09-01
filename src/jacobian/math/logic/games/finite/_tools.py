"""Finite game theory operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.logic.games.finite._models import (
    MAX_EXACT_EQUILIBRIUM_WORK,
    BestResponseResult,
    DeterministicTerminalGameRequest,
    NashEquilibriumRequest,
    NashEquilibriumResult,
    ZeroSumGameRequest,
)
from jacobian.math.logic.games.finite.operations import (
    best_response,
    nash_equilibrium,
    solve_terminal_game,
)
from jacobian.math.logic.games.finite.values import DeterministicTerminalGameSolution


def _run_best_response(request: ZeroSumGameRequest) -> BestResponseResult:
    return best_response(request.payoff_matrix)


def _run_nash_equilibrium(request: NashEquilibriumRequest) -> NashEquilibriumResult:
    return nash_equilibrium(request.payoff_matrix)


def _run_deterministic_terminal_game(
    request: DeterministicTerminalGameRequest,
) -> DeterministicTerminalGameSolution:
    return solve_terminal_game(request.game)


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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="game_theory.best_response.compute",
        title="Compute a pure best response for the row player",
        description="Return the row with the greatest worst-case payoff and its exact value "
        "for a finite zero-sum payoff matrix.",
        request_type=ZeroSumGameRequest,
        result_type=BestResponseResult,
        run=_run_best_response,
        tags=("game-theory", "best-response", "zero-sum", "exact"),
        examples=(
            OperationExample(
                name="simple_2x2_best_response",
                description="The first row has worst-case payoff 0 and the second row has worst-case payoff 2.",
                input=GAME_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="game_theory.nash_equilibrium.compute",
        title="Compute Nash equilibrium of a zero-sum game",
        description="Find the Nash equilibrium of a 2-player zero-sum game using "
        "exact rational primal and dual linear programs. Payoff entries are "
        "row-major with n_rows * n_cols entries; exact-equilibrium admission "
        "requires its published coupled work measure to be at most "
        f"{MAX_EXACT_EQUILIBRIUM_WORK}.",
        request_type=NashEquilibriumRequest,
        result_type=NashEquilibriumResult,
        run=_run_nash_equilibrium,
        tags=("game-theory", "nash-equilibrium", "zero-sum", "exact"),
        examples=(
            OperationExample(
                name="simple_2x2_nash",
                description="Nash equilibrium of a 2x2 zero-sum game.",
                input=GAME_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="game.deterministic_terminal.solve",
        title="Solve a finite deterministic terminal-payoff game",
        description="Compute every position's exact minimax payoff and one canonical "
        "optimal stationary strategy for each player in a materialized finite "
        "turn-based arena. Terminal positions carry exact rational payoffs to "
        "MAX, and every infinite play has the declared draw payoff.",
        request_type=DeterministicTerminalGameRequest,
        result_type=DeterministicTerminalGameSolution,
        run=_run_deterministic_terminal_game,
        tags=(
            "game-theory",
            "deterministic-game",
            "terminal-payoff",
            "stationary-strategy",
            "exact",
        ),
        examples=(
            OperationExample(
                name="owned_cycle_and_two_terminals",
                description="Solve every position of an owned cyclic arena; positions must "
                "partition into MAX, MIN, and terminal owners, every nonterminal "
                "must have a move, and moves must use declared-position order.",
                input=DETERMINISTIC_TERMINAL_GAME_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
