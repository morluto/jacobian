"""Finite game theory operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.finite_game_theory import (
    BestResponseResult,
    NashEquilibriumResult,
    ZeroSumGameRequest,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.finite_game_theory.operations import (
    compute_best_response,
    compute_nash_equilibrium,
)
from jacobian.math_tools import MathTool


def _op[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
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
        "exact rational arithmetic via support enumeration.",
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
)


__all__ = ["FINITE_GAME_THEORY_OPERATIONS"]
