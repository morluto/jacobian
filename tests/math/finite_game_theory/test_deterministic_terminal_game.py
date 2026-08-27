"""Exact contract tests for deterministic terminal-payoff games."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math import finite_game_theory
from jacobian.math.finite_game_theory import (
    DeterministicGameMove,
    DeterministicGamePosition,
    DeterministicTerminalGame,
    DeterministicTerminalGameSolution,
    StationaryChoice,
    TerminalGameValueClass,
    solve_terminal_game,
)
from jacobian.math.finite_game_theory import _operations as operation_adapter
from jacobian.math.finite_game_theory import operations as terminal_operations
from jacobian.math.finite_game_theory._models import (
    DeterministicTerminalGameRequest,
)
from jacobian.math.finite_game_theory._operations import (
    compute_deterministic_terminal_game,
    verify_deterministic_terminal_game_solution,
)
from jacobian.math.finite_game_theory._tools import TOOLS


def _r(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(numerator, denominator)


def _paper_game() -> DeterministicTerminalGame:
    return DeterministicTerminalGame(
        positions=(
            DeterministicGamePosition(label="s", owner="MIN"),
            DeterministicGamePosition(label="u", owner="MAX"),
            DeterministicGamePosition(label="v", owner="MIN"),
            DeterministicGamePosition(label="t1", owner="TERMINAL", payoff=_r(1)),
            DeterministicGamePosition(label="t2", owner="TERMINAL", payoff=_r(2)),
        ),
        moves=(
            DeterministicGameMove(source="s", target="u"),
            DeterministicGameMove(source="s", target="v"),
            DeterministicGameMove(source="u", target="u"),
            DeterministicGameMove(source="u", target="t1"),
            DeterministicGameMove(source="v", target="t2"),
        ),
        draw_payoff=_r(0),
    )


def _all_stationary_strategies(
    game: DeterministicTerminalGame, owner: str
) -> tuple[dict[str, str], ...]:
    successors: dict[str, list[str]] = {
        position.label: [] for position in game.positions
    }
    for move in game.moves:
        successors[move.source].append(move.target)
    controlled = tuple(
        position.label for position in game.positions if position.owner == owner
    )
    return tuple(
        dict(zip(controlled, choices, strict=True))
        for choices in product(*(successors[position] for position in controlled))
    )


def _play_payoff(
    game: DeterministicTerminalGame,
    start: str,
    max_strategy: dict[str, str],
    min_strategy: dict[str, str],
) -> Fraction:
    positions = {position.label: position for position in game.positions}
    current = start
    seen: set[str] = set()
    while True:
        position = positions[current]
        if position.owner == "TERMINAL":
            assert position.payoff is not None
            return position.payoff.as_fraction()
        if current in seen:
            return game.draw_payoff.as_fraction()
        seen.add(current)
        current = (
            max_strategy[current] if position.owner == "MAX" else min_strategy[current]
        )


def _profile(
    game: DeterministicTerminalGame,
    max_strategy: dict[str, str],
    min_strategy: dict[str, str],
) -> tuple[Fraction, ...]:
    return tuple(
        _play_payoff(game, position.label, max_strategy, min_strategy)
        for position in game.positions
    )


def test_source_example_returns_values_and_stationary_witnesses() -> None:
    game = _paper_game()
    result = solve_terminal_game(game)

    assert tuple(
        (entry.payoff.as_fraction(), entry.positions) for entry in result.value_classes
    ) == (
        (Fraction(1), ("s", "u", "t1")),
        (Fraction(2), ("v", "t2")),
    )
    assert result.max_strategy == (StationaryChoice(position="u", target="t1"),)
    assert result.min_strategy == (
        StationaryChoice(position="s", target="u"),
        StationaryChoice(position="v", target="t2"),
    )


def test_all_four_source_strategy_outcomes_are_reproduced() -> None:
    game = _paper_game()

    outcomes = tuple(
        _profile(game, max_strategy, min_strategy)
        for max_strategy in _all_stationary_strategies(game, "MAX")
        for min_strategy in _all_stationary_strategies(game, "MIN")
    )

    assert outcomes == (
        tuple(map(Fraction, (0, 0, 2, 1, 2))),
        tuple(map(Fraction, (2, 0, 2, 1, 2))),
        tuple(map(Fraction, (1, 1, 2, 1, 2))),
        tuple(map(Fraction, (2, 1, 2, 1, 2))),
    )


def test_solution_matches_independent_stationary_strategy_enumeration() -> None:
    game = _paper_game()
    result = solve_terminal_game(game)
    max_strategies = _all_stationary_strategies(game, "MAX")
    min_strategies = _all_stationary_strategies(game, "MIN")
    result_max = {choice.position: choice.target for choice in result.max_strategy}
    result_min = {choice.position: choice.target for choice in result.min_strategy}

    expected_values = {
        position: value_class.payoff.as_fraction()
        for value_class in result.value_classes
        for position in value_class.positions
    }
    for start in (position.label for position in game.positions):
        maximin = max(
            min(
                _play_payoff(game, start, max_strategy, min_strategy)
                for min_strategy in min_strategies
            )
            for max_strategy in max_strategies
        )
        minimax = min(
            max(
                _play_payoff(game, start, max_strategy, min_strategy)
                for max_strategy in max_strategies
            )
            for min_strategy in min_strategies
        )
        witnessed_lower = min(
            _play_payoff(game, start, result_max, min_strategy)
            for min_strategy in min_strategies
        )
        witnessed_upper = max(
            _play_payoff(game, start, max_strategy, result_min)
            for max_strategy in max_strategies
        )

        assert maximin == minimax == expected_values[start]
        assert witnessed_lower == witnessed_upper == expected_values[start]


def test_infinite_play_is_an_exact_value_not_a_missing_witness() -> None:
    game = DeterministicTerminalGame(
        positions=(
            DeterministicGamePosition(label="x", owner="MAX"),
            DeterministicGamePosition(label="y", owner="MIN"),
        ),
        moves=(
            DeterministicGameMove(source="x", target="x"),
            DeterministicGameMove(source="x", target="y"),
            DeterministicGameMove(source="y", target="x"),
        ),
        draw_payoff=_r(3, 2),
    )

    result = solve_terminal_game(game)

    assert result.value_classes == (
        TerminalGameValueClass(payoff=_r(3, 2), positions=("x", "y")),
    )
    assert result.max_strategy == (StationaryChoice(position="x", target="x"),)
    assert result.min_strategy == (StationaryChoice(position="y", target="x"),)


def test_exact_fractional_thresholds_choose_reachability_over_draw() -> None:
    game = DeterministicTerminalGame(
        positions=(
            DeterministicGamePosition(label="x", owner="MAX"),
            DeterministicGamePosition(label="high", owner="TERMINAL", payoff=_r(2, 3)),
        ),
        moves=(
            DeterministicGameMove(source="x", target="x"),
            DeterministicGameMove(source="x", target="high"),
        ),
        draw_payoff=_r(1, 2),
    )

    result = solve_terminal_game(game)

    assert result.value_classes[-1].payoff.as_fraction() == Fraction(2, 3)
    assert result.max_strategy == (StationaryChoice(position="x", target="high"),)


def test_reachability_strategy_uses_canonical_strict_progress_rank() -> None:
    game = DeterministicTerminalGame(
        positions=(
            DeterministicGamePosition(label="x", owner="MAX"),
            DeterministicGamePosition(label="y", owner="MAX"),
            DeterministicGamePosition(label="t", owner="TERMINAL", payoff=_r(1)),
        ),
        moves=(
            DeterministicGameMove(source="x", target="y"),
            DeterministicGameMove(source="x", target="t"),
            DeterministicGameMove(source="y", target="t"),
        ),
        draw_payoff=_r(0),
    )

    result = solve_terminal_game(game)

    assert result.value_classes == (
        TerminalGameValueClass(payoff=_r(1), positions=("x", "y", "t")),
    )
    assert result.max_strategy == (
        StationaryChoice(position="x", target="t"),
        StationaryChoice(position="y", target="t"),
    )


def test_below_draw_reachability_and_at_draw_safety_choose_progress() -> None:
    game = DeterministicTerminalGame(
        positions=(
            DeterministicGamePosition(label="m", owner="MIN"),
            DeterministicGamePosition(label="x", owner="MAX"),
            DeterministicGamePosition(label="low", owner="TERMINAL", payoff=_r(-1)),
        ),
        moves=(
            DeterministicGameMove(source="m", target="m"),
            DeterministicGameMove(source="m", target="low"),
            DeterministicGameMove(source="x", target="x"),
            DeterministicGameMove(source="x", target="low"),
        ),
        draw_payoff=_r(0),
    )

    result = solve_terminal_game(game)

    assert result.value_classes == (
        TerminalGameValueClass(payoff=_r(-1), positions=("m", "low")),
        TerminalGameValueClass(payoff=_r(0), positions=("x",)),
    )
    assert result.max_strategy == (StationaryChoice(position="x", target="x"),)
    assert result.min_strategy == (StationaryChoice(position="m", target="low"),)


@pytest.mark.parametrize(
    ("payload", "error_type"),
    [
        (
            {
                "positions": [{"label": "t", "owner": "TERMINAL"}],
                "moves": [],
                "draw_payoff": {"num": "0", "den": "1"},
            },
            "finite_game.terminal_payoff_required",
        ),
        (
            {
                "positions": [
                    {
                        "label": "x",
                        "owner": "MAX",
                        "payoff": {"num": "0", "den": "1"},
                    }
                ],
                "moves": [{"source": "x", "target": "x"}],
                "draw_payoff": {"num": "0", "den": "1"},
            },
            "finite_game.nonterminal_payoff_forbidden",
        ),
        (
            {
                "positions": [
                    {
                        "label": "t",
                        "owner": "TERMINAL",
                        "payoff": {"num": "0", "den": "1"},
                    }
                ],
                "moves": [{"source": "t", "target": "t"}],
                "draw_payoff": {"num": "0", "den": "1"},
            },
            "finite_game.terminal_has_moves",
        ),
        (
            {
                "positions": [{"label": "x", "owner": "MAX"}],
                "moves": [{"source": "x", "target": "missing"}],
                "draw_payoff": {"num": "0", "den": "1"},
            },
            "finite_game.move_endpoint_unknown",
        ),
        (
            {
                "positions": [
                    {"label": "x", "owner": "MAX"},
                    {"label": "y", "owner": "MIN"},
                ],
                "moves": [
                    {"source": "y", "target": "x"},
                    {"source": "x", "target": "y"},
                ],
                "draw_payoff": {"num": "0", "den": "1"},
            },
            "finite_game.moves_not_canonical",
        ),
    ],
)
def test_malformed_arenas_fail_at_the_typed_boundary(
    payload: dict[str, object], error_type: str
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        DeterministicTerminalGame.model_validate(payload)
    assert exc_info.value.errors()[0]["type"] == error_type


def test_threshold_work_is_admitted_and_rejected_before_solving() -> None:
    def terminal_game(size: int) -> dict[str, object]:
        return {
            "positions": [
                {
                    "label": f"t{index}",
                    "owner": "TERMINAL",
                    "payoff": {"num": str(index), "den": "1"},
                }
                for index in range(size)
            ],
            "moves": [],
            "draw_payoff": {"num": "0", "den": "1"},
        }

    DeterministicTerminalGame.model_validate(terminal_game(256))
    with pytest.raises(ValidationError) as exc_info:
        DeterministicTerminalGame.model_validate(terminal_game(400))
    assert exc_info.value.errors()[0]["type"] == "finite_game.threshold_work_exceeded"


def test_result_size_is_rejected_independently_of_threshold_work() -> None:
    size = 170
    labels = tuple(f"v{index:04d}" + "x" * 59 for index in range(size))
    payload = {
        "positions": [{"label": label, "owner": "MAX"} for label in labels],
        "moves": [
            {"source": source, "target": target}
            for source in labels
            for target in labels
        ],
        "draw_payoff": {"num": "0", "den": "1"},
    }

    with pytest.raises(ValidationError) as exc_info:
        DeterministicTerminalGame.model_validate(payload)
    assert exc_info.value.errors()[0]["type"] == "finite_game.result_size_exceeded"


def test_solution_claims_are_structural_and_verifier_rejects_mutations() -> None:
    result = solve_terminal_game(_paper_game())

    bad_classes = (
        TerminalGameValueClass(payoff=_r(0), positions=("s", "u", "t1")),
        *result.value_classes[1:],
    )
    bad_value_classes = DeterministicTerminalGameSolution(
        game=result.game,
        value_classes=bad_classes,
        max_strategy=result.max_strategy,
        min_strategy=result.min_strategy,
    )
    assert not verify_deterministic_terminal_game_solution(bad_value_classes)

    bad_strategy = DeterministicTerminalGameSolution(
        game=result.game,
        value_classes=result.value_classes,
        max_strategy=(StationaryChoice(position="u", target="u"),),
        min_strategy=result.min_strategy,
    )
    assert not verify_deterministic_terminal_game_solution(bad_strategy)

    draw_mutation = result.game.model_dump(mode="json")
    draw_mutation["draw_payoff"] = {"num": "3", "den": "1"}
    claim = DeterministicTerminalGameSolution(
        game=DeterministicTerminalGame.model_validate(draw_mutation),
        value_classes=result.value_classes,
        max_strategy=result.max_strategy,
        min_strategy=result.min_strategy,
    )
    assert not verify_deterministic_terminal_game_solution(claim)


def test_public_request_and_example_return_the_declared_result() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "game.deterministic_terminal.solve"
    )
    request = DeterministicTerminalGameRequest.model_validate(
        operation.examples[0].input
    )

    result = compute_deterministic_terminal_game(request)

    assert isinstance(result, DeterministicTerminalGameSolution)
    assert result.value_classes[0].payoff.as_fraction() == 1


def test_trusted_terminal_game_producers_run_the_minimax_kernel_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = _paper_game()
    request = DeterministicTerminalGameRequest(game=game)
    calls = 0
    original = terminal_operations._solve_terminal_game_data

    def counted_native(*args: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(
        terminal_operations, "_solve_terminal_game_data", counted_native
    )
    solve_terminal_game(game)
    assert calls == 1

    calls = 0
    original_adapter = operation_adapter._solve_terminal_game_data

    def counted_adapter(*args: object) -> object:
        nonlocal calls
        calls += 1
        return original_adapter(*args)

    monkeypatch.setattr(operation_adapter, "_solve_terminal_game_data", counted_adapter)
    compute_deterministic_terminal_game(request)
    assert calls == 1


def test_native_api_is_explicit() -> None:
    assert tuple(finite_game_theory.__all__) == (
        "DeterministicGameMove",
        "DeterministicGamePosition",
        "DeterministicTerminalGame",
        "DeterministicTerminalGameSolution",
        "StationaryChoice",
        "TerminalGameValueClass",
        "solve_terminal_game",
    )
    assert all(hasattr(finite_game_theory, name) for name in finite_game_theory.__all__)
