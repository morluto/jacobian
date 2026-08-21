"""Known-answer and adversarial tests for the disjunctive sum operation."""

import pytest
from pydantic import ValidationError

from jacobian.math.impartial_games._models import (
    DisjunctiveSumRequest,
)
from jacobian.math.impartial_games._operations import compute_disjunctive_sum
from jacobian.math.impartial_games._tools import TOOLS

# -- helpers -----------------------------------------------------------------

_TERMINAL = {"positions": ["start"], "moves": []}

_SINGLE_MOVE = {
    "positions": ["a", "b"],
    "moves": [{"source": "a", "target": "b"}],
}

# Grundy value 2 at position "a"
_G2 = {
    "positions": ["a", "b", "c", "d"],
    "moves": [
        {"source": "a", "target": "b"},
        {"source": "a", "target": "c"},
        {"source": "c", "target": "d"},
    ],
}

# Grundy value 3 at position "e"
_G3 = {
    "positions": ["e", "f", "g", "h", "i", "j", "k", "l"],
    "moves": [
        {"source": "e", "target": "f"},
        {"source": "e", "target": "g"},
        {"source": "g", "target": "i"},
        {"source": "e", "target": "h"},
        {"source": "h", "target": "j"},
        {"source": "h", "target": "k"},
        {"source": "k", "target": "l"},
    ],
}


class TestDisjunctiveSum:
    def test_two_terminal_games_xor_to_zero(self) -> None:
        request = DisjunctiveSumRequest(
            components=[_TERMINAL, _TERMINAL],
            start_positions=["start", "start"],
        )
        result = compute_disjunctive_sum(request)
        assert result.grundy_value == 0
        assert result.is_p_position is True
        assert result.component_grundy_values == (0, 0)
        assert result.component_count == 2

    def test_terminal_plus_single_move_xor_to_one(self) -> None:
        request = DisjunctiveSumRequest(
            components=[_TERMINAL, _SINGLE_MOVE],
            start_positions=["start", "a"],
        )
        result = compute_disjunctive_sum(request)
        assert result.grundy_value == 1
        assert result.is_p_position is False

    def test_two_single_moves_xor_to_zero(self) -> None:
        request = DisjunctiveSumRequest(
            components=[_SINGLE_MOVE, _SINGLE_MOVE],
            start_positions=["a", "a"],
        )
        result = compute_disjunctive_sum(request)
        assert result.grundy_value == 0
        assert result.is_p_position is True

    def test_grundy_one_two_three_xor_to_zero(self) -> None:
        """1 ^ 2 ^ 3 = 0."""
        request = DisjunctiveSumRequest(
            components=[_SINGLE_MOVE, _G2, _G3],
            start_positions=["a", "a", "e"],
        )
        result = compute_disjunctive_sum(request)
        assert result.component_grundy_values == (1, 2, 3)
        assert result.grundy_value == 0
        assert result.is_p_position is True

    def test_single_component(self) -> None:
        request = DisjunctiveSumRequest(
            components=[_SINGLE_MOVE],
            start_positions=["a"],
        )
        result = compute_disjunctive_sum(request)
        assert result.grundy_value == 1
        assert result.component_count == 1

    def test_result_fields(self) -> None:
        request = DisjunctiveSumRequest(
            components=[_TERMINAL, _SINGLE_MOVE],
            start_positions=["start", "a"],
        )
        result = compute_disjunctive_sum(request)
        assert result.grundy_value == 1
        assert result.component_grundy_values == (0, 1)
        assert result.is_p_position is False
        assert result.component_count == 2


class TestDisjunctiveSumValidation:
    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValidationError, match="equal length"):
            DisjunctiveSumRequest(
                components=[_TERMINAL],
                start_positions=["start", "start"],
            )

    def test_start_position_not_in_game_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not in component"):
            DisjunctiveSumRequest(
                components=[_TERMINAL],
                start_positions=["nonexistent"],
            )

    def test_empty_components_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DisjunctiveSumRequest(components=[], start_positions=[])

    def test_cyclic_component_rejected(self) -> None:
        with pytest.raises(ValidationError, match="acyclic"):
            DisjunctiveSumRequest(
                components=[
                    {
                        "positions": ["a", "b"],
                        "moves": [
                            {"source": "a", "target": "b"},
                            {"source": "b", "target": "a"},
                        ],
                    }
                ],
                start_positions=["a"],
            )


class TestToolRegistration:
    def test_tool_is_registered(self) -> None:
        operation_ids = {t.operation_id for t in TOOLS}
        assert "game.impartial.disjunctive_sum.compute" in operation_ids

    def test_tool_has_correct_tags(self) -> None:
        tool = next(
            t
            for t in TOOLS
            if t.operation_id == "game.impartial.disjunctive_sum.compute"
        )
        assert "disjunctive-sum" in tool.tags
        assert "impartial" in tool.tags
        assert "exact" in tool.tags
        assert "game-theory" in tool.tags
