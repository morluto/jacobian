"""Known-answer tests for nim sum and outcome profile operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.impartial_games._models import NimSumRequest, OutcomeProfileRequest
from jacobian.math.impartial_games._operations import (
    compute_nim_sum,
    compute_outcome_profile,
)

_GAME = {
    "positions": ["0", "1", "2", "3"],
    "moves": [
        {"source": "3", "target": "2"},
        {"source": "3", "target": "1"},
        {"source": "2", "target": "1"},
        {"source": "2", "target": "0"},
        {"source": "1", "target": "0"},
    ],
}


class TestNimSum:
    def test_empty_heaps(self) -> None:
        result = compute_nim_sum(NimSumRequest(heaps=()))
        assert result.nim_sum == 0
        assert result.is_p_position is True

    def test_single_heap(self) -> None:
        result = compute_nim_sum(NimSumRequest(heaps=(5,)))
        assert result.nim_sum == 5
        assert result.is_p_position is False

    def test_xor_identity(self) -> None:
        result = compute_nim_sum(NimSumRequest(heaps=(5, 5)))
        assert result.nim_sum == 0
        assert result.is_p_position is True

    def test_1_2_3_is_zero(self) -> None:
        result = compute_nim_sum(NimSumRequest(heaps=(1, 2, 3)))
        assert result.nim_sum == 0
        assert result.is_p_position is True

    def test_1_2_3_4_5(self) -> None:
        result = compute_nim_sum(NimSumRequest(heaps=(1, 2, 3, 4, 5)))
        assert result.nim_sum == 1

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError, match="nonnegative"):
            NimSumRequest(heaps=(-1,))

    def test_heaps_preserved(self) -> None:
        heaps = (7, 3, 11)
        result = compute_nim_sum(NimSumRequest(heaps=heaps))
        assert result.heaps == heaps


class TestOutcomeProfile:
    def test_p_positions(self) -> None:
        request = OutcomeProfileRequest(game=_GAME)
        result = compute_outcome_profile(request)
        assert "0" in result.p_positions
        assert "3" in result.p_positions

    def test_n_positions(self) -> None:
        request = OutcomeProfileRequest(game=_GAME)
        result = compute_outcome_profile(request)
        assert "1" in result.n_positions
        assert "2" in result.n_positions

    def test_terminal_position(self) -> None:
        request = OutcomeProfileRequest(game=_GAME)
        result = compute_outcome_profile(request)
        assert "0" in result.terminal_positions

    def test_grundy_values(self) -> None:
        request = OutcomeProfileRequest(game=_GAME)
        result = compute_outcome_profile(request)
        grundy_map = dict(result.grundy_values)
        assert grundy_map["0"] == 0
        assert grundy_map["1"] == 1
        assert grundy_map["2"] == 2
        assert grundy_map["3"] == 0
