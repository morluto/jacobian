"""Known-answer tests for nim sum and outcome profile operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.impartial_games._models import NimSumRequest, OutcomeProfileRequest
from jacobian.math.impartial_games._operations import (
    compute_nim_sum,
    compute_outcome_profile,
)
from jacobian.math.impartial_games.values import NimPosition

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
        result = compute_nim_sum(NimSumRequest(position=NimPosition(heaps=())))
        assert result.nim_sum == 0
        assert result.is_p_position is True

    def test_single_heap(self) -> None:
        result = compute_nim_sum(NimSumRequest(position=NimPosition(heaps=(5,))))
        assert result.nim_sum == 5
        assert result.is_p_position is False

    def test_xor_identity(self) -> None:
        result = compute_nim_sum(NimSumRequest(position=NimPosition(heaps=(5, 5))))
        assert result.nim_sum == 0
        assert result.is_p_position is True

    def test_1_2_3_is_zero(self) -> None:
        result = compute_nim_sum(NimSumRequest(position=NimPosition(heaps=(1, 2, 3))))
        assert result.nim_sum == 0
        assert result.is_p_position is True

    def test_1_2_3_4_5(self) -> None:
        result = compute_nim_sum(
            NimSumRequest(position=NimPosition(heaps=(1, 2, 3, 4, 5)))
        )
        assert result.nim_sum == 1

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            NimSumRequest(position=NimPosition(heaps=(-1,)))

    def test_heaps_preserved(self) -> None:
        position = NimPosition(heaps=(3, 7, 11))
        result = compute_nim_sum(NimSumRequest(position=position))
        assert result.position == position

    def test_result_rejects_source_and_decision_mutations(self) -> None:
        result = compute_nim_sum(NimSumRequest(position=NimPosition(heaps=(1, 2, 4))))

        payload = result.model_dump(mode="json")
        payload["position"]["heaps"] = [1, 2, 5]
        with pytest.raises(ValidationError, match="exact xor"):
            type(result).model_validate(payload)

        payload = result.model_dump(mode="json")
        payload["is_p_position"] = True
        with pytest.raises(ValidationError, match="is_p_position"):
            type(result).model_validate(payload)


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
