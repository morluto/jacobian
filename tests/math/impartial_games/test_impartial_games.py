"""Known-answer and adversarial tests for finite impartial games."""

import pytest
from pydantic import ValidationError

from jacobian.math.impartial_games import (
    GameMove,
    ImpartialGame,
    birthdays,
    grundy_classes,
    grundy_table,
    mex,
    nim_options,
    nim_sum,
    outcome_profile,
    position_grundy,
    subtraction_game,
)
from jacobian.math.impartial_games._models import (
    BirthdayRequest,
    BirthdayResult,
    GrundyEntry,
    GrundyTableRequest,
    GrundyTableResult,
    SubtractionGrundyPrefixRequest,
    SubtractionGrundyPrefixResult,
)
from jacobian.math.impartial_games._operations import (
    compute_birthday,
    compute_grundy_table,
    compute_subtraction_grundy_prefix,
)
from jacobian.math.impartial_games._tools import TOOLS


def _game() -> ImpartialGame:
    return ImpartialGame(
        positions=("0", "1", "2", "3"),
        moves=(
            GameMove(source="3", target="2"),
            GameMove(source="3", target="1"),
            GameMove(source="2", target="1"),
            GameMove(source="2", target="0"),
            GameMove(source="1", target="0"),
        ),
    )


class TestGrundyTable:
    def test_known_complete_table(self) -> None:
        result = compute_grundy_table(GrundyTableRequest(game=_game()))

        assert tuple((entry.position, entry.grundy) for entry in result.entries) == (
            ("0", 0),
            ("1", 1),
            ("2", 2),
            ("3", 0),
        )
        assert result.histogram == (2, 1, 1)
        assert result.max_grundy == 2
        assert result.complete is True

    def test_option_grundy_values_are_a_set_not_a_multiset(self) -> None:
        game = ImpartialGame(
            positions=("a", "b", "root"),
            moves=(
                GameMove(source="root", target="a"),
                GameMove(source="root", target="b"),
            ),
        )

        analysis = grundy_table(game)

        assert dict(analysis.option_value_sets)["root"] == (0,)
        assert dict(analysis.values)["root"] == 1

    def test_result_binds_the_complete_source_game(self) -> None:
        with pytest.raises(ValidationError, match="exact complete"):
            GrundyTableResult(
                game=_game(),
                entries=(GrundyEntry(position="0", grundy=0, option_grundy_set=()),),
                max_grundy=0,
                histogram=(1,),
                topological_order=("0",),
            )

    def test_cycle_is_rejected_during_request_model_validation(self) -> None:
        with pytest.raises(ValidationError, match="acyclic"):
            ImpartialGame(
                positions=("a", "b"),
                moves=(
                    GameMove(source="a", target="b"),
                    GameMove(source="b", target="a"),
                ),
            )

    @pytest.mark.parametrize(
        "moves",
        [
            (GameMove(source="a", target="a"),),
            (GameMove(source="a", target="missing"),),
            (
                GameMove(source="a", target="b"),
                GameMove(source="a", target="b"),
            ),
        ],
    )
    def test_invalid_move_relations_fail_closed(
        self, moves: tuple[GameMove, ...]
    ) -> None:
        with pytest.raises(ValidationError):
            ImpartialGame(positions=("a", "b"), moves=moves)


class TestBirthdays:
    def test_known_birthday_table(self) -> None:
        result = compute_birthday(BirthdayRequest(game=_game()))

        assert result.birthdays == (("0", 0), ("1", 1), ("2", 2), ("3", 3))
        assert result.complete is True

    def test_false_birthday_table_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exact complete"):
            BirthdayResult(game=_game(), birthdays=(("0", 0),))


class TestSubtractionGrundyPrefix:
    def test_known_bounded_prefix(self) -> None:
        request = SubtractionGrundyPrefixRequest(subtraction_set=(1, 3), max_heap=5)

        result = compute_subtraction_grundy_prefix(request)

        assert result.grundy_values == (0, 1, 0, 1, 0, 1)
        assert result.option_sets == ((), (0,), (1,), (0,), (1,), (0,))
        assert result.p_positions == (0, 2, 4)
        assert result.n_positions == (1, 3, 5)
        assert result.scope == "HEAPS_ZERO_THROUGH_MAX_HEAP"
        assert result.complete is True

    @pytest.mark.parametrize("values", [(3, 1), (1, 1), (0, 1), (1, 501)])
    def test_subtraction_set_must_be_canonical_and_bounded(
        self, values: tuple[int, ...]
    ) -> None:
        with pytest.raises(ValidationError):
            SubtractionGrundyPrefixRequest(subtraction_set=values, max_heap=5)

    def test_work_bound_is_rejected_before_dynamic_programming(self) -> None:
        with pytest.raises(ValidationError, match="work bound"):
            SubtractionGrundyPrefixRequest(
                subtraction_set=tuple(range(1, 501)), max_heap=500
            )

    def test_false_prefix_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exact complete"):
            SubtractionGrundyPrefixResult(
                subtraction_set=(1,),
                max_heap=1,
                grundy_values=(0, 0),
                option_sets=((), (0,)),
                p_positions=(0, 1),
                n_positions=(),
            )


class TestNativePortfolio:
    def test_projection_helpers_reuse_complete_grundy_semantics(self) -> None:
        game = _game()

        assert position_grundy(game, "2") == 2
        assert outcome_profile(game) == (("0", "3"), ("1", "2"))
        assert grundy_classes(game) == ((0, ("0", "3")), (1, ("1",)), (2, ("2",)))

    def test_nim_helpers_are_exact_and_bounded(self) -> None:
        assert mex((0, 1, 1, 3)) == 2
        assert nim_sum((3, 4, 5)) == 2
        assert nim_options((2,)) == ((0,), (1,))

        with pytest.raises(ValueError, match="output bound"):
            nim_options((5001,))

    def test_subtraction_dag_fails_closed_at_move_bound(self) -> None:
        with pytest.raises(ValueError, match="move bound"):
            subtraction_game(tuple(range(1, 101)), 100)

    def test_only_five_audited_outcomes_are_public(self) -> None:
        assert {tool.operation_id for tool in TOOLS} == {
            "game.impartial.birthday.compute",
            "game.impartial.grundy_table.compute",
            "game.subtraction.grundy_prefix.compute",
            "game.nim.nim_sum.compute",
            "game.impartial.outcome_profile.compute",
        }

    def test_native_birthdays_equal_public_kernel(self) -> None:
        assert birthdays(_game()) == (("0", 0), ("1", 1), ("2", 2), ("3", 3))
