"""Tests for impartial combinatorial game operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.impartial_games._models import (
    BirthdayRequest,
    GameMove,
    GrundyClassesRequest,
    GrundyTableRequest,
    ImpartialGameDAGRequest,
    MexRequest,
    NimEquivalentRequest,
    NimOptionsRequest,
    NimSumRequest,
    OutcomeProfileRequest,
    PositionGrundyRequest,
    SubtractionDAGRequest,
    SubtractionGrundyPrefixRequest,
)
from jacobian.math.impartial_games._operations import (
    compute_birthday,
    compute_grundy_classes,
    compute_grundy_table,
    compute_mex,
    compute_nim_equivalent,
    compute_nim_options,
    compute_nim_sum,
    compute_outcome_profile,
    compute_position_grundy,
    compute_subtraction_dag,
    compute_subtraction_grundy_prefix,
)


def _chain_game() -> ImpartialGameDAGRequest:
    """A simple chain: 3->2, 3->1, 2->1, 2->0, 1->0.

    Grundy values: 0->0, 1->1, 2->2, 3->mex(1,2)=0.
    """
    return ImpartialGameDAGRequest(
        positions=("0", "1", "2", "3"),
        moves=(
            GameMove(source="3", target="2"),
            GameMove(source="3", target="1"),
            GameMove(source="2", target="1"),
            GameMove(source="2", target="0"),
            GameMove(source="1", target="0"),
        ),
    )


def _pure_chain() -> ImpartialGameDAGRequest:
    """Pure chain: 3->2->1->0. Grundy values: 0,1,2,3."""
    return ImpartialGameDAGRequest(
        positions=("0", "1", "2", "3"),
        moves=(
            GameMove(source="3", target="2"),
            GameMove(source="2", target="1"),
            GameMove(source="1", target="0"),
        ),
    )


class TestMex:
    def test_simple(self):
        req = MexRequest(values=[0, 1, 3, 4])
        result = compute_mex(req)
        assert result.mex == 2
        assert result.membership_prefix == (0, 1)
        assert result.first_gap == 2

    def test_empty(self):
        req = MexRequest(values=[])
        result = compute_mex(req)
        assert result.mex == 0

    def test_contiguous(self):
        req = MexRequest(values=[0, 1, 2, 3])
        result = compute_mex(req)
        assert result.mex == 4

    def test_with_duplicates(self):
        req = MexRequest(values=[0, 0, 1, 1, 3])
        result = compute_mex(req)
        assert result.mex == 2

    def test_rejects_negative(self):
        with pytest.raises(ValidationError, match="non-negative"):
            MexRequest(values=[-1])


class TestGrundyTable:
    def test_pure_chain(self):
        """A pure chain 3->2->1->0 has Grundy values [0, 1, 0, 1] (alternating)."""
        req = GrundyTableRequest(game=_pure_chain())
        result = compute_grundy_table(req)
        grundy_map = {e.position: e.grundy for e in result.entry_map}
        assert grundy_map["0"] == 0
        assert grundy_map["1"] == 1
        assert grundy_map["2"] == 0
        assert grundy_map["3"] == 1
        assert result.max_grundy == 1

    def test_chain_with_branch(self):
        """Chain with branch: 3->2, 3->1, 2->1, 2->0, 1->0.
        Grundy(0)=0, Grundy(1)=1, Grundy(2)=mex(1,0)=2, Grundy(3)=mex(2,1)=0."""
        req = GrundyTableRequest(game=_chain_game())
        result = compute_grundy_table(req)
        grundy_map = {e.position: e.grundy for e in result.entry_map}
        assert grundy_map["0"] == 0
        assert grundy_map["1"] == 1
        assert grundy_map["2"] == 2
        assert grundy_map["3"] == 0
        assert result.max_grundy == 2

    def test_terminal_positions(self):
        """Terminal positions have Grundy value 0."""
        req = GrundyTableRequest(game=_chain_game())
        result = compute_grundy_table(req)
        assert result.entry_map[0].grundy == 0  # position "0"
        assert result.entry_map[0].option_grundy_set == ()

        # Find position "3" entry
        pos3 = next(e for e in result.entry_map if e.position == "3")
        assert set(pos3.option_grundy_set) == {1, 2}

    def test_histogram(self):
        req = GrundyTableRequest(game=_chain_game())
        result = compute_grundy_table(req)
        # values are 0, 1, 2, 0 -> histogram = [2, 1, 1]
        assert result.histogram == (2, 1, 1)

    def test_topological_order(self):
        req = GrundyTableRequest(game=_chain_game())
        result = compute_grundy_table(req)
        # In topological order, "0" (terminal) should come last
        order = result.topological_order
        # topo sort goes from source to sink
        assert order.index("0") > order.index("1")
        assert order.index("1") > order.index("2")

    def test_diamond_game(self):
        """Diamond: 3->2, 3->1, 2->0, 1->0. Grundy(0)=0, Grundy(1)=1,
        Grundy(2)=1, Grundy(3)=mex(1,1)=0."""
        moves = (
            GameMove(source="3", target="2"),
            GameMove(source="3", target="1"),
            GameMove(source="2", target="0"),
            GameMove(source="1", target="0"),
        )
        req = GrundyTableRequest(
            game=ImpartialGameDAGRequest(
                positions=("0", "1", "2", "3"), moves=moves
            )
        )
        result = compute_grundy_table(req)
        grundy_map = {e.position: e.grundy for e in result.entry_map}
        assert grundy_map["0"] == 0
        assert grundy_map["1"] == 1
        assert grundy_map["2"] == 1
        assert grundy_map["3"] == 0

    def test_rejects_cycle(self):
        moves = (
            GameMove(source="a", target="b"),
            GameMove(source="b", target="a"),
        )
        req = GrundyTableRequest(
            game=ImpartialGameDAGRequest(positions=("a", "b"), moves=moves)
        )
        with pytest.raises(ValueError, match="acyclic"):
            compute_grundy_table(req)


class TestPositionGrundy:
    def test_single_position(self):
        req = PositionGrundyRequest(game=_pure_chain(), position="3")
        result = compute_position_grundy(req)
        assert result.grundy == 1
        assert "3" in result.reachable_positions

    def test_terminal(self):
        req = PositionGrundyRequest(game=_chain_game(), position="0")
        result = compute_position_grundy(req)
        assert result.grundy == 0


class TestOutcomeProfile:
    def test_chain_partition(self):
        """Chain game: positions 0 and 3 are P (Grundy 0), 1 and 2 are N."""
        req = OutcomeProfileRequest(game=_chain_game())
        result = compute_outcome_profile(req)
        assert "0" in result.p_positions
        assert "3" in result.p_positions
        assert "1" in result.n_positions
        assert "2" in result.n_positions
        assert "0" in result.terminal_positions


class TestNimEquivalent:
    def test_pure_chain(self):
        req = NimEquivalentRequest(game=_pure_chain(), position="3")
        result = compute_nim_equivalent(req)
        assert result.heap_size == 1
        assert result.position == "3"


class TestGrundyClasses:
    def test_pure_chain(self):
        req = GrundyClassesRequest(game=_pure_chain())
        result = compute_grundy_classes(req)
        # Pure chain has 2 classes: {0,2} with grundy 0, {1,3} with grundy 1
        assert len(result.classes) == 2
        grundy_0 = next(c for c in result.classes if c.grundy == 0)
        assert set(grundy_0.positions) == {"0", "2"}
        grundy_1 = next(c for c in result.classes if c.grundy == 1)
        assert set(grundy_1.positions) == {"1", "3"}

    def test_chain_with_branch(self):
        """Chain game: positions 0 and 3 have Grundy 0."""
        req = GrundyClassesRequest(game=_chain_game())
        result = compute_grundy_classes(req)
        # Should have 3 classes: {0, 3} with grundy 0, {1} with grundy 1, {2} with grundy 2
        grundy_0 = next(c for c in result.classes if c.grundy == 0)
        assert set(grundy_0.positions) == {"0", "3"}


class TestBirthday:
    def test_pure_chain(self):
        req = BirthdayRequest(game=_pure_chain())
        result = compute_birthday(req)
        bdays = dict(result.birthdays)
        assert bdays["0"] == 0
        assert bdays["1"] == 1
        assert bdays["2"] == 2
        assert bdays["3"] == 3


class TestNimSum:
    def test_simple(self):
        req = NimSumRequest(heaps=(3, 4, 5))
        result = compute_nim_sum(req)
        assert result.nim_sum == 3 ^ 4 ^ 5
        assert result.is_p_position is False

    def test_p_position(self):
        req = NimSumRequest(heaps=(1, 1))
        result = compute_nim_sum(req)
        assert result.nim_sum == 0
        assert result.is_p_position is True

    def test_single_heap(self):
        req = NimSumRequest(heaps=(7,))
        result = compute_nim_sum(req)
        assert result.nim_sum == 7

    def test_zero_heaps(self):
        req = NimSumRequest(heaps=(0, 0))
        result = compute_nim_sum(req)
        assert result.nim_sum == 0
        assert result.is_p_position is True

    def test_rejects_negative(self):
        with pytest.raises(ValidationError, match="non-negative"):
            NimSumRequest(heaps=(-1,))


class TestNimOptions:
    def test_simple(self):
        req = NimOptionsRequest(heaps=(2, 1))
        result = compute_nim_options(req)
        # heap 0: can go to 0 or 1; heap 1: can go to 0
        # heap 0 -> 0: (0, 1)
        # heap 0 -> 1: (1, 1)
        # heap 1 -> 0: (2, 0)
        assert len(result.options) == 3

    def test_zero_heap(self):
        req = NimOptionsRequest(heaps=(0,))
        result = compute_nim_options(req)
        assert len(result.options) == 0

    def test_option_correctness(self):
        req = NimOptionsRequest(heaps=(3, 2))
        result = compute_nim_options(req)
        for opt in result.options:
            assert opt.new_size < opt.old_size
            assert opt.resulting_heaps[opt.heap_index] == opt.new_size


class TestSubtractionDAG:
    def test_simple(self):
        req = SubtractionDAGRequest(subtraction_set=(1, 3), max_heap=5)
        result = compute_subtraction_dag(req)
        assert len(result.positions) == 6
        # Position 0 has no moves (can't subtract 1 or 3)
        assert "0" in result.terminal_positions

    def test_moves(self):
        req = SubtractionDAGRequest(subtraction_set=(1, 3), max_heap=5)
        result = compute_subtraction_dag(req)
        # From position 3: can go to 2 (subtract 1) and 0 (subtract 3)
        sources = [m.source for m in result.moves if m.source == "3"]
        assert set(sources) == {"3"}
        targets_3 = {m.target for m in result.moves if m.source == "3"}
        assert targets_3 == {"2", "0"}


class TestSubtractionGrundyPrefix:
    def test_sub_1(self):
        """Subtraction set {1}: g(i) = i mod 2."""
        req = SubtractionGrundyPrefixRequest(subtraction_set=(1,), max_heap=10)
        result = compute_subtraction_grundy_prefix(req)
        for i in range(11):
            assert result.grundy_values[i] == i % 2

    def test_sub_1_3(self):
        """Subtraction set {1, 3}: known sequence g = 0, 1, 0, 1, 0, 1, ..."""
        req = SubtractionGrundyPrefixRequest(subtraction_set=(1, 3), max_heap=10)
        result = compute_subtraction_grundy_prefix(req)
        for i in range(11):
            assert result.grundy_values[i] in (0, 1)

    def test_sub_2_3(self):
        """Subtraction set {2, 3}: g = 0, 0, 1, 1, 2, 0, 3, ..."""
        req = SubtractionGrundyPrefixRequest(subtraction_set=(2, 3), max_heap=10)
        result = compute_subtraction_grundy_prefix(req)
        assert result.grundy_values[0] == 0
        assert result.grundy_values[1] == 0
        assert result.grundy_values[2] == 1
        assert result.grundy_values[3] == 1

    def test_p_n_positions(self):
        req = SubtractionGrundyPrefixRequest(subtraction_set=(1,), max_heap=5)
        result = compute_subtraction_grundy_prefix(req)
        for p in result.p_positions:
            assert result.grundy_values[p] == 0
        for n in result.n_positions:
            assert result.grundy_values[n] > 0


class TestValidation:
    def test_duplicate_positions(self):
        with pytest.raises(ValidationError, match="unique"):
            ImpartialGameDAGRequest(positions=("a", "a"), moves=())

    def test_self_loop(self):
        with pytest.raises(ValidationError, match="self-loops"):
            ImpartialGameDAGRequest(
                positions=("a", "b"),
                moves=(GameMove(source="a", target="a"),),
            )

    def test_duplicate_move(self):
        with pytest.raises(ValidationError, match="duplicate"):
            ImpartialGameDAGRequest(
                positions=("a", "b"),
                moves=(
                    GameMove(source="a", target="b"),
                    GameMove(source="a", target="b"),
                ),
            )

    def test_undeclared_target(self):
        with pytest.raises(ValidationError, match="target"):
            ImpartialGameDAGRequest(
                positions=("a",),
                moves=(GameMove(source="a", target="b"),),
            )

    def test_position_not_in_game(self):
        with pytest.raises(ValidationError, match="not in the game"):
            PositionGrundyRequest(game=_chain_game(), position="99")
