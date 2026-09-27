"""Tests for bottom-up tree-automaton determinization (#3762)."""

from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.math.logic.automata.tree._models import (
    TreeDeterminizeRequest,
    TreeDeterminizeResult,
)
from jacobian.math.logic.automata.tree._tools import (
    TOOLS,
    compute_tree_automaton_determinize,
)
from jacobian.math.logic.automata.tree.operations import (
    complement_tree_automaton,
    complete_deterministic_tree_automaton,
    determinize_tree_automaton,
    run_tree_automaton,
    verify_determinization,
)
from jacobian.math.logic.automata.tree.values import (
    BottomUpTreeAutomaton,
    DeterministicBottomUpTreeAutomaton,
    RankedTree,
    TreeAutomatonTransition,
)


def _and_automaton() -> BottomUpTreeAutomaton:
    """Nondeterministic automaton for boolean AND-trees with an aliased true."""
    return BottomUpTreeAutomaton(
        state_count=3,
        arity=(0, 0, 2),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(), target_state=1),
            TreeAutomatonTransition(symbol=1, child_states=(), target_state=2),
            TreeAutomatonTransition(symbol=2, child_states=(0, 0), target_state=0),
            TreeAutomatonTransition(symbol=2, child_states=(0, 1), target_state=0),
            TreeAutomatonTransition(symbol=2, child_states=(0, 2), target_state=0),
            TreeAutomatonTransition(symbol=2, child_states=(1, 0), target_state=0),
            TreeAutomatonTransition(symbol=2, child_states=(2, 0), target_state=0),
            TreeAutomatonTransition(symbol=2, child_states=(1, 1), target_state=1),
            TreeAutomatonTransition(symbol=2, child_states=(1, 2), target_state=1),
            TreeAutomatonTransition(symbol=2, child_states=(2, 1), target_state=1),
            TreeAutomatonTransition(symbol=2, child_states=(2, 2), target_state=1),
            TreeAutomatonTransition(symbol=2, child_states=(2, 2), target_state=2),
        ),
        final_states=(1,),
    )


def _blowup_automaton() -> BottomUpTreeAutomaton:
    """Nondeterministic automaton whose subset construction needs 3 states."""
    return BottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 0, 2),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(), target_state=1),
            TreeAutomatonTransition(symbol=2, child_states=(0, 0), target_state=0),
            TreeAutomatonTransition(symbol=2, child_states=(0, 1), target_state=0),
            TreeAutomatonTransition(symbol=2, child_states=(0, 1), target_state=1),
            TreeAutomatonTransition(symbol=2, child_states=(1, 0), target_state=1),
            TreeAutomatonTransition(symbol=2, child_states=(1, 1), target_state=0),
        ),
        final_states=(1,),
    )


def _leaf(symbol: int) -> RankedTree:
    return RankedTree(symbol=symbol, children=())


def _fork(left: RankedTree, right: RankedTree) -> RankedTree:
    return RankedTree(symbol=2, children=(left, right))


class TestKnownAnswer:
    def test_and_trees_determinize_to_two_states(self) -> None:
        result = determinize_tree_automaton(_and_automaton(), 64, 3)

        assert isinstance(result, TreeDeterminizeResult)
        assert result.status == "COMPLETE"
        assert result.truncation_reason == "NONE"
        assert result.subset_map == ((0,), (1, 2))
        assert result.deterministic.state_count == 2
        assert result.deterministic.final_states == (1,)
        assert result.equivalence_claim
        assert result.sample_agreement

    def test_closure_certificate_replays_every_row(self) -> None:
        result = determinize_tree_automaton(_and_automaton(), 64, 3)

        assert result.closure_rows_checked == len(result.deterministic.transitions)
        assert result.closure_rows_checked == 6
        assert result.combos_evaluated == 6

    def test_deterministic_machine_is_deterministic(self) -> None:
        result = determinize_tree_automaton(_and_automaton(), 64, 3)
        keys = [
            (transition.symbol, transition.child_states)
            for transition in result.deterministic.transitions
        ]

        assert len(set(keys)) == len(keys)


class TestPreservationReplay:
    def test_acceptance_agrees_on_hand_trees(self) -> None:
        source = _and_automaton()
        result = determinize_tree_automaton(source, 64, 2)
        source_finals = set(source.final_states)
        deterministic_finals = set(result.deterministic.final_states)
        trees = (
            _leaf(0),
            _leaf(1),
            _fork(_leaf(0), _leaf(0)),
            _fork(_leaf(1), _leaf(1)),
            _fork(_leaf(1), _leaf(0)),
            _fork(_fork(_leaf(1), _leaf(1)), _leaf(1)),
            _fork(_fork(_leaf(1), _leaf(1)), _leaf(0)),
        )

        assert result.sample_trees_checked > len(trees)
        for tree in trees:
            assert bool(run_tree_automaton(source, tree) & source_finals) == bool(
                run_tree_automaton(result.deterministic, tree) & deterministic_finals
            ), tree

    def test_subset_finals_meet_source_finals(self) -> None:
        source = _and_automaton()
        result = determinize_tree_automaton(source, 64, 2)
        source_finals = set(source.final_states)

        assert set(result.deterministic.final_states) == {
            index
            for index, subset in enumerate(result.subset_map)
            if set(subset) & source_finals
        }


class TestBoundary:
    def test_subset_budget_truncation_never_claims_equivalence(self) -> None:
        result = determinize_tree_automaton(_blowup_automaton(), 2, 2)

        assert result.status == "TRUNCATED"
        assert result.truncation_reason == "STATE_BUDGET"
        assert not result.equivalence_claim
        assert not result.sample_agreement
        assert result.sample_trees_checked == 0
        assert result.closure_rows_checked == 0

    def test_full_budget_completes_with_three_subsets(self) -> None:
        result = determinize_tree_automaton(_blowup_automaton(), 64, 2)

        assert result.status == "COMPLETE"
        assert result.subset_map == ((0,), (0, 1), (1,))

    def test_truncated_construction_stays_deterministic(self) -> None:
        result = determinize_tree_automaton(_blowup_automaton(), 2, 2)
        keys = [
            (transition.symbol, transition.child_states)
            for transition in result.deterministic.transitions
        ]

        assert len(set(keys)) == len(keys)
        assert len(result.subset_map) == result.deterministic.state_count

    def test_empty_language_determinizes_to_the_placeholder(self) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=1,
            arity=(0, 2),
            transitions=(
                TreeAutomatonTransition(symbol=1, child_states=(0, 0), target_state=0),
            ),
            final_states=(0,),
        )
        result = determinize_tree_automaton(automaton, 64, 2)

        assert result.status == "COMPLETE"
        assert result.subset_map == ((),)
        assert result.deterministic.final_states == ()
        assert result.sample_trees_checked == 0

    def test_subset_budget_outside_schema_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TreeDeterminizeRequest(
                automaton=_and_automaton(),
                max_subset_states=0,
                sample_max_height=2,
            )

    def test_sample_height_outside_schema_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TreeDeterminizeRequest(
                automaton=_and_automaton(),
                max_subset_states=64,
                sample_max_height=6,
            )


class TestAdversarial:
    def test_malformed_transition_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BottomUpTreeAutomaton.model_validate(
                {
                    "state_count": 1,
                    "arity": [0, 2],
                    "transitions": [
                        {"symbol": 1, "child_states": [0], "target_state": 0},
                    ],
                    "final_states": [0],
                }
            )

    def test_unknown_symbol_transition_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BottomUpTreeAutomaton.model_validate(
                {
                    "state_count": 1,
                    "arity": [0],
                    "transitions": [
                        {"symbol": 4, "child_states": [], "target_state": 0},
                    ],
                    "final_states": [0],
                }
            )


class TestNativeCatalogParity:
    def test_native_kernel_and_owner_adapter_agree(self) -> None:
        request = TreeDeterminizeRequest(
            automaton=_and_automaton(),
            max_subset_states=64,
            sample_max_height=3,
        )

        assert compute_tree_automaton_determinize(request) == (
            determinize_tree_automaton(
                request.automaton,
                request.max_subset_states,
                request.sample_max_height,
            )
        )

    def test_declared_example_executes_through_the_owner_adapter(self) -> None:
        operation = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "tree_automaton.determinize.compute"
        )
        request = operation.request_type.model_validate_json(
            encode_strict_json(operation.examples[0].input), strict=True
        )
        result = operation.run(request)

        assert result.status == "COMPLETE"
        assert result.deterministic.state_count == 2


class TestComposition:
    @staticmethod
    def _partial_source() -> BottomUpTreeAutomaton:
        # A nondeterministic automaton whose determinization is partial: the
        # mixed-subset row f((0,), (1, 2)) has no image.
        return BottomUpTreeAutomaton(
            state_count=3,
            arity=(0, 0, 2),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
                TreeAutomatonTransition(symbol=1, child_states=(), target_state=1),
                TreeAutomatonTransition(symbol=1, child_states=(), target_state=2),
                TreeAutomatonTransition(symbol=2, child_states=(0, 0), target_state=0),
                TreeAutomatonTransition(symbol=2, child_states=(1, 1), target_state=2),
                TreeAutomatonTransition(symbol=2, child_states=(2, 2), target_state=1),
            ),
            final_states=(2,),
        )

    @staticmethod
    def _ground_trees(arity: tuple[int, ...], height: int) -> list[RankedTree]:
        levels: list[list[RankedTree]] = []
        current = [
            RankedTree(symbol=symbol) for symbol, rank in enumerate(arity) if rank == 0
        ]
        levels.append(current)
        trees = list(current)
        for _ in range(height):
            following: list[RankedTree] = []
            for symbol, rank in enumerate(arity):
                if rank == 0:
                    continue
                for children in itertools.product(levels[-1], repeat=rank):
                    node = RankedTree(symbol=symbol, children=children)
                    following.append(node)
                    trees.append(node)
            if not following:
                break
            levels.append(following)
        return trees

    @staticmethod
    def _nfa_accepts(machine: BottomUpTreeAutomaton, tree: RankedTree) -> bool:
        return bool(run_tree_automaton(machine, tree) & set(machine.final_states))

    @staticmethod
    def _partial_dfa_accepts(machine: BottomUpTreeAutomaton, tree: RankedTree) -> bool:
        table = {
            (row.symbol, row.child_states): row.target_state
            for row in machine.transitions
        }

        def state(node: RankedTree) -> int | None:
            children = tuple(state(child) for child in node.children)
            if any(child is None for child in children):
                return None
            return table.get(
                (node.symbol, tuple(child for child in children if child is not None))
            )

        root = state(tree)
        return root is not None and root in machine.final_states

    def test_determinized_result_flows_into_completion_and_complement(
        self,
    ) -> None:
        source = self._partial_source()

        result = determinize_tree_automaton(source, 64, 3)

        assert result.status == "COMPLETE"
        assert isinstance(result.deterministic, DeterministicBottomUpTreeAutomaton)
        completion = complete_deterministic_tree_automaton(result.deterministic)
        assert completion.sink_state is not None
        complemented = complement_tree_automaton(completion.completed)
        decoded = TreeDeterminizeResult.model_validate_json(result.model_dump_json())
        assert isinstance(decoded.deterministic, DeterministicBottomUpTreeAutomaton)
        for tree in self._ground_trees(source.arity, 3):
            source_accepted = self._nfa_accepts(source, tree)
            assert source_accepted == self._partial_dfa_accepts(
                result.deterministic, tree
            ), tree
            assert source_accepted == self._partial_dfa_accepts(
                completion.completed, tree
            ), tree
            assert (
                self._partial_dfa_accepts(complemented.complement, tree)
                is not source_accepted
            ), tree

    def test_truncated_result_carries_the_deterministic_carrier(self) -> None:
        result = determinize_tree_automaton(_blowup_automaton(), 2, 2)

        assert result.status == "TRUNCATED"
        assert isinstance(result.deterministic, DeterministicBottomUpTreeAutomaton)


class TestSerialization:
    def test_strict_json_round_trip_preserves_the_construction(self) -> None:
        result = compute_tree_automaton_determinize(
            TreeDeterminizeRequest(
                automaton=_and_automaton(),
                max_subset_states=64,
                sample_max_height=2,
            )
        )
        decoded = TreeDeterminizeResult.model_validate_json(result.model_dump_json())

        assert decoded == result
        assert verify_determinization(decoded)

    def test_forged_construction_does_not_verify(self) -> None:
        result = compute_tree_automaton_determinize(
            TreeDeterminizeRequest(
                automaton=_and_automaton(),
                max_subset_states=64,
                sample_max_height=2,
            )
        )
        forged = result.model_copy(update={"sample_trees_checked": 0})

        assert not verify_determinization(forged)

    def test_truncated_result_round_trips(self) -> None:
        result = determinize_tree_automaton(_blowup_automaton(), 2, 2)
        decoded = TreeDeterminizeResult.model_validate_json(result.model_dump_json())

        assert decoded == result
        assert decoded.status == "TRUNCATED"
        assert verify_determinization(decoded)
