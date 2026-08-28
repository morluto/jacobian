"""Tests for tree automaton operations."""

from math import comb

import pytest
from pydantic import ValidationError

from jacobian.math.logic.automata.tree import (
    ReachableStateProfile,
    reachable_state_profile,
)
from jacobian.math.logic.automata.tree._models import (
    AcceptedTreeCountRequest,
    TreeAutomatonReachabilityRequest,
    TreeRunRequest,
)
from jacobian.math.logic.automata.tree._operations import (
    compute_accepted_tree_count,
    compute_tree_automaton_reachability,
    compute_tree_run,
)
from jacobian.math.logic.automata.tree._tools import TOOLS
from jacobian.math.logic.automata.tree.operations import (
    accepted_tree_count,
    run_tree_automaton,
)
from jacobian.math.logic.automata.tree.values import (
    MAX_REACHABILITY_WITNESS_NODES,
    MAX_TA_STATES,
    MAX_TA_SYMBOLS,
    MAX_TA_TRANSITIONS,
    MAX_TREE_AUTOMATON_REACHABILITY_WORK,
    BottomUpTreeAutomaton,
    RankedTree,
    TreeAutomatonTransition,
    ranked_tree_node_count,
    reachability_execution_work_bound,
    reachability_price_components,
)


def _assert_validation_code(
    exc: pytest.ExceptionInfo[ValidationError], code: str
) -> None:
    assert exc.value.errors()[0]["type"] == code


# Helpers
def _leaf() -> RankedTree:
    return RankedTree(symbol=0, children=())


def _node(left: RankedTree, right: RankedTree) -> RankedTree:
    return RankedTree(symbol=1, children=(left, right))


def _simple_automaton() -> BottomUpTreeAutomaton:
    """2-state automaton: accepts balanced binary trees."""
    return BottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 2),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0, 0), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(1, 0), target_state=1),
            TreeAutomatonTransition(symbol=1, child_states=(0, 1), target_state=1),
            TreeAutomatonTransition(symbol=1, child_states=(1, 1), target_state=1),
        ),
        final_states=(0,),
    )


def _seeded_chain_with_padded_rows(chain_states: int) -> BottomUpTreeAutomaton:
    """One nullary-seeded chain padded to 4096 rows by unreachable arity-16 rows."""
    transitions = [
        TreeAutomatonTransition(symbol=1, child_states=(), target_state=0),
        *(
            TreeAutomatonTransition(
                symbol=0, child_states=(state,), target_state=state + 1
            )
            for state in range(chain_states - 1)
        ),
        *(
            TreeAutomatonTransition(
                symbol=2,
                child_states=(
                    *(((index // 64**position) % 64) for position in range(15)),
                    63,
                ),
                target_state=index % 64,
            )
            for index in range(4096 - chain_states)
        ),
    ]
    return BottomUpTreeAutomaton(
        state_count=64,
        arity=(1, 0, 16),
        transitions=tuple(transitions),
        final_states=(),
    )


def _nullary_saturated_with_wide_rows() -> BottomUpTreeAutomaton:
    """64 nullary seeds padded to 4096 rows by wide rows over seeded states.

    Every state gains a one-node witness in the first scan and no wide row
    improves anything afterwards, so saturation stabilizes after exactly two
    scans regardless of the number of constructible states.
    """
    transitions = [
        *(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=state)
            for state in range(32)
        ),
        *(
            TreeAutomatonTransition(symbol=2, child_states=(), target_state=state)
            for state in range(32, 64)
        ),
        *(
            TreeAutomatonTransition(
                symbol=1,
                child_states=tuple(
                    (index // 64**position) % 64 for position in range(16)
                ),
                target_state=index % 64,
            )
            for index in range(4032)
        ),
    ]
    return BottomUpTreeAutomaton(
        state_count=64,
        arity=(0, 16, 0),
        transitions=tuple(transitions),
        final_states=(),
    )


class TestRun:
    def test_accepted_leaf(self) -> None:
        automaton = _simple_automaton()
        tree = _leaf()
        states = run_tree_automaton(automaton, tree)
        assert states == {0}

    def test_accepted_balanced_tree(self) -> None:
        automaton = _simple_automaton()
        tree = _node(_leaf(), _leaf())
        states = run_tree_automaton(automaton, tree)
        assert states == {0}

    def test_run_request_accepts(self) -> None:
        automaton = _simple_automaton()
        tree = _node(_leaf(), _leaf())
        result = compute_tree_run(TreeRunRequest(automaton=automaton, tree=tree))
        assert result.accepted is True
        assert result.root_states == (0,)

    def test_run_request_rejects(self) -> None:
        # Use automaton where state 1 is final but not state 0
        automaton = BottomUpTreeAutomaton(
            state_count=2,
            arity=(0, 2),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
                TreeAutomatonTransition(symbol=1, child_states=(1, 1), target_state=0),
            ),
            final_states=(0,),
        )
        # Tree f(a, a) with a -> state 1, f(1, 1) -> state 0 (accepted)
        tree = _node(_leaf(), _leaf())
        result = compute_tree_run(TreeRunRequest(automaton=automaton, tree=tree))
        assert result.accepted is True
        assert result.root_states == (0,)
        # Leaf alone: a -> state 1 (not final, rejected)
        leaf_result = compute_tree_run(
            TreeRunRequest(automaton=automaton, tree=_leaf())
        )
        assert leaf_result.accepted is False
        assert leaf_result.root_states == (1,)

    def test_nondeterministic_run_returns_every_reachable_root_state(self) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=2,
            arity=(0,),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
            ),
            final_states=(1,),
        )

        result = compute_tree_run(TreeRunRequest(automaton=automaton, tree=_leaf()))

        assert result.root_states == (0, 1)
        assert result.state_chart == (((), (0, 1)),)
        assert result.accepted is True
        assert result.node_count == 1
        assert result.complete is True

        payload = result.model_dump()
        payload["state_chart"] = (((), (0,)),)
        forged = type(result).model_validate(payload)
        assert forged.state_chart != result.state_chart

    def test_native_run_rejects_invalid_nested_rank(self) -> None:
        automaton = _simple_automaton()
        invalid_tree = _node(RankedTree(symbol=1), _leaf())

        with pytest.raises(ValueError, match="every tree node"):
            run_tree_automaton(automaton, invalid_tree)


class TestAcceptedTreeCount:
    def test_empty_ranked_alphabet_has_exact_empty_language(self) -> None:
        """The canonical empty alphabet has no ground trees of any positive size."""
        automaton = BottomUpTreeAutomaton(
            state_count=2,
            arity=(),
            transitions=(),
            final_states=(),
        )

        assert (
            BottomUpTreeAutomaton.model_validate(automaton.model_dump(mode="json"))
            == automaton
        )

        count = compute_accepted_tree_count(
            AcceptedTreeCountRequest(automaton=automaton, tree_size=1)
        )
        assert count.count == "0"
        assert count.estimated_work_bound == 0

        profile = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )
        assert profile.reachable_states == ()
        assert profile.unreachable_states == (0, 1)
        assert profile.witnesses == ()

        arity_schema = BottomUpTreeAutomaton.model_json_schema()["properties"]["arity"]
        assert "minItems" not in arity_schema
        assert "empty ranked alphabet" in arity_schema["description"]

    def test_count_size_1(self) -> None:
        automaton = _simple_automaton()
        result = compute_accepted_tree_count(
            AcceptedTreeCountRequest(automaton=automaton, tree_size=1)
        )
        assert result.count == "1"

    def test_count_size_3(self) -> None:
        automaton = _simple_automaton()
        result = compute_accepted_tree_count(
            AcceptedTreeCountRequest(automaton=automaton, tree_size=3)
        )
        assert result.count == "1"

    def test_nondeterminism_counts_trees_not_accepting_runs(self) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=3,
            arity=(0, 1),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
                TreeAutomatonTransition(symbol=1, child_states=(0,), target_state=2),
                TreeAutomatonTransition(symbol=1, child_states=(1,), target_state=2),
            ),
            final_states=(2,),
        )

        assert accepted_tree_count(automaton, 2) == 1

    def test_one_tree_reaching_two_final_states_is_counted_once(self) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=2,
            arity=(0,),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
            ),
            final_states=(0, 1),
        )

        assert accepted_tree_count(automaton, 1) == 1

    def test_distinct_nullary_symbols_are_distinct_trees(self) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=1,
            arity=(0, 0),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
                TreeAutomatonTransition(symbol=1, child_states=(), target_state=0),
            ),
            final_states=(0,),
        )

        assert accepted_tree_count(automaton, 1) == 2

    def test_full_binary_tree_boundary_count_is_complete(self) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=1,
            arity=(0, 2),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
                TreeAutomatonTransition(symbol=1, child_states=(0, 0), target_state=0),
            ),
            final_states=(0,),
        )
        result = compute_accepted_tree_count(
            AcceptedTreeCountRequest(automaton=automaton, tree_size=99)
        )

        assert result.count == str(comb(98, 49) // 50)
        assert result.tree_size == 99
        assert result.complete is True
        assert result.estimated_work_bound <= 2_000_000

    def test_impossible_binary_tree_size_has_exact_zero_count(self) -> None:
        automaton = _simple_automaton()

        result = compute_accepted_tree_count(
            AcceptedTreeCountRequest(automaton=automaton, tree_size=2)
        )

        assert result.count == "0"
        assert result.complete is True


class TestReachableStates:
    def test_native_reachable_state_profile_returns_canonical_value(self) -> None:
        automaton = _simple_automaton()

        profile = reachable_state_profile(automaton)

        assert isinstance(profile, ReachableStateProfile)
        assert profile.automaton == automaton
        assert profile.reachable_states == (0,)
        assert profile.unreachable_states == (1,)
        for state, tree in profile.witnesses:
            assert run_tree_automaton(automaton, tree) == {state}

        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )
        assert isinstance(result, ReachableStateProfile)
        assert result == profile

    def test_no_nullary_seed_has_an_empty_reachable_profile(self) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=2,
            arity=(1,),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(0,), target_state=1),
            ),
            final_states=(1,),
        )

        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )

        assert result.reachable_states == ()
        assert result.unreachable_states == (0, 1)
        assert result.witnesses == ()

    def test_hyperedge_fixed_point_returns_minimum_witnesses(self) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=5,
            arity=(0, 0, 1, 2),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
                TreeAutomatonTransition(symbol=1, child_states=(), target_state=1),
                TreeAutomatonTransition(symbol=3, child_states=(0, 1), target_state=2),
                TreeAutomatonTransition(symbol=2, child_states=(2,), target_state=3),
                TreeAutomatonTransition(symbol=3, child_states=(3, 3), target_state=4),
            ),
            final_states=(4,),
        )

        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )

        assert result.reachable_states == (0, 1, 2, 3, 4)
        assert result.unreachable_states == ()
        assert tuple(ranked_tree_node_count(tree) for _, tree in result.witnesses) == (
            1,
            1,
            3,
            4,
            9,
        )
        for state, tree in result.witnesses:
            assert state in run_tree_automaton(automaton, tree)

    def test_reachability_requires_every_hyperedge_child(self) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=3,
            arity=(0, 2),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
                TreeAutomatonTransition(symbol=1, child_states=(0, 2), target_state=1),
            ),
            final_states=(1,),
        )

        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )

        assert result.reachable_states == (0,)
        assert result.unreachable_states == (1, 2)
        assert tuple(state for state, _ in result.witnesses) == (0,)

    def test_profile_is_independent_of_transition_wire_order(self) -> None:
        transitions = (
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(), target_state=1),
            TreeAutomatonTransition(symbol=2, child_states=(0, 1), target_state=2),
        )
        first = BottomUpTreeAutomaton(
            state_count=3, arity=(0, 0, 2), transitions=transitions, final_states=()
        )
        second = first.model_copy(update={"transitions": tuple(reversed(transitions))})

        first_result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=first)
        )
        second_result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=second)
        )
        assert first_result.reachable_states == second_result.reachable_states
        assert first_result.unreachable_states == second_result.unreachable_states
        assert first_result.witnesses == second_result.witnesses

    def test_witnesses_choose_minimum_node_count_then_canonical_transition(
        self,
    ) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=3,
            arity=(0, 0, 0, 1),
            transitions=(
                TreeAutomatonTransition(symbol=3, child_states=(0,), target_state=1),
                TreeAutomatonTransition(symbol=2, child_states=(), target_state=1),
                TreeAutomatonTransition(symbol=1, child_states=(), target_state=2),
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=2),
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            ),
            final_states=(),
        )

        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )

        assert tuple(ranked_tree_node_count(tree) for _, tree in result.witnesses) == (
            1,
            1,
            1,
        )
        assert tuple(tree.symbol for _, tree in result.witnesses) == (0, 2, 0)

    def test_equal_node_tie_breaks_by_smallest_root_symbol(self) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=2,
            arity=(0, 0),
            transitions=(
                TreeAutomatonTransition(symbol=1, child_states=(), target_state=0),
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
            ),
            final_states=(),
        )

        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )

        assert result.witnesses[0][1] == RankedTree(symbol=0)
        assert result.witnesses[1][1] == RankedTree(symbol=0)

        forged_payload = result.model_dump()
        forged_payload["witnesses"] = [[0, {"symbol": 1, "children": []}], [1, _leaf()]]
        forged = ReachableStateProfile.model_validate(forged_payload)
        assert forged.witnesses != result.witnesses

    def test_equal_node_tie_breaks_by_smallest_child_state_tuple(self) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=3,
            arity=(0, 2),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
                TreeAutomatonTransition(symbol=1, child_states=(1, 0), target_state=2),
                TreeAutomatonTransition(symbol=1, child_states=(0, 1), target_state=2),
            ),
            final_states=(),
        )

        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )

        # Both arity-2 rows derive state 2 with three nodes; (0, 1) < (1, 0).
        assert result.witnesses[2][1] == RankedTree(
            symbol=1,
            children=(RankedTree(symbol=0), RankedTree(symbol=0)),
        )

    def test_witness_schema_publishes_canonical_tie_break(self) -> None:
        witnesses_description = ReachableStateProfile.model_json_schema()["properties"][
            "witnesses"
        ]["description"]

        assert "(symbol, child_states, target_state)" in witnesses_description
        assert "lexicographically smallest" in witnesses_description
        assert "recursively" in witnesses_description

        reachability_tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "tree_automaton.states.reachable.compute"
        )
        assert "lexicographically smallest" in reachability_tool.description
        assert "(symbol, child_states, target_state)" in reachability_tool.description

    def test_result_rejects_forged_reachable_states(self) -> None:
        automaton = _simple_automaton()
        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )
        payload = result.model_dump()
        payload["reachable_states"] = (1,)

        with pytest.raises(ValidationError) as exc:
            ReachableStateProfile.model_validate(payload)
        _assert_validation_code(exc, "tree_automata.states_not_disjoint")

    def test_result_model_accepts_structural_witness_data(self) -> None:
        automaton = _simple_automaton()
        profile = reachable_state_profile(automaton)
        forged_payload = {
            **profile.model_dump(),
            "witnesses": [
                [
                    0,
                    {
                        "symbol": 1,
                        "children": [
                            {"symbol": 0, "children": []},
                            {"symbol": 0, "children": []},
                        ],
                    },
                ]
            ],
        }

        forged = ReachableStateProfile.model_validate(forged_payload)
        assert forged.witnesses != profile.witnesses

    def test_deserialized_profile_is_structural(self) -> None:
        automaton = _simple_automaton()
        profile = reachable_state_profile(automaton)

        payload = profile.model_dump()
        revalidated = ReachableStateProfile.model_validate(payload)

        assert revalidated == profile

    def test_profile_shape_validators_reject_independent_forgeries(self) -> None:
        automaton = _simple_automaton()
        payload = reachable_state_profile(automaton).model_dump()

        unsorted = {**payload, "unreachable_states": ()}
        with pytest.raises(ValidationError) as exc:
            ReachableStateProfile.model_validate(unsorted)
        _assert_validation_code(exc, "tree_automata.states_do_not_partition")

        misaligned = {
            **payload,
            "reachable_states": (),
            "unreachable_states": (0, 1),
        }
        with pytest.raises(ValidationError) as exc:
            ReachableStateProfile.model_validate(misaligned)
        _assert_validation_code(exc, "tree_automata.witnesses_not_aligned")

        foreign_alphabet = {
            **payload,
            "witnesses": [[0, {"symbol": 1, "children": []}]],
        }
        with pytest.raises(ValidationError) as exc:
            ReachableStateProfile.model_validate(foreign_alphabet)
        _assert_validation_code(exc, "tree_automata.witness_arity_mismatch")


class TestValidation:
    def test_reachability_accepts_large_state_domain_with_compact_witnesses(
        self,
    ) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=64,
            arity=(0,),
            transitions=tuple(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=state)
                for state in range(64)
            ),
            final_states=(),
        )

        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )

        assert result.reachable_states == tuple(range(64))
        assert len(result.witnesses) == 64

    @pytest.mark.scale
    def test_nullary_free_child_slot_scans_admit_immediately_stable_profile(
        self,
    ) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=64,
            arity=(16,),
            transitions=tuple(
                TreeAutomatonTransition(
                    symbol=0,
                    child_states=tuple(
                        (index // 64**position) % 64 for position in range(16)
                    ),
                    target_state=index % 64,
                )
                for index in range(4096)
            ),
            final_states=(),
        )

        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )
        profile = reachable_state_profile(automaton)

        assert result.reachable_states == ()
        assert result.unreachable_states == tuple(range(64))
        assert result.witnesses == ()
        assert isinstance(profile, ReachableStateProfile)
        assert profile.automaton == automaton
        assert profile.reachable_states == ()
        assert profile.unreachable_states == tuple(range(64))
        assert profile.witnesses == ()

    @pytest.mark.scale
    def test_nullary_seeded_child_slot_scans_admit_saturated_profile(self) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=64,
            arity=(16, 0),
            transitions=(
                TreeAutomatonTransition(symbol=1, child_states=(), target_state=0),
                *tuple(
                    TreeAutomatonTransition(
                        symbol=0,
                        child_states=tuple(
                            (index // 64**position) % 64 for position in range(16)
                        ),
                        target_state=index % 64,
                    )
                    for index in range(4095)
                ),
            ),
            final_states=(),
        )

        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )
        profile = reachable_state_profile(automaton)

        assert result.reachable_states == (0,)
        assert result.unreachable_states == tuple(range(1, 64))
        assert tuple(ranked_tree_node_count(tree) for _, tree in result.witnesses) == (
            1,
        )
        assert isinstance(profile, ReachableStateProfile)
        assert profile.automaton == automaton
        assert profile.reachable_states == (0,)
        assert profile.unreachable_states == tuple(range(1, 64))

    @pytest.mark.scale
    def test_two_scan_saturation_is_admitted_beyond_constructible_state_count(
        self,
    ) -> None:
        automaton = _nullary_saturated_with_wide_rows()

        _, per_scan_work, scan_rounds = reachability_price_components(automaton)
        assert scan_rounds == 2

        per_profile = reachability_execution_work_bound(automaton)
        assert per_profile == 983_040 + 2 * per_scan_work + 3 * 4096
        assert per_profile == 1_560_832
        assert per_profile <= 30_000_000

        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )

        assert result.reachable_states == tuple(range(64))
        assert result.unreachable_states == ()
        assert (
            tuple(ranked_tree_node_count(tree) for _, tree in result.witnesses)
            == (1,) * 64
        )

        profile = reachable_state_profile(automaton)
        assert isinstance(profile, ReachableStateProfile)
        assert profile.reachable_states == tuple(range(64))

    def test_improvement_lag_stabilizes_within_measured_convergence_bound(self) -> None:
        transitions = [
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0)
        ]
        for state in range(1, 13):
            transitions.append(
                TreeAutomatonTransition(
                    symbol=1,
                    child_states=(0,) * 16,
                    target_state=state,
                )
            )
            transitions.append(
                TreeAutomatonTransition(
                    symbol=2,
                    child_states=(state - 1,),
                    target_state=state,
                )
            )
        automaton = BottomUpTreeAutomaton(
            state_count=13,
            arity=(0, 16, 1),
            transitions=tuple(transitions),
            final_states=(),
        )

        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )

        assert result.reachable_states == tuple(range(13))
        assert tuple(
            ranked_tree_node_count(tree) for _, tree in result.witnesses
        ) == tuple(range(1, 14))

    @pytest.mark.scale
    def test_reachability_admits_seeded_deep_chain_within_one_pass_work_bound(
        self,
    ) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=64,
            arity=(16, 1, 0),
            transitions=(
                TreeAutomatonTransition(symbol=2, child_states=(), target_state=0),
                *tuple(
                    TreeAutomatonTransition(
                        symbol=1,
                        child_states=(state,),
                        target_state=state + 1,
                    )
                    for state in range(63)
                ),
                *tuple(
                    TreeAutomatonTransition(
                        symbol=0,
                        child_states=tuple(
                            (index // 64**position) % 64 for position in range(16)
                        ),
                        target_state=index % 64,
                    )
                    for index in range(4032)
                ),
            ),
            final_states=(),
        )

        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )
        assert result.reachable_states == tuple(range(64))

    @pytest.mark.scale
    def test_native_profile_prices_measured_convergence_within_shared_envelope(
        self,
    ) -> None:
        automaton = _seeded_chain_with_padded_rows(16)

        _, per_scan_work, scan_rounds = reachability_price_components(automaton)
        assert scan_rounds == 17
        per_profile = reachability_execution_work_bound(automaton)
        assert per_profile == 983_040 + 17 * per_scan_work + 3 * 4096
        assert per_profile == 5_855_356
        assert per_profile <= 30_000_000

        profile = reachable_state_profile(automaton)
        assert profile.reachable_states == tuple(range(16))
        assert tuple(
            ranked_tree_node_count(tree) for _, tree in profile.witnesses
        ) == tuple(range(1, 17))

    @pytest.mark.scale
    def test_public_request_reuses_the_execution_envelope(self) -> None:
        automaton = _seeded_chain_with_padded_rows(31)

        _, per_scan_work, scan_rounds = reachability_price_components(automaton)
        assert scan_rounds == 32
        per_profile = reachability_execution_work_bound(automaton)
        assert per_profile == 983_040 + 32 * per_scan_work + 3 * 4096
        assert per_profile <= 30_000_000

        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )
        assert result.reachable_states == tuple(range(31))

    @pytest.mark.scale
    def test_near_envelope_rows_are_priced_exactly_and_admitted_only_when_they_fit(
        self,
    ) -> None:
        """4,096-row chain fixtures stay within the one-pass envelope.

        The public operation executes one sort and measured saturation.  The
        next deeper sibling remains admissible because it no longer pays for
        request-time profile materialization a second time.
        """

        def advertised_public_bound(automaton: BottomUpTreeAutomaton) -> int:
            sort_work, per_scan_work, scan_rounds = reachability_price_components(
                automaton
            )
            return sort_work + scan_rounds * per_scan_work + 3 * 4096

        admitted = _seeded_chain_with_padded_rows(30)
        rejected = _seeded_chain_with_padded_rows(31)

        assert reachability_execution_work_bound(admitted) == 9_831_692
        assert reachability_execution_work_bound(admitted) == (
            advertised_public_bound(admitted)
        )
        assert reachability_execution_work_bound(admitted) <= 30_000_000

        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=admitted)
        )
        assert result.reachable_states == tuple(range(30))

        assert reachability_execution_work_bound(rejected) == 10_114_816
        assert reachability_execution_work_bound(rejected) == (
            advertised_public_bound(rejected)
        )
        assert reachability_execution_work_bound(rejected) <= 30_000_000
        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=rejected)
        )
        assert result.reachable_states == tuple(range(31))

    @pytest.mark.scale
    def test_deepest_native_profile_and_public_operation_fit_one_envelope(self) -> None:
        automaton = _seeded_chain_with_padded_rows(64)

        _, per_scan_work, scan_rounds = reachability_price_components(automaton)
        assert scan_rounds == 65
        per_profile = reachability_execution_work_bound(automaton)
        assert per_profile == 983_040 + 65 * per_scan_work + 3 * 4096
        assert per_profile == 19_390_588
        assert per_profile <= 30_000_000

        profile = reachable_state_profile(automaton)
        assert isinstance(profile, ReachableStateProfile)
        assert profile.reachable_states == tuple(range(64))
        assert tuple(
            ranked_tree_node_count(tree) for _, tree in profile.witnesses
        ) == tuple(range(1, 65))

        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=automaton)
        )
        assert result.reachable_states == tuple(range(64))

    def test_reachability_schema_publishes_coupled_admission_envelope(self) -> None:
        request_schema = TreeAutomatonReachabilityRequest.model_json_schema()
        result_schema = ReachableStateProfile.model_json_schema()

        request_description = request_schema["description"]
        assert "MAX_TREE_AUTOMATON_REACHABILITY_WORK" in request_description
        assert f"{MAX_TREE_AUTOMATON_REACHABILITY_WORK:,}" in request_description
        assert "MAX_REACHABILITY_WITNESS_NODES" in request_description
        assert str(MAX_REACHABILITY_WITNESS_NODES) in request_description
        assert "summed" in request_description

        automaton_description = request_schema["properties"]["automaton"]["description"]
        assert f"{MAX_TA_STATES} states" in automaton_description
        assert f"{MAX_TA_SYMBOLS} ranked symbols" in automaton_description
        assert f"{MAX_TA_TRANSITIONS} unique transitions" in automaton_description
        assert "{MAX_REACHABILITY_WITNESS_NODES}" not in automaton_description
        assert (
            f"{MAX_TREE_AUTOMATON_REACHABILITY_WORK:,} units" in automaton_description
        )
        assert "summed" in automaton_description

        witnesses_description = result_schema["properties"]["witnesses"]["description"]
        assert (
            f"{MAX_REACHABILITY_WITNESS_NODES} nodes summed over all reachable states"
            in witnesses_description
        )

    def test_reachability_rejects_materialized_witnesses_beyond_output_bound(
        self,
    ) -> None:
        transitions = [
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0)
        ]
        transitions.extend(
            TreeAutomatonTransition(
                symbol=1,
                child_states=(state - 1, state - 1),
                target_state=state,
            )
            for state in range(1, 13)
        )
        automaton = BottomUpTreeAutomaton(
            state_count=13,
            arity=(0, 2),
            transitions=tuple(transitions),
            final_states=(),
        )

        request = TreeAutomatonReachabilityRequest(automaton=automaton)
        with pytest.raises(ValueError, match="witness output"):
            compute_tree_automaton_reachability(request)

    def test_arity_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BottomUpTreeAutomaton(
                state_count=1,
                arity=(0, 2),
                transitions=(
                    TreeAutomatonTransition(
                        symbol=0, child_states=(0,), target_state=0
                    ),
                ),
                final_states=(0,),
            )

    def test_symbol_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BottomUpTreeAutomaton(
                state_count=1,
                arity=(0,),
                transitions=(
                    TreeAutomatonTransition(symbol=5, child_states=(), target_state=0),
                ),
                final_states=(0,),
            )

    def test_final_state_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BottomUpTreeAutomaton(
                state_count=1,
                arity=(0,),
                transitions=(),
                final_states=(5,),
            )

    def test_nested_tree_arity_is_validated_before_execution(self) -> None:
        invalid_tree = _node(RankedTree(symbol=1), _leaf())

        request = TreeRunRequest(automaton=_simple_automaton(), tree=invalid_tree)
        with pytest.raises(ValueError, match="arity"):
            compute_tree_run(request)

    def test_duplicate_transitions_are_rejected(self) -> None:
        transition = TreeAutomatonTransition(symbol=0, child_states=(), target_state=0)

        with pytest.raises(ValidationError) as exc:
            BottomUpTreeAutomaton(
                state_count=1,
                arity=(0,),
                transitions=(transition, transition),
                final_states=(0,),
            )
        _assert_validation_code(exc, "tree_automata.transitions_not_unique")

    def test_duplicate_final_states_are_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc:
            BottomUpTreeAutomaton(
                state_count=1,
                arity=(0,),
                transitions=(),
                final_states=(0, 0),
            )
        _assert_validation_code(exc, "tree_automata.final_states_not_unique")

    def test_count_request_rejects_work_beyond_complete_bound(self) -> None:
        automaton = BottomUpTreeAutomaton(
            state_count=6,
            arity=(0, 2),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
                TreeAutomatonTransition(symbol=1, child_states=(0, 0), target_state=0),
            ),
            final_states=(0,),
        )

        request = AcceptedTreeCountRequest(automaton=automaton, tree_size=100)
        with pytest.raises(ValueError, match="work"):
            compute_accepted_tree_count(request)
