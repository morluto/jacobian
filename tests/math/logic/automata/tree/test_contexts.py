from itertools import product

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.tree.contexts import (
    FiniteTreeContext,
    TreeContextFrame,
)
from jacobian.math.logic.automata.tree.operations import (
    map_tree_context_states,
    plug_tree_context_operation,
)
from jacobian.math.logic.automata.tree.values import (
    CompleteDeterministicBottomUpTreeAutomaton,
    RankedTree,
    TreeAutomatonTransition,
)


def test_context_accepts_canonical_frames_and_sibling_values():
    sibling = RankedTree(symbol=0)
    frame = TreeContextFrame(symbol=1, hole_child=0, siblings=(sibling,))
    context = FiniteTreeContext(arity=(0, 2), frames=(frame,))
    assert context.frames == (frame,)
    assert context.frames[0].siblings == (sibling,)


def test_context_rejects_nested_arity_entries_before_canonicalization():
    with pytest.raises(
        ValueError, match="context arities must be in the supported range"
    ):
        FiniteTreeContext.model_validate({"arity": [[0] * 10000], "frames": []})


def test_context_plugging_preserves_ranked_tree_structure():
    context = FiniteTreeContext.model_validate(
        {
            "arity": [0, 1, 2],
            "frames": [
                {
                    "symbol": 2,
                    "hole_child": 1,
                    "siblings": [
                        {"symbol": 1, "children": [{"symbol": 0, "children": []}]},
                    ],
                },
                {"symbol": 1, "hole_child": 0, "siblings": []},
            ],
        }
    )
    leaf = RankedTree(symbol=0, children=())
    result = plug_tree_context_operation(context, leaf)
    assert result.plugged_tree == RankedTree(
        symbol=2,
        children=(
            RankedTree(symbol=1, children=(leaf,)),
            RankedTree(symbol=1, children=(leaf,)),
        ),
    )


def test_plug_admits_result_depth_using_hole_path_and_fixed_siblings():
    sibling = RankedTree(symbol=0)
    for _ in range(126):
        sibling = RankedTree(symbol=1, children=(sibling,))
    context = FiniteTreeContext(
        arity=(0, 1, 2),
        frames=({"symbol": 2, "hole_child": 0, "siblings": (sibling,)},),
    )
    result = plug_tree_context_operation(context, RankedTree(symbol=0))
    assert result.plugged_tree.children[1] == sibling


def test_induced_state_map_matches_independent_direct_evaluation():
    # Complete DTA over three constants and a binary `f`, with transition
    # f(x,y) = (x + 2y) mod 3. The oracle plugs each state at the hole and
    # evaluates bottom-up independently from the context-spine algorithm.
    transitions = [
        TreeAutomatonTransition(symbol=i, child_states=(), target_state=i)
        for i in range(3)
    ]
    transitions += [
        TreeAutomatonTransition(
            symbol=3, child_states=(x, y), target_state=(x + 2 * y) % 3
        )
        for x, y in product(range(3), repeat=2)
    ]
    machine = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=3,
        arity=(0, 0, 0, 2),
        transitions=tuple(transitions),
        final_states=(0,),
    )
    context = FiniteTreeContext(
        arity=(0, 0, 0, 2),
        frames=(
            {"symbol": 3, "hole_child": 1, "siblings": (RankedTree(symbol=0),)},
            {"symbol": 3, "hole_child": 0, "siblings": (RankedTree(symbol=0),)},
        ),
    )
    actual = map_tree_context_states(machine, context).state_map

    table = {
        (row.symbol, row.child_states): row.target_state for row in machine.transitions
    }

    class Hole:
        def __init__(self, state):
            self.state = state

    def direct_context_value(state):
        tree = Hole(state)
        for frame in reversed(context.frames):
            siblings = iter(frame.siblings)
            children = tuple(
                tree if child_index == frame.hole_child else next(siblings)
                for child_index in range(context.arity[frame.symbol])
            )
            tree = (frame.symbol, children)

        def eval_with_hole(node):
            if isinstance(node, Hole):
                return node.state
            if isinstance(node, RankedTree):
                symbol, children = node.symbol, node.children
            else:
                symbol, children = node
            return table[(symbol, tuple(eval_with_hole(child) for child in children))]

        return eval_with_hole(tree)

    direct = tuple(direct_context_value(state) for state in range(3))
    assert actual == direct


def test_context_rank_mismatch_and_plug_growth_are_rejected():
    with pytest.raises(ValueError):
        FiniteTreeContext(
            arity=(0, 2),
            frames=({"symbol": 1, "hole_child": 1, "siblings": ()},),
        )
    oversized = RankedTree(symbol=0, children=())
    # The 127-frame context plus a depth-2 input exceeds the depth envelope.
    large_context = FiniteTreeContext(
        arity=(0, 1),
        frames=({"symbol": 1, "hole_child": 0, "siblings": ()},) * 127,
    )
    tree = RankedTree(symbol=1, children=(oversized,))
    with pytest.raises(OperationResourceAdmissionError):
        plug_tree_context_operation(large_context, tree)


def test_native_operations_revalidate_forged_nested_values():
    forged_context = FiniteTreeContext.model_construct(
        arity=(0, 2),
        frames=(TreeContextFrame.model_construct(symbol=1, hole_child=0, siblings=()),),
    )
    with pytest.raises(OperationDomainValidationError):
        plug_tree_context_operation(forged_context, RankedTree(symbol=0))

    machine = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=1,
        arity=(0,),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
        ),
        final_states=(0,),
    )
    with pytest.raises(OperationDomainValidationError):
        map_tree_context_states(machine, forged_context)


def test_context_json_admission_bounds_tree_growth_before_parsing():
    sibling = {"symbol": 0, "children": []}
    for _ in range(128):
        sibling = {"symbol": 1, "children": [sibling]}
    with pytest.raises(ValueError, match="context exceeds the supported tree depth"):
        FiniteTreeContext.model_validate(
            {
                "arity": [0, 1],
                "frames": [{"symbol": 1, "hole_child": 0, "siblings": [sibling]}],
            }
        )


def test_native_tree_and_context_bounds_run_before_serialization(monkeypatch):
    tree = RankedTree.model_construct(symbol=0, children=())
    for _ in range(128):
        tree = RankedTree.model_construct(symbol=1, children=(tree,))
    context = FiniteTreeContext(arity=(0, 1), frames=())

    def serialization_must_not_run(*_args, **_kwargs):
        raise AssertionError("oversized value was serialized before admission")

    monkeypatch.setattr(RankedTree, "model_dump", serialization_must_not_run)
    with pytest.raises(OperationResourceAdmissionError):
        plug_tree_context_operation(context, tree)


def test_state_map_bounds_automaton_rows_before_serialization(monkeypatch):
    transition = TreeAutomatonTransition.model_construct(
        symbol=0, child_states=(), target_state=0
    )
    automaton = CompleteDeterministicBottomUpTreeAutomaton.model_construct(
        state_count=1,
        arity=(0,),
        transitions=(transition,) * 4097,
        final_states=(0,),
    )
    context = FiniteTreeContext(arity=(0,), frames=())

    def serialization_must_not_run(*_args, **_kwargs):
        raise AssertionError("oversized value was serialized before admission")

    monkeypatch.setattr(
        CompleteDeterministicBottomUpTreeAutomaton,
        "model_dump",
        serialization_must_not_run,
    )
    with pytest.raises(OperationResourceAdmissionError):
        map_tree_context_states(automaton, context)

    sibling = RankedTree.model_construct(symbol=0, children=())
    for _ in range(128):
        sibling = RankedTree.model_construct(symbol=1, children=(sibling,))
    oversized_context = FiniteTreeContext.model_construct(
        arity=(0, 1, 2),
        frames=(
            TreeContextFrame.model_construct(
                symbol=2, hole_child=0, siblings=(sibling,)
            ),
        ),
    )
    monkeypatch.setattr(FiniteTreeContext, "model_dump", serialization_must_not_run)
    with pytest.raises(OperationResourceAdmissionError):
        plug_tree_context_operation(oversized_context, RankedTree(symbol=0))
