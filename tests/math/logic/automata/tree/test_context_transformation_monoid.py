from itertools import product
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.tree import (
    CompleteDeterministicBottomUpTreeAutomaton,
    TreeAutomatonTransition,
    map_tree_context_states,
    tree_context_transformation_monoid,
    values,
)
from jacobian.math.logic.automata.tree._models import (
    TreeContextTransformationMonoidResult,
)
from jacobian.math.logic.automata.tree.contexts import (
    FiniteTreeContext,
    TreeContextFrame,
)


def _unary_automaton() -> CompleteDeterministicBottomUpTreeAutomaton:
    return CompleteDeterministicBottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0,), target_state=1),
            TreeAutomatonTransition(symbol=1, child_states=(1,), target_state=1),
        ),
        final_states=(1,),
    )


def test_unary_context_monoid_is_closed_and_has_composable_witnesses() -> None:
    automaton = _unary_automaton()
    result = tree_context_transformation_monoid(automaton, max_elements=8)

    maps = tuple(element.state_map for element in result.elements)
    assert maps == ((0, 1), (1, 1))
    assert maps[result.identity_index] == (0, 1)
    for left_index, left in enumerate(maps):
        for right_index, right in enumerate(maps):
            composed = tuple(left[right[q]] for q in range(automaton.state_count))
            product_index = result.multiplication_table[left_index][right_index]
            assert maps[product_index] == composed
    for element in result.elements:
        assert (
            map_tree_context_states(automaton, element.context).state_map
            == element.state_map
        )
    for depth in range(6):
        context = FiniteTreeContext.model_construct(
            arity=automaton.arity,
            frames=tuple(
                TreeContextFrame.model_construct(symbol=1, hole_child=0, siblings=())
                for _ in range(depth)
            ),
        )
        assert map_tree_context_states(automaton, context).state_map in maps


def test_multibranch_generators_use_reachable_sibling_witnesses() -> None:
    automaton = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 2),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            *(
                TreeAutomatonTransition(
                    symbol=1,
                    child_states=children,
                    target_state=children[0] ^ children[1],
                )
                for children in product(range(2), repeat=2)
            ),
        ),
        final_states=(1,),
    )
    result = tree_context_transformation_monoid(automaton, max_elements=16)
    assert (0, 1) in tuple(element.state_map for element in result.elements)
    for element in result.elements:
        assert (
            map_tree_context_states(automaton, element.context).state_map
            == element.state_map
        )


def test_exact_closure_refuses_when_element_cap_is_too_small() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        tree_context_transformation_monoid(_unary_automaton(), max_elements=1)


def test_monoid_preflights_reachability_before_running_saturation() -> None:
    # 63 states with one binary symbol needs 3970 complete transition rows, so
    # the conservative preflight charge alone exceeds the monoid work envelope
    # even though the exact saturation converges after a couple of scans.
    automaton = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=63,
        arity=(0, 2),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            *(
                TreeAutomatonTransition(
                    symbol=1, child_states=(left, right), target_state=left
                )
                for left in range(63)
                for right in range(63)
            ),
        ),
        final_states=(0,),
    )
    with (
        patch.object(
            values, "_priced_saturation", wraps=values._priced_saturation
        ) as run,
        pytest.raises(OperationResourceAdmissionError) as raised,
    ):
        tree_context_transformation_monoid(automaton, max_elements=8)

    assert raised.value.errors()[0]["type"] == (
        "tree_context.monoid.reachability_work_bound"
    )
    assert run.call_count == 0


def test_monoid_charges_one_reachability_pass_before_closure() -> None:
    with patch.object(
        values, "_priced_saturation", wraps=values._priced_saturation
    ) as run:
        tree_context_transformation_monoid(_unary_automaton(), max_elements=8)

    assert run.call_count == 1


def test_monoid_rejects_witness_contexts_with_mismatched_signature() -> None:
    result = tree_context_transformation_monoid(_unary_automaton(), max_elements=8)
    payload = result.model_dump(mode="json")
    payload["elements"][0]["context"]["arity"] = [0, 1, 1]

    with pytest.raises(ValidationError) as raised:
        TreeContextTransformationMonoidResult.model_validate(payload)

    assert raised.value.errors()[0]["type"] == (
        "tree_automata.context_monoid_signature"
    )
