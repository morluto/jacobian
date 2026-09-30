"""Context structural admission preserves native and public composition."""

import json

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.logic.automata.tree.contexts import (
    FiniteTreeContext,
    TreeContextFrame,
)
from jacobian.math.logic.automata.tree.operations import (
    map_tree_context_states,
    plug_tree_context_operation,
)
from jacobian.math.logic.automata.tree.values import (
    MAX_RUN_TREE_NODES,
    CompleteDeterministicBottomUpTreeAutomaton,
    RankedTree,
    TreeAutomatonTransition,
)


def test_negative_hole_is_rejected_natively_and_by_public_parsing() -> None:
    leaf = RankedTree(symbol=0)
    frame = TreeContextFrame.model_construct(symbol=1, hole_child=-1, siblings=(leaf,))
    context = FiniteTreeContext.model_construct(arity=(0, 2), frames=(frame,))
    with pytest.raises(OperationDomainValidationError) as native:
        plug_tree_context_operation(context, leaf)
    with pytest.raises(OperationRequestValidationError) as public:
        invoke_operation(
            "ranked_tree.context.plug.compute",
            {
                "context": context.model_dump(mode="json"),
                "tree": leaf.model_dump(mode="json"),
            },
            Catalog.open(),
        )
    assert native.value.errors()[0]["type"] == "tree_context.frame_rank"
    assert public.value.errors()[0]["type"] == "tree_context.frame_rank"


def test_full_node_boundary_context_round_trips_into_state_map() -> None:
    sibling = RankedTree(symbol=0)
    for _ in range(11):
        sibling = RankedTree(symbol=1, children=(sibling, sibling))
    # The 4,095-node binary sibling plus its frame reaches the 4,096-node carrier.
    assert MAX_RUN_TREE_NODES == 2**12
    context = FiniteTreeContext(
        arity=(0, 2),
        frames=(TreeContextFrame(symbol=1, hole_child=0, siblings=(sibling,)),),
    )
    automaton = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=1,
        arity=(0, 2),
        final_states=(0,),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0, 0), target_state=0),
        ),
    )
    context_payload = json.loads(context.model_dump_json())
    revived = FiniteTreeContext.model_validate_json(json.dumps(context_payload))
    native = map_tree_context_states(automaton, revived)
    public = invoke_operation(
        "tree_automaton.context.state_map.compute",
        {"automaton": automaton.model_dump(mode="json"), "context": context_payload},
        Catalog.open(),
    )
    assert native.state_map == (0,)
    assert public.output == native.model_dump(mode="json")
