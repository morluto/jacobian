"""Tree-count admission and serialized resource boundaries."""

from fractions import Fraction
from typing import Any

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.tree._models import AcceptedTreeCountResult
from jacobian.math.logic.automata.tree.operations import (
    accepted_tree_count,
    verify_accepted_tree_count,
)
from jacobian.math.logic.automata.tree.values import (
    BottomUpTreeAutomaton,
    TreeAutomatonTransition,
)


def test_native_tree_count_handles_nullary_sizes_without_allocation() -> None:
    automaton = BottomUpTreeAutomaton(
        state_count=1,
        arity=(0,),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
        ),
        final_states=(0,),
    )
    assert accepted_tree_count(automaton, 0) == 0
    assert accepted_tree_count(automaton, 1) == 1
    assert accepted_tree_count(automaton, 101) == 0
    assert accepted_tree_count(automaton, 10**100) == 0


def test_tree_count_claim_cannot_refute_unadmitted_work() -> None:
    automaton = BottomUpTreeAutomaton(
        state_count=32,
        arity=(0, 2),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0, 0), target_state=0),
        ),
        final_states=(0,),
    )
    with pytest.raises(OperationResourceAdmissionError, match="size"):
        accepted_tree_count(automaton, 101)
    claim = AcceptedTreeCountResult(
        automaton=automaton, tree_size=100, count=0, estimated_work_bound=0
    )
    with pytest.raises(OperationResourceAdmissionError, match="work"):
        verify_accepted_tree_count(claim)


@pytest.mark.parametrize("size", [False, True, 1.5, Fraction(1, 1)])
def test_nullary_tree_count_requires_integer_size(size: Any) -> None:
    from jacobian.math.logic.automata.tree.values import accepted_tree_count_work_bound

    automaton = BottomUpTreeAutomaton(
        state_count=1, arity=(0,), transitions=(), final_states=()
    )
    with pytest.raises(OperationDomainValidationError, match="integer"):
        accepted_tree_count(automaton, size)
    with pytest.raises(OperationDomainValidationError, match="integer"):
        accepted_tree_count_work_bound(automaton, size)
